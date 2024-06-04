"""
ComparativeEncoder module, trains a model comparatively using distances.
"""
import os
import shutil
import pickle
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
import tensorflow.math as M

from .encoders import ModelBuilder
from .distance import Hyperbolic, Euclidean, Cosine


def _run_tf_fn(init_message=None, print_time=False):
    def fit_dec(fn):
        def fit_output_mgmt(self, *args, **kwargs):
            start_time = time.time()
            if self.quiet:
                tf.keras.utils.disable_interactive_logging()
            elif init_message:
                print(init_message)
            result = fn(self, *args, **kwargs)
            if self.quiet:
                tf.keras.utils.enable_interactive_logging()
            elif print_time:
                print(f'Total time taken: {time.time() - start_time} seconds.')
            return result
        return fit_output_mgmt
    return fit_dec


def _prepare_tf_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y) if y is not None else x)
    dataset = dataset.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    return dataset


class ComparativeModel:
    """
    Abstract ComparativeModel class. Stores some useful common functions.
    """
    def __init__(self, v_scope='model', dist=None, embed_dist='euclidean', model=None,
                 strategy=None, history=None, quiet=False, properties=None, random_seed=None,
                 **kwargs):
        self.strategy = strategy or tf.distribute.get_strategy()
        self.distance = dist
        self.quiet = quiet
        self.properties = {} if properties is None else properties
        self.history = history or {}
        self.embed_dist = embed_dist
        self.rng = np.random.default_rng(seed=random_seed)
        with tf.name_scope(v_scope):
            with self.strategy.scope():
                self.model = model or self.create_model(**kwargs)

    def create_model(self):
        """
        Create a model. Scopes automatically applied.
        """
        return None

    def select_strategy(self, strategy):
        """
        Select either the given strategy or the default strategy.
        """
        return strategy or tf.distribute.get_strategy()

    # Subclass must override
    def train_step(self) -> dict:
        """
        Single epoch of training.
        """
        return {}

    def random_set(self, x: np.ndarray, y: np.ndarray, epoch_factor=1) -> tuple[np.ndarray]:
        p1 = np.concatenate([self.rng.permutation(x.shape[0]) for _ in range(epoch_factor)])
        p2 = np.concatenate([self.rng.permutation(x.shape[0]) for _ in range(epoch_factor)])
        breakpoint()
        return x[p1], x[p2], y[p1], y[p2]

    @_run_tf_fn(print_time=True)
    def fit(self, *args, epochs=100, early_stop=True, min_delta=0, patience=3, **kwargs):
        """
        Train the model based on the given parameters. Extra arguments are passed to train_step.
        @param epochs: epochs to train for.
        @param min_delta: Minimum change required to qualify as an improvement.
        @param patience: How many epochs with no improvement before giving up. patience=0 disables.
        """
        patience = patience or epochs + 1  # If patience==0, do not early stop
        if patience < 1:
            raise ValueError('Patience value must be >1.')
        wait = 0
        best_weights = self.model.get_weights()
        for i in range(epochs):
            start = time.time()
            if not self.quiet:
                print(f'Epoch {i + 1}:')
            this_history = self.train_step(*args, **kwargs)
            if not self.quiet:
                print(f'Epoch time: {time.time() - start}')
            self.history = {k: v + this_history[k] for k, v in self.history.items()} if \
                self.history else this_history
            if not early_stop or i == 0:
                continue
            prev_best = min(self.history['loss'][:-1])
            this_loss = self.history['loss'][-1]
            if this_loss < prev_best - min_delta:
                best_weights = self.model.get_weights()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                print('Stopping early due to lack of improvement!')
                self.model.set_weights(best_weights)
                break
        return self.history

    def transform(self, data: np.ndarray):
        """
        Transform the given data.
        """
        return data

    def save(self, path: str, model=None):
        """
        Save the model to the given path.
        @param path: path to save to.
        """
        try:
            os.makedirs(path)
        except FileExistsError:
            print("WARN: Directory exists, overwriting...")
            shutil.rmtree(path)
            os.makedirs(path)
        model = model or self.model
        model.save(os.path.join(path, 'model.h5'))
        with open(os.path.join(path, 'distance.pkl'), 'wb') as f:
            pickle.dump(self.distance, f)
        with open(os.path.join(path, 'embed_dist.txt'), 'w') as f:
            f.write(self.embed_dist)
        if self.history:
            with open(os.path.join(path, 'history.json'), 'w') as f:
                json.dump(self.history, f)

    @classmethod
    def load(cls, path: str, v_scope: str, strategy=None, model=None, **kwargs):
        """
        Load the model from the filesystem.
        """
        contents = os.listdir(path)
        if not model:
            if 'model.h5' not in contents:
                raise ValueError('Model save file is necessary for loading a ComparativeModel!')
            strategy = strategy or tf.distribute.get_strategy()
            with tf.name_scope(v_scope):
                with strategy.scope():
                    model = tf.keras.models.load_model(os.path.join(path, 'model.h5'))

        if 'distance.pkl' not in contents:
            print('Warning: distance save file missing!')
            dist = None
        else:
            with open(os.path.join(path, 'distance.pkl'), 'rb') as f:
                dist = pickle.load(f)
        if 'embed_dist.txt' not in contents:
            print('Warning: embedding distance save file missing, assuming Euclidean')
            dist = 'euclidean'
        else:
            with open(os.path.join(path, 'embed_dist.txt'), 'r') as f:
                embed_dist = f.read().strip()
        history = None
        if 'history.json' in contents:
            with open(os.path.join(path, 'history.json'), 'r') as f:
                history = json.load(f)
        return cls(v_scope=v_scope, dist=dist, model=model, strategy=strategy, history=history,
                   embed_dist=embed_dist, **kwargs)


class ComparativeEncoder(ComparativeModel):
    """
    Generic comparative encoder that can fit to data and transform sequences.
    """
    def __init__(self, model: tf.keras.Model, v_scope='encoder', **kwargs):
        """
        @param encoder: TensorFlow model that must support .train() and .predict() at minimum.
        @param dist: distance metric to use when comparing two sequences.
        """
        properties = {
            'input_shape': model.layers[0].output.shape[1:],
            'input_dtype': model.layers[0].dtype,
            'repr_size': model.layers[-1].output.shape[1],
            'depth': len(model.layers),
        }
        self.encoder = model
        super().__init__(v_scope, properties=properties, **kwargs)

    def create_model(self, loss='corr_coef'):
        inputa = tf.keras.layers.Input(self.properties['input_shape'], name='input_a',
                                       dtype=self.properties['input_dtype'])
        inputb = tf.keras.layers.Input(self.properties['input_shape'], name='input_b',
                                       dtype=self.properties['input_dtype'])
        # Set embedding distance calculation layer
        if self.embed_dist.lower() == 'euclidean':
            dist_cl = self.EuclideanDistanceLayer()
        elif self.embed_dist.lower() == 'hyperbolic':
            dist_cl = self.HyperbolicDistanceLayer()
        else:
            raise ValueError('Invalid embedding distance provided!')
        distances = dist_cl(
            self.encoder(inputa),
            self.encoder(inputb),
        )
        loss_kwargs = {'loss': 'mse', 'metrics': ['mae']} if loss == 'mse' else \
            {'loss': self.correlation_coefficient_loss}
        comparative_model = tf.keras.Model(inputs=[inputa, inputb], outputs=distances)
        comparative_model.compile(optimizer='adam', **loss_kwargs)
        return comparative_model

    @classmethod
    def from_model_builder(cls, builder: ModelBuilder, repr_size=None, **kwargs):
        """
        Initialize a ComparativeEncoder from a ModelBuilder object. Easy way to propagate the
        distribute strategy and variable scope.
        """
        encoder = builder.compile(repr_size=repr_size) if repr_size else builder.compile()
        return cls(encoder, strategy=builder.strategy, v_scope=builder.v_scope, **kwargs)

    @staticmethod
    class EuclideanDistanceLayer(tf.keras.layers.Layer):
        """
        This layer computes the distance between its two prior layers.
        """
        def call(self, a, b):
            return tf.reduce_sum(tf.square(a - b), -1)

    @staticmethod
    class HyperbolicDistanceLayer(tf.keras.layers.Layer):
        def call(self, a, b):
            """
            Computes hyperbolic distance in Poincaré ball model.
            """
            # Partially adapted from https://github.com/kousun12/tf_hyperbolic
            sq_norm = lambda v: tf.clip_by_value(
                tf.reduce_sum(v ** 2),
                clip_value_min=tf.keras.backend.epsilon(),
                clip_value_max=1 - tf.keras.backend.epsilon()
            )
            numerator = tf.reduce_sum(tf.pow(a - b, 2), axis=-1)
            denominator_a = 1 - sq_norm(a)
            denominator_b = 1 - sq_norm(b)
            frac = numerator / (denominator_a * denominator_b)
            return tf.math.acosh(1 + 2 * frac)

    @staticmethod
    def correlation_coefficient_loss(y_true, y_pred):
        """
        Correlation coefficient loss function for ComparativeEncoder.
        """
        x, y = y_true, y_pred
        mx, my = M.reduce_mean(x), M.reduce_mean(y)
        xm, ym = x - mx, y - my
        r_num = M.reduce_sum(M.multiply(xm, ym))
        r_den = M.sqrt(M.multiply(M.reduce_sum(M.square(xm)), M.reduce_sum(M.square(ym))))
        r = r_num / r_den
        r = M.maximum(M.minimum(r, 1.0), -1.0)
        return 1 - r

    # pylint: disable=arguments-differ
    def train_step(self, batch_size: int, data: np.ndarray, distance_on: np.ndarray, epoch_factor=1):
        """
        Train a single randomized epoch on data and distance_on.
        @param data: data to train model on.
        @param distance_on: np.ndarray of data to use for distance computations. Allows for distance
        to be based on secondary properties of each sequence, or on a string representation of the
        sequence (e.g. for alignment comparison methods).
        @param batch_size: batch size for TensorFlow.
        @param jobs: number of CPU jobs to use.
        @param chunksize: chunksize for Python multiprocessing.
        """
        # It's common to input pandas series from Dataset instead of numpy array
        data = data.to_numpy() if isinstance(data, pd.Series) else data
        distance_on = distance_on.to_numpy() if isinstance(distance_on, pd.Series) else distance_on
        x1, x2, y1, y2 = self.random_set(data, distance_on, epoch_factor=epoch_factor)

        y = self.distance.transform_multi(y1, y2)

        train_data = _prepare_tf_dataset({'input_a': x1, 'input_b': x2}, y, batch_size)

        return self.model.fit(train_data, epochs=1).history

    def fit(self, *args, distance_on=None, **kwargs):
        distance_on = distance_on if distance_on is not None else args[1]
        super().fit(*args, distance_on, **kwargs)

    @_run_tf_fn()
    def transform(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all sequences to transform.
        @param batch_size: Batch size for .predict().
        @return np.ndarray: Representations for all sequences in data.
        """
        dataset = _prepare_tf_dataset(data, None, batch_size)
        return self.encoder.predict(dataset, batch_size=batch_size)

    def save(self, path: str):
        super().save(path, model=self.encoder)

    @classmethod
    def load(cls, path: str, v_scope='encoder', **kwargs):
        custom_objects = {'correlation_coefficient_loss': cls.correlation_coefficient_loss}
        with tf.keras.utils.custom_object_scope(custom_objects):
            return super().load(path, v_scope, **kwargs)

    def summary(self):
        """
        Prints a summary of the encoder.
        """
        self.encoder.summary()


class Decoder(ComparativeModel):
    """
    Abstract Decoder for encoding distances.
    """
    def __init__(self, v_scope='decoder', embed_dist_args=None, **kwargs):
        super().__init__(v_scope, **kwargs)
        embed_dist_args = embed_dist_args or {}
        if self.embed_dist == 'hyperbolic':
            self.embed_dist_calc = Hyperbolic(**embed_dist_args)
        elif self.embed_dist == 'euclidean':
            self.embed_dist_calc = Euclidean(**embed_dist_args)
        elif self.embed_dist == 'cosine':
            self.embed_dist_calc = Cosine(**embed_dist_args)
        else:  # Should never happen
            raise ValueError(f'Invalid embedding distance for decoder: {self.embed_dist}.')

    def random_distance_set(self, encodings: np.ndarray, distance_on: np.ndarray, epoch_factor=1):
        """
        Create a random set of distance data from the inputs.
        """
        x1, x2, y1, y2 = self.random_set(encodings, distance_on, epoch_factor=epoch_factor)

        if not self.quiet:
            print(f'Calculating embedding distances')
        x = self.embed_dist_calc.transform_multi(x1, x2)
        if not self.quiet:
            print('Calculating true distances')
        y = self.distance.transform_multi(y1, y2)
        return x, y

    def evaluate(self, encodings: np.ndarray, distance_on: np.ndarray, sample_size=None):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: predicted distances, true distances
        """
        sample_size = sample_size or len(encodings)
        x, y = self.random_distance_set(encodings, distance_on,
                                        epoch_factor=int(sample_size / len(encodings))+1)
        if not self.quiet:
            print('Predicting true distances...')
        x = self.transform(x)
        y = self.distance.invert_postprocessing(y)

        mse = mean_squared_error(y, x)
        r2 = r2_score(y, x)
        mape = mean_absolute_percentage_error(y, x)
        if not self.quiet:
            print(f'Mean squared error of distances: {mse}')
            print(f'R-squared correlation coefficient: {r2}')
            print(f'Mean absolute percentage error: {mape}')
        return x, y

    def save(self, path: str):
        super().save(path)
        with open(os.path.join(path, 'embed_dist_calc.pkl'), 'wb') as f:
            pickle.dump(self.embed_dist_calc, f)

    @staticmethod
    def load(path: str, v_scope='decoder', **kwargs):
        # pylint: disable=arguments-differ
        contents = os.listdir(path)
        if 'model.h5.pkl' in contents:
            obj = LinearDecoder.load(path)
        else:
            obj = DenseDecoder.load(path, **kwargs)
        with open(os.path.join(path, 'embed_dist_calc.pkl'), 'rb') as f:
            obj.embed_dist_calc = pickle.load(f)
        return obj


class _LinearRegressionModel(LinearRegression):
    def save(self, path: str):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self, f)


class LinearDecoder(Decoder):
    """
    Linear model of a decoder. Far more efficient and useful in cases where ComparativeEncoder
    achives very low loss values.
    """
    # pylint: disable=arguments-differ
    def create_model(self):
        return _LinearRegressionModel()

    def fit(self, encodings: np.ndarray, distance_on: np.ndarray, *args, **kwargs):
        """
        Fit the LinearDecoder to the given data.
        """
        # It's common to input pandas series from Dataset instead of numpy array
        distance_on = distance_on.to_numpy() if isinstance(distance_on, pd.Series) else distance_on
        x, y = self.random_distance_set(encodings, distance_on, *args, **kwargs)
        self.model.fit(x.reshape((-1, 1)), y)

    def transform(self, data: np.ndarray):
        """
        Transform the given data.
        """
        return self.distance.invert_postprocessing(self.model.predict(data.reshape(-1, 1)))

    @classmethod
    def load(cls, path: str, **kwargs):
        with open(os.path.join(path, 'model.h5.pkl'), 'rb') as f:
            model = pickle.load(f)
        return super(Decoder, cls).load(path, 'decoder', model=model, **kwargs)


class DenseDecoder(Decoder):
    """
    Decoder model to convert generated distances into true distances.
    """
    def __init__(self, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    # pylint: disable=arguments-differ
    def create_model(self):
        dec_input = tf.keras.layers.Input((1,))
        x = tf.keras.layers.Dense(10, activation='relu')(dec_input)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=.1)(x)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation='relu')(x)
        decoder = tf.keras.Model(inputs=dec_input, outputs=x)
        decoder.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsolutePercentageError())
        return decoder

    def fit(self, *args, epochs=25, **kwargs):
        """
        The decoder is probably an afterthought, 25 epochs seems like a sensible default to avoid
        adding too much overhead.
        """
        return super().fit(*args, epochs=epochs, **kwargs)

    def train_step(self, encodings: np.ndarray, distance_on: np.ndarray, epoch_factor=1):
        # It's common to input pandas series from Dataset instead of numpy array
        distance_on = distance_on.to_numpy() if isinstance(distance_on, pd.Series) else distance_on

        x, y = self.random_distance_set(encodings, distance_on, epoch_factor=epoch_factor)
        train_data = _prepare_tf_dataset(x, y, self.batch_size)

        return self.model.fit(train_data, epochs=1).history

    @_run_tf_fn()
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the given distances between this model's encodings into predicted true distances.
        """
        dataset = _prepare_tf_dataset(data, None, self.batch_size)
        return self.distance.invert_postprocessing(self.model.predict(dataset))

    def summary(self):
        """
        Prints a summary of the decoder.
        """
        self.model.summary()

    @classmethod
    def load(cls, path: str, v_scope='decoder', **kwargs):
        return super(Decoder, cls()).load(path, v_scope, **kwargs)

