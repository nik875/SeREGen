"""
ComparativeEncoder module, trains a model comparatively using distances.
"""
import os
import pickle
import json
import time
import multiprocessing as mp

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

from .encoders import ModelBuilder


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


class ComparativeModel:
    """
    Abstract ComparativeModel class. Stores some useful common functions.
    """
    def __init__(self, v_scope='model', dist=None, model=None, strategy=None,
                 history=None, quiet=False, properties=None, **kwargs):
        self.strategy = strategy or tf.distribute.get_strategy()
        self.distance = dist
        self.quiet = quiet
        self.properties = {} if properties is None else properties
        self.history = history or {}
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

    @_run_tf_fn(print_time=True)
    def fit(self, *args, epochs=100, early_stop=True, min_delta=0, patience=3, **kwargs):
        """
        Train the model based on the given parameters. Extra arguments are passed to train_step.
        @param epochs: epochs to train for.
        @param min_delta: Minimum change required to qualify as an improvement.
        @param patience: How many epochs with no improvement before giving up.
        """
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
        os.makedirs(path)
        model = model or self.model
        model.save(os.path.join(path, 'model'))
        with open(os.path.join(path, 'distance.pkl'), 'wb') as f:
            pickle.dump(self.distance, f)
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
            if 'model' not in contents:
                raise ValueError('Model save file is necessary for loading a ComparativeModel!')
            strategy = strategy or tf.distribute.get_strategy()
            with tf.name_scope(v_scope):
                with strategy.scope():
                    model = tf.keras.models.load_model(os.path.join(path, 'model'))

        if 'distance.pkl' not in contents:
            print('Warning: distance save file missing!')
            dist = None
        else:
            with open(os.path.join(path, 'distance.pkl'), 'rb') as f:
                dist = pickle.load(f)
        history = None
        if 'history.json' in contents:
            with open(os.path.join(path, 'history.json'), 'r') as f:
                history = json.load(f)
        return cls(v_scope=v_scope, dist=dist, model=model, strategy=strategy, history=history,
                   **kwargs)


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
            'input_shape': model.layers[0].output_shape[0][1:],
            'input_dtype': model.layers[0].dtype,
            'repr_size': model.layers[-1].output_shape[1],
            'depth': len(model.layers),
        }
        self.encoder = model
        super().__init__(v_scope, properties=properties, **kwargs)

    def create_model(self, loss='corr_coef'):
        inputa = tf.keras.layers.Input(self.properties['input_shape'], name='input_a',
                                       dtype=self.properties['input_dtype'])
        inputb = tf.keras.layers.Input(self.properties['input_shape'], name='input_b',
                                       dtype=self.properties['input_dtype'])
        distances = self.DistanceLayer()(
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

    class DistanceLayer(tf.keras.layers.Layer):
        """
        This layer computes the distance between its two prior layers.
        """
        # pylint: disable=missing-function-docstring
        def call(self, a, b):
            return tf.reduce_sum(tf.square(a - b), -1)

    @staticmethod
    def correlation_coefficient_loss(y_true, y_pred):
        """
        Correlation coefficient loss function for ComparativeEncoder.
        """
        x, y = y_true, y_pred
        mx, my = K.mean(x), K.mean(y)
        xm, ym = x - mx, y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den
        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return 1 - r

    # pylint: disable=arguments-differ
    def train_step(self, data: np.ndarray, distance_on: np.ndarray, batch_size=100, jobs=1,
                   chunksize=1):
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
        rng = np.random.default_rng()
        p1 = rng.permutation(data.shape[0])
        x1 = data[p1]
        y1 = distance_on[p1]
        p2 = rng.permutation(data.shape[0])
        x2 = data[p2]
        y2 = distance_on[p2]

        if jobs == 1:
            it = tqdm(zip(y1, y2)) if self.quiet else zip(y1, y2)
            y = np.fromiter((self.distance.transform(i) for i in it), dtype=np.float64)
        else:
            with mp.Pool(jobs) as p:
                it = p.imap(self.distance.transform, zip(y1, y2), chunksize=chunksize)
                y = np.fromiter((it if self.quiet else tqdm(it, total=len(y1))), dtype=np.float64)
        y = self.distance.postprocessor(y)  # Vectorized transformations are applied here

        train_data = tf.data.Dataset.from_tensor_slices(({'input_a': x1, 'input_b': x2}, y))
        train_data = train_data.batch(batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)

        return self.model.fit(train_data, epochs=1).history

    def fit(self, *args, distance_on=None, **kwargs):
        distance_on = distance_on if distance_on is not None else args[0]
        super().fit(*args, distance_on, **kwargs)

    @_run_tf_fn()
    def transform(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all sequences to transform.
        @param batch_size: Batch size for .predict().
        @return np.ndarray: Representations for all sequences in data.
        """
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
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
    def __init__(self, v_scope='decoder', **kwargs):
        super().__init__(v_scope, **kwargs)

    def random_set(self, encodings: np.ndarray, distance_on: np.ndarray, jobs=1, chunksize=1):
        """
        Create a random set of distance data from the inputs.
        """
        rng = np.random.default_rng()
        p1 = rng.permutation(distance_on.shape[0])
        y1 = distance_on[p1]
        p2 = rng.permutation(distance_on.shape[0])
        y2 = distance_on[p2]

        with mp.Pool(jobs) as p:
            it = p.imap(self.distance.transform, zip(y1, y2), chunksize=chunksize)
            y = np.fromiter((it if self.quiet else tqdm(it, total=len(y1))), dtype=np.float64)
        # Do not postprocess distances. The idea is that transform should provide a meaningful
        # distance, even if the postprocessed distances only have meaning in context of the current
        # dataset because of normalization.
        x1, x2 = encodings[p1], encodings[p2]
        x = np.fromiter((euclidean(x1[i], x2[i]) for i in range(len(y))), dtype=np.float64)
        return x, y

    def fit(self, *args, **kwargs):
        pass

    # pylint: disable=arguments-differ
    @staticmethod
    def load(path: str, v_scope='decoder', **kwargs):
        contents = os.listdir(path)
        if 'model.pkl' in contents:
            return LinearDecoder.load(path)
        return DenseDecoder.load(path, **kwargs)


class _LinearRegressionModel(LinearRegression):
    # pylint: disable=missing-class-docstring,missing-function-docstring
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

    def fit(self, encodings: np.ndarray, distance_on: np.ndarray, *args, jobs=1, chunksize=1,
            **kwargs):
        """
        Fit the LinearDecoder to the given data.
        """
        x, y = self.random_set(encodings, distance_on, jobs=jobs, chunksize=chunksize)
        self.model.fit(x.reshape((-1, 1)), y)

    def transform(self, data: np.ndarray):
        """
        Transform the given data.
        """
        return self.model.predict(data.reshape(-1, 1))

    @classmethod
    def load(cls, path: str, **kwargs):
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        return super(Decoder, cls()).load(path, 'decoder', model=model, **kwargs)


class DenseDecoder(Decoder):
    """
    Decoder model to convert generated distances into true distances.
    """
    # pylint: disable=arguments-differ
    def create_model(self):
        dec_input = tf.keras.layers.Input((1,))
        x = tf.keras.layers.Dense(100, activation='relu')(dec_input)
        x = tf.keras.layers.Dense(100, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=.1)(x)
        x = tf.keras.layers.Dense(100, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation='relu')(x)
        decoder = tf.keras.Model(inputs=dec_input, outputs=x)
        decoder.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsolutePercentageError())
        return decoder

    def train_step(self, encodings: np.ndarray, distance_on: np.ndarray, batch_size=1000, jobs=1,
                   chunksize=1):
        x, y = self.random_set(encodings, distance_on, jobs=jobs, chunksize=chunksize)
        train_data = tf.data.Dataset.from_tensor_slices((x, y))
        train_data = train_data.batch(batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)

        return self.model.fit(train_data, epochs=1).history

    @_run_tf_fn()
    def transform(self, data: np.ndarray, batch_size=256) -> np.ndarray:
        """
        Transform the given distances between this model's encodings into predicted true distances.
        """
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(batch_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        return self.model.predict(dataset)

    def summary(self):
        """
        Prints a summary of the decoder.
        """
        self.model.summary()

    @classmethod
    def load(cls, path: str, v_scope='decoder', **kwargs):
        return super(Decoder, cls()).load(path, v_scope, **kwargs)

