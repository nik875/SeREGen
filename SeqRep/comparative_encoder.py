"""
ComparativeEncoder module, trains a model comparatively using distances.
"""
import os
import pickle
import time
import multiprocessing as mp

import numpy as np
from scipy.spatial.distance import euclidean
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

from . import distance
from .encoders import ModelBuilder


class Model:
    """
    Generic model class. Stores some useful common functions.
    """
    def __init__(self, model: tf.keras.Model, strategy=None, quiet=False, properties=None):
        self.model = model
        self.strategy = strategy or tf.distribute.get_strategy()
        self.quiet = quiet
        self.properties = {} if properties is None else properties

    @staticmethod
    def _run_tf_fn(init_message=None):
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
                else:
                    print(f'Total time taken: {time.time() - start_time} seconds.')
                return result
            return fit_output_mgmt
        return fit_dec


class ComparativeEncoder(Model):
    """
    Generic comparative encoder that can fit to data and transform sequences.
    """
    def __init__(self, model: tf.keras.Model, dist=None, **kwargs):
        """
        @param encoder: TensorFlow model that must support .train() and .__call__() at minimum.
        .predict() required for progress bar when transforming data.
        @param dist: distance metric to use when comparing two sequences.
        """
        properties = {
            'input_shape': model.layers[0].output_shape[0][1:],
            'input_dtype': model.layers[0].dtype,
            'repr_size': model.layers[-1].output_shape[1:],
            'depth': len(model.layers)
        }
        self.encoder = model
        self.distance = dist or distance.Distance()
        strategy = kwargs['strategy'] if 'strategy' in kwargs else tf.distribute.get_strategy()

        with strategy.scope():
            inputa = tf.keras.layers.Input(properties['input_shape'], name='input_a',
                                           dtype=properties['input_dtype'])
            inputb = tf.keras.layers.Input(properties['input_shape'], name='input_b',
                                           dtype=properties['input_dtype'])
            distances = self.DistanceLayer()(
                self.encoder(inputa),
                self.encoder(inputb),
            )
            comparative_model = tf.keras.Model(inputs=[inputa, inputb], outputs=distances)
            comparative_model.compile(optimizer='adam', loss=self.correlation_coefficient_loss)
        super().__init__(comparative_model, properties=properties, **kwargs)

    @classmethod
    def from_model_builder(cls, builder: ModelBuilder, repr_size=None, **kwargs):
        """
        Initialize a ComparativeEncoder from a ModelBuilder object. Easy way to propagate the
        distribute strategy.
        """
        encoder = builder.compile(repr_size=repr_size) if repr_size else builder.compile()
        return cls(encoder, strategy=builder.strategy, **kwargs)

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

    def _randomized_epoch(self, data: np.ndarray, distance_on: np.ndarray, jobs: int,
                          chunksize: int, batch_size: int):
        """
        Train a single randomized epoch on data and distance_on.
        @param data: data to train model on.
        @param distance_on: data to use for distance computations.
        @param jobs: number of CPU jobs to use.
        @param chunksize: chunksize for Python multiprocessing.
        @param batch_size: batch size for TensorFlow.
        """
        rng = np.random.default_rng()
        p1 = rng.permutation(data.shape[0])
        x1 = data[p1]
        y1 = distance_on[p1]
        p2 = rng.permutation(data.shape[0])
        x2 = data[p2]
        y2 = distance_on[p2]

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

    @Model._run_tf_fn('Training ComparativeEncoder...')
    def fit(self, data: np.ndarray, distance_on=None, batch_size=100, epochs=10, jobs=1,
            chunksize=1, min_delta=0, patience=0):
        """
        Fit the ComparativeEncoder to the given data.
        @param data: np.ndarray to train on.
        @param distance_on: np.ndarray of data to use for distance computations. Allows for distance
        to be based on secondary properties of each sequence, or on a string representation of the
        sequence (e.g. for alignment comparison methods).
        @param batch_size: batch size for TensorFlow.
        @param epochs: epochs to train for.
        @param jobs: number of CPU jobs to use for distance calculations (not GPU optimized).
        @param chunksize: chunksize for Python multiprocessing.
        @param min_delta: Minimum change required to qualify as an improvement.
        @param patience: How many epochs with no improvement before giving up.
        """
        distance_on = distance_on if distance_on is not None else data
        history = {}
        for i in range(epochs):
            start = time.time()
            if not self.quiet:
                print(f'Epoch {i + 1}:')
            this_history = self._randomized_epoch(data, distance_on, jobs, chunksize, batch_size)
            if not self.quiet:
                print(f'Epoch time: {time.time() - start}')
            history = {k: v + this_history[k] for k, v in history} if history else this_history
            if not patience:  # Disable early stopping by setting to None or 0
                continue
            past_losses = history['loss'][-patience - 1:]
            if past_losses[-1] - past_losses[0] > min_delta:
                print('Stopping early due to lack of improvement!')
                break
        return history

    @Model._run_tf_fn()
    def transform(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all sequences to transform.
        @param batch_size: Batch size for .predict(), required for progress bar. Slows execution.
        @return np.ndarray: Representations for all sequences in data.
        """
        return self.encoder.predict(data, batch_size=batch_size)

    def save(self, path: str):
        """
        Save the encoder model to the given path.
        @param path: path to save to.
        """
        os.makedirs(path)
        self.encoder.save(os.path.join(path, 'encoder'))
        with open(os.path.join(path, 'distance.pkl'), 'wb') as f:
            pickle.dump(self.distance, f)

    @classmethod
    def load(cls, path: str, strategy=None, **kwargs):
        """
        Load an encoder model and create a new ComparativeEncoder.
        @param path: path where model is saved.
        @param dist: distance metric to use for new ComparativeEncoder.
        @return: ComparativeEncoder object
        """
        if not os.path.exists(os.path.join(path, 'encoder')):
            raise ValueError('Encoder save file is necessary for loading a model!')
        custom_objects = {'correlation_coefficient_loss': cls.correlation_coefficient_loss}
        strategy = strategy or tf.distribute.get_strategy()
        with strategy.scope():
            with tf.keras.utils.custom_object_scope(custom_objects):
                encoder = tf.keras.models.load_model(os.path.join(path, 'encoder'))

        if not os.path.exists(os.path.join(path, 'distance.pkl')):
            print('Warning: distance save file missing! Inferencing is possible, but training and '
                  'decoder evaluation is not!')
            dist = None
        else:
            with open(os.path.join(path, 'distance.pkl'), 'rb') as f:
                dist = pickle.load(f)
        return cls(encoder, dist=dist, strategy=strategy, **kwargs)

    def summary(self):
        """
        Prints a summary of the encoder.
        """
        self.encoder.summary()


class DistanceDecoder(Model):
    """
    Decoder model to convert generated distances into true distances.
    """
    def __init__(self, model=None, dist=None, strategy=None, **kwargs):
        model = model or self.default_decoder(strategy=strategy)
        super().__init__(model, strategy=strategy, **kwargs)
        self.distance = dist or distance.Distance()

    @staticmethod
    def default_decoder(size=100, strategy=None):
        """
        Create a simple decoder.
        """
        strategy = strategy or tf.distribute.get_strategy()
        with strategy.scope():
            dec_input = tf.keras.layers.Input((1,))
            x = tf.keras.layers.Dense(size, activation='relu')(dec_input)
            x = tf.keras.layers.Dense(size, activation='relu')(x)
            x = tf.keras.layers.Dropout(rate=.1)(x)
            x = tf.keras.layers.Dense(size, activation='relu')(x)
            x = tf.keras.layers.Dense(1, activation='relu')(x)
            decoder = tf.keras.Model(inputs=dec_input, outputs=x)
            decoder.compile(optimizer='adam',
                            loss=tf.keras.losses.MeanAbsolutePercentageError())
        return decoder

    @Model._run_tf_fn('Training DistanceDecoder...')
    def fit(self, encodings: np.ndarray, distance_on: np.ndarray, batch_size=1000, epoch_limit=100,
            patience=1, jobs=1, chunksize=1):
        """
        Fit the decoder to the given data.
        """
        rng = np.random.default_rng()
        p1 = rng.permutation(distance_on.shape[0])
        y1 = distance_on[p1]
        p2 = rng.permutation(distance_on.shape[0])
        y2 = distance_on[p2]

        print('Calculating distances between model inputs...')
        with mp.Pool(jobs) as p:
            it = p.imap(self.distance.transform, zip(y1, y2), chunksize=chunksize)
            y = np.fromiter((it if self.quiet else tqdm(it, total=len(y1))), dtype=np.float64)
        # Do not postprocess distances. The idea is that transform should provide a meaningful
        # distance, even if the postprocessed distances only have meaning in context of the current
        # dataset because of normalization.
        x1, x2 = encodings[p1], encodings[p2]
        if not self.quiet:
            print('Calculating euclidean distances between encodings...')
        x = np.fromiter((euclidean(x1[i], x2[i]) for i in (range(len(y)) if self.quiet else
                                                        tqdm(range(len(y))))), dtype=np.float64)

        print('Training model...')
        return self.model.fit(x, y, epochs=epoch_limit, batch_size=batch_size,
                              validation_split=.1,
                              callbacks=[tf.keras.callbacks.EarlyStopping(
                                  monitor='val_loss', patience=patience)])

    @Model._run_tf_fn()
    def transform(self, data: np.ndarray, batch_size=256) -> np.ndarray:
        """
        Transform the given distances between this model's encodings into predicted true distances.
        """
        return self.model.predict(data, batch_size=batch_size)

    def save(self, path: str):
        """
        Save the decoder model to the given path.
        @param path: path to save to.
        """
        os.makedirs(path)
        self.model.save(os.path.join(path, 'decoder'))
        with open(os.path.join(path, 'distance.pkl'), 'wb') as f:
            pickle.dump(self.distance, f)

    @classmethod
    def load(cls, path: str, strategy=None, **kwargs):
        """
        Load an decoder model and create a new DistanceDecoder.
        @param path: path where model is saved.
        @param dist: distance metric to use for new DistanceDecoder.
        @return: DistanceDecoder object
        """
        if not os.path.exists(os.path.join(path, 'decoder')):
            raise ValueError('Decoder save file is necessary for loading a model!')
        strategy = strategy or tf.distribute.get_strategy()
        with strategy.scope():
            model = tf.keras.models.load_model(os.path.join(path, 'decoder'))

        if not os.path.exists(os.path.join(path, 'distance.pkl')):
            print('Warning: distance save file missing! Inferencing is possible, but training and '
                  'evaluation is not!')
            dist = None
        else:
            with open(os.path.join(path, 'distance.pkl'), 'rb') as f:
                dist = pickle.load(f)
        return cls(model=model, dist=dist, strategy=strategy, **kwargs)

    def summary(self):
        """
        Prints a summary of the decoder.
        """
        self.model.summary()

