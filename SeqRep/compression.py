"""
Library for input compression before data is passed into a model.
"""
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .dataset_builder import Dataset
from .kmers import KMerCounter


class Compressor:
    """
    Abstract Compressor class used for compressing input data.
    """
    def fit(self, data: np.ndarray):
        """
        Fit the compressor to the given data.
        @param data: data to fit to.
        """

    def compress(self, data: np.ndarray) -> np.ndarray:
        """
        Compress the given data.
        @param data: data to compress.
        @return np.ndarray: compressed data.
        """
        return data


class PCACompressor(Compressor):
    """
    Use PCA to compress the input data.
    """
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    def fit(self, data: np.ndarray):
        data = StandardScaler().fit_transform(data)
        self.pca.fit(data)

    # pylint: disable=unused-argument
    def compress(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.pca.transform(data)


class AECompressor(Compressor):
    """
    Train an autoencoder to compress the input data.
    """
    def __init__(self, inputs: tf.keras.layers.Layer,
                 reprs: tf.keras.layers.Layer,
                 outputs: tf.keras.layers.Layer,
                 loss='mse'):
        self.encoder = tf.keras.Model(inputs=inputs, outputs=reprs)
        self.decoder = tf.keras.Model(inputs=reprs, outputs=outputs)
        self.ae = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.ae.compile(optimizer='adam', loss=loss)

    @classmethod
    def auto(cls, data: np.ndarray, repr_size: int, output_activation=None,
             loss='mse'):
        """
        Automatically generate an autoencoder based on the input data. Recommended way to create
        an AECompressor.
        """
        inputs = tf.keras.layers.Input(data.shape[1:])
        x = tf.keras.layers.Dense(data.shape[-1], activation='relu')(inputs)
        reprs = tf.keras.layers.Dense(repr_size)(x)
        x = tf.keras.layers.Dense(data.shape[-1], activation='relu')(reprs)
        outputs = tf.keras.layers.Dense(data.shape[-1],
                                        activation=output_activation)(x)
        return cls(inputs, reprs, outputs, loss=loss)

    def summary(self):
        """
        Print a summary of this autoencoder.
        """
        self.ae.summary()

    def fit(self, data: np.ndarray, epochs=1, batch_size=1):
        self.ae.fit(data, data, epochs=epochs, batch_size=batch_size)

    def compress(self, data: np.ndarray, progress=True) -> np.ndarray:
        return self.encoder.predict(data) if progress else self.encoder(data)

    def decode(self, data: np.ndarray, progress=True) -> np.ndarray:
        """
        Apply the decoder to transform compressed data back to predicted K-Mer counts.
        """
        return self.decoder.predict(data) if progress else self.decoder(data)


class _KMersMPWrapper:
    def __init__(self, counter: KMerCounter, comp: PCACompressor):
        self.counter = counter
        self.comp = comp

    # pylint: disable=missing-docstring
    def count_kmers(self, seq: str):
        kmers = self.counter.str_to_kmer_counts(seq)
        return self.comp.compress([kmers], progress=False)[0]


def count_kmers_mp(K: int, comp: Compressor, ds: Dataset, jobs=1, chunksize=1,
                   progress=True, debug=False) -> np.ndarray:
    """
    Count K-Mers for all sequences in a given Dataset, automatically applying compression using
    multiprocessing. Intended for use with PCACompressor.
    """
    counter = KMerCounter(K)
    obj = _KMersMPWrapper(counter, comp)
    if debug:
        return np.array([obj.count_kmers([i]) for i in tqdm(ds['seqs'])])
    with mp.Pool(jobs) as p:
        it = tqdm(p.imap(obj.count_kmers, ds['seqs'], chunksize=chunksize), total=len(ds)) \
            if progress else p.imap(obj.count_kmers, ds['seqs'], chunksize=chunksize)
        return np.array(list(it))


def count_kmers_batched(K: int, comp: Compressor, ds: Dataset, batch_size=1,
                        jobs=1, chunksize=1, progress=True) -> np.ndarray:
    """
    Count K-Mers for all sequences in given Dataset, automatically applying compression in a batched
    fashion. Intended for use with AECompressor.
    """
    counter = KMerCounter(K, jobs=jobs, chunksize=chunksize)
    full_batches = len(ds) // batch_size
    encodings = []
    for i in (tqdm(range(full_batches)) if progress else range(full_batches)):
        this_batch = ds['seqs'].iloc[batch_size * i:batch_size * (i + 1)]
        kmer_counts = counter.kmer_counts(this_batch.to_numpy(), quiet=True)
        encodings.append(comp.compress(kmer_counts, progress=False))
    last_batch = ds['seqs'].iloc[batch_size * full_batches:]
    kmer_counts = counter.kmer_counts(last_batch.to_numpy(), quiet=True)
    encodings.append(comp.compress(kmer_counts, progress=False))
    return np.array(encodings)

