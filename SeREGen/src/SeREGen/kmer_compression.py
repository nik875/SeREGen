"""
Library for input compression before data is passed into a model.
"""
import os
import shutil
import pickle
import copy
import multiprocessing as mp


import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA, IncrementalPCA


from .kmers import KMerCounter


class KMerCountCompressor(KMerCounter):
    # pylint: disable=unused-argument
    """
    Abstract Compressor class used for compressing input data.
    """
    _SAVE_EXCLUDE_VARS = []

    def __init__(self, counter: KMerCounter, compress_to: int):
        super().__init__(counter.k, jobs=counter.jobs, chunksize=counter.chunksize,
                         debug=counter.debug, silence=counter.silence)
        self.compress_to = compress_to
        self.fit_called = False

    def save(self, savedir: str):
        """
        Save the Compressor to the filesystem.
        """
        shutil.rmtree(savedir, ignore_errors=True)
        os.makedirs(savedir)

        to_pkl = copy.copy(self)  # Efficient shallow copy for pickling
        for i in self._SAVE_EXCLUDE_VARS:  # Don't pickle attrs in _SAVE_EXCLUDE_VARS
            delattr(to_pkl, i)

        with open(os.path.join(savedir, 'compressor.pkl'), 'wb') as f:
            pickle.dump(to_pkl, f)

    @staticmethod
    def load(savedir: str):
        """
        Load the Compressor from the filesystem.
        """
        if not os.path.exists(savedir) or not os.path.exists(savedir):
            raise ValueError("Directory doesn't exist!")
        if 'compressor.pkl' not in os.listdir(savedir):
            raise ValueError('compressor.pkl is necessary!')
        with open(os.path.join(savedir, 'compressor.pkl'), 'rb') as f:
            obj = pickle.load(f)
        # pylint: disable=protected-access
        obj._load_special(savedir)
        return obj

    def _load_special(self, savedir: str):
        """
        Load any special variables from the savedir for this object. Called by Compressor.load().
        """

    def fit(self, data: np.ndarray):
        """
        Fit the compressor to the given data.
        @param data: data to fit to.
        @param silence: whether to print output
        """
        self.fit_called = True

    def raw_transform(self, data: np.ndarray) -> np.ndarray:
        """
        The most basic transform operation, after kmer counting. Must be implemented.
        """
        return data

    def _transform_with_kmer_counts(self, data: np.ndarray) -> np.ndarray:
        data = self.kmer_counts(data, silence=True, jobs=1, chunksize=1)
        return self.raw_transform(data)

    def transform(self, data: np.ndarray, silence=False) -> np.ndarray:
        # pylint: disable=unused-argument
        """
        Compress an array of data elements.
        @param data: data to compress.
        @param silence: additional option to silence output of this function.
        @return np.ndarray: compressed data.
        """
        return data

    def inverse_transform(self, data: np.ndarray, silence=False) -> np.ndarray:
        # pylint: disable=unused-argument
        """
        Decodes the compressed data back to original.
        @param data: data to decode.
        @param silence: additional option to silence output of this function.
        @return np.ndarray: uncompressed data.
        """
        return data


class _PCACompressor(KMerCountCompressor):
    """
    Abstract PCA compressor, conserves code.
    """

    def __init__(self, pca, *args, jobs=1, chunksize=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca = pca
        self.scaler = StandardScaler()
        self.compress_jobs = jobs
        self.compress_chunksize = chunksize

    def _batch_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fully_batchable_data = data[:len(data) - len(data) % self.compress_chunksize]
        full_batches = np.reshape(fully_batchable_data,
                                  (-1, self.compress_chunksize, *fully_batchable_data.shape[1:]))
        last_batch = data[len(data) - len(data) % self.compress_chunksize:]
        return full_batches, last_batch

    def _mp_map_over_batches(self, fn: callable, data: np.ndarray, silence=False) -> np.ndarray:
        full_batches, last_batch = self._batch_data(data)
        with mp.Pool(self.compress_jobs) as p:
            it = p.imap_unordered(fn, full_batches) if self.silence or silence else tqdm(
                p.imap_unordered(fn, full_batches), total=len(full_batches))
            result = list(it)
        if len(last_batch) > 0:
            result.append(fn(last_batch))
        return np.concatenate(result) if len(result) > 0 and isinstance(result[0], np.ndarray) \
            else result

    def raw_transform(self, data: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(data))

    def transform(self, data: np.ndarray, silence=False) -> np.ndarray:
        if isinstance(data[0], str):  # Calling on list of strings
            data = np.array(data, dtype=object)
            transform_fn = self._transform_with_kmer_counts
        else:  # Calling on kmer counts
            transform_fn = self.raw_transform
        return self._mp_map_over_batches(transform_fn, data, silence)

    def _raw_inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(self.pca.inverse_transform(data))

    def inverse_transform(self, data: np.ndarray, silence=False):
        return self._mp_map_over_batches(self._raw_inverse_transform, data, silence)


class KMerCountPCA(_PCACompressor):
    """
    Use PCA to compress input data. Suppoorts parallelization on transform, not fit.
    """

    def __init__(self, counter: KMerCounter, n_components: int, **kwargs):
        pca = SKPCA(n_components=n_components)
        super().__init__(pca, counter, n_components, **kwargs)

    def fit(self, data: np.ndarray):
        super().fit(data)
        self.pca.fit(self.scaler.fit_transform(data))


class KMerCountIPCA(_PCACompressor):
    """
    Use PCA to compress the input data. Supports fit parallelization over multiple CPUs.
    """

    def __init__(self, counter: KMerCounter, n_components: int, **kwargs):
        """
        Uses jobs, chunksize defined by KMerCounter
        """
        super().__init__(None, counter, n_components, **kwargs)
        self.pca = IncrementalPCA(n_components=n_components, batch_size=self.compress_chunksize)

    def fit(self, data: np.ndarray):
        super().fit(data)
        if not self.silence:
            print(f'Fitting IPCA Compressor using CPUs: {self.compress_jobs}...')
        data = self.scaler.fit_transform(data)
        full_batches, last_batch = self._batch_data(data)
        if len(last_batch) < self.compress_to:  # Drop last batch if not enough data
            last_batch = full_batches[-1]
            full_batches = full_batches[:-1]
        self._mp_map_over_batches(self.pca.partial_fit, np.concatenate(full_batches))
        # Use normal fit on last batch so sklearn doesn't trigger a fit not called error
        self.pca.fit(last_batch)
