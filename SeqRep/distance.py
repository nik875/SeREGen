import multiprocessing as mp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from kpal.metrics import euclidean
from Bio import pairwise2
from tqdm import tqdm as tqdm


class Distance:
    """
    Abstract class representing a distance metric for two sequences.
    Downstream extensions must implement transform.
    """
    def __init__(self):
        """
        Any initial parameters for the distance metric.
        """
        # This parameter allows for a distance metric to either be fit manually before
        # being passed to a ComparativeEncoder, or for the ComparativeEncoder to autodetect
        # that fit has not been called and automatically call fit whenever it receives data.
        self.fit_called = False

    def fit(self, data):
        """
        Fit the distance metric to the given data. Does nothing by default.
        @param data: data to fit to
        """
        self.fit_called = True

    def transform(self, pair: tuple) -> int:
        """
        Transform a pair of elements into a single integer distance between those elements.
        @param pair: two-element tuple containing elements to compute distance between.
        @return int: distance value
        """
        return 0

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        """
        Postprocess a full array of distances. Does nothing by default.
        @param data: np.ndarray
        @return np.ndarray
        """
        return data


class _RandomDistanceCalculator:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.rng = np.random.default_rng()

    def gen(_) -> float:
        return euclidean(self.rng.choice(self.data), self.rng.choice(self.data))


class Euclidean(Distance):
    """
    Normalized Euclidean distance implementation between two arrays of numbers.
    Sensitive to non-normal distributions of distances! Always check plot before use.
    """
    def __init__(self, max_zscore_dev=3):
        super().__init__()  # Update self.fit_called
        self.max_zscore_dev = max_zscore_dev
        self.mean = -1; self.std = -1; self.min_val = -1; self.sampled_zscores = None

    def fit(self, data, jobs=1, chunksize=1, sample_size=1000):
        """
        Fits the distance metric to a given dataset. Estimates mean, standard deviation,
        and minimum value to allow for accurate normalized score calculation.
        @param data: np.ndarray of numbers that could be passed into transform.
        @param sample_size: number of times to repeat random sampling and euclidean
        distance calculation before generating mean, std, and min_val parameters.
        """
        super().fit(data)  # Update self.fit_called
        calculator = _RandomDistanceCalculator(data)
        with mp.Pool(jobs) as p:
            result = np.fromiter(tqdm(p.imap_unordered(calculator.gen, range(sample_size)), total=sample_size), dtype=np.float)
        # Reduce bias by trimming all zscores more than self.max_zscore_dev away from mean
        mask = np.logical_and(zscores > -1 * self.max_zscore_dev, zscores < self.max_zscore_dev)
        result = result[mask]
        zscores = zscores[mask]

        self.mean = result.mean()
        self.std = result.std()
        self.min_val = zscores.min()
        self.sampled_zscores = zscores

    def plot_zscores(self):
        """
        Plots the random zscore sample generated by fit as a histogram.
        """
        assert self.sampled_zscores != None
        plt.hist(self.sampled_zscores - self.min_val)
        plt.show()

    def transform(self, pair: tuple) -> int:
        """
        Transforms a given pair of integer arrays into a single normalized Euclidean distance.
        fit() must have been called before calling transform, otherwise an exception is raised.
        @param pair: tuple of integer arrays.
        @return int: normalized Euclidean distance.
        """
        super().transform(pair)
        assert self.mean != -1 or self.std != -1 or self.min_val != -1  # TODO: CREATE BETTER ERROR
        zscore = (euclidean(*pair) - self.mean) / self.std
        return zscore - self.min_val


class Alignment(Distance):
    """
    Normalized alignment distance between two textual DNA sequences. Sequences must
    all have equal lengths.
    """
    MAX_DIST = 2 ** .5  # Sqrt(2) for max dist means a grid selection area with side length 1
    AVERAGE_DIST = 0.5232711374270173  # Empirically determined for this MAX_DIST

    def fit(self, data, **kwargs):
        """
        Fits the distance metric to a given dataset.
        @param data: np.ndarray of string sequences.
        """
        super().fit(data)

    def transform(self, pair: tuple) -> int:
        """
        Transforms a single pair of strings into a similarity score.
        @param pair: tuple of two strings
        @return int: normalized alignment distance
        """
        super().transform(pair)
        return pairwise2.align.localxx(pair[0], pair[1], score_only=True)

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        """
        Converts similarity scores into normalized distances for output.
        @param data: np.ndarray
        @return np.ndarray
        """
        data = np.max(data) - data
        zscores = (data - np.mean(data)) / np.std(data)
        max_zscore = np.max(zscores)
        return self.MAX_DIST / (np.max(zscores) - np.min(zscores)) * zscores + self.AVERAGE_DIST
