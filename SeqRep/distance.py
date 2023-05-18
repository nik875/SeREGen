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
    Downstream subclasses must implement transform.
    """
    MAX_DIST = 2 ** .5
    AVERAGE_DIST = 0.5232711374270173  # Empirically determined for this MAX_DIST

    def __init__(self, transform_fn=None, postprocessor_fn=None):
        """
        Allows a functional method for creating new distance metrics.
        """
        self.transform = transform_fn or self.transform
        self.postprocessor = postprocessor_fn or self.postprocessor

    def transform(self, pair: tuple) -> int:
        """
        Transform a pair of elements into a single integer distance between those elements.
        @param pair: two-element tuple containing elements to compute distance between.
        @return int: distance value
        """
        return 0

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        """
        Postprocess a full array of distances. Does a basic normalization by default.
        @param data: np.ndarray
        @return np.ndarray
        """
        zscores = stats.zscore(data)
        max_zscore = np.max(zscores)
        return self.MAX_DIST / (np.max(zscores) - np.min(zscores)) * zscores + self.AVERAGE_DIST


class Euclidean(Distance):
    """
    Normalized Euclidean distance implementation between two arrays of numbers.
    Sensitive to non-normal distributions of distances! Always check plot before use.
    """
    def transform(self, pair: tuple) -> int:
        """
        Transforms a given pair of integer arrays into a single Euclidean distance.
        @param pair: tuple of integer arrays.
        @return int: Euclidean distance.
        """
        super().transform(pair)
        return euclidean(*pair)


class Alignment(Distance):
    """
    Normalized alignment distance between two textual DNA sequences. Sequences must
    all have equal lengths.
    """
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
        return super().postprocessor(data)

