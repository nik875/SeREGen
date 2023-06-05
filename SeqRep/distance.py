"""
Contains distance metrics used for training ComparativeEncoders.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean as sceuclidean, cosine as sccosine
from Bio import pairwise2
from .kmers import KMerCounter


def unwrap_tuple(fn):
    """
    Call a function on a tuple pair, expanding the tuple into arguments.
    """
    def wrapper(tup):
        return fn(*tup)
    return wrapper


# pylint: disable=method-hidden, unused-argument
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
        self.transform = unwrap_tuple(transform_fn) if transform_fn else self.transform
        self.postprocessor = unwrap_tuple(postprocessor_fn) if postprocessor_fn \
            else self.postprocessor

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
        return self.MAX_DIST / (np.max(zscores) - np.min(zscores)) * zscores + self.AVERAGE_DIST

euclidean = Distance(transform_fn=sceuclidean)
cosine = Distance(transform_fn=sccosine)


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
        return pairwise2.align.localxx(pair[0], pair[1], score_only=True)

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        """
        Converts similarity scores into normalized distances for output.
        @param data: np.ndarray
        @return np.ndarray
        """
        data = np.max(data) - data  # Convert similarity scores into distances
        return super().postprocessor(data)


class IncrementalDistance(Distance):
    """
    Incrementally applies a regular K-Mers based distance metric over raw sequences.
    Use when not enough memory exists to fully encode a dataset into K-Mers with the specified K.
    """
    def __init__(self, distance: Distance, counter: KMerCounter):
        super().__init__()
        self.distance = distance
        self.counter = counter

    def transform(self, pair: tuple) -> int:
        """
        Converts the pair of sequences to K-Mers, then finds the distance.
        """
        kmer_pair = self.counter.str_to_kmer_counts(pair[0]), \
            self.counter.str_to_kmer_counts(pair[1])
        return self.distance.transform(kmer_pair)

