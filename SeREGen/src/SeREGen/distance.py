"""
Contains distance metrics used for training ComparativeEncoders.
"""
import numpy as np
from scipy.spatial.distance import euclidean as sceuclidean, cosine as sccosine
from scipy.spatial import distance_matrix
from Bio.Align import PairwiseAligner
import Levenshtein
import py_stringmatching as sm
from rdkit.Chem import AllChem, DataStructs
from .kmers import KMerCounter


# pylint: disable=method-hidden
class Distance:
    """
    Abstract class representing a distance metric for two sequences.
    Downstream subclasses must implement transform.
    """
    def __init__(self, repr_size: int, sample_size=1000):
        self.repr_size = repr_size
        rng = np.random.default_rng()
        a = rng.random((sample_size, repr_size))
        b = rng.random((sample_size, repr_size))
        mat = distance_matrix(a, b)
        dists = mat.flatten()
        self.enc_space_mean = np.mean(dists)
        self.enc_space_std = np.std(dists)
        self.dist_space_mean, self.dist_space_std = None, None

    #pylint: disable=unused-argument
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
        self.dist_space_mean = np.mean(data)
        self.dist_space_std = np.std(data)
        zscores = (data - self.dist_space_mean) / self.dist_space_std
        return zscores * self.enc_space_std + self.enc_space_mean

    def invert_postprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Inverts the postprocessing of data to return raw transform results.
        @param data: np.ndarray
        @return np.ndarray
        """
        zscores = (data - self.enc_space_mean) / self.enc_space_std
        return zscores * self.dist_space_std + self.dist_space_mean


class Euclidean(Distance):
    """
    Basic Euclidean distance implementation
    """
    def transform(self, pair: tuple) -> int:
        return sceuclidean(*pair)


class EuclideanWithoutNorm(Euclidean):
    def postprocessor(self, data):
        return data
    
    def invert_postprocessing(self, data):
        return data


class Cosine(Distance):
    """
    Cosine distance implementation.
    """
    def transform(self, pair: tuple) -> int:
        return sccosine(*pair)


class CosineWithoutNorm(Cosine):
    def postprocessor(self, data):
        return data
    
    def invert_postprocessing(self, data):
        return data


class IncrementalDistance(Distance):
    """
    Incrementally applies a regular K-Mers based distance metric over raw sequences.
    Use when not enough memory exists to fully encode a dataset into K-Mers with the specified K.
    """
    def __init__(self, distance: Distance, counter: KMerCounter):
        super().__init__(distance.repr_size)
        self.distance = distance
        self.counter = counter

    def transform(self, pair: tuple) -> int:
        kmer_pair = self.counter.str_to_kmer_counts(pair[0]), \
            self.counter.str_to_kmer_counts(pair[1])
        return self.distance.transform(kmer_pair)


class EditDistance(Distance):
    """
    Normalized Levenshtein edit distance between textual DNA sequences.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligner = Levenshtein.distance

    def transform(self, pair: tuple) -> int:
        return self.aligner(*pair) / max(map(len, pair))


class SmithWaterman(Distance):
    """
    Normalized alignment distance between two textual DNA sequences. Distance is computed from
    Smith-Waterman local alignment similarity scores. Unpublished, not recommended unless this
    legacy functionality is necessary.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligner = PairwiseAligner()

    def transform(self, pair: tuple) -> int:
        return self.aligner.align(*pair).score / max(map(len, pair))

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        data = 1 - data  # Convert similarity scores into distances
        return super().postprocessor(data)

    def invert_postprocessing(self, data: np.ndarray) -> np.ndarray:
        return 1 - super().invert_postprocessing(data)


class CompoundDistance(Distance):
    """
    Distance between two chemical compounds.
    """
    def transform(self, pair: tuple):
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pair[0], 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(pair[1], 2, nBits=1024)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        data = 1 - data  # Convert similarity scores into distances
        return super().postprocessor(data)
