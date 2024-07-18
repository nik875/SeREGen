"""
Contains distance metrics used for training ComparativeEncoders.
"""

import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from Bio.Align import PairwiseAligner
import Levenshtein
from rdkit.Chem import AllChem, DataStructs
from .kmers import KMerCounter


# pylint: disable=method-hidden
class Distance:
    """
    Abstract class representing a distance metric for two sequences.
    Downstream subclasses must implement transform.
    """

    def __init__(self, jobs=1, chunksize=1, silence=False):
        self.jobs = jobs
        self.chunksize = chunksize
        self.silence = silence

    # pylint: disable=unused-argument
    def transform(self, pair: tuple) -> int:
        """
        Transform a pair of elements into a single integer distance between those elements. This can
        return a non-distance score if postprocessor() and invert_postprocessing are implemented.
        @param pair: two-element tuple containing elements to compute distance between.
        @return int: distance value
        """
        return 0

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        """
        Postprocess the output of transform() into a distance. Allows for metrics that don't
        explicitly calculate distance via postprocessing similarity scores/other results into
        distance. Must be invertible.
        @param data: np.ndarray
        @return np.ndarray
        """
        return data

    def transform_multi(self, y1: np.ndarray, y2: np.ndarray, silence=False) -> np.ndarray:
        """
        Transform two large arrays of data.
        """
        if self.jobs == 1 or len(y1) < self.chunksize:
            it = (zip(y1, y2) if self.silence or silence else tqdm(zip(y1, y2), total=len(y1)))
            y = np.fromiter((self.transform(i) for i in it), dtype=np.float64)
        else:
            with mp.Pool(self.jobs) as p:
                it = p.imap(self.transform, zip(y1, y2), chunksize=self.chunksize)
                y = np.fromiter(
                    (it if self.silence or silence else tqdm(
                        it, total=len(y1))), dtype=np.float64,)
        y = self.postprocessor(y)  # Vectorized transformations are applied here
        return y

    def invert_postprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Inverts the postprocessing of data to return raw transform results.
        @param data: np.ndarray
        @return np.ndarray
        """
        return data


class VectorizedDistance(Distance):
    """
    Distance with a vectorized implementation. Takes advantage of multiprocessing and
    vectorization. transform() must take in pairs of single elements OR pairs of arrays of elements.
    """

    def transform_multi(self, y1, y2, silence=False):
        if self.jobs == 1 or len(y1) < self.chunksize:
            return self.postprocessor(self.transform((y1, y2)))
        y1_split = np.array_split(y1, len(y1) // self.chunksize)
        y2_split = np.array_split(y2, len(y1) // self.chunksize)
        with mp.Pool(self.jobs) as p:
            it = p.imap(self.transform, zip(y1_split, y2_split), chunksize=1)
            y = list(it if self.silence or silence else tqdm(it, total=len(y1_split)))
            y = np.concatenate(y)
        y = self.postprocessor(y)
        return y


class Euclidean(VectorizedDistance):
    """
    Basic Euclidean distance implementation
    """

    def transform(self, pair: tuple) -> int:
        return np.linalg.norm(pair[0] - pair[1], axis=-1)


class Cosine(VectorizedDistance):
    """
    Cosine distance implementation.
    """

    def transform(self, pair: tuple) -> int:
        # Subtracting from 1 to convert similarity to distance
        return np.sum(pair[0] * pair[1], axis=-1) / (
            np.linalg.norm(pair[0], axis=-1) * np.linalg.norm(pair[1], axis=-1))

    def postprocessor(self, data):
        return 1 - super().postprocessor(data)

    def invert_postprocessing(self, data):
        return 1 - super().invert_postprocessing(data)


class Hyperbolic(VectorizedDistance):
    """
    Computes hyperbolic distance between two arrays of points in Poincaré ball model.
    Numpy implementation.
    """

    def transform(self, pair):
        a, b = pair
        a = a.astype(np.float64)  # Both arrays must be the same type
        b = b.astype(np.float64)
        eps = np.finfo(np.float64).eps  # Machine epsilon

        def sq_norm(v):
            return np.clip(np.sum(v**2, axis=-1), eps, 1 - eps)

        numerator = np.sum((a - b) ** 2, axis=-1)
        denominator_a = 1 - sq_norm(a)
        denominator_b = 1 - sq_norm(b)
        frac = numerator / (denominator_a * denominator_b)
        return np.arccosh(1 + 2 * frac)


class IncrementalDistance(VectorizedDistance):
    """
    Incrementally applies a regular K-Mers based distance metric over raw sequences.
    Use when not enough memory exists to fully encode a dataset into K-Mers with the specified K.
    Meant for use with VectorizedDistances. distance must have a transform() that can handle pairs
    of arrays of elements, and these arrays can have length 1.
    """

    def __init__(self, distance: VectorizedDistance, counter: KMerCounter):
        super().__init__()
        self.distance = distance
        self.counter = counter
        self.jobs = self.distance.jobs
        self.chunksize = self.distance.chunksize

    def transform(self, pair: tuple) -> int:
        pair = [i if isinstance(i, np.ndarray) else np.array([i]) for i in pair]
        kmer_pair = [self.counter.kmer_counts(i, jobs=1, chunksize=1, silence=True) for i in pair]
        return self.distance.transform(tuple(kmer_pair))


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
    Smith-Waterman local alignment similarity scores. Deprecated, not recommended unless this
    legacy functionality is necessary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aligner = PairwiseAligner()

    def transform(self, pair: tuple) -> int:
        return self.aligner.align(*pair).score / max(map(len, pair))

    def postprocessor(self, data: np.ndarray) -> np.ndarray:
        return 1 - super().postprocessor(data)

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
        return 1 - super().postprocessor(data)

    def invert_postprocessing(self, data):
        return 1 - super().invert_postprocessing(data)
