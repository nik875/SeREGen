import re
import numpy as np
import multiprocessing as mp
from tqdm import tqdm as tqdm


class KMerCounter:
    def __init__(self, k: int, jobs=1, chunksize=1, debug=False):
        """
        KMer Counter class that can convert DNA/RNA sequences into sequences of kmers or kmer frequency tables.
        Wraps logic for Python multiprocessing using jobs and chunksize.
        @param debug: disables multiprocessing to allow better tracebacks.
        """
        self.k = k
        self.debug = debug
        self.jobs = jobs
        self.chunksize = chunksize
        self.alphabet = np.array(['A', 'C', 'G', 'T', 'U'])
        self.alphabet_pattern = re.compile(f'[^{"".join(self.alphabet)}]')
        alphabet_view = self.alphabet.view(np.int32)

        # Make a lookup table with an entry for every possible data byte
        self.lookup_table = np.zeros(256, dtype=np.uint32)
        self.lookup_table[alphabet_view[0]] = 0
        self.lookup_table[alphabet_view[1]] = 1
        self.lookup_table[alphabet_view[2]] = 2
        self.lookup_table[alphabet_view[3]] = 3
        self.lookup_table[alphabet_view[4]] = 3  # U = T

    def _split_str(self, seq: str) -> list[str]:
        """
        Splits the input using the self.alphabet_pattern regex pattern.
        Filters out anything smaller than k. Acts as a generator.
        """
        for p in self.alphabet_pattern.split(seq):
            if len(p) >= self.k:
                yield np.array([p]).view(np.uint32)

    def _seq_to_kmers(self, seq: np.ndarray) -> np.ndarray:
        """
        Convert a sequence of base pair bytes to a sequence of integer kmers.
        SEQ MUST NOT HAVE BASE PAIRS OUTSIDE OF ACGT/U!
        """
        binary_converted = self.lookup_table[seq]  # Convert seq to integers based on encoding scheme
        stride = np.lib.stride_tricks.sliding_window_view(binary_converted, self.k)
        # If every base pair is assigned a unique two-bit code, a kmer is simply the concatenation of these codes
        # In other words, kmer = sum(val << (kmer_len - idx) * 2 for idx, val in kmer_window) where val is a 2-bit sequence
        kmers = np.copy(stride[:, -1])  # Start with a new array to store the sum
        for i in range(stride.shape[1] - 2, -1, -1):  # Iterate over columns in reverse
            kmers += stride[:, i] << (stride.shape[1] - i - 1) * 2  # Add this column << (kmer_len - idx) * 2 to sum
        return kmers

    def str_to_kmers(self, seq: str) -> np.ndarray:
        """
        Convert a string sequence to integer kmers. Works around invalid base pairs.
        """
        # Split given string into valid parts and concatenate resulting kmer sequences for each part
        return np.concatenate([self._seq_to_kmers(i) for i in self._split_str(seq)])

    def _seq_to_kmer_counts(self, seq: np.ndarray) -> np.ndarray:
        """
        Convert an array of base pair bytes to an array of kmer frequencies.
        SEQ MUST NOT HAVE BASE PAIRS OUTSIDE OF ACGT/U!
        """
        kmers = self._seq_to_kmers(seq)
        kmer_counts = np.zeros(4 ** self.k)
        uniques, counts = np.unique(kmers, return_counts=True)
        kmer_counts[uniques] = counts
        return kmer_counts

    def str_to_kmer_counts(self, seq: str) -> np.ndarray:
        """
        Convert a string sequence to kmer counts. Works around invalid base pairs.
        """
        # Split given string into parts and take sum of each kmer's total occurrences in each part
        return np.sum([self._seq_to_kmer_counts(i) for i in self._split_str(seq)], axis=0)

    def _gen_kmers(self, seqs: np.ndarray, func: callable, quiet: bool, use_mp: bool) -> list:
        """
        Avoids duplication of logic for kmer sequence/count generation.
        """
        if use_mp:
            with mp.Pool(self.jobs) as p:
                it = p.imap(func, seqs, chunksize=self.chunksize) if quiet else \
                    tqdm(p.imap(func, seqs, chunksize=self.chunksize), total=len(seqs))
                return list(it)
        else:
            it = seqs if quiet else tqdm(seqs)
            return [func(i) for i in it]

    def kmer_sequences(self, seqs: list, quiet=False) -> list:
        """
        Generate kmer sequences for a given array of string sequences.
        Sequences do not need to be uniform lengths. Invalid/unknown base pairs will be ignored.
        """
        return self._gen_kmers(seqs, self.str_to_kmers, quiet, not self.debug)

    def kmer_counts(self, seqs: np.ndarray, quiet=False) -> np.ndarray:
        """
        Generate kmer counts for a given array of string sequences.
        Sequences do not need to be uniform lengths. Invalid/unknown base pairs will be ignored.
        """
        return self._gen_kmers(seqs, self.str_to_kmer_counts, quiet, not self.debug)
    
    def _ohe_seq(self, seq: np.ndarray) -> np.ndarray:
        """
        One-hot encode a numerical sequence.
        """
        s = np.zeros((len(seq), 4 ** self.k))
        s[np.arange(len(seq)), seq] = 1
        return s

    def kmer_sequences_ohe(self, seqs: list, trim_to=None, quiet=False) -> list:
        """
        Generate kmer sequences for a given array of string sequences. Result is one-hot encoded.
        Sequences do not need to be uniform length. Invalid/unknown base pairs will be ignored.
        """
        kmers = self.kmer_sequences(seqs, quiet=quiet)
        if trim_to is None:
            return self._gen_kmers(kmers, self._ohe_seq, quiet, False)
        assert all(len(i) >= trim_to for i in kmers)
        kmers = np.stack([i[:trim_to] for i in kmers])
        result = np.zeros((*kmers.shape, 4 ** self.k))
        indices = np.concatenate(np.array(np.meshgrid(np.arange(kmers.shape[0]), np.arange(kmers.shape[1]))).T)
        result[indices[:, 0], indices[:, 1], kmers.flatten()] = 1
        return result
