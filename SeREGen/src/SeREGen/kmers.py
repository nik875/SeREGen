"""
KMer library that handles KMer counting and KMer sequence encoding.
"""
import re
import multiprocessing as mp
import numpy as np
from tqdm import tqdm


class KMerCounter:
    """
    KMer Counter class that can convert DNA/RNA sequences into sequences of kmers or kmer
    frequency tables. Wraps logic for Python multiprocessing using jobs and chunksize.
    """

    def __init__(self, k: int, jobs=1, chunksize=1, debug=False, silence=False):
        """
        KMer Counter class that can convert DNA/RNA sequences into sequences of kmers or kmer
        frequency tables. Wraps logic for Python multiprocessing using jobs and chunksize.
        @param debug: disables multiprocessing to allow better tracebacks.
        """
        self.k = k
        self.jobs = jobs
        self.chunksize = chunksize
        self.debug = debug
        self.silence = silence
        alphabet = np.array(['A', 'C', 'G', 'T', 'U'])
        self.alphabet_pattern = re.compile(f'[^{"".join(alphabet)}]')

        # Make a lookup table with an entry for every possible data byte
        self.lookup_table = np.zeros(256, dtype=np.uint32)
        # A = 0 is implied by np.zeros
        self.lookup_table[ord('C')] = 1
        self.lookup_table[ord('G')] = 2
        self.lookup_table[ord('T')] = 3
        self.lookup_table[ord('U')] = 3  # U = T

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
        binary_converted = self.lookup_table[seq]  # Convert seq to integers
        stride = np.lib.stride_tricks.sliding_window_view(binary_converted, self.k)
        # If every base pair is a unique two-bit code, a kmer is the concatenation of these codes
        # In other words, kmer = sum(val << (kmer_len - idx) * 2 for idx, val in kmer_window)
        # where val is a 2-bit sequence
        kmers = np.copy(stride[:, -1])  # Start with a new array to store the sum
        for i in range(stride.shape[1] - 2, -1, -1):  # Iterate over columns in reverse
            # Add this column << (kmer_len - idx) * 2 to sum
            kmers += stride[:, i] << (stride.shape[1] - i - 1) * 2
        return kmers

    def str_to_kmers(self, seq: str) -> np.ndarray:
        """
        Convert a string sequence to integer kmers. Works around invalid base pairs.
        Remember that values of 0 and 1 are also used by keras for pad and OOV tokens! Be sure to
        add 2 before passing in a kmer sequence directly to a model.
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

    def _apply(self, func: callable, seqs: np.ndarray, jobs=None, chunksize=None,
               silence=False) -> list:
        """
        Avoids duplication of logic for kmer sequence/count generation.
        """
        jobs = jobs or self.jobs
        chunksize = chunksize or self.chunksize
        if not self.debug and jobs > 1:
            with mp.Pool(jobs) as p:
                it = p.imap(func, seqs, chunksize=chunksize)
                return list(it if self.silence or silence else tqdm(it, total=len(seqs)))
        else:
            return [func(i) for i in (seqs if self.silence or silence else tqdm(seqs))]

    def kmer_sequences(self, seqs: list[str], **kwargs) -> list[list[int]]:
        """
        Generate kmer sequences for a given array of string sequences.
        Sequences do not need to be uniform lengths. Invalid/unknown base pairs will be ignored.
        Remember that values of 0 and 1 are also used by keras for pad and OOV tokens! Be sure to
        add 2 before passing in a kmer sequence directly to a model.
        """
        return self._apply(self.str_to_kmers, seqs, **kwargs)

    def kmer_counts(self, seqs: list[str], **kwargs) -> np.ndarray:
        """
        Generate kmer counts for a given array of string sequences.
        Sequences do not need to be uniform lengths. Invalid/unknown base pairs will be ignored.
        """
        return np.array(self._apply(self.str_to_kmer_counts, seqs, **kwargs))


class Nucleotide_AA(KMerCounter):
    """
    Convert a nucleotide sequence to a "comprehensive" amino acid sequence containing all three
    possible start shifts. 'J', 'O', 'U', 'X' are letters unused in standard amino acid one-letter
    codes. We thus use X to represent all three kinds of stop codon.
    """
    AA_LOOKUP = np.array([
        'K',  # AAA
        'N',  # AAC
        'K',  # AAG
        'N',  # AAU
        'T', 'T', 'T', 'T',  # AC*
        'R',  # AGA
        'S',  # AGC
        'R',  # AGG
        'S',  # AGU
        'I',  # AUA
        'I',  # AUC
        'M',  # AUG
        'I',  # AUU
        'Q',  # CAA
        'H',  # CAC
        'Q',  # CAG
        'H',  # CAU
        'P', 'P', 'P', 'P',  # CC*
        'R', 'R', 'R', 'R',  # CG*
        'L', 'L', 'L', 'L',  # CU*
        'E',  # GAA
        'D',  # GAC
        'E',  # GAG
        'D',  # GAU
        'A', 'A', 'A', 'A',  # GC*
        'G', 'G', 'G', 'G',  # GG*
        'V', 'V', 'V', 'V',  # GU*
        'X',  # UAA (STOP)
        'Y',  # UAC
        'X',  # UAG (STOP)
        'Y',  # UAU
        'S', 'S', 'S', 'S',  # UC*
        'X',  # UGA  (STOP)
        'C',  # UGC
        'W',  # UGG
        'C',  # UGU
        'L',  # UUA
        'F',  # UUC
        'L',  # UUG
        'F'  # UUU
    ])

    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)

    def _kmer_seq_to_aa(self, seq: np.ndarray) -> str:
        return ''.join(self.AA_LOOKUP[seq])

    def transform(self, seqs: list[str]) -> np.ndarray:
        """
        Convert the given nucleotide sequences to comprehensive amino acid sequences.
        @param seqs: nucleotide sequences to convert.
        """
        kmers = self.kmer_sequences(seqs)
        if not self.silence:
            print('Converting to amino acid sequences...')
        return np.array(self._apply(self._kmer_seq_to_aa, kmers), dtype=str)
