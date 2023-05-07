import numpy as np
import multiprocessing as mp
from tqdm import tqdm as tqdm


class KMerCounter:
    def __init__(self, k: int, jobs=1, chunksize=1, rna=False):
        self.k = k
        self.jobs = jobs
        self.chunksize = chunksize
        self.alphabet = np.array(['A', 'C', 'G', 'U'] if rna else ['A', 'C', 'G', 'T'])
        self.alphabet_pattern = re.compile(f'[^{"".join(self.alphabet)}]')
        alphabet_view = alphabet.view(np.int32)
        
        # Make a lookup table with an entry for every possible data byte
        self.lookup_table = np.zeros(256, dtype=np.uint32)
        self.lookup_table[alphabet_view[0]] = 0
        self.lookup_table[alphabet_view[1]] = 1
        self.lookup_table[alphabet_view[2]] = 2
        self.lookup_table[alphabet_view[3]] = 3

    def seq_to_kmers(self, seq: np.ndarray) -> np.ndarray:
        binary_converted = self.lookup_table[seq]  # Convert seq to integers based on encoding scheme
        stride = np.lib.stride_tricks.sliding_window_view(binary_converted, 2)
        kmers = np.copy(stride[:, -1])
        for i in range(stride.shape[1] - 2, -1, -1):
            kmers += stride[:, i] << (stride.shape[1] - i + 1) * 2
        return kmers

    @staticmethod
    def _kmer_counts(seq: np.ndarray) -> np.ndarray:
        return np.unique(seq, return_counts=True)[1]

    def kmer_sequences(self, seqs: np.ndarray, quiet=False, use_mp=True) -> list:
        int_seqs = [np.array([i]).view(np.uint32) for i in seqs]
        if not quiet:
            print('Generating kmer sequences...')
        if use_mp:
            with mp.Pool(self.jobs) as p:
                it = p.imap(self.seq_to_kmers, int_seqs, chunksize=self.chunksize) if quiet else \
                    tqdm(p.imap(self.seq_to_kmers, int_seqs, chunksize=self.chunksize), total=len(int_seqs))
                return list(it)
        else:
            it = int_seqs if quiet else tqdm(int_seqs)
            return [self.seq_to_kmers(i) for i in it]

    def kmer_counts(self, seqs: np.ndarray, quiet=False) -> list:
        kmers = self.kmer_sequences(seqs, quiet=quiet, use_mp=False)
        if not quiet:
            print('Counting unique kmers...')
        with mp.Pool(self.jobs) as p:
            it = p.imap(self._kmer_counts, kmers, chunksize=self.chunksize) if quiet else \
                tqdm(p.imap(self._kmer_counts, kmers, chunksize=self.chunksize), total=len(kmers))
            return list(it)
