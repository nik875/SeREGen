import re
import numpy as np
import multiprocessing as mp
from tqdm import tqdm as tqdm


class OneHotEncoder:
    def __init__(self, alphabet=None, jobs=1, chunksize=1):
        self.jobs, self.chunksize = jobs, chunksize
        self.alphabet = alphabet if alphabet is not None else {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        alphabet_view = np.array(list(self.alphabet.keys())).view(np.uint32)
        self.unk_val = max(self.alphabet.values()) + 1  # Add an extra token for unknown base pairs
        self.lookup_table = np.full((256,), self.unk_val, dtype=np.uint32)  # Create a lookup table for base pair bytes
        self.lookup_table[alphabet_view] = list(self.alphabet.values())
        self.arr_size = self.unk_val + 1
    
    def _encode_seq(self, seq: np.ndarray) -> np.ndarray:
        result = np.zeros((len(seq), self.arr_size))
        enc = self.lookup_table[seq]
        result[np.arange(len(seq)), enc] = 1
        return result
    
    def encode_str(self, seq: str) -> np.ndarray:
        return self._encode_seq(np.array([seq]).view(np.uint32))
    
    def encode_seqs(self, seqs: list[str], trim_to=None, quiet=False) -> list[np.ndarray]:
        if trim_to is None and not all(len(i) == len(seqs[0]) for i in seqs):
            with mp.Pool(self.jobs) as p:
                it = p.imap(self.encode_str, seqs, chunksize=self.chunksize) if quiet else \
                    tqdm(p.imap(self.encode_str, seqs, chunksize=self.chunksize), total=len(seqs))
                return list(it)
        if trim_to is not None:
            assert all(len(i) >= trim_to for i in seqs)
            seqs = [i[:trim_to] for i in seqs]
        seqs = np.stack([np.array([i]) for i in seqs]).view(np.uint32)
        encodings = self.lookup_table[seqs.flatten()]
        ind = np.concatenate(np.array(np.meshgrid(np.arange(seqs.shape[0]), np.arange(seqs.shape[1]))).T)
        result = np.zeros((seqs.shape[0], seqs.shape[1], self.arr_size))
        result[ind[:, 0], ind[:, 1], encodings] = 1
        return result
