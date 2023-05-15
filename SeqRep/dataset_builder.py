import typing
import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

from .kmers import KMerCounter
tqdm.pandas()


# Avoid creating encoder object every single time encode is called
_ENC = LabelBinarizer(sparse_output=False)
_ENC.fit(['A', 'C', 'G', 'N', 'U'])


class _SequenceTrimmer:
    """
    Helper class for Python multiprocessing library, allows for sequence trimming and padding
    to be multiprocessed with arbitrary trim lengths.
    """
    def __init__(self, length: int):
        self.length = length

    def trim(self, seq: str) -> str:
        seq = seq[:self.length]
        return seq + ('N' * (self.length - len(seq)))


class TaxSeries(pd.Series):
    _metadata = ['tax']

    @property
    def _constructor(self):
        return TaxSeries

    def tax_mask(self, level: typing.Union[str, int], value: str) -> pd.Series:
        """
        Returns a mask for all values of this series where the given taxonomic level equals the value.
        @param level: taxonomic rank to target search
        @param value: search query
        @return pd.Series: boolean mask
        """
        level = self.tax.index(level) if isinstance(level, str) else level
        assert level != -1
        return self.apply(lambda i: i[level] == value)

    def correct_length_mask(self) -> pd.Series:
        """
        Return a mask for all values of this series with a correct number of elements.
        @return pd.Series: boolean mask
        """
        return self.apply(len) == len(self.tax)

    def index_of_level(self, value: str) -> int:
        """
        Given a taxonomic level, return the index of that level.
        @param value: taxonomic level
        @return int: index
        """
        return self.tax.index(value)


class DatasetBuilder:
    @staticmethod
    def _read_fasta(path: str):
        """
        Reads a fasta file using BioPython and returns a list of tuples: (name, sequence).
        """
        headers, seqs = [], []
        with open(path, 'r') as f:
            for header, seq in SeqIO.FastaIO.SimpleFastaParser(f):
                headers.append(header)
                seqs.append(seq)
        return np.array(headers), np.array(seqs)

    @staticmethod
    def _dataset_decorator(cls):
        class DatasetDecorated(Dataset):
            @property
            def _constructor(self):
                return DatasetDecorated

            @property
            def _constructor_sliced(self):
                return cls
        return DatasetDecorated

    def from_fasta(self, paths: list[str]):
        """
        Factory function that builds a dataset from a fasta file. reads in all sequences from all fasta files in list.
        @param paths: list of fasta file paths.
        @return dataset: new dataset object
        """
        raw_headers, seqs = [], []
        for i in paths:
            headers, s = self._read_fasta(i)
            raw_headers.append(headers)
            seqs.append(s)
        raw_headers = np.concatenate(raw_headers)
        seqs = np.concatenate(seqs)
        tax = self.header_parser(raw_headers) if self.header_parser else None
        cls = self._dataset_decorator(type(tax)) if tax is not None else Dataset
        return cls({'orig_seqs': seqs, 'seqs': seqs, 'raw_headers': raw_headers, 'tax': tax})

    def __init__(self, header_parser=None):
        """
        @param header_parser: HeaderParser object for header parsing
        """
        self.header_parser = header_parser


class Dataset(pd.DataFrame):
    """
    Useful class for handling sequence data. Underlying storage container is a pandas DataFrame.
    Columns:
    orig_seqs: raw, unprocessed sequence data. acts like a "backup" when performing transformations on sequences.
    seqs: sequences that can be transformed by built-in functions
    raw_headers: raw, unprocessed header data.
    tax: taxonomic information, present only if HeaderParser passed to DatasetBuilder or if manually added
    """
    @property
    def _constructor(self):
        return Dataset

    def add_tax_data(self, tax_col: pd.Series, tax_hierarchy: list[str]):
        """
        Add taxonomic classification data to the dataframe after dataset creation. Allows for other tools to be used.
        Returns a new Dataset with tax data added.
        to taxonomically classify previously unclassified data.
        @param tax_col: _TaxSeries column.
        @param tax_hierarchy: list of taxonomic levels represented by each index.
        @return Dataset: subclass of Dataset with tax data added.
        """
        tax_series_type = HeaderParser._tax_series_decorator(tax_hierarchy)
        series = tax_series_type(tax_col)
        dataset_type = DatasetBuilder._dataset_decorator(tax_series_type)
        return dataset_type({'orig_seqs': self['orig_seqs'], 'seqs': self['seqs'],
                             'raw_headers': self['raw_headers'], 'tax': series})

    def drop_bad_headers(self):
        """
        Drop all elements which have a header of the wrong length. Returns new DataFrame.
        """
        assert self['tax'] is not None
        return self.loc[self['tax'].correct_length_mask()]

    def length_dist(self, progress=True):
        """
        Plots a histogram of the sequence lengths in this dataset. Helpful for selecting a trim length.
        @param progress: show progress bar during length calculations
        """
        plt.hist(self['seqs'].progress_apply(len) if progress else self['seqs'].apply(len))
        plt.show()

    def trim_seqs(self, length: int):
        """
        Trim all sequences to the given length and pad sequences with N which are too short.
        @param length: length to trim to.
        """
        trimmer = _SequenceTrimmer(length)
        self['seqs'] = self['orig_seqs'].apply(trimmer.trim)

    @staticmethod
    def _encode_sequence(seq: str) -> np.ndarray:
        """
        One hot encode a single sequence.
        @param seq: string sequence with padding.
        @return np.ndarray: Encoded sequence
        """
        return _ENC.transform(list(seq))

    def one_hot_encode(self, jobs=1, chunksize=1, progress=True):
        """
        One hot encode all sequences in this dataset.
        @param jobs: number of multiprocessing jobs
        @param chunksize: chunksize for multiprocessing
        @param progress: optional progress bar
        """
        with mp.Pool(jobs) as p:
            imap = p.imap(self._encode_sequence, self['seqs'], chunksize=chunksize)
            return np.array(list(tqdm(imap, total=len(self)) if progress else imap))

    def gen_kmer_seqs(self, K, jobs=1, chunksize=1, progress=True) -> list:
        """
        Convert all sequences to kmer sequences.
        """
        counter = KMerCounter(K, jobs=jobs, chunksize=chunksize)
        return counter.kmer_sequences(self['seqs'].to_numpy, quiet=progress)

    def count_kmers(self, K: int, jobs=1, chunksize=1, progress=True) -> list:
        """
        Count kmers for all sequences.
        @param K: Length of sequences to match.
        @param jobs: number of multiprocessing jobs
        @param chunksize: chunksize for multiprocessing
        @param progress: optional progress bar
        @return list: counts of each kmer for all sequences
        """
        counter = KMerCounter(K, jobs=jobs, chunksize=chunksize)
        return counter.kmer_counts(self['seqs'].to_numpy(), quiet=progress)


class HeaderParser:
    def __init__(self, tax_extractor: callable, tax_hierarchy: list[str]):
        """
        It's easily possible to create a new custom HeaderParser with a custom tax_extractor function
        and a custom tax_hierarchy list. No subclassing necessary.
        @param tax_extractor: function that takes in a raw header string and outputs a list of taxonomic elements.
        @param tax_hierarchy: list of strings that store what each position in the taxonomic list represents.
        """
        self.tax_extractor = tax_extractor
        self.tax_hierarchy = tax_hierarchy

    @staticmethod
    def _tax_series_decorator(tax_hierarchy: list[str]) -> type:
        """
        Hidden decorator function that returns a subclass of TaxSeries with the 'tax' attribute defined.
        @param tax_hierarchy: list of taxonomic levels
        """
        class TaxSeriesDecorated(TaxSeries):
            tax = tax_hierarchy

            @property
            def _constructor(self):
                return TaxSeriesDecorated
        return TaxSeriesDecorated

    def __call__(self, data: list[str]) -> TaxSeries:
        """
        __call__ is used to actually build the header dataset.
        @param data: list of unparsed headers
        @return _TaxSeries: custom pandas series for header data
        """
        cls = self._tax_series_decorator(self.tax_hierarchy)
        return cls([self.tax_extractor(i) for i in data])


class SILVAHeaderParser(HeaderParser):
    """
    Predefined header parser for the SILVA dataset.
    """
    def __init__(self):
        """
        Passes a custom tax_extractor and tax_hierarchy to superclass.
        """
        def tax_extractor(header: str):
            return np.array(' '.join(header.split(' ')[1:]).split(';'))
        super().__init__(tax_extractor, ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
