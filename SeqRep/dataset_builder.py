import typing
import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from .kmers import KMerCounter
from .ohe import OneHotEncoder
tqdm.pandas()


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


class LabelSeries(pd.Series):
    _metadata = ['labels']

    @property
    def _constructor(self):
        return LabelSeries

    def label_mask(self, label: typing.Union[str, int], value: str) -> pd.Series:
        """
        Returns a mask for all values of this series where the given label equals the value.
        @param label: label to target search
        @param value: search query
        @return pd.Series: boolean mask
        """
        label = self.labels.index(label) if isinstance(label, str) else label
        if label == -1:
            raise ValueError('Label not in dataset!')
        return self.apply(lambda i: i[label] == value)

    def correct_length_mask(self) -> pd.Series:
        """
        Return a mask for all values of this series with a correct number of elements.
        @return pd.Series: boolean mask
        """
        return self.apply(len) == len(self.labels)

    def index_of_label(self, label: str) -> int:
        """
        Given a label, return the index of that label.
        @param label: label to search for
        @return int: index
        """
        return self.labels.index(label)


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
        Factory function that builds a dataset from a fasta file. Reads in all sequences from all fasta files in list.
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
        labels = self.header_parser(raw_headers) if self.header_parser else None
        cls = self._dataset_decorator(type(labels)) if labels is not None else Dataset
        return cls({'orig_seqs': seqs, 'seqs': seqs, 'raw_headers': raw_headers, 'labels': labels})

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
    labels: label data, present only if HeaderParser passed to DatasetBuilder or if manually added
    """
    @property
    def _constructor(self):
        return Dataset

    def add_labels(self, lbl_rows: pd.Series, lbl_cols: list[str]):
        """
        Add label data to the dataframe after dataset creation. Allows for other methods of label parsing.
        Returns a new Dataset with label data added.
        @param lbl_rows: _LabelSeries column.
        @param lbl_cols: list of labels represented by each index.
        @return Dataset: subclass of Dataset with label data added.
        """
        lbl_series_type = HeaderParser._lbl_series_decorator(lbl_cols)
        series = lbl_series_type(lbl_rows)
        dataset_type = DatasetBuilder._dataset_decorator(lbl_series_type)
        return dataset_type({'orig_seqs': self['orig_seqs'], 'seqs': self['seqs'],
                             'raw_headers': self['raw_headers'], 'labels': series})

    def drop_bad_headers(self):
        """
        Drop all elements which have a header of the wrong length. Returns new DataFrame.
        """
        assert self['labels'] is not None
        return self.loc[self['labels'].correct_length_mask()]

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
        enc = OneHotEncoder()
        return enc.encode_str(seq)

    def one_hot_encode(self, jobs=1, chunksize=1, trim_to=None, progress=True):
        """
        One hot encode all sequences in this dataset.
        @param jobs: number of multiprocessing jobs
        @param chunksize: chunksize for multiprocessing
        @param progress: optional progress bar
        """
        enc = OneHotEncoder(jobs=jobs, chunksize=chunksize)
        return enc.encode_seqs(self['seqs'].to_numpy(), trim_to=trim_to, quiet=not progress)

    def gen_kmer_seqs(self, K, jobs=1, chunksize=1, trim_to=None, progress=True) -> list:
        """
        Convert all sequences to one-hot encoded kmer sequences.
        """
        counter = KMerCounter(K, jobs=jobs, chunksize=chunksize)
        return counter.kmer_sequences_ohe(self['seqs'].to_numpy(), trim_to=trim_to, quiet=not progress)

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
        return counter.kmer_counts(self['seqs'].to_numpy(), quiet=not progress)


class HeaderParser:
    def __init__(self, label_extractor: callable, label_cols: list[str]):
        """
        It's easily possible to create a new custom HeaderParser with a custom label_extractor function
        and a custom label_cols list. No subclassing necessary.
        @param label_extractor: function that takes in a raw header string and outputs a list of labels.
        @param label_cols: list of strings that store what each position in the label list represents.
        """
        self.label_extractor = label_extractor
        self.label_cols = label_cols

    @staticmethod
    def _lbl_series_decorator(label_cols: list[str]) -> type:
        """
        Hidden decorator function that returns a subclass of LabelSeries with the 'labels' attribute defined.
        @param label_cols: list of labels
        """
        class LabelSeriesDecorated(LabelSeries):
            labels = label_cols

            @property
            def _constructor(self):
                return LabelSeriesDecorated
        return LabelSeriesDecorated

    def __call__(self, data: list[str]) -> LabelSeries:
        """
        __call__ is used to actually build the header dataset.
        @param data: list of unparsed headers
        @return LabelSeries: custom pandas series for header data
        """
        cls = self._lbl_series_decorator(self.label_cols)
        return cls([self.label_extractor(i) for i in data])


class SILVAHeaderParser(HeaderParser):
    """
    Predefined header parser for the SILVA dataset.
    """
    def __init__(self):
        """
        Passes a custom label_extractor and label_cols to superclass.
        """
        def tax_extractor(header: str):
            return np.array(' '.join(header.split(' ')[1:]).split(';'))
        super().__init__(tax_extractor, ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])


class COVIDDataParser(HeaderParser):
    """
    Predefined header parser for COVID nucleotide sequence data downloads from NCBI
    """
    def __init__(self):
        """
        Passes a custom label_extractor and label_cols to superclass.
        """
        def label_extractor(header: str):
            return np.array([header.split('|')[2]], dtype=str)
        super().__init__(label_extractor, ['Variant'])

