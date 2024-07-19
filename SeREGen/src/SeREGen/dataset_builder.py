"""
Classes to help load and preprocess a dataset from the filesystem.
"""
import typing
import re
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .kmers import KMerCounter, Nucleotide_AA
tqdm.pandas()


class _SequenceTrimmer:
    """
    Helper class for Python multiprocessing library, allows for sequence trimming and padding
    to be multiprocessed with arbitrary trim lengths.
    """

    def __init__(self, length: int):
        self.length = length

    def trim(self, seq: str) -> str:
        """
        Performs trimming.
        """
        seq = seq[:self.length]
        return seq + ('N' * (self.length - len(seq)))


class LabelSeries(pd.Series):
    """
    Extends pd.Series, contains additional methods to generate masks based on labels.
    """
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
    """
    Constructs a Dataset from input files.
    """

    def __init__(self, header_parser=None):
        """
        @param header_parser: HeaderParser object for header parsing
        """
        self.header_parser = header_parser

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
    def _dataset_decorator(cls_type):
        """
        Adds necessary _constructor and _constructor_sliced properties to new Dataset objects.
        """
        class DatasetDecorated(Dataset):
            """
            Defines _constructor and _constructor_sliced for Dataset.
            """
            @property
            def _constructor(self):
                return DatasetDecorated

            @property
            def _constructor_sliced(self):
                return cls_type
        return DatasetDecorated

    def from_fasta(self, paths: list[str], max_rows=None):
        """
        Factory function that builds a dataset from a fasta file. Reads in all sequences from all
        fasta files in list.
        @param paths: list of fasta file paths.
        @return dataset: new dataset object
        """
        raw_headers, seqs = [], []
        for i in paths:
            headers, s = self._read_fasta(i)
            raw_headers.append(headers)
            seqs.append(s)
        raw_headers = np.concatenate(raw_headers)[:max_rows or -1]
        seqs = np.concatenate(seqs)[:max_rows or -1]
        labels = self.header_parser(raw_headers) if self.header_parser else None
        cls = self._dataset_decorator(type(labels)) if labels is not None else Dataset
        return cls({'orig_seqs': seqs, 'seqs': seqs, 'raw_headers': raw_headers, 'labels': labels})


class Dataset(pd.DataFrame):
    """
    Useful class for handling sequence data. Underlying storage container is a pandas DataFrame.
    Columns:
    orig_seqs: raw, unprocessed sequence data. acts like a "backup" when performing transformations
    on sequences.
    seqs: sequences that can be transformed by built-in functions
    raw_headers: raw, unprocessed header data.
    labels: label data, present only if HeaderParser passed to DatasetBuilder or if manually added
    """
    @property
    def _constructor(self):
        return Dataset

    def add_labels(self, lbl_rows: pd.Series, lbl_cols: list[str]):
        """
        Add label data to the dataframe after dataset creation. Allows for other methods of label
        parsing. Returns a new Dataset with label data added.
        @param lbl_rows: _LabelSeries column.
        @param lbl_cols: list of labels represented by each index.
        @return Dataset: subclass of Dataset with label data added.
        """
        # pylint: disable=protected-access
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
        return self.loc[self['labels'].correct_length_mask()].dropna()

    def length_dist(self, progress=True):
        """
        Plots a histogram of the sequence lengths in this dataset.
        Helpful for selecting a trim length.
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

    def replace_unknown_nucls(self):
        """
        Replace all unknown nucleotide base pairs in all sequences with N.
        """
        self['seqs'] = self['seqs'].apply(lambda i: re.sub('[^ATGCUN]', 'N', i))

    def gen_kmer_seqs(self, K=-1, jobs=1, chunksize=1, max_len=None, progress=True,
                      avoid_pad=True, avoid_oov=False, counter=None) -> list[np.ndarray]:
        """
        Convert all sequences to ordinal encoded kmer sequences.
        @param max_len: If given, trims and pads sequences to this length.
        @param avoid_pad: Adds 1 to all kmer values to dodge pad token. True by default.
        @param avoid_oov: Additionally adds 1 to all kmer values to dodge oov token. False by
        default.
        """
        if counter is None and K == -1:
            raise ValueError('Either K must be specified or a KMerCounter object must be passed!')
        counter = counter or KMerCounter(K, jobs=jobs, chunksize=chunksize, silence=not progress)
        kmers = counter.kmer_sequences(self['seqs'].to_numpy())
        # Ensures no kmer values conflict with special tokens.
        kmers = [i + int(avoid_pad) + int(avoid_oov) for i in kmers]
        if not max_len:
            return kmers

        padded_kmers = pad_sequence(kmers, batch_first=True, padding_value=0)
        # Trim to max_len
        if padded_kmers.shape[1] > max_len:
            padded_kmers = padded_kmers[:, :max_len]
        return padded_kmers.numpy()

    def aa_seqs(self, jobs=1, chunksize=1, progress=True, converter=None) -> list[str]:
        """
        Returns amino acid sequences based on the sliding window kmers method. Includes all possible
        interpretations of the underlying nucleotide sequences. Non-invertible.
        """
        converter = converter or Nucleotide_AA(jobs=jobs, chunksize=chunksize, progress=progress)
        return converter.transform(self['seqs'].to_numpy())

    def count_kmers(self, K=-1, jobs=1, chunksize=1, progress=True, counter=None) -> np.ndarray:
        """
        Count kmers for all sequences.
        @param K: Length of sequences to match.
        @param jobs: number of multiprocessing jobs
        @param chunksize: chunksize for multiprocessing
        @param progress: optional progress bar
        @return np.ndarray: counts of each kmer for all sequences
        """
        if counter is None and K == -1:
            raise ValueError('Either K must be specified or a KMerCounter object must be passed!')
        counter = counter or KMerCounter(K, jobs=jobs, chunksize=chunksize, silence=not progress)
        return counter.kmer_counts(self['seqs'].to_numpy())


class HeaderParser:
    """
    HeaderParser class that can be extended to parse headers from FASTA files into labels.
    """

    def __init__(self, label_extractor: callable, label_cols: list[str]):
        """
        It's easily possible to create a new custom HeaderParser with a custom label_extractor
        function and a custom label_cols list. No subclassing necessary.
        @param label_extractor: function that takes in a raw header string and outputs a list of
        labels.
        @param label_cols: list of strings that store what each position in the label list
        represents.
        """
        self.label_extractor = label_extractor
        self.label_cols = label_cols

    @staticmethod
    def _lbl_series_decorator(label_cols: list[str]) -> type:
        """
        Hidden decorator function that returns a subclass of LabelSeries with the 'labels' attribute
        defined.
        @param label_cols: list of labels
        """
        class LabelSeriesDecorated(LabelSeries):
            """
            Defines _constructor for LabelSeries class.
            """
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


def _silva_tax_extractor(header: str):
    return np.array(' '.join(header.split(' ')[1:]).split(';'))


SILVA_header_parser = HeaderParser(_silva_tax_extractor,
                                   ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus',
                                    'Species'])


def _covid_variant_extractor(header: str):
    return np.array([header.split('|')[2]], dtype=str)


COVID_header_parser = HeaderParser(_covid_variant_extractor, ['Variant'])
