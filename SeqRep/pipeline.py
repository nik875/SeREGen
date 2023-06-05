"""
Automated pipelines for sequence representation generation.
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import BallTree
from .dataset_builder import Dataset, DatasetBuilder, SILVAHeaderParser, COVIDHeaderParser
from .visualize import repr_scatterplot
from .kmers import KMerCounter
from .compression import PCA, IPCA, AE, Compressor
from .encoders import ModelBuilder
from .comparative_encoder import ComparativeEncoder
from .distance import cosine, IncrementalDistance


class Pipeline:
    """
    An abstract automated pipeline for sequence representation generation.
    """
    def __init__(self, paths: list[str], header_parser='None', quiet=False):
        """
        An automated pipeline for sequence representation generation.
        @param paths: FASTA file paths to read.
        @param header_parser: Header parser to use, defaults to 'None'.
        """
        if not isinstance(header_parser, str):
            builder = DatasetBuilder(header_parser)
        elif header_parser == 'SILVA':
            builder = DatasetBuilder(SILVAHeaderParser())
        elif header_parser == 'COVID':
            builder = DatasetBuilder(COVIDHeaderParser())
        else:
            builder = DatasetBuilder()
        self.dataset = builder.from_fasta(paths)
        self.quiet = quiet
        self.model = None
        self.preproc_reprs = None
        self.reprs = None
        self.index = None

    # Must be implemented by subclass
    # pylint: disable=unused-argument
    def preprocess_seq(self, seq):
        """
        Preprocesses a string sequence.
        @param seq: Sequence to preprocess.
        @return np.ndarray: Returns None by default.
        """
        return None

    # Should be overriden by subclass for efficient preprocessing
    def preprocess_seqs(self, seqs: list) -> list:
        """
        Preprocesses a list of sequences.
        @param seqs: Sequences to preprocess.
        @return list: Returns an array of preprocessed sequences.
        """
        return [self.preprocess_seq(i) for i in (tqdm(seqs) if not self.quiet else seqs)]

    # Must be implemented by subclass, super method must be called by implementation.
    # This super method preprocesses the dataset into self.preproc_reprs.
    # This variable is used to determine whether fit was called and to avoid preprocessing the
    # dataset twice between fit and transform_dataset.
    def fit(self, **kwargs):
        """
        Fit the model to the dataset.
        """
        if not self.quiet:
            print('Preprocessing dataset...')
        self.preproc_reprs = self.preprocess_seqs(self.dataset['seqs'].to_numpy(), **kwargs)

    def _fit_called_check(self):
        if self.preproc_reprs is None:
            raise ValueError('Fit must be called before transform!')

    # Must be implemented by subclass.
    def transform_after_preproc(self, data):
        """
        Convert preprocessed sequence representations into final encodings.
        """
        self._fit_called_check()
        return data

    def transform(self, seqs: list) -> list:
        """
        Transform an array of string sequences to learned representations.
        @param seqs: List of string sequences to transform.
        @return list: Sequence representations.
        """
        self._fit_called_check()
        return self.transform_after_preproc(self.preprocess_seqs(seqs))

    def transform_dataset(self) -> np.ndarray:
        """
        Transforms the loaded dataset into representations. Saves as self.reprs and returns result.
        Deletes any existing search tree.
        """
        self._fit_called_check()
        self.reprs = self.transform_after_preproc(self.preproc_reprs)
        self.index = None  # Delete existing search tree because we assume reprs have changed.
        return self.reprs

    def _reprs_check(self):
        """
        Wraps logic to check that reprs exist.
        """
        if self.reprs is None:
            raise ValueError('transform_dataset must be called first!')

    def visualize_axes(self, x: int, y: int, **kwargs):
        """
        Visualizes two axes of the dataset representations on a simple scatterplot.
        @param x: which axis to use as x.
        @param y: which axis to use as y.
        @param kwargs: Accepts additional keyword arguments for visualize.repr_scatterplot().
        """
        self._reprs_check()
        repr_scatterplot(np.stack([self.reprs[:, x], self.reprs[:, y]], axis=1), **kwargs)

    def visualize_2D(self, **kwargs):
        """
        Visualizes 2D dataset as a scatterplot. Keyword arguments to repr_scatterplot are accepted.
        """
        self._reprs_check()
        if len(self.reprs.shape) != 2 or self.reprs.shape[1] != 2:
            raise ValueError('Incompatible representation dimensions!')
        self.visualize_axes(0, 1, **kwargs)

    def search(self, query: list[str], n_neighbors=1) -> tuple[np.ndarray, list[pd.Series]]:
        """
        Search the dataset for the most similar sequences to the query.
        @param query: List of string sequences to find similar sequences to.
        @param n_neighbors: Number of neighbors to find for each sequence. Defaults to 1.
        @return np.ndarray: Search results.
        """
        self._reprs_check()
        if self.index is None:  # If index hasn't been created, create it.
            if not self.quiet:
                print('Creating search index (this could take a while)...')
            self.index = BallTree(self.reprs)
        enc = self.preprocess_seqs(query)
        dists, ind = self.index.query(enc, k=n_neighbors)
        matches = [self.dataset.iloc[i] for i in ind]
        return dists, matches


class KMerCountsPipeline(Pipeline):
    """
    Automated pipeline using KMer Counts. Optionally compresses input data before training model.
    """
    def __init__(self, K: int, paths: list[str], repr_size=2, header_parser='None', quiet=False,
                 depth=3, counter_jobs=1, counter_chunksize=1, trim_to=0, compressor=None,
                 comp_fit_sample_frac=1, comp_repr_size=0, **comp_args):
        super().__init__(paths, header_parser=header_parser, quiet=quiet)
        if trim_to:  # Optional sequence trimming
            self.dataset.trim_seqs(trim_to)
        self.counter = KMerCounter(K, jobs=counter_jobs, chunksize=counter_chunksize, quiet=quiet)

        # Compression (defaults to 80% if a compressor is passed, otherwise none is applied)
        r = np.random.default_rng()
        sample = r.permutation(len(self.dataset))[:int(len(self.dataset) *
                                                       comp_fit_sample_frac)]
        sample = self.counter.kmer_counts(self.dataset['seqs'].to_numpy()[sample])
        postcomp_len = comp_repr_size or 4 ** K // 10 * 2
        if compressor == 'PCA':
            self.compressor = PCA(postcomp_len, quiet=quiet, **comp_args)
        elif compressor == 'IPCA':
            self.compressor = IPCA(postcomp_len, quiet=quiet, **comp_args)
        elif compressor == 'AE':
            self.compressor = AE.auto(sample, postcomp_len, **comp_args)
            if not quiet:
                print('AE Compressor Summary:')
                self.compressor.summary()
        else:
            self.compressor = Compressor(postcomp_len := 4 ** K, quiet)
        self.compressor.fit(sample)

        # Model (training is distributed by default across all available GPUs)
        builder = ModelBuilder((postcomp_len,), tf.distribute.MirroredStrategy())
        builder.dense(postcomp_len, depth=depth)
        self.model = ComparativeEncoder.from_model_builder(builder, dist=cosine,
                                                           output_dim=repr_size, quiet=quiet)
        if not quiet:
            self.model.summary()

    @classmethod
    def from_objs(cls, K: int, ds: Dataset, counter: KMerCounter, compressor=None, **kwargs):
        """
        Build a KMerCountsPipeline from Dataset, KMerCounter, and optional Compressor objects.
        """
        obj = cls(K, [], quiet=True, **kwargs)
        obj.quiet = kwargs['quiet'] if 'quiet' in kwargs else False
        obj.dataset = ds
        obj.counter = counter
        obj.compressor = compressor or Compressor(4 ** K, obj.quiet)

        builder = ModelBuilder((obj.compressor.postcomp_len,), tf.distribute.MirroredStrategy())
        builder.dense(obj.compressor.postcomp_len, depth=obj.model.depth)
        obj.model = ComparativeEncoder.from_model_builder(builder, dist=obj.model.distance,
                                                          output_dim=obj.model.repr_size,
                                                          quiet=obj.quiet)
        if not obj.quiet:
            obj.model.summary()
        return obj

    def preprocess_seq(self, seq: str) -> np.ndarray:
        counts = self.counter.str_to_kmer_counts(seq)
        return self.compressor.transform(np.array([counts]))[0]

    def preprocess_seqs(self, seqs: list[str], batch_size=0) -> np.ndarray:
        return self.compressor.count_kmers(self.counter, seqs, batch_size)

    def fit(self, preproc_batch_size=0, dist_on_preproc=False, incremental_dist=False,
            **model_fit_args):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        """
        super().fit(batch_size=preproc_batch_size)
        if incremental_dist:
            self.model.distance = IncrementalDistance(self.model.distance, self.counter)
            distance_on = self.dataset['seqs']
        elif dist_on_preproc:
            distance_on = self.preproc_reprs
        else:
            distance_on = self.counter.kmer_counts(self.dataset['seqs'].to_numpy())
        # For AECompressor, distances between encodings are meaningless
        distance_on = self.counter.kmer_counts(self.dataset['seqs'].to_numpy()) \
            if isinstance(self.compressor, AE) else self.preproc_reprs
        if not self.quiet:
            print('Training model...')
        self.model.fit(self.preproc_reprs, distance_on=distance_on, **model_fit_args)

    def transform_after_preproc(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform the given sequences. Accepts keyword arguments for ComparativeEncoder.transform().
        """
        super().transform_after_preproc(data)
        return self.model.transform(data, **kwargs)

