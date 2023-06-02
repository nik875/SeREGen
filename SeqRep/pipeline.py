"""
Automated pipelines for sequence representation generation.
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import BallTree
from .dataset_builder import DatasetBuilder, SILVAHeaderParser, COVIDHeaderParser
from .visualize import repr_scatterplot
from .kmers import KMerCounter
from .compression import PCACompressor, AECompressor, Compressor, count_kmers_mp, \
    count_kmers_batched
from .encoders import ModelBuilder
from .comparative_encoder import ComparativeEncoder
from .distance import Euclidean


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
        self.reprs = None
        self.index = None

    # pylint: disable=unused-argument
    def preprocess_seq(self, seq: str) -> np.ndarray:
        """
        Preprocesses a string sequence. Must be implemented by subclass.
        @param seq: Sequence to preprocess.
        @return np.ndarray: Returns None by default.
        """
        return None

    def preprocess_seqs(self, seqs: list[str]) -> list:
        """
        Preprocesses a list of sequences. Can be overriden by subclass.
        @param seqs: Sequences to preprocess.
        @return list: Returns a list of None by default.
        """
        return [self.preprocess_seq(i) for i in (tqdm(seqs) if not self.quiet else seqs)]

    def fit(self):
        """
        Fit the model to the dataset. Implemented by subclass.
        """

    def transform(self, seqs: list[str]) -> list:
        """
        Transform an array of string sequences to learned representations.
        Implemented by subclass. Same as preprocess_seqs by default.
        @param seqs: List of string sequences to transform.
        @return list: Sequence representations (list of None by default).
        """
        return self.preprocess_seqs(seqs)

    def transform_dataset(self) -> np.ndarray:
        """
        Transforms the loaded dataset into representations. Saves as self.reprs and returns result.
        Deletes any existing search tree.
        """
        self.reprs = self.transform(self.dataset['seqs'].to_numpy())
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
    def __init__(self, paths: list[str], K: int, repr_size=2, header_parser='None', quiet=False,
                 depth=3, jobs=1, chunksize=1, trim_to=0, compressor=None,
                 compressor_fit_sample_frac=1, comp_repr_size=0, ae_comp_fit_epochs=10,
                 ae_comp_batch_size=100):
        super().__init__(paths, header_parser=header_parser, quiet=quiet)
        if trim_to:  # Optional sequence trimming
            self.dataset.trim_seqs(trim_to)
        self.K = K
        self.jobs = jobs
        self.chunksize = chunksize
        self.ae_comp_batch_size = ae_comp_batch_size
        self.counter = KMerCounter(K, jobs=jobs, chunksize=chunksize)

        # Compression (defaults to 80%)
        r = np.random.default_rng()
        sample = r.permutation(len(self.dataset))[:int(len(self.dataset) *
                                                       compressor_fit_sample_frac)]
        sample = self.counter.kmer_counts(self.dataset['seqs'].to_numpy()[sample], quiet=self.quiet)
        postcomp_len = comp_repr_size or 4 ** K // 10 * 2
        if compressor == 'PCA':
            self.compressor = PCACompressor(postcomp_len)
            if not self.quiet:
                print('Training PCA compressor...')
            self.compressor.fit(sample)
        elif compressor == 'AE':
            self.compressor = AECompressor.auto(sample, postcomp_len)
            self.compressor: AECompressor
            if not self.quiet:
                print('AE Compressor Summary:')
                self.compressor.summary()
                print('Training compressor...')
            self.compressor.fit(sample, epochs=ae_comp_fit_epochs, batch_size=ae_comp_batch_size)
        else:
            postcomp_len = 4 ** K
            self.compressor = Compressor()
            self.compressor.fit(sample)

        # Model (training is distributed by default across all available GPUs)
        builder = ModelBuilder((postcomp_len,), tf.distribute.MirroredStrategy())
        builder.dense(postcomp_len, depth=depth)
        self.model = ComparativeEncoder.from_model_builder(builder, dist=Euclidean(),
                                                           output_dim=repr_size)
        if not quiet:
            self.model.summary()

    def preprocess_seq(self, seq: str) -> np.ndarray:
        counts = self.counter.str_to_kmer_counts(seq)
        return self.compressor.compress(np.array([counts]))[0]

    def preprocess_seqs(self, seqs: list[str]) -> np.ndarray:
        if isinstance(self.compressor, PCACompressor):
            return count_kmers_mp(self.K, self.compressor, self.dataset, self.jobs, self.chunksize)
        if isinstance(self.compressor, AECompressor):
            return count_kmers_batched(self.K, self.compressor, self.dataset,
                                       self.ae_comp_batch_size, self.jobs, self.chunksize,
                                       not self.quiet)
        return self.counter.kmer_counts(seqs, quiet=self.quiet)

    def fit(self, **kwargs):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        """
        if not self.quiet:
            print('Preprocessing dataset...')
        model_input = self.preprocess_seqs(self.dataset['seqs'].to_numpy())
        # For AECompressor, distances between encodings are meaningless
        distance_on = self.counter.kmer_counts(self.dataset['seqs'].to_numpy(), quiet=self.quiet) \
            if isinstance(self.compressor, AECompressor) else model_input
        if not self.quiet:
            print('Training model...')
        self.model.fit(model_input, distance_on=distance_on, silent=self.quiet, **kwargs)

    def transform(self, seqs: list[str], **kwargs) -> np.ndarray:
        """
        Transform the given sequences. Accepts keyword arguments for ComparativeEncoder.transform().
        """
        enc = self.preprocess_seqs(seqs)
        return self.model.transform(enc, progress=not self.quiet, **kwargs)

