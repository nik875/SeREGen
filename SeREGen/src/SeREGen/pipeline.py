"""
Automated pipelines for sequence representation generation.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.neighbors import BallTree
import torch
from tqdm import tqdm

from .dataset_builder import DatasetBuilder, SILVA_header_parser, COVID_header_parser
from .visualize import repr_scatterplot
from .kmers import KMerCounter, Nucleotide_AA
from .kmer_compression import KMerCountPCA, KMerCountIPCA, KMerCountCompressor
from .encoders import ModelBuilder
from .comparative_encoder import ComparativeEncoder
from .distance import IncrementalDistance, EditDistance, Cosine, Euclidean
from ._saving import _create_save_directory, _save_object, _load_object

# pylint: disable=arguments-differ


class Pipeline:
    # pylint: disable=too-many-instance-attributes
    """
    An abstract automated pipeline for sequence representation generation.
    """

    def __init__(self, model=None, dataset=None, preproc_reprs=None, reprs=None,
                 silence=False, random_seed=None):
        self.model = model
        self.dataset, self.preproc_reprs, self.reprs = dataset, preproc_reprs, reprs
        self.silence = silence
        self.index = None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)
        if random_seed:
            torch.manual_seed(random_seed)  # Set global random seed

    def load_dataset(self, paths: list[str], header_parser='None', trim_to=0, max_rows=None):
        """
        Load a dataset into memory from a list of FASTA files.
        """
        if not isinstance(header_parser, str):
            builder = DatasetBuilder(header_parser)
        elif header_parser == 'SILVA':
            builder = DatasetBuilder(SILVA_header_parser)
        elif header_parser == 'COVID':
            builder = DatasetBuilder(COVID_header_parser)
        else:
            builder = DatasetBuilder()
        self.dataset = builder.from_fasta(paths, max_rows=max_rows)
        self.dataset.replace_unknown_nucls()
        if trim_to:
            self.dataset.trim_seqs(trim_to)

    # Subclass must override.
    def preprocess_seqs(self, seqs) -> np.ndarray:
        """
        Preprocesses a list of sequences.
        @param seqs: Sequences to preprocess.
        @return np.ndarray: Returns an array of preprocessed sequences.
        """
        if isinstance(seqs, list):
            return np.array(seqs)
        if isinstance(seqs, pd.Series):
            return seqs.to_numpy()
        return seqs

    # Must be implemented by subclass, super method must be called by implementation.
    # This super method preprocesses the dataset into self.preproc_reprs.
    # This variable is used to determine whether fit was called and to avoid preprocessing the
    # dataset twice between fit and transform_dataset. Returns indices of unique sequences.
    def fit(self, **kwargs):
        """
        Fit the model to the dataset.
        """
        if self.dataset is None:
            raise ValueError('Must load dataset before calling fit!')
        if not self.silence:
            print('Preprocessing dataset...')
        _, unique_inds = np.unique(self.dataset['seqs'], return_index=True)
        self.preprocess_dataset(**kwargs)
        return unique_inds

    def _fit_called_check(self):
        if self.preproc_reprs is None:
            raise ValueError('Fit must be called before transform!')

    def transform(self, seqs: list) -> list:
        """
        Transform an array of string sequences to learned representations.
        @param seqs: List of string sequences to transform.
        @return list: Sequence representations.
        """
        self._fit_called_check()
        return self.model.transform(self.preprocess_seqs(seqs))

    def preprocess_dataset(self, **kwargs) -> np.ndarray:
        """
        Preprocesses all sequences in dataset.
        """
        self.preproc_reprs = self.preprocess_seqs(self.dataset['seqs'], **kwargs)

    def transform_dataset(self, **kwargs) -> np.ndarray:
        """
        Transforms the loaded dataset into representations. Saves as self.reprs and returns result.
        Deletes any existing search tree.
        """
        self._fit_called_check()
        self.reprs = self.model.transform(self.preproc_reprs, **kwargs)
        self.index = None  # Delete existing search tree because we assume reprs have changed.
        return self.reprs

    def _reprs_check(self):
        """
        Wraps logic to check that reprs exist.
        """
        if self.reprs is None:
            raise ValueError('transform_dataset must be called first!')

    def plot_training_history(self, savepath=None):
        """
        Plot the training history of the trained model. Converts 1 - r loss into r^2.
        """
        self._fit_called_check()
        data = self.model.history['loss']
        plt.plot(np.arange(len(data)), data)
        plt.title('ComparativeEncoder Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Model Loss')
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def visualize_axes(self, x: int, y: int, **kwargs):
        """
        Visualizes two axes of the dataset representations on a simple scatterplot.
        @param x: which axis to use as x.
        @param y: which axis to use as y.
        @param kwargs: Accepts additional keyword arguments for visualize.repr_scatterplot().
        """
        self._reprs_check()
        repr_scatterplot(np.stack([self.reprs[:, x], self.reprs[:, y]], axis=1), **kwargs)

    def search(self, query: list[str], n_neighbors=1) -> tuple[np.ndarray, list[pd.Series]]:
        """
        Search the dataset for the most similar sequences to the query.
        @param query: List of string sequences to find similar sequences to.
        @param n_neighbors: Number of neighbors to find for each sequence. Defaults to 1.
        @return np.ndarray: Search results.
        """
        # TODO: distance decoding back to original via linear regression
        self._reprs_check()
        if self.model.embed_dist == 'euclidean':
            if self.index is None:  # If index hasn't been created, create it.
                if not self.silence:
                    print('Creating search index...')
                self.index = BallTree(self.reprs)
            query_enc = self.transform([query], 1)
            dists, ind = self.index.query(query_enc, k=n_neighbors)
            matches = self.dataset.iloc[ind[0]]
            # return self.decoder.transform(dists[0], **kwargs).flatten(), matches
            return matches
        if self.model.embed_dist == 'hyperbolic':  # TODO: BALL TREE
            query_enc = self.transform([query], 1)
            x = np.repeat(query_enc, len(self.reprs), axis=0)
            dists = self.model.transform(x, self.reprs)
            s = np.argsort(dists)[:n_neighbors]
            # return self.decoder.transform(dists, **kwargs)[s], self.dataset.iloc[s]
            return self.dataset.iloc[s]
        raise ValueError('Invalid embedding distance!')  # Should never happen

    # Must be overridden, pass model.evaluate arguments up to here
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: predicted distances, true distances
        """
        return self.model.evaluate(*args, **kwargs)

    def save(self, path):
        _create_save_directory(path)
        self.model.save(os.path.join(path, "model"))
        _save_object(self.preproc_reprs, path, "preproc_reprs.pkl")
        _save_object(self.reprs, path, "reprs.pkl")
        _save_object(self.silence, path, "silence.pkl")
        _save_object(self.random_seed, path, "random_seed.pkl")

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        Pipeline.__init__(obj,
                          model=ComparativeEncoder.load(os.path.join(path, "model")),
                          preproc_reprs=_load_object(path, "preproc_reprs.pkl"),
                          reprs=_load_object(path, "reprs.pkl"),
                          silence=_load_object(path, "silence.pkl"),
                          random_seed=_load_object(path, "random_seed.pkl"))
        return obj


class KMerCountsPipeline(Pipeline):
    """
    Automated pipeline using KMer Counts. Optionally compresses input data before training model.
    """
    DISTS = {
        'cosine': Cosine,
        'euclidean': Euclidean,
        'edit': EditDistance
    }

    def __init__(self, counter=None, compressor=None, _super_init=True, **kwargs):
        if _super_init:
            super().__init__(**kwargs)
        self.counter = counter
        self.K_ = self.counter.k if self.counter else None
        self.compressor = compressor

    def create_kmer_counter(self, K: int, **kwargs):
        """
        Add a KMerCounter to this KMerCountsPipeline.
        """
        self.counter = KMerCounter(K, silence=self.silence, **kwargs)
        self.K_ = self.counter.k

    def create_compressor(self, compressor: str, repr_size=0, fit_sample_frac=1, **init_args):
        """
        Add a Compressor to this KMerCountsPipeline.
        """
        if not self.counter:
            raise ValueError('KMerCounter needs to be created before running! \
                             Use create_kmer_counter().')
        if compressor == 'None' or not compressor:
            self.compressor = None
            return
        if compressor not in ['PCA', 'IPCA']:
            raise ValueError('Invalid Compressor Provided')

        sample = self.rng.permutation(len(self.dataset))[:int(len(self.dataset) * fit_sample_frac)]
        print('Counting k-mers in compressor fit sample...')
        sample = self.counter.kmer_counts(self.dataset['seqs'].to_numpy()[sample])
        compress_to = repr_size or 4 ** self.K_ // 10 * 2

        if compressor == 'PCA':
            self.compressor = KMerCountPCA(self.counter, compress_to, **init_args)
        elif compressor == 'IPCA':
            self.compressor = KMerCountIPCA(self.counter, compress_to, **init_args)

        if self.model is not None and not self.silence:
            print('Creating a compressor after the model is not recommended! Consider running  \
                  create_model again.')
        print('Fitting compressor...')
        self.compressor.fit(sample)

    def create_model(self, depth=3, dist='cosine', dist_args=None,
                     float_type=torch.float64, **kwargs):
        """
        Create a Model for this KMerCountsPipeline. Uses all available GPUs.
        """
        # Argument validation
        if not self.counter:
            raise ValueError('KMerCounter needs to be created before running! \
                             Use create_kmer_counter().')
        if dist not in self.DISTS:
            raise ValueError('Invalid argument: dist. Must be one of "cosine", "edit", "euclidean"')
        dist = self.DISTS[dist](silence=self.silence, **(dist_args or {}))

        compress_to = self.compressor.compress_to if self.compressor else 4 ** self.K_
        builder = ModelBuilder((compress_to,), input_dtype=float_type)
        builder.dense(compress_to, depth=depth)
        self.model = ComparativeEncoder.from_model_builder(builder, dist=dist,
                                                           silence=self.silence,
                                                           random_seed=self.rng.integers(2**32),
                                                           **kwargs)
        if not self.silence:
            self.model.summary()

    def preprocess_seqs(self, seqs: list[str], **kwargs) -> np.ndarray:
        if self.compressor is not None:
            return self.compressor.transform(seqs, **kwargs)
        return self.counter.transform(seqs, **kwargs)

    def _get_distance_on(self, update_dist=False):
        if isinstance(self.model.properties["dist"], (Euclidean, Cosine)):
            if self.compressor is None:  # If k-mer count based distance and not compressing
                return self.preproc_reprs  # Feed kmer counts as input
            if update_dist:  # Use an IncrementalDistance to reduce memory usage
                self.model.distance = IncrementalDistance(self.model.distance, self.counter)
            return self.dataset['seqs'].to_numpy()
        return self.dataset['seqs'].to_numpy()

    def fit(self, preproc_args=None, repr_size=2, loss="r2", **kwargs):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        Automatically calls create_model() with default arguments if not already called.
        """
        if not self.model:
            self.create_model()

        # Always preprocess (with compression) since this is necessary for model.
        unique_inds = super().fit(**(preproc_args or {}))
        distance_on = self._get_distance_on(update_dist=True)

        self.model.fit(
            self.preproc_reprs[unique_inds],
            distance_on=distance_on[unique_inds],
            loss=loss,
            repr_size=repr_size,
            **kwargs)
        self.transform_dataset()

    def evaluate(self, **kwargs):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: predicted distances, true distances
        """
        _, unique_inds = np.unique(self.preproc_reprs, return_index=True)
        distance_on = self._get_distance_on()
        return super().evaluate(self.preproc_reprs[unique_inds], distance_on[unique_inds], **kwargs)

    def save(self, path: str):
        super().save(path)
        _save_object(self.counter, path, "counter.pkl")
        if self.compressor is not None:
            self.compressor.save(os.path.join(path, 'compressor'))

    @classmethod
    def load(cls, path: str):
        obj = super().load(path)
        cls.__init__(obj,
                     _load_object(path, "counter.pkl"),
                     KMerCountCompressor.load(os.path.join(path, "compressor")),
                     super_init=False)
        return obj


class SequencePipeline(Pipeline):
    """
    Abstract sequence alignment estimator. Define VOCAB in subclass.
    """
    VOCAB = []  # MUST be defined by subclass; chr(0) is reserved

    def __init__(self, *args, seq_len=.9, _super_init=True, **kwargs):
        if _super_init:
            super().__init__(*args, **kwargs)
        self.seq_len = seq_len

        alphabet = np.array(self.VOCAB)
        self.alphabet_pattern = re.compile(f'[^{"".join(alphabet)}]')
        # Make a lookup table with an entry for every possible data byte
        self.lookup_table = np.zeros(256, dtype=np.uint32)
        for idx, val in enumerate(self.VOCAB):
            self.lookup_table[ord(val)] = idx + 1

    def load_dataset(self, *args, **kwargs):
        super().load_dataset(*args, **kwargs)
        if self.seq_len < 1:
            target_zscore = st.norm.ppf(self.seq_len)
            lengths = self.dataset['seqs'].apply(len)
            mean = np.mean(lengths)
            std = np.std(lengths)
            self.seq_len = int(target_zscore * std + mean)

    def preprocess_seqs(self, seqs: list[str]):
        """
        Ordinally encodes, pads, and trims sequences to seq_len
        """
        seqs = super().preprocess_seqs(seqs)

        def ord_encode(s):
            arr = np.zeros((self.seq_len,), dtype="<U1")
            trimmed = s[:self.seq_len]
            arr[:len(trimmed)] = list(trimmed)
            return self.lookup_table[arr.view(np.uint32)]
        result = [ord_encode(i) for i in tqdm(seqs)]
        return np.stack(result)

    def create_model(self, res='low', repr_size=2, dist_args=None, **kwargs):
        """
        Create a model for the Pipeline.
        @param res: Resolution of the model's encoding output. Available options are:
            'low' (default): Basic dense neural network operating on top of learned embeddings for
            input sequences.
            'medium': Convolutional layer operating on 1/4 the length of input sequences.
            'high': Convolutional layer + attention block operating on 1/4 the length of input
            sequences.
            'ultra': Convolutional layer + attention block operating on full length of input
            sequences.
        @param seq_len: Specifies input length of sequences to model. Three possibilities:
            seq_len == None: Auto-detect the maximum sequence length and use as model input size.
            0 < seq_len < 1: Ensure that this fraction of the total dataset is NOT truncated.
            seq_len >= 1: Trim and pad directly to this length.
        @param dist_args: Arguments for distance metric (jobs, chunksize)
        @param **kwargs: Everything else passed to ComparativeEncoder.from_model_builder
        """

        dist_args = dist_args or {}
        dist = EditDistance(silence=self.silence, **dist_args)

        if res == 'low':
            self.model = self.low_res_model(
                self.seq_len,
                dist=dist,
                random_seed=self.rng.integers(2**32),
                repr_size=repr_size,
                **kwargs)
        elif res == 'medium':
            self.model = self.medium_res_model(
                self.seq_len,
                dist=dist,
                random_seed=self.rng.integers(2**32),
                repr_size=repr_size,
                **kwargs)
        elif res == 'high':
            self.model = self.high_res_model(
                self.seq_len,
                dist=dist,
                random_seed=self.rng.integers(2**32),
                repr_size=repr_size,
                **kwargs)
        elif res == 'ultra':
            self.model = self.ultra_res_model(
                self.seq_len,
                dist=dist,
                random_seed=self.rng.integers(2**32),
                repr_size=repr_size,
                **kwargs)
        else:
            raise ValueError(
                'Invalid argument: res must be one of "low", "medium", "high", "ultra"')

        if not self.silence:
            self.model.summary()

    @classmethod
    def _init_builder(cls, seq_len, float_type, embed_dim, use_embedding_layer):
        builder = ModelBuilder((seq_len,), input_dtype=float_type)
        if use_embedding_layer:
            builder.embedding(len(cls.VOCAB), embed_dim, 0)
        else:
            builder.one_hot_encoding(len(cls.VOCAB))
            builder.dense(embed_dim)
        return builder

    @classmethod
    def low_res_model(cls, seq_len: int, compress_factor=1, depth=3, embed_dim=8,
                      float_type=torch.float64, use_embedding_layer=True, **kwargs):
        """
        Basic dense neural network operating on top of learned embeddings for input sequences.
        """
        builder = cls._init_builder(seq_len, float_type, embed_dim, use_embedding_layer)
        builder.transpose()
        builder.dense(seq_len // compress_factor, depth=1)
        builder.dense(seq_len // compress_factor, depth=depth - 1, residual=True)
        builder.transpose()
        return ComparativeEncoder.from_model_builder(builder, **kwargs)

    @classmethod
    def medium_res_model(cls, seq_len: int, compress_factor=4, conv_filters=16, conv_kernel_size=6,
                         dense_depth=3, embed_dim=12, float_type=torch.float64,
                         use_embedding_layer=True, **kwargs):
        """
        Convolutional layer operating on 1/4 the length of input sequences.
        """
        builder = cls._init_builder(seq_len, float_type, embed_dim, use_embedding_layer)
        builder.transpose()
        if dense_depth:
            builder.dense(seq_len, depth=1)
            builder.dense(seq_len, depth=dense_depth - 1, residual=True)
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, residual=True)
        return ComparativeEncoder.from_model_builder(builder, **kwargs)

    @classmethod
    def high_res_model(cls, seq_len: int, compress_factor=4, conv_filters=32, conv_kernel_size=8,
                       attn_heads=2, dense_depth=3, embed_dim=12, float_type=torch.float64,
                       use_embedding_layer=True, **kwargs):
        """
        Convolutional layer + attention block operating on 1/4 the length of input sequences.
        """
        builder = cls._init_builder(seq_len, float_type, embed_dim, use_embedding_layer)
        builder.transpose()
        if dense_depth:
            builder.dense(seq_len, depth=1)
            builder.dense(seq_len, depth=dense_depth - 1, residual=True)
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, residual=True)
        builder.dense(seq_len // compress_factor * 4)
        builder.reshape((*builder.shape()[:-2], builder.shape()[-1] // 4, 4))
        builder.attention(attn_heads, seq_len // compress_factor, residual=True)
        return ComparativeEncoder.from_model_builder(builder, **kwargs)

    @classmethod
    def ultra_res_model(cls, seq_len: int, compress_factor=1, conv_filters=64, conv_kernel_size=16,
                        attn_heads=4, dense_depth=3, embed_dim=16, **kwargs):
        """
        Convolutional layer + attention block operating on full length of input sequences.
        """
        return cls.high_res_model(seq_len, compress_factor, conv_filters,
                                  conv_kernel_size, attn_heads, dense_depth, embed_dim, **kwargs)

    def fit(self, preproc_args=None, loss="r2", suppress_grad_warn=None, **kwargs):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        Automatically calls create_model() with default arguments if not already called.
        """
        if not self.model:
            print('Warning: using default low-res model...')
            self.create_model()
        unique_inds = super().fit(**(preproc_args or {}))

        suppress_grad_warn = suppress_grad_warn or []
        if loss == "corr_coef" and "vanishing" not in suppress_grad_warn:
            suppress_grad_warn.append("vanishing")

        self.model.fit(
            self.preproc_reprs[unique_inds],
            loss=loss,
            suppress_grad_warn=suppress_grad_warn,
            **kwargs)
        self.transform_dataset()

    def evaluate(self, **kwargs):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: predicted distances, true distances
        """
        return super().evaluate(self.preproc_reprs, self.preproc_reprs, **kwargs)

    def save(self, path):
        super().save(path)
        _save_object(self.seq_len, path, "seq_len.pkl")

    @classmethod
    def load(cls, path):
        obj = super().load(path)
        cls.__init__(obj, seq_len=_load_object(path, "seq_len.pkl"), _super_init=False)
        return obj


class DNASequencePipeline(SequencePipeline):
    """
    Edit distance estimator for DNA sequences.
    """
    VOCAB = ['A', 'C', 'G', 'T']


class RNASequencePipeline(SequencePipeline):
    """
    Edit distance estimator for RNA sequences.
    """
    VOCAB = ['A', 'C', 'G', 'U']


class HomologousSequencePipeline(SequencePipeline):
    """
    Edit distance of 3 possible forward reading frames.
    """
    VOCAB = np.unique(Nucleotide_AA.AA_LOOKUP)

    def __init__(self, converter=None, _super_init=True, **kwargs):
        if _super_init:
            super().__init__(**kwargs)
        self.converter = converter

    def create_converter(self, *args, **kwargs):
        """
        Create a Nucleotide_AA converter for the Pipeline. Directly wraps constructor.
        """
        self.converter = Nucleotide_AA(*args, **kwargs)

    def preprocess_seqs(self, seqs: list[str]):
        if self.converter is None:
            print('Warning: default converter being used...')
            self.create_converter()
        return self.converter.transform(seqs)

    def save(self, path: str):
        super().save(path)
        _save_object(self.converter, path, "converter.pkl")

    @classmethod
    def load(cls, path: str):
        obj = super().load(path)
        cls.__init__(obj, _load_object(path, "converter.pkl"), _super_init=False)
        return obj
