"""
Automated pipelines for sequence representation generation.
"""
import os
import shutil
import pickle
import json
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from scipy.spatial.distance import euclidean
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error, r2_score

from .dataset_builder import DatasetBuilder, SILVA_header_parser, COVID_header_parser
from .visualize import repr_scatterplot
from .kmers import KMerCounter, Nucleotide_AA
from .compression import PCA, IPCA, AE, Compressor
from .encoders import ModelBuilder
from .comparative_encoder import ComparativeEncoder
from .distance import cosine, IncrementalDistance, alignment


class Pipeline:
    """
    An abstract automated pipeline for sequence representation generation.
    """
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.dataset = None
        self.model = None
        self.preproc_reprs = None
        self.reprs = None
        self.index = None

    def load_dataset(self, paths: list[str], header_parser='None', trim_to=0):
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
        self.dataset = builder.from_fasta(paths)
        if trim_to:
            self.dataset.trim_seqs(trim_to)

    # Should be implemented by subclass unless strings are passed directly as input
    # pylint: disable=unused-argument
    def preprocess_seq(self, seq):
        """
        Preprocesses a string sequence.
        @param seq: Sequence to preprocess.
        @return np.ndarray: Returns seq by default.
        """
        return seq

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
    def transform_after_preproc(self, data, **kwargs):
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

    def transform_dataset(self, **kwargs) -> np.ndarray:
        """
        Transforms the loaded dataset into representations. Saves as self.reprs and returns result.
        Deletes any existing search tree.
        """
        self._fit_called_check()
        self.reprs = self.transform_after_preproc(self.preproc_reprs)
        self.index = None  # Delete existing search tree because we assume reprs have changed.
        return self.reprs

    def save(self, savedir: str):
        """
        Save the Pipeline to the given directory.
        """
        shutil.rmtree(savedir, ignore_errors=True)
        os.makedirs(savedir)
        if self.model is not None:
            self.model.save(os.path.join(savedir, 'model'))
        with open(os.path.join(savedir, 'distance.pkl'), 'wb') as f:
            pickle.dump(self.model.distance, f)
        if self.preproc_reprs is not None:
            np.save(os.path.join(savedir, 'preproc_reprs.npy'), self.preproc_reprs)
        if self.reprs is not None:
            np.save(os.path.join(savedir, 'reprs.npy'), self.reprs)
        kwargs = self._save_special_kwargs()
        kwargs['quiet'] = self.quiet
        with open(os.path.join(savedir, 'kwargs.json'), 'w') as f:
            json.dump(kwargs, f)
        return kwargs

    def _save_special_kwargs(self) -> dict:
        """
        Returns all special keyword arguments to save for this Pipeline.
        """
        return {}

    @classmethod
    def load(cls, savedir: str):
        """
        Load a Pipeline from the savedir.
        """
        if not os.path.exists(savedir):
            raise ValueError("Directory doesn't exist!")
        contents = os.listdir(savedir)
        if 'kwargs.json' in contents:
            with open(os.path.join(savedir, 'kwargs.json'), 'r') as f:
                kwargs = json.load(f)
        else:
            raise ValueError('kwargs.json file necessary!')
        kwargs.update(cls._load_special(savedir))
        obj = cls(**kwargs)
        if 'distance.pkl' in contents:
            with open(os.path.join(savedir, 'distance.pkl'), 'rb') as f:
                dist = pickle.load(f)
        else:
            raise ValueError('distance.pkl file necessary!')
        if 'model' in contents:
            obj.model = ComparativeEncoder.load(os.path.join(savedir, 'model'), dist=dist)
        if 'preproc_reprs.npy' in contents:
            obj.preproc_reprs = np.load(os.path.join(savedir, 'preproc_reprs.npy'))
        if 'reprs.npy' in contents:
            obj.reprs = np.load(os.path.join(savedir, 'reprs.npy'))
        return obj

    @staticmethod
    def _load_special(savedir: str) -> dict:
        """
        Returns a dictionary of all loaded special constructor arguments for this Pipeline.
        """
        return {}

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

    def search(self, query: list[str], n_neighbors=1,
               **kwargs) -> tuple[np.ndarray, list[pd.Series]]:
        """
        Search the dataset for the most similar sequences to the query. Accepts keyword arguments to
        ComparativeEncoder.transform_distances().
        @param query: List of string sequences to find similar sequences to.
        @param n_neighbors: Number of neighbors to find for each sequence. Defaults to 1.
        @return np.ndarray: Search results.
        """
        self._reprs_check()
        if self.index is None:  # If index hasn't been created, create it.
            if not self.quiet:
                print('Creating search index...')
            self.index = BallTree(self.reprs)
        query_enc = self.transform([query])
        dists, ind = self.index.query(query_enc, k=n_neighbors)
        matches = [self.dataset.iloc[i] for i in ind[0]]
        return self.model.transform_distances(dists[0]), matches

    def evaluate(self, sample_size=None, jobs=1, chunksize=1, distance_transform_batch_size=0):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: predicted distances, true distances
        """
        self._reprs_check()
        sample_size = sample_size or len(self.dataset)
        rng = np.random.default_rng()
        sample = rng.permutation(len(self.dataset))[:sample_size]
        encs = self.reprs[sample]
        inputs = self.preproc_reprs[sample]
        p1 = rng.permutation(len(encs))
        p2 = rng.permutation(len(encs))
        x1, x2 = encs[p1], encs[p2]
        y1, y2 = inputs[p1], inputs[p2]
        if not self.quiet:
            print('Calculating distances between encodings...')
        x = np.fromiter((euclidean(x1[i], x2[i]) for i in (range(len(x1)) if self.quiet else
                                                           tqdm(range(len(x1))))), dtype=np.floatc)
        x = self.model.transform_distances(x, batch_size=distance_transform_batch_size)
        if not self.quiet:
            print('Calculating distances between model inputs...')
        with mp.Pool(jobs) as p:
            it = p.imap(self.model.distance.transform, zip(y1, y2), chunksize=chunksize)
            y = np.fromiter((it if self.quiet else tqdm(it, total=len(y1))), dtype=np.floatc)
        mse = mean_squared_error(y, x)
        r2 = r2_score(y, x)
        print(f'Mean squared error of distances: {mse}')
        print(f'R-squared correlation coefficient: {r2}')
        return x, y


class KMerCountsPipeline(Pipeline):
    """
    Automated pipeline using KMer Counts. Optionally compresses input data before training model.
    """
    def __init__(self, counter=None, model=None, compressor=None, quiet=False):
        super().__init__(quiet)
        self.counter = counter
        self.K_ = self.counter.k if self.counter else None
        self.model = model
        self.repr_size_ = self.model.repr_size if self.model else None
        self.compressor = compressor

    def create_kmer_counter(self, K: int, jobs=1, chunksize=1):
        """
        Add a KMerCounter to this KMerCountsPipeline.
        """
        self.counter = KMerCounter(K, jobs=jobs, chunksize=chunksize, quiet=self.quiet)
        self.K_ = self.counter.k

    def create_compressor(self, compressor: str, repr_size=0, fit_sample_frac=1, **init_args):
        """
        Add a Compressor to this KMerCountsPipeline.
        """
        if not self.counter:
            raise ValueError('KMerCounter needs to be created before running! \
                             Use create_kmer_counter().')
        r = np.random.default_rng()
        sample = r.permutation(len(self.dataset))[:int(len(self.dataset) * fit_sample_frac)]
        sample = self.counter.kmer_counts(self.dataset['seqs'].to_numpy()[sample])
        postcomp_len = repr_size or 4 ** self.K_ // 10 * 2
        if compressor == 'PCA':
            self.compressor = PCA(postcomp_len, quiet=self.quiet, **init_args)
        elif compressor == 'IPCA':
            self.compressor = IPCA(postcomp_len, quiet=self.quiet, **init_args)
        elif compressor == 'AE':
            self.compressor = AE.auto(sample, postcomp_len, quiet=self.quiet, **init_args)
            if not self.quiet:
                print('AE Compressor Summary:')
                self.compressor.summary()
        else:
            self.compressor = Compressor(postcomp_len := 4 ** self.K_, self.quiet)
        # pylint: disable=unidiomatic-typecheck
        # Strict type check needed here for this conditional
        if self.model is not None and not self.quiet and type(self.compressor) != Compressor:
            print('Creating a compressor after the model is not recommended! Consider running  \
                  create_model again.')
        self.compressor.fit(sample)

    def create_model(self, repr_size=2, depth=3, dist=None):
        """
        Create a Model for this KMerCountsPipeline. Uses all available GPUs.
        """
        if not self.counter:
            raise ValueError('KMerCounter needs to be created before running! \
                             Use create_kmer_counter().')
        if not self.compressor:
            self.create_compressor('None')

        builder = ModelBuilder((self.compressor.postcomp_len,), tf.distribute.MirroredStrategy())
        builder.dense(self.compressor.postcomp_len, depth=depth)
        self.model = ComparativeEncoder.from_model_builder(builder, dist=dist or cosine,
                                                           output_dim=repr_size, quiet=self.quiet)
        if not self.quiet:
            self.model.summary()

    def preprocess_seq(self, seq: str) -> np.ndarray:
        counts = self.counter.str_to_kmer_counts(seq)
        return self.compressor.transform(np.array([counts]))[0]

    def preprocess_seqs(self, seqs: list[str], batch_size=0) -> np.ndarray:
        return self.compressor.count_kmers(self.counter, seqs, batch_size)

    def fit(self, preproc_batch_size=0, dist_on_preproc=False, incremental_dist=True,
            **model_fit_args):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        Automatically calls create_model() with default arguments if not already called.
        """
        if not self.model:
            self.create_model()
        super().fit(batch_size=preproc_batch_size)
        if incremental_dist:
            self.model.distance = IncrementalDistance(self.model.distance, self.counter)
            distance_on = self.dataset['seqs']
        elif dist_on_preproc:
            distance_on = self.preproc_reprs
        else:
            distance_on = self.counter.kmer_counts(self.dataset['seqs'].to_numpy())
        if not self.quiet:
            print('Training model...')
        self.model.fit(self.preproc_reprs, distance_on=distance_on, **model_fit_args)
        dec_fit_args = {k:v for k, v in model_fit_args if k in ['jobs', 'chunksize']}
        self.model.fit_decoder(self.preproc_reprs, distance_on=distance_on, epoch_limit=1,
                               **dec_fit_args)

    def transform_after_preproc(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform the given sequences. Accepts keyword arguments for ComparativeEncoder.transform().
        """
        super().transform_after_preproc(data)
        return self.model.transform(data, **kwargs)

    def save(self, savedir: str):
        super().save(savedir)
        with open(os.path.join(savedir, 'counter.pkl'), 'wb') as f:
            pickle.dump(self.counter, f)
        if self.compressor is not None:
            self.compressor.save(os.path.join(savedir, 'compressor'))

    @staticmethod
    def _load_special(savedir: str):
        result = {}
        contents = os.listdir(savedir)
        if 'counter.pkl' not in contents:
            raise ValueError('counter is necessary!')
        with open(os.path.join(savedir, 'counter.pkl'), 'rb') as f:
            result['counter'] = pickle.load(f)
        if 'compressor' in contents:
            result['compressor'] = Compressor.load(os.path.join(savedir, 'compressor'))
        return result


class HomologousSequencePipeline(Pipeline):
    """
    Sequence-based alignment estimator that factors in homologous sequences (those with the same or
    similar protein outputs despite minor mutations).
    """
    VOCAB = np.unique(Nucleotide_AA.AA_LOOKUP)

    def __init__(self, converter=None, model=None, quiet=False):
        super().__init__(quiet)
        self.converter = converter
        self.model = model

    def create_converter(self, *args, **kwargs):
        """
        Create a Nucleotide_AA converter for the Pipeline. Directly wraps constructor.
        """
        self.converter = Nucleotide_AA(*args, **kwargs)

    def create_model(self, res='low', seq_len=.85, output_dim=2):
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
        @param output_dim: Number of dimensions in output encodings (default 2).
        """
        if (seq_len is None or seq_len < 1) and self.dataset is None:
            raise ValueError('Dataset must be loaded before autodetection of sequence length!')
        if seq_len is not None and seq_len < 1:
            target_zscore = st.norm.ppf(seq_len)
            lengths = self.dataset['seqs'].apply(len)
            mean = np.mean(lengths)
            std = np.std(lengths)
            seq_len = int(target_zscore * std + mean)
        if res == 'low':
            self.model = self.low_res_model(seq_len, output_dim)
        elif res == 'medium':
            self.model = self.medium_res_model(seq_len, output_dim)
        elif res == 'high':
            self.model = self.high_res_model(seq_len, output_dim)
        elif res == 'ultra':
            self.model = self.ultra_res_model(seq_len, output_dim)
        if not self.quiet:
            self.model.summary()

    @classmethod
    def low_res_model(cls, seq_len: int, output_dim: int, compress_factor=1, depth=3):
        """
        Basic dense neural network operating on top of learned embeddings for input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=8, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        builder.dense(seq_len // compress_factor, depth=depth)
        builder.transpose()
        model = ComparativeEncoder.from_model_builder(builder, dist=alignment,
                                                      output_dim=output_dim)
        return model

    @classmethod
    def medium_res_model(cls, seq_len: int, output_dim: int, compress_factor=4, conv_filters=16,
                         conv_kernel_size=6):
        """
        Convolutional layer operating on 1/4 the length of input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=12, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, seq_len // compress_factor)
        model = ComparativeEncoder.from_model_builder(builder, dist=alignment,
                                                      output_dim=output_dim)
        return model

    @classmethod
    def high_res_model(cls, seq_len: int, output_dim: int, compress_factor=4, conv_filters=32,
                       conv_kernel_size=8, attn_heads=2):
        """
        Convolutional layer + attention block operating on 1/4 the length of input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=16, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, seq_len // compress_factor)
        builder.attention(attn_heads, seq_len // compress_factor)
        model = ComparativeEncoder.from_model_builder(builder, dist=alignment,
                                                      output_dim=output_dim)
        return model

    @classmethod
    def ultra_res_model(cls, seq_len: int, output_dim: int, compress_factor=1, conv_filters=64,
                        conv_kernel_size=16, attn_heads=4):
        """
        Convolutional layer + attention block operating on full length of input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=20, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, seq_len // compress_factor)
        builder.attention(attn_heads, seq_len // compress_factor)
        model = ComparativeEncoder.from_model_builder(builder, dist=alignment,
                                                      output_dim=output_dim)
        return model

    def preprocess_seq(self, seq: str) -> str:
        if self.converter is None:
            print('Warning: default converter being used...')
            self.create_converter()
        return self.converter.transform([seq])[0]

    def preprocess_seqs(self, seqs: list[str]):
        if self.converter is None:
            print('Warning: default converter being used...')
            self.create_converter()
        return self.converter.transform(seqs)

    def fit(self, **kwargs):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        Automatically calls create_model() with default arguments if not already called.
        """
        if not self.model:
            print('Warning: using default low-res model...')
            self.create_model()
        super().fit()
        if not self.quiet:
            print('Training model...')
        self.model.fit(self.preproc_reprs, **kwargs)
        kwargs = {k:v for k, v in kwargs if k in ['jobs', 'chunksize']}
        self.model.fit_decoder(self.preproc_reprs, epoch_limit=1, **kwargs)

    def transform_after_preproc(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform the given sequences. Accepts keyword arguments for ComparativeEncoder.transform().
        """
        super().transform_after_preproc(data)
        return self.model.transform(data, **kwargs)

    def save(self, savedir: str):
        super().save(savedir)
        with open(os.path.join(savedir, 'converter.pkl'), 'wb') as f:
            pickle.dump(self.converter, f)

    @staticmethod
    def _load_special(savedir: str):
        result = {}
        contents = os.listdir(savedir)
        if 'converter.pkl' not in contents:
            raise ValueError('converter is necessary!')
        with open(os.path.join(savedir, 'converter.pkl'), 'rb') as f:
            result['converter'] = pickle.load(f)
        return result

