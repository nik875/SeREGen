"""
Automated pipelines for sequence representation generation.
"""
import os
import shutil
import pickle
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.spatial.distance import euclidean
from sklearn.neighbors import BallTree

from .dataset_builder import DatasetBuilder, SILVA_header_parser, COVID_header_parser
from .visualize import repr_scatterplot
from .kmers import KMerCounter, Nucleotide_AA
from .kmer_compression import KMerCountPCA, KMerCountIPCA, KMerCountAE, KMerCountCompressor
from .encoders import ModelBuilder
from .comparative_encoder import ComparativeEncoder, Decoder, DenseDecoder, LinearDecoder
from .distance import IncrementalDistance, EditDistance, Distance, Cosine, Euclidean


class Pipeline:
    """
    An abstract automated pipeline for sequence representation generation.
    """
    AVAILABLE_DECODERS = {
        'default': Decoder,
        'dense': DenseDecoder,
        'linear': LinearDecoder
    }
    def __init__(self, model=None, decoder=None, dataset=None, preproc_reprs=None, reprs=None,
                 quiet=False, random_seed=None):
        self.model, self.decoder = model, decoder
        self.dataset, self.preproc_reprs, self.reprs = dataset, preproc_reprs, reprs
        self.quiet = quiet
        self.index = None
        self.rng = np.random.default_rng(seed=random_seed)
        tf.random.set_seed(random_seed)  # Set global random seed

    def set_decoder(self, decoder, *args, **kwargs):
        """
        Manually set the decoder.
        """
        self.decoder = decoder if isinstance(decoder, Decoder) else \
            self.AVAILABLE_DECODERS[decoder](*args, **kwargs)

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
        self.dataset.replace_unknown_nucls()
        if trim_to:
            self.dataset.trim_seqs(trim_to)

    def sample_data(self, size: int):
        """
        Randomly samples the Dataset and all representations down to size. Irreversible.
        """
        sample = self.rng.permutation(size)[:size]
        self.dataset = self.dataset.iloc[sample]
        if self.preproc_reprs is not None:
            self.preproc_reprs = self.preproc_reprs[sample]
        if self.reprs is not None:
            self.reprs = self.reprs[sample]

    # Subclass must override.
    @staticmethod
    def preprocess_seqs(seqs) -> np.ndarray:
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
        if not self.quiet:
            print('Preprocessing dataset...')
        _, unique_inds = np.unique(self.dataset['seqs'], return_index=True)
        self.preprocess_dataset(**kwargs)
        return unique_inds

    def fit_decoder(self, distance_on: np.ndarray, transform_batch_size: int, **kwargs):
        """
        Fit the decoder based on the model's representations.
        """
        if self.dataset is None:
            raise ValueError('Must load dataset before fitting decoder!')
        # Always transform dataset in case encoder has changed.
        if not self.quiet:
            print('Transforming dataset...')
        self.transform_dataset(transform_batch_size)
        if not self.quiet:
            print("Training Distance Decoder...")
        self.decoder.fit(self.reprs, distance_on, **kwargs)

    def _fit_decoder(self, dec_args: dict, batch_size: int, *args, **kwargs):
        """
        Adds defaults to dec_args before calling fit_decoder.
        """
        dec_args = dec_args or {}
        if 'transform_batch_size' not in dec_args:
            # Use the encoder's batch_size to transform
            dec_args['transform_batch_size'] = batch_size
        self.fit_decoder(*args, **(kwargs | dec_args))

    def _fit_called_check(self):
        if self.preproc_reprs is None:
            raise ValueError('Fit must be called before transform!')

    def plot_training_history(self, savepath=None):
        """
        Plot the training history of the trained model. Converts 1 - r loss into r^2.
        """
        data = self.model.history['loss']
        plt.plot(np.arange(len(data)), data)
        plt.title('ComparativeEncoder Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Model Loss')
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def transform(self, seqs: list, batch_size: int) -> list:
        """
        Transform an array of string sequences to learned representations.
        @param seqs: List of string sequences to transform.
        @return list: Sequence representations.
        """
        self._fit_called_check()
        return self.model.transform(self.preprocess_seqs(seqs), batch_size)

    def preprocess_dataset(self, **kwargs) -> np.ndarray:
        """
        Preprocesses all sequences in dataset.
        """
        self.preproc_reprs = self.preprocess_seqs(self.dataset['seqs'], **kwargs)

    def transform_dataset(self, batch_size: int, **kwargs) -> np.ndarray:
        """
        Transforms the loaded dataset into representations. Saves as self.reprs and returns result.
        Deletes any existing search tree.
        """
        self._fit_called_check()
        self.reprs = self.model.transform(self.preproc_reprs, batch_size, **kwargs)
        self.index = None  # Delete existing search tree because we assume reprs have changed.
        return self.reprs

    def save(self, savedir: str):
        """
        Save the Pipeline to the given directory.
        """
        shutil.rmtree(savedir, ignore_errors=True)
        os.makedirs(savedir)
        models_path = os.path.join(savedir, 'models')
        if self.model is not None:
            self.model.save(os.path.join(models_path, 'encoder'))
        # pylint: disable=unidiomatic-typecheck
        if self.decoder is not None and not type(self.decoder) == Decoder:
            self.decoder.save(os.path.join(models_path, 'decoder'))
        if self.preproc_reprs is not None:
            np.save(os.path.join(savedir, 'preproc_reprs.npy'), self.preproc_reprs)
        if self.reprs is not None:
            np.save(os.path.join(savedir, 'reprs.npy'), self.reprs)

    @classmethod
    def load(cls, savedir: str, strategy=None, quiet=False, **custom_kwargs):
        """
        Load a Pipeline from the savedir. Keyword arguments passed to ComparativeEncoder load.
        """
        # pylint: disable=broad-exception-caught
        if not os.path.exists(savedir):
            raise ValueError("Directory doesn't exist!")
        contents = os.listdir(savedir)
        kwargs = cls._load_special(savedir)
        kwargs['quiet'] = quiet
        if 'models' in contents:
            thisdir = os.path.join(savedir, 'models')
            if os.path.exists(os.path.join(thisdir, 'encoder')):
                kwargs['model'] = ComparativeEncoder.load(os.path.join(thisdir, 'encoder'),
                                                          strategy=strategy, quiet=quiet)
            else:
                print('Warning: encoder missing!')
            if os.path.exists(os.path.join(thisdir, 'decoder')):
                kwargs['decoder'] = Decoder.load(os.path.join(thisdir, 'decoder'), quiet=quiet)
            else:
                kwargs['decoder'] = Decoder()
        else:
            print('Warning: models missing!')
        if 'preproc_reprs.npy' in contents:
            try:
                kwargs['preproc_reprs'] = np.load(os.path.join(savedir, 'preproc_reprs.npy'),
                                                  allow_pickle=True)
            except Exception as e:
                print('Warning: exception in loading preproc_reprs: ' + e)
        if 'reprs.npy' in contents:
            try:
                kwargs['reprs'] = np.load(os.path.join(savedir, 'reprs.npy'), allow_pickle=True)
            except Exception as e:
                print('Warning: exception in loading preproc_reprs: ' + e)
        kwargs.update(custom_kwargs)
        return cls(**kwargs)

    @staticmethod
    def _load_special(savedir: str) -> dict:
        # pylint: disable=unused-argument
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
        Decoder.transform().
        @param query: List of string sequences to find similar sequences to.
        @param n_neighbors: Number of neighbors to find for each sequence. Defaults to 1.
        @return np.ndarray: Search results.
        """
        self._reprs_check()
        if self.model.embed_dist == 'euclidean':
            if self.index is None:  # If index hasn't been created, create it.
                if not self.quiet:
                    print('Creating search index...')
                self.index = BallTree(self.reprs)
            query_enc = self.transform([query], 1)
            dists, ind = self.index.query(query_enc, k=n_neighbors)
            matches = self.dataset.iloc[ind[0]]
            return self.decoder.transform(dists[0], **kwargs).flatten(), matches
        if self.model.embed_dist == 'hyperbolic':  # TODO: BALL TREE
            query_enc = self.transform([query], 1)
            x = np.repeat(query_enc, len(self.reprs), axis=0)
            dists = self.decoder.embed_dist_calc.transform_multi(x, self.reprs)
            s = np.argsort(dists)[:n_neighbors]
            return self.decoder.transform(dists, **kwargs)[s], self.dataset.iloc[s]
        raise ValueError('Invalid embedding distance!')  # Should never happen

    def evaluate(self, **kwargs):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: predicted distances, true distances
        """
        self._reprs_check()
        return self.decoder.evaluate(self.reprs, self.preproc_reprs, **kwargs)


class KMerCountsPipeline(Pipeline):
    """
    Automated pipeline using KMer Counts. Optionally compresses input data before training model.
    """
    DISTS = {
        'cosine': Cosine,
        'euclidean': Euclidean,
        'edit': EditDistance
    }
    def __init__(self, counter=None, compressor=None, **kwargs):
        super().__init__(**kwargs)
        self.counter = counter
        self.K_ = self.counter.k if self.counter else None
        self.repr_size_ = self.model.properties['repr_size'] if self.model else None
        self.compressor = compressor

    def create_kmer_counter(self, K: int, **kwargs):
        """
        Add a KMerCounter to this KMerCountsPipeline.
        """
        self.counter = KMerCounter(K, quiet=self.quiet, **kwargs)
        self.K_ = self.counter.k

    def create_compressor(self, compressor: str, repr_size=0, fit_sample_frac=1, **init_args):
        """
        Add a Compressor to this KMerCountsPipeline.
        """
        if not self.counter:
            raise ValueError('KMerCounter needs to be created before running! \
                             Use create_kmer_counter().')
        if compressor == 'None' or not compressor:
            self.compressor = KMerCountCompressor(self.counter, 4 ** self.K_, jobs=self.counter.jobs,
                                                  chunksize=self.counter.chunksize)
            return
        elif compressor not in ['PCA', 'IPCA', 'AE']:
            raise ValueError('Invalid Compressor Provided')
        sample = self.rng.permutation(len(self.dataset))[:int(len(self.dataset) * fit_sample_frac)]
        print('Counting k-mers in compressor fit sample...')
        sample = self.counter.kmer_counts(self.dataset['seqs'].to_numpy()[sample])
        compress_to = repr_size or 4 ** self.K_ // 10 * 2
        if compressor == 'PCA':
            self.compressor = KMerCountPCA(self.counter, compress_to, **init_args)
        elif compressor == 'IPCA':
            self.compressor = KMerCountIPCA(self.counter, compress_to, **init_args)
        elif compressor == 'AE':
            if 'batch_size' not in init_args:
                raise ValueError('batch_size must be provided in init_args for AE compressor!')
            self.compressor = KMerCountAE.auto(self.counter, sample, compress_to, **init_args)
            if not self.quiet:
                print('AE Compressor Summary:')
                self.compressor.summary()
        # pylint: disable=unidiomatic-typecheck
        # Strict type check needed here for this conditional
        if self.model is not None and not self.quiet and type(self.compressor) != Compressor:
            print('Creating a compressor after the model is not recommended! Consider running  \
                  create_model again.')
        print('Fitting compressor...')
        self.compressor.fit(sample)

    def create_model(self, repr_size=2, embed_dist='euclidean', depth=3, decoder='linear', dist='cosine',
                     distribute_strategy=None, dist_args=None, dec_args=None, embed_dist_args=None):
        """
        Create a Model for this KMerCountsPipeline. Uses all available GPUs.
        """
        # Argument validation
        if not self.counter:
            raise ValueError('KMerCounter needs to be created before running! \
                             Use create_kmer_counter().')
        if dist not in self.DISTS:
            raise ValueError('Invalid argument: dist. Must be one of "cosine", "edit", "euclidean"')
        dist = self.DISTS[dist](quiet=self.quiet, **(dist_args or {}))

        if not self.compressor:  # Create default (blank) compressor if needed
            self.create_compressor('None')

        dec_args = dec_args or {}
        if decoder == 'dense' and 'batch_size' not in dec_args:
            raise ValueError('Must pass "batch_size" in param dec_args for dense decoder')
        if 'embed_dist_args' not in dec_args:
            dec_args['embed_dist_args'] = embed_dist_args
        self.set_decoder(decoder, dist=dist, embed_dist=embed_dist, **dec_args)  # Create decoder
        builder = ModelBuilder((self.compressor.compress_to,),
                               distribute_strategy=distribute_strategy or
                               tf.distribute.MirroredStrategy())
        builder.dense(self.compressor.compress_to, depth=depth)
        self.model = ComparativeEncoder.from_model_builder(builder, dist=dist, repr_size=repr_size,
                                                           quiet=self.quiet, embed_dist=embed_dist,
                                                           random_seed=self.rng.integers(2**32))
        if not self.quiet:
            self.model.summary()

    def preprocess_seqs(self, seqs: list[str], **kwargs) -> np.ndarray:
        return self.compressor.transform(seqs, **kwargs)

    def fit(self, batch_size: int, preproc_args=None, dec_args=None, **kwargs):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        Automatically calls create_model() with default arguments if not already called.
        """
        if not self.model:
            self.create_model()

        # Always preprocess (with compression) since this is necessary for model.
        unique_inds = super().fit(**(preproc_args or {}))
        # If k-mer count based distance and not compressing
        if isinstance(self.model.distance, (Euclidean, Cosine)):
            if type(self.compressor) == KMerCountCompressor:
                distance_on = self.preproc_reprs  # Feed kmer counts as input
            else:  # kmer based distance with compression
                # Use an IncrementalDistance to reduce memory usage
                self.model.distance = IncrementalDistance(self.model.distance, self.counter)
                self.decoder.distance = IncrementalDistance(self.decoder.distance, self.counter)
                distance_on = self.dataset['seqs'].to_numpy()
        else:
            # IncrementalDistance takes sequences as input
            distance_on = self.dataset['seqs'].to_numpy()

        self.model.fit(batch_size, self.preproc_reprs[unique_inds],
                       distance_on=distance_on[unique_inds], **kwargs)
        self._fit_decoder(dec_args, batch_size, distance_on)


    def save(self, savedir: str):
        super().save(savedir)
        with open(os.path.join(savedir, 'counter.pkl'), 'wb') as f:
            pickle.dump(self.counter, f)
        if self.compressor is not None:
            self.compressor.save(os.path.join(savedir, 'compressor'))

    @classmethod
    def load(cls, *args, **kwargs):
        return super().load(*args, strategy=tf.distribute.MirroredStrategy(), **kwargs)

    @staticmethod
    def _load_special(savedir: str):
        result = {}
        contents = os.listdir(savedir)
        if 'counter.pkl' not in contents:
            raise ValueError('counter is necessary!')
        with open(os.path.join(savedir, 'counter.pkl'), 'rb') as f:
            result['counter'] = pickle.load(f)
        if 'compressor' in contents:
            result['compressor'] = KMerCountCompressor.load(os.path.join(savedir, 'compressor'))
        return result

class SequencePipeline(Pipeline):
    """
    Abstract sequence alignment estimator.
    """
    VOCAB = []  # MUST be defined by subclass
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_model(self, res='low', seq_len=.9, repr_size=2, embed_dist='euclidean', decoder='linear',
                     dec_args=None, dist_args=None, embed_dist_args=None, **kwargs):
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
        @param repr_size: Number of dimensions in output encodings (default 2).
        @param embed_dist: Embedding space geometry, 'euclidean' or 'hyperbolic'
        @param dist_args: Arguments for distance metric (jobs, chunksize)
        @param **kwargs: Everything else passed to ComparativeEncoder.from_model_builder
        """
        if (seq_len is None or seq_len < 1) and self.dataset is None:
            raise ValueError('Dataset must be loaded before autodetection of sequence length!')
        if seq_len is not None and seq_len < 1:
            target_zscore = st.norm.ppf(seq_len)
            lengths = self.dataset['seqs'].apply(len)
            mean = np.mean(lengths)
            std = np.std(lengths)
            seq_len = int(target_zscore * std + mean)

        dist_args = dist_args or {}
        dist = EditDistance(quiet=self.quiet, **dist_args)

        dec_args = dec_args or {}
        if decoder == 'dense' and 'batch_size' not in dec_args:
            raise ValueError('Must pass "batch_size" in param dec_args for dense decoder')
        if 'embed_dist_args' not in dec_args:
            dec_args['embed_dist_args'] = embed_dist_args
        self.set_decoder(decoder, dist=dist, embed_dist=embed_dist, **dec_args)  # Create decoder

        if res == 'low':
            self.model = self.low_res_model(seq_len, repr_size=repr_size, dist=dist,
                                            random_seed=self.rng.integers(2**32),
                                            embed_dist=embed_dist, **kwargs)
        elif res == 'medium':
            self.model = self.medium_res_model(seq_len, repr_size=repr_size, dist=dist,
                                            random_seed=self.rng.integers(2**32),
                                            embed_dist=embed_dist, **kwargs)
        elif res == 'high':
            self.model = self.high_res_model(seq_len, repr_size=repr_size, dist=dist,
                                            random_seed=self.rng.integers(2**32),
                                            embed_dist=embed_dist, **kwargs)
        elif res == 'ultra':
            self.model = self.ultra_res_model(seq_len, repr_size=repr_size, dist=dist,
                                            random_seed=self.rng.integers(2**32),
                                            embed_dist=embed_dist, **kwargs)
        else:
            raise ValueError('Invalid argument: res must be one of "low", "medium", "high", "ultra"')

        if not self.quiet:
            self.model.summary()

    @classmethod
    def low_res_model(cls, seq_len: int, compress_factor=1, depth=3, **kwargs):
        """
        Basic dense neural network operating on top of learned embeddings for input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=8, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        builder.dense(seq_len // compress_factor, depth=depth)
        builder.transpose()
        model = ComparativeEncoder.from_model_builder(builder, **kwargs)
        return model

    @classmethod
    def medium_res_model(cls, seq_len: int, compress_factor=4, conv_filters=16, conv_kernel_size=6,
                         dense_depth=3, **kwargs):
        """
        Convolutional layer operating on 1/4 the length of input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=12, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        if dense_depth:
            builder.dense(seq_len, depth=dense_depth)
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, seq_len // compress_factor)
        model = ComparativeEncoder.from_model_builder(builder, **kwargs)
        return model

    @classmethod
    def high_res_model(cls, seq_len: int, compress_factor=4, conv_filters=32, conv_kernel_size=8,
                       attn_heads=2, dense_depth=3, **kwargs):
        """
        Convolutional layer + attention block operating on 1/4 the length of input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=16, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        if dense_depth:
            builder.dense(seq_len, depth=dense_depth)
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, seq_len // compress_factor * 4)
        builder.reshape((*builder.shape()[:-2], builder.shape()[-1] // 4, 4))
        builder.attention(attn_heads, seq_len // compress_factor)
        model = ComparativeEncoder.from_model_builder(builder, **kwargs)
        return model

    @classmethod
    def ultra_res_model(cls, seq_len: int, compress_factor=1, conv_filters=64, conv_kernel_size=16,
                        attn_heads=4, dense_depth=3, **kwargs):
        """
        Convolutional layer + attention block operating on full length of input sequences.
        """
        builder = ModelBuilder.text_input(cls.VOCAB, embed_dim=20, max_len=seq_len,
                                          distribute_strategy=tf.distribute.MirroredStrategy())
        builder.transpose()
        if dense_depth:
            builder.dense(seq_len, depth=dense_depth)
        builder.dense(seq_len // compress_factor)
        builder.transpose()
        builder.conv1D(conv_filters, conv_kernel_size, seq_len // compress_factor * 4)
        builder.reshape((*builder.shape()[:-2], builder.shape()[-1] // 4, 4))
        builder.attention(attn_heads, seq_len // compress_factor)
        model = ComparativeEncoder.from_model_builder(builder, **kwargs)
        return model

    def fit(self, batch_size=256, dec_args=None, **kwargs):
        """
        Fit model to loaded dataset. Accepts keyword arguments for ComparativeEncoder.fit().
        Automatically calls create_model() with default arguments if not already called.
        """
        if not self.model:
            print('Warning: using default low-res model...')
            self.create_model()
        unique_inds = super().fit()

        self.model.fit(batch_size, self.preproc_reprs[unique_inds], **kwargs)
        self._fit_decoder(dec_args, batch_size, self.preproc_reprs)

    @classmethod
    def load(cls, *args, **kwargs):
        return super().load(*args, strategy=tf.distribute.MirroredStrategy(), **kwargs)


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
    def __init__(self, converter=None, **kwargs):
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

    def save(self, savedir: str):
        super().save(savedir)
        with open(os.path.join(savedir, 'converter.pkl'), 'wb') as f:
            pickle.dump(self.converter, f)

    @staticmethod
    def _load_special(savedir: str):
        result = {}
        contents = os.listdir(savedir)
        if 'converter.pkl' in contents:
            with open(os.path.join(savedir, 'converter.pkl'), 'rb') as f:
                result['converter'] = pickle.load(f)
        return result

