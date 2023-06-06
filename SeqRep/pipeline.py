"""
Automated pipelines for sequence representation generation.
"""
import os
import shutil
import pickle
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import BallTree
from .dataset_builder import DatasetBuilder, SILVAHeaderParser, COVIDHeaderParser
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
            builder = DatasetBuilder(SILVAHeaderParser())
        elif header_parser == 'COVID':
            builder = DatasetBuilder(COVIDHeaderParser())
        else:
            builder = DatasetBuilder()
        self.dataset = builder.from_fasta(paths)
        if trim_to:
            self.dataset.trim_seqs(trim_to)

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
        kwargs += cls._load_special(savedir)
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

    def search(self, query: list[str], n_neighbors=1) -> tuple[np.ndarray, list[pd.Series]]:
        """
        Search the dataset for the most similar sequences to the query.
        @param query: List of string sequences to find similar sequences to.
        @param n_neighbors: Number of neighbors to find for each sequence. Defaults to 1.
        @return np.ndarray: Search results.
        """
        self._reprs_check()
        self._fit_called_check()
        if self.index is None:  # If index hasn't been created, create it.
            if not self.quiet:
                print('Creating search index...')
            self.index = BallTree(self.reprs)
        query_enc = self.transform([query])
        dists, ind = self.index.query(query_enc, k=n_neighbors)
        matches = [self.dataset.iloc[i] for i in ind[0]]
        return dists[0], matches


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

    def fit(self, preproc_batch_size=0, dist_on_preproc=False, incremental_dist=False,
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

