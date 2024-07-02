"""
ComparativeEncoder module, trains a model comparatively using distances.
"""
import os
import shutil
import pickle
import json
import time
import math
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch import nn
from torchinfo import summary
from tqdm import tqdm
from geomstats.geometry.poincare_ball import PoincareBall

from .encoders import ModelBuilder
from .distance import Hyperbolic, Euclidean, Cosine


class _NormalizedDistanceLayer(nn.Module):
    """
    Adds a scaling parameter that's set to 1 / average distance on the first iteration.
    Output WILL be normalized.
    """

    def __init__(self, **kwargs):
        super().__init__(name='dist', **kwargs)
        self.scaling_param = nn.Parameter(torch.ones(1))

    def norm(self, dists):
        """
        Normalize the distances with scaling and set scaling if first time.
        """
        return dists * self.scaling_param


class EuclideanDistanceLayer(_NormalizedDistanceLayer):
    """
    This layer computes the distance between its two prior layers.
    """

    def forward(self, a, b):
        return self.norm(torch.sum((a - b) ** 2, dim=-1))


class HyperbolicDistanceLayer(_NormalizedDistanceLayer):
    """
    Computes hyperbolic distance in Poincaré ball model.
    """

    def __init__(self, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.hyperbolic_metric = PoincareBall(embedding_size).metric.dist

    def forward(self, a, b):
        return self.norm(self.hyperbolic_metric(a, b))


class ComparativeLayer(nn.Module):
    """
    Encode two inputs and calculate the embedding distance between them.
    """

    def __init__(self, encoder, embed_dist, reg_dims, embedding_size):
        """
        @param encoder: PyTorch model to use as the encoder.
        @param embed_dist: Distance metric to use when comparing two sequences.
        @param reg_dims: Whether to return encoder output for regularization.
        """
        super().__init__()
        self.encoder = encoder
        self.reg_dims = reg_dims

        if embed_dist.lower() == 'euclidean':
            self.dist_layer = EuclideanDistanceLayer()
        elif embed_dist.lower() == 'hyperbolic':
            self.dist_layer = HyperbolicDistanceLayer(embedding_size=embedding_size)
        else:
            raise ValueError('Invalid embedding distance provided!')

    def forward(self, inputa, inputb):
        """
        Forward pass of the comparative layer.

        @param inputa: First input tensor.
        @param inputb: Second input tensor.
        @return: Distances between encoded inputs, and optionally the encoded representation of
        inputa.
        """
        encodera = self.encoder(inputa)
        encoderb = self.encoder(inputb)
        distances = self.dist_layer(encodera, encoderb)

        if self.reg_dims:
            return {'dist': distances, 'encoder': encodera}
        return {'dist': distances}


class ModelTrainer:
    """
    Contains all the necessary code to train a torch model with a nice fit() loop. train_step()
    must be implemented by subclass.
    """

    def __init__(self, model: nn.Module, losses=None, optimizer=None, history=None, silence=False,
                 properties=None, random_seed=None):
        self.model = model
        self.history = history or {}
        self.silence = silence
        self.properties = properties or {}
        self.rng = np.random.default_rng(seed=random_seed)
        self.losses = losses
        self.optimizer = optimizer

    @staticmethod
    def prepare_torch_dataset(x, y=None, batch_size=256):
        """
        Prepare a PyTorch dataset and dataloader from input tensors.

        @param x: Input tensor(s) or numpy array(s). Can be a single input or a list/tuple of inputs.
        @param y: Target tensor or numpy array. Optional for cases where there's no target.
        @param batch_size: Batch size for the dataloader
        @return: PyTorch DataLoader
        """
        def to_torch_tensor(data):
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
            if isinstance(data, torch.Tensor):
                return data.float()
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Handle single or multiple x inputs
        if isinstance(x, (list, tuple)):
            x = [to_torch_tensor(xi) for xi in x]
        else:
            x = [to_torch_tensor(x)]

        # Handle y if provided
        if y is not None:
            y = to_torch_tensor(y)
            dataset = torch.utils.data.TensorDataset(*x, y)
        else:
            dataset = torch.utils.data.TensorDataset(*x)

        # Create dataloader
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_step(self, x, y, batch_size) -> dict:
        """
        Abstract single epoch of training.
        """
        dataloader = _prepare_torch_dataset(x, y, batch_size)
        epoch_loss = 0.0
        self.model.train()
        progress_bar = dataloader if self.silence else tqdm(
            dataloader, total=len(dataloader), desc="Training model...")

        for batch in progress_bar:
            batch_x1, batch_x2, batch_y = batch
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x1, batch_x2)

            # Compute loss
            loss = self.losses['dist'](outputs['dist'], batch_y)
            if self.properties['reg_dims']:
                loss += self.losses['encoder'](outputs['encoder'])

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            running_loss = epoch_loss / (progress_bar.n + 1)
            if not self.silence:
                progress_bar.set_description(f"Training model (loss: {running_loss:.4f})")

        if not self.silence:
            progress_bar.close()
        return {'loss': epoch_loss / len(dataloader)}

    def first_epoch(self, *args, lr=.1, **kwargs):
        """
        In the first epoch, modify only the scaling parameter with a higher LR.
        """
        orig_lr = self.model.optimizer.param_groups[0]['lr']
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = lr
        self.train_step(*args, **kwargs)
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = orig_lr

    def fit(self, *args, epochs=100, early_stop=True,
            min_delta=0, patience=3, first_ep_lr=0, **kwargs):
        """
        Train the model based on the given parameters. Extra arguments are passed to train_step.
        @param epochs: epochs to train for.
        @param min_delta: Minimum change required to qualify as an improvement.
        @param patience: How many epochs with no improvement before giving up. patience=0 disables.
        @param first_ep_lr: Learning rate for first epoch, when scaling param is being trained.
        """
        train_start = time.time()
        patience = patience or epochs + 1  # If patience==0, do not early stop
        if patience < 1:
            raise ValueError('Patience value must be >1.')
        if first_ep_lr:
            self._print('Running fast first epoch...')
            self.first_epoch(*args, lr=first_ep_lr, **kwargs)

        wait = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for i in range(epochs):  # Iterate over epochs
            start = time.time()
            self._print(f'Epoch {i + 1}:')

            this_history = self.train_step(*args, **kwargs)  # Run the train step
            self._print(f'Epoch time: {time.time() - start}')

            self.history = {k: v + this_history[k] for k, v in self.history.items()} if \
                self.history else this_history  # Update the loss history

            prev_best = min(self.history['loss'][:-1])
            this_loss = self.history['loss'][-1]

            if math.isnan(this_loss) or this_loss == 0:  # Divergence detection
                self._print('Stopping due to numerical instability, loss converges (0) or ' +
                            'diverges (Nan)')
                self.model.load_state_dict(best_state_dict)
                break

            if not early_stop or i == 0:  # If not early stopping, ignore the following
                continue

            if this_loss < prev_best - min_delta:  # Early stopping
                best_state_dict = self.model.get_state_dict()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                self._print('Stopping early due to lack of improvement!')
                self.model.load_state_dict(best_state_dict)
                break

        self._print(f"Completed fit() in {time.time() - train_start:.2f}s")
        return self.history

    def transform(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all values to transform.
        @param batch_size: Batch size for DataLoader.
        @return np.ndarray: Model output for all inputs.
        """
        dataset = _prepare_torch_dataset(data, None, batch_size)
        results = []
        for batch in tqdm(dataset):
            results.append(self.model(batch))
        return np.concatenate(results, axis=0)

    def summary(self):
        return summary(self.model)


class ComparativeModel(ModelTrainer):
    """
    Abstract ComparativeModel class. Stores some useful common functions.
    """

    def __init__(self, dist=None, embed_dist='euclidean', model=None, history=None,
                 silence=False, properties=None, random_seed=None, **kwargs):
        """
        model: tuple of (model, losses, optimizer)
        """
        model, losses, optimizer = model or self.create_model(**kwargs)
        super().__init__(
            model,
            losses,
            optimizer,
            history,
            silence,
            properties,
            random_seed,
        )
        self.distance = dist
        self.embed_dist = embed_dist

    def _print(self, *args, **kwargs):
        if not self.silence:
            print(*args, **kwargs)

    def create_model(self):
        """
        Create a model. Scopes automatically applied.
        """
        return None, None, None

    def random_set(self, x: np.ndarray, y: np.ndarray, epoch_factor=1) -> tuple[np.ndarray]:
        p1 = np.concatenate([self.rng.permutation(x.shape[0])
                            for _ in range(epoch_factor)])
        p2 = np.concatenate([self.rng.permutation(x.shape[0])
                            for _ in range(epoch_factor)])
        return x[p1], x[p2], y[p1], y[p2]

    def save(self, path: str, model=None):
        """
        Save the model to the given path.
        @param path: path to save to.
        """
        try:
            os.makedirs(path)
        except FileExistsError:
            print("WARN: Directory exists, overwriting...")
            shutil.rmtree(path)
            os.makedirs(path)
        model = model or self.model
        model.save(os.path.join(path, 'model.h5'))
        with open(os.path.join(path, 'distance.pkl'), 'wb') as f:
            pickle.dump(self.distance, f)
        with open(os.path.join(path, 'embed_dist.txt'), 'w') as f:
            f.write(self.embed_dist)
        if self.history:
            with open(os.path.join(path, 'history.json'), 'w') as f:
                json.dump(self.history, f)
        with open(os.path.join(path, 'properties.json'), 'w') as f:
            json.dump(self.properties, f)

    @classmethod
    def load(cls, path: str, v_scope: str, strategy=None, model=None, **kwargs):
        """
        Load the model from the filesystem.
        """
        contents = os.listdir(path)
        if not model:
            if 'model.h5' not in contents:
                raise ValueError(
                    'Model save file is necessary for loading a ComparativeModel!')
            strategy = strategy or tf.distribute.get_strategy()
            with tf.name_scope(v_scope):
                with strategy.scope():
                    model = tf.keras.models.load_model(
                        os.path.join(path, 'model.h5'))

        if 'distance.pkl' not in contents:
            print('Warning: distance save file missing!')
            dist = None
        else:
            with open(os.path.join(path, 'distance.pkl'), 'rb') as f:
                dist = pickle.load(f)
        if 'embed_dist.txt' not in contents:
            print('Warning: embedding distance save file missing, assuming Euclidean')
            embed_dist = 'euclidean'
        else:
            with open(os.path.join(path, 'embed_dist.txt'), 'r') as f:
                embed_dist = f.read().strip()
        history = None
        if 'history.json' in contents:
            with open(os.path.join(path, 'history.json'), 'r') as f:
                history = json.load(f)
        properties = None
        if 'properties.json' in contents:
            with open(os.path.join(path, 'properties.json'), 'w') as f:
                properties = json.load(f)
        return cls(v_scope=v_scope, dist=dist, model=model, strategy=strategy, history=history,
                   embed_dist=embed_dist, properties=properties, **kwargs)


class ComparativeEncoder(ComparativeModel):
    """
    Generic comparative encoder that can fit to data and transform sequences.
    """

    def __init__(self, model: nn.Module, reg_dims=False, properties=None, **kwargs):
        """
        @param model: PyTorch model that must support .train() and .eval() at minimum.
        @param reg_dims: Whether to use regularization on dimensions.
        @param properties: Custom properties to override defaults.
        """
        default_properties = {
            'input_shape': model.layers[0].output.shape[1:],
            'input_dtype': model.layers[0].dtype,
            'repr_size': model.layers[-1].output.shape[1],
            'depth': len(model.layers),
            'reg_dims': reg_dims
        }
        self.encoder = model
        super().__init__(properties=properties or default_properties, **kwargs)

    def create_model(self, loss='corr_coef', lr=.001, **adam_kwargs):
        comparative_model = ComparativeLayer(
            self.encoder,
            self.embed_dist,
            self.properties['reg_dims'],
            self.properties['repr_size'])

        if loss == 'corr_coef':
            losses = {'dist': self.correlation_coefficient_loss}
        elif loss == 'r2':
            losses = {'dist': self.r2_loss}
        else:
            losses = {'dist': loss}

        if self.properties['reg_dims']:
            losses['encoder'] = self.reg_loss

        optimizer = torch.optim.Adam(comparative_model.parameters(), lr=lr, **adam_kwargs)
        return comparative_model, losses, optimizer

    @classmethod
    def from_model_builder(cls, builder: ModelBuilder, repr_size=None, norm_type='soft_clip',
                           embed_dist='euclidean', **kwargs):
        """
        Initialize a ComparativeEncoder from a ModelBuilder object. Easy way to propagate the
        distribute strategy and variable scope. Also automatically adds a clip_norm for hyperbolic.
        """
        encoder = builder.compile(repr_size=repr_size, norm_type=norm_type, embed_space=embed_dist)
        return cls(encoder, strategy=builder.strategy, embed_dist=embed_dist, **kwargs)

    @staticmethod
    def reg_loss(y_pred):
        # Calculate the absolute values of the columns
        abs_columns = torch.abs(y_pred)
        # Calculate the mean of each column
        column_means = torch.mean(abs_columns, dim=0)
        # Apply a continuous function to the column means
        # Here, we use the exponential function e^(-x)
        exp_column_means = torch.exp(-column_means)
        # Calculate the average of the exponential column means
        avg_exp_column_means = torch.mean(exp_column_means)
        # Subtract the average from 1 to get the loss
        loss = 1.0 - avg_exp_column_means
        return loss

    @staticmethod
    def correlation_coefficient_loss(y_true, y_pred):
        """
        Correlation coefficient loss function for ComparativeEncoder.
        """
        x, y = y_true, y_pred
        mx, my = torch.mean(x), torch.mean(y)
        xm, ym = x - mx, y - my
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2))
        r = r_num / (r_den + 1e-8)  # Adding a small epsilon to avoid division by zero
        r = torch.clamp(r, min=-1.0, max=1.0)
        return 1 - r

    @staticmethod
    def r2_loss(y_true, y_pred):
        """
        Pearson's R^2 correlation, retaining the sign of the original R.
        """
        r = 1 - ComparativeEncoder.correlation_coefficient_loss(y_true, y_pred)
        r2 = r ** 2 * (r / torch.abs(r))
        return 1 - r2

    # pylint: disable=arguments-differ
    def train_step(self, data: np.ndarray, distance_on: np.ndarray, batch_size=256, epoch_factor=1):
        """
        Train a single randomized epoch on data and distance_on.
        @param data: data to train model on.
        @param distance_on: np.ndarray of data to use for distance computations. Allows for distance
        to be based on secondary properties of each sequence, or on a string representation of the
        sequence (e.g. for alignment comparison methods).
        @param batch_size: batch size for TensorFlow.
        @param jobs: number of CPU jobs to use.
        @param chunksize: chunksize for Python multiprocessing.
        """
        # It's common to input pandas series from Dataset instead of numpy array
        data = data.to_numpy() if isinstance(data, pd.Series) else data
        distance_on = distance_on.to_numpy() if isinstance(distance_on, pd.Series) else distance_on
        x1, x2, y1, y2 = self.random_set(data, distance_on, epoch_factor=epoch_factor)
        y = self.distance.transform_multi(y1, y2)
        return super().train_step((x1, x2), y, batch_size)


    def fit(self, *args, distance_on=None, **kwargs):
        distance_on = distance_on if distance_on is not None else args[1]
        super().fit(*args, distance_on, **kwargs)

    def transform(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all sequences to transform.
        @param batch_size: Batch size for .predict().
        @return np.ndarray: Representations for all sequences in data.
        """
        dataset = _prepare_torch_dataset(data, None, batch_size)
        reprs = []
        for batch in tqdm(dataset):
            reprs.append(self.encoder(batch))
        return np.concatenate(reprs, axis=0)

    def save(self, path: str):
        super().save(path, model=self.encoder)

    @classmethod
    def load(cls, path: str, v_scope='encoder', **kwargs):
        custom_objects = {
            'correlation_coefficient_loss': cls.correlation_coefficient_loss}
        with tf.keras.utils.custom_object_scope(custom_objects):
            return super().load(path, v_scope, **kwargs)

    def summary(self):
        """
        Prints a summary of the encoder.
        """
        summary(self.encoder)


class Decoder(ComparativeModel):
    """
    Abstract Decoder for encoding distances.
    """

    def __init__(self, v_scope='decoder', embed_dist_args=None, **kwargs):
        super().__init__(v_scope, **kwargs)
        embed_dist_args = embed_dist_args or {}
        if self.embed_dist == 'hyperbolic':
            self.embed_dist_calc = Hyperbolic(**embed_dist_args)
        elif self.embed_dist == 'euclidean':
            self.embed_dist_calc = Euclidean(**embed_dist_args)
        elif self.embed_dist == 'cosine':
            self.embed_dist_calc = Cosine(**embed_dist_args)
        else:  # Should never happen
            raise ValueError(
                f'Invalid embedding distance for decoder: {self.embed_dist}.')

    def random_distance_set(self, encodings: np.ndarray, distance_on: np.ndarray, epoch_factor=1):
        """
        Create a random set of distance data from the inputs.
        """
        x1, x2, y1, y2 = self.random_set(
            encodings, distance_on, epoch_factor=epoch_factor)

        self._print('Calculating embedding distances')
        x = self.embed_dist_calc.transform_multi(x1, x2)
        self._print('Calculating true distances')
        y = self.distance.transform_multi(y1, y2)
        return x, y

    def evaluate(self, encodings: np.ndarray, distance_on: np.ndarray, sample_size=None):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: true distances, predicted distances
        """
        sample_size = sample_size or len(encodings)
        x, y = self.random_distance_set(encodings, distance_on,
                                        epoch_factor=int(sample_size / len(encodings)) + 1)
        self._print('Predicting true distances...')
        x = self.transform(x)
        y = self.distance.invert_postprocessing(y)

        r2 = r2_score(y, x)
        mse = mean_squared_error(y, x)
        self._print(f'Mean squared error of distances: {mse}')
        self._print(f'R-squared correlation coefficient: {r2}')
        return y, x

    def save(self, path: str):
        super().save(path)
        with open(os.path.join(path, 'embed_dist_calc.pkl'), 'wb') as f:
            pickle.dump(self.embed_dist_calc, f)

    @staticmethod
    def load(path: str, v_scope='decoder', **kwargs):
        # pylint: disable=arguments-differ
        contents = os.listdir(path)
        if 'model.h5.pkl' in contents:
            obj = LinearDecoder.load(path)
        else:
            obj = DenseDecoder.load(path, **kwargs)
        with open(os.path.join(path, 'embed_dist_calc.pkl'), 'rb') as f:
            obj.embed_dist_calc = pickle.load(f)
        return obj


class _LinearRegressionModel(LinearRegression):
    def save(self, path: str):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self, f)


class LinearDecoder(Decoder):
    """
    Linear model of a decoder. Useful with correlation coefficient loss.
    """
    def create_model(self):
        return _LinearRegressionModel()

    def fit(self, encodings: np.ndarray, distance_on: np.ndarray, *args, **kwargs):
        """
        Fit the LinearDecoder to the given data.
        """
        # It's common to input pandas series from Dataset instead of numpy array
        distance_on = distance_on.to_numpy() if isinstance(
            distance_on, pd.Series) else distance_on
        x, y = self.random_distance_set(
            encodings, distance_on, *args, **kwargs)
        self.model.fit(x.reshape((-1, 1)), y)

    def transform(self, data: np.ndarray):
        """
        Transform the given data.
        """
        return self.distance.invert_postprocessing(self.model.predict(data.reshape(-1, 1)))

    @classmethod
    def load(cls, path: str, **kwargs):
        with open(os.path.join(path, 'model.h5.pkl'), 'rb') as f:
            model = pickle.load(f)
        return super(Decoder, cls).load(path, 'decoder', model=model, **kwargs)


class DenseDecoder(Decoder):
    """
    Decoder model to convert generated distances into true distances.
    """

    def __init__(self, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def create_model(self):
        optimizer = torch.optim.Adam(comparative_model.parameters(), lr=lr, **adam_kwargs)
        losses = {'loss': nn.MSELoss()}
        return nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Dropout(.1),
            nn.Linear(10, 10),
            nn.ReLU()
        ), optimizer, losses

    def fit(self, *args, epochs=25, **kwargs):
        """
        The decoder is probably an afterthought, 25 epochs seems like a sensible default to avoid
        adding too much overhead.
        """
        return super().fit(*args, epochs=epochs, **kwargs)

    def train_step(self, encodings: np.ndarray, distance_on: np.ndarray, epoch_factor=1):
        # It's common to input pandas series from Dataset instead of numpy array
        distance_on = distance_on.to_numpy() if isinstance(distance_on, pd.Series) else distance_on
        x, y = self.random_distance_set(encodings, distance_on, epoch_factor=epoch_factor)
        return super().train_step(x, y, self.batch_size)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the given distances between this model's encodings into predicted true distances.
        """
        return self.distance.invert_postprocessing(super().transform(data))

    @classmethod
    def load(cls, path: str, v_scope='decoder', **kwargs):
        return super(Decoder, cls()).load(path, v_scope, **kwargs)
