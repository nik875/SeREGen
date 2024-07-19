"""
ComparativeEncoder module, trains a model comparatively using distances.
"""

import time
import math
import copy

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from torch import nn
from torchinfo import summary
from tqdm import tqdm
from geomstats.geometry.poincare_ball import PoincareBall

from .encoders import ModelBuilder
from .distance import Distance
from ._saving import _save_object, _save_torch, _load_object, _load_torch, _create_save_directory


def check_grad_nan(model, suppress=False):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                if not suppress:
                    print(f"NaN gradient detected in {name}")
                return True
    return False


def check_grad_explosion(model, threshold, suppress=False):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.abs(param.grad).max() > threshold:
                if not suppress:
                    print(
                        f"Exploding gradient detected in {name}: {torch.abs(param.grad).max()}"
                    )
                return True
    return False


def check_grad_vanishing(model, threshold, suppress=False):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.abs(param.grad).max() < threshold:
                if not suppress:
                    print(f"Vanishing gradient detected in {name}: {torch.abs(param.grad).max()}")
                return True
    return False


def check_gradients(model, explosion_threshold=1e4, vanishing_threshold=1e-7, suppress=None):
    suppress = suppress or []
    return {
        "nan": check_grad_nan(model, "nan" in suppress),
        "exploding": check_grad_explosion(model, explosion_threshold, "exploding" in suppress),
        "vanishing": check_grad_vanishing(model, vanishing_threshold, "vanishing" in suppress),
    }


class _NormalizedDistanceLayer(nn.Module):
    """
    Adds a scaling parameter that's set to 1 / average distance on the first iteration.
    Output WILL be normalized.
    """

    def __init__(self, trainable_scaling=True, **kwargs):
        super().__init__(**kwargs)
        self.scaling_param = nn.Parameter(torch.ones(1), requires_grad=trainable_scaling)

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

    def __init__(self, encoder, embed_dist, embedding_size):
        """
        @param encoder: PyTorch model to use as the encoder.
        @param embed_dist: Distance metric to use when comparing two sequences.
        """
        super().__init__()
        self.encoder = encoder

        if embed_dist.lower() == "euclidean":
            self.dist_layer = EuclideanDistanceLayer()
        elif embed_dist.lower() == "hyperbolic":
            self.dist_layer = HyperbolicDistanceLayer(embedding_size=embedding_size)
        else:
            raise ValueError("Invalid embedding distance provided!")

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
        return distances, encodera, encoderb


class CustomLosses:
    """
    Functions that apply custom losses
    """

    @staticmethod
    def corr_coef(y_true, y_pred):
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
    def r2(y_true, y_pred):
        """
        Pearson's R^2 correlation, retaining the sign of the original R.
        """
        r = 1 - CustomLosses.corr_coef(y_true, y_pred)
        r2 = r**2 * (r / (torch.abs(r) + 1e-8))
        return 1 - r2


class ComparativeEncoder:
    """
    Contains all the necessary code to train a torch model with a nice fit() loop. train_step()
    must be implemented by subclass. Can be reused to train different model types.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dist: Distance,
        repr_size=None,
        embed_dist="euclidean",
        batch_size=128,
        device=None,
        silence=False,
        input_dtype=torch.float64,
        output_dtype=torch.float64,
        properties=None,
        random_seed=None,
        force_dtype=False
    ):
        self.properties = properties or {}
        if repr_size is not None:
            self.properties["repr_size"] = repr_size
        elif "repr_size" not in self.properties:
            raise ValueError(
                "repr_size must be provided as argument or in properties dict"
            )
        self.properties["silence"] = silence
        self.properties["input_dtype"] = input_dtype
        self.properties["output_dtype"] = output_dtype
        self.properties["device"] = device or self.get_device()
        self.properties["seed"] = random_seed
        self.properties["force_dtype"] = force_dtype
        if self.properties["force_dtype"]:
            encoder = encoder.to(self.properties["output_dtype"])
        self.encoder = encoder.to(self.properties["device"])
        self.model = self.create_model()
        if "dist" not in self.properties:
            if not isinstance(dist, Distance):
                raise ValueError(f"Argument 'dist' should be type Distance, received {type(dist)}")
            self.properties["dist"] = dist
        if "embed_dist" not in self.properties:
            self.properties["embed_dist"] = embed_dist
        if "batch_size" not in self.properties:
            self.properties["batch_size"] = batch_size
        self.rng = np.random.default_rng(seed=random_seed)
        self.history = {}

    def create_model(self):
        model = ComparativeLayer(
            self.encoder,
            self.properties["embed_dist"],
            self.properties["repr_size"],
        )
        if self.properties["force_dtype"]:
            model = model.to(self.properties["output_dtype"])
        model = model.to(self.properties["device"])
        return model

    @classmethod
    def from_model_builder(
        cls,
        builder: ModelBuilder,
        repr_size=None,
        norm_type="soft_clip",
        embed_dist="euclidean",
        **kwargs,
    ):
        """
        Initialize a ComparativeEncoder from a ModelBuilder object. Easy way to propagate the
        distribute strategy and variable scope. Also automatically adds a clip_norm for hyperbolic.
        """
        encoder, properties = builder.compile(
            repr_size=repr_size, norm_type=norm_type, embed_space=embed_dist)
        return cls(encoder, properties=properties, embed_dist=embed_dist,
                   input_dtype=builder.input_dtype, output_dtype=builder.float_, **kwargs)

    def _print(self, *args, **kwargs):
        if not self.properties["silence"]:
            print(*args, **kwargs)

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            if self.properties["input_dtype"] == torch.float64:
                self.properties["input_dtype"] = torch.float32  # MPS uses float32
            return torch.device("mps")
        self._print("WARN: GPU not detected, defaulting to CPU")
        return torch.device("cpu")

    def prepare_torch_dataset(self, x, y=None, shuffle=True):
        """
        Prepare a PyTorch dataset and dataloader from input tensors.

        @param x: Input tensor(s) or numpy array(s). Can be a single input or a list/tuple of inputs
        @param y: Target tensor or numpy array. Optional for cases where there's no target.
        @param batch_size: Batch size for the dataloader
        @return: PyTorch DataLoader
        """

        def to_torch_tensor(data, dtype):
            if isinstance(data, np.ndarray):
                if dtype == str:
                    data = data.to()
                return torch.from_numpy(data).to(dtype)
            if isinstance(data, torch.Tensor):
                return data.to(dtype)
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Handle single or multiple x inputs
        if isinstance(x, (list, tuple)):
            x = [to_torch_tensor(xi, self.properties["input_dtype"]) for xi in x]
        else:
            x = [to_torch_tensor(x, self.properties["input_dtype"])]

        # Handle y if provided
        if y is not None:
            y = to_torch_tensor(y, self.properties["output_dtype"])
            dataset = torch.utils.data.TensorDataset(*x, y)
        else:
            dataset = torch.utils.data.TensorDataset(*x)

        # Create dataloader
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.properties["batch_size"], shuffle=shuffle)

    def train_step(self, x, y, loss="corr_coef", lr=.001,
                   suppress_grad_warn=None, clip_grad=True) -> dict:
        """
        Abstract single epoch of training.
        """
        match loss:
            case "corr_coef":
                loss_fn = CustomLosses.corr_coef
            case "r2":
                loss_fn = CustomLosses.r2
            case "mse":
                loss_fn = nn.MSELoss()
            case _:
                if isinstance(loss, str) and loss != '':
                    raise ValueError(f"Unknown loss string: {loss}")
                loss_fn = loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        suppress_grad_warn = suppress_grad_warn or []
        suppress_grad_warn = [i.lower() for i in suppress_grad_warn]
        dataloader = self.prepare_torch_dataset(x, y)
        epoch_loss, n = 0, 0
        self.model = self.model.to(self.properties["device"])
        self.model.train()
        if not self.properties["silence"]:
            dataloader = tqdm(dataloader, total=len(dataloader), desc="Training model...")

        for b_x1, b_x2, b_y in dataloader:
            b_x1 = b_x1.to(self.properties["input_dtype"]).to(self.properties["device"])
            b_x2 = b_x2.to(self.properties["input_dtype"]).to(self.properties["device"])
            b_y = b_y.to(self.properties["output_dtype"]).to(self.properties["device"])
            optimizer.zero_grad()

            # Forward pass
            outputs, *_ = self.model(b_x1, b_x2)

            # Compute loss
            loss = loss_fn(outputs, b_y)

            if math.isnan(loss):
                if "nan" in suppress_grad_warn:
                    print("WARN: nan loss")
                    continue
                print("ERR: nan loss")
                return {"loss": math.nan}  # Return nan to stop training at higher level

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e4)

            errs = check_gradients(self.model, suppress=suppress_grad_warn)
            if errs["nan"]:
                if "nan" in suppress_grad_warn:
                    print("WARN: nan gradient")
                    continue
                print("ERR: nan gradient")
                return {"loss": math.nan}  # Return nan to stop training at higher level

            optimizer.step()
            epoch_loss += loss.item() * b_x1.shape[0]
            n += b_x1.shape[0]
            running_loss = epoch_loss / n
            if not self.properties["silence"]:
                dataloader.set_description(f"Training model (loss: {running_loss:.4e})")

        if not self.properties["silence"]:
            dataloader.close()
        return {"loss": epoch_loss / n}

    def random_set(self, x: np.ndarray, y: np.ndarray, epoch_factor=1) -> tuple[np.ndarray]:
        total_samples = x.shape[0] * epoch_factor
        p1 = np.empty(total_samples, dtype=int)
        p2 = np.empty(total_samples, dtype=int)

        idx = 0
        while idx < total_samples:
            to_draw = min(x.shape[0], total_samples - idx)

            new_p1 = self.rng.permutation(x.shape[0])[:to_draw]
            new_p2 = self.rng.permutation(x.shape[0])[:to_draw]

            # Remove matching pairs
            mask = new_p1 != new_p2
            new_p1, new_p2 = new_p1[mask], new_p2[mask]

            # Add to p1 and p2
            end_idx = idx + new_p1.shape[0]
            p1[idx:end_idx] = new_p1
            p2[idx:end_idx] = new_p2
            idx = end_idx

        # Trim to exact size if we've overshot
        p1, p2 = p1[:total_samples], p2[:total_samples]
        return x[p1], x[p2], y[p1], y[p2]

    def random_epoch(self, data: np.ndarray, distance_on: np.ndarray, epoch_factor=1, **kwargs):
        # pylint: disable=arguments-differ
        """
        Train a single randomized epoch on data and distance_on.
        @param data: data to train model on.
        @param distance_on: np.ndarray of data to use for distance computations. Allows for distance
        to be based on secondary properties of each sequence, or on a string representation of the
        sequence (e.g. for alignment comparison methods).
        @param jobs: number of CPU jobs to use.
        @param chunksize: chunksize for Python multiprocessing.
        """
        # It's common to input pandas series from Dataset instead of numpy array
        data = data.to_numpy() if isinstance(data, pd.Series) else data
        if isinstance(distance_on, pd.Series):
            distance_on = distance_on.to_numpy()
        x1, x2, y1, y2 = self.random_set(data, distance_on, epoch_factor=epoch_factor)
        y = self.properties["dist"].transform_multi(y1, y2)
        return self.train_step((x1, x2), y, **kwargs)

    def fit(
        self,
        data: np.ndarray,
        distance_on=None,
        epoch_factor=1,
        epochs=100,
        early_stop=True,
        min_delta=0,
        patience=3,
        first_ep_lr=0,
        **kwargs,
    ):
        """
        Train the model based on the given parameters. Extra arguments are passed to train_step.
        @param epochs: epochs to train for.
        @param min_delta: Minimum change required to qualify as an improvement.
        @param patience: How many epochs with no improvement before giving up. patience=0 disables.
        @param first_ep_lr: Learning rate for first epoch, when scaling param is being trained.
        """
        train_start = time.time()
        distance_on = distance_on if distance_on is not None else data
        patience = patience or epochs + 1  # If patience==0, do not early stop
        if patience < 1:
            raise ValueError("Patience value must be >1.")
        if "loss" not in self.history or not isinstance(self.history["loss"], list):
            self.history["loss"] = []
        if first_ep_lr:
            self._print("Running fast first epoch...")
            orig_lr = kwargs["lr"] if "lr" in kwargs else .001
            kwargs["lr"] = first_ep_lr
            first_history = self.random_epoch(data, distance_on, epoch_factor, **kwargs)
            kwargs["lr"] = orig_lr
            if math.isnan(first_history["loss"]):
                self._print(
                    "Stopping due to numerical instability, loss converges (0) or diverges (Nan)")
                return first_history
            self.history["loss"].append(first_history["loss"])

        wait = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for i in range(epochs):  # Iterate over epochs
            start = time.time()
            self._print(f"Epoch {i + 1}:")

            # Run the train step
            this_history = self.random_epoch(data, distance_on, epoch_factor, **kwargs)
            self._print(f"Epoch time: {time.time() - start}")

            self.history["loss"].append(this_history["loss"])

            this_loss = self.history["loss"][-1]
            # Make sure prev_best is the same as this_loss at beginning
            prev_best = min(self.history["loss"][:-1]) \
                if len(self.history["loss"]) > 1 else this_loss

            # Divergence detection
            if math.isnan(this_loss) or this_loss == 0:
                self._print(
                    "Stopping due to numerical instability, loss converges (0) or diverges (Nan)")
                self.model.load_state_dict(best_state_dict)
                break

            if not early_stop or i == 0:  # If not early stopping, ignore the following
                continue

            if this_loss < prev_best - min_delta:  # Early stopping
                best_state_dict = self.model.state_dict()
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                self._print("Stopping early due to lack of improvement!")
                self.model.load_state_dict(best_state_dict)
                break

        self._print(f"Completed fit() in {time.time() - train_start:.2f}s")
        return self.history

    def transform(self, x1: np.ndarray, x2=None) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all values to transform.
        @param batch_size: Batch size for DataLoader.
        @return np.ndarray: Model output for all inputs.
        """
        model, x = (self.encoder, x1) if x2 is None else (self.model, (x1, x2))
        model.to(self.properties["device"])
        model.eval()
        dataset = self.prepare_torch_dataset(x, None, shuffle=False)
        if self.properties["silence"]:
            dataset = tqdm(dataset, total=len(dataset))
        results = []
        with torch.no_grad():
            for batch in dataset:
                model_output = model(*[i.to(self.properties["device"]) for i in batch])
                results.append(model_output if x2 is None else model_output[0])
        return torch.cat(results, dim=0).detach().cpu().numpy()

    def comparative_model_summary(self, _model=None):
        model = _model or self.model
        return summary(model, input_shape=(
            1, *self.properties["input_shape"]), dtypes=[self.properties["input_dtype"]])

    def summary(self):
        return self.comparative_model_summary(_model=self.encoder)

    def save(self, path):
        _create_save_directory(path)
        _save_torch(self.encoder, path, "model.pth")
        _save_object(self.history, path, "history.pkl")
        _save_object(self.properties, path, "properties.pkl")

    @classmethod
    def load(cls, path):
        trainer = cls.__new__(cls)
        trainer.encoder = _load_torch(path, "model.pth")
        trainer.history = _load_object(path, "history.pkl")
        trainer.properties = _load_object(path, "properties.pkl")
        trainer.model = trainer.create_model()
        trainer.rng = np.random.default_rng(
            seed=trainer.properties["random_seed"] if "random_seed" in trainer.properties else None)
        return trainer

    def random_distance_set(self, data: np.ndarray, distance_on: np.ndarray, epoch_factor=1):
        """
        Create a random set of distance data from the inputs.
        """
        self.model.eval()
        x1, x2, y1, y2 = self.random_set(data, distance_on, epoch_factor=epoch_factor)
        self._print("Calculating embedding distances")
        x = self.transform(x1, x2)
        self._print("Calculating true distances")
        y = self.properties["dist"].transform_multi(y1, y2)
        return x, y

    def evaluate(self, data: np.ndarray, distance_on: np.ndarray, sample_size=None):
        """
        Evaluate the performance of the model by seeing how well we can predict true sequence
        dissimilarity from encoding distances.
        @param sample_size: Number of sequences to use for evaluation. All in dataset by default.
        @return np.ndarray, np.ndarray: true distances, predicted distances
        """
        sample_size = sample_size or len(data)
        x, y = self.random_distance_set(
            data, distance_on, epoch_factor=sample_size // len(data) + 1)

        r2 = pearsonr(x, y).statistic ** 2
        mse = np.mean((x - y) ** 2)
        self._print(f"Mean squared error of distances: {mse}")
        self._print(f"R-squared correlation coefficient: {r2}")
        return x, y
