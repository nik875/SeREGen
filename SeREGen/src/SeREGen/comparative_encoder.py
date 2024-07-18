"""
ComparativeEncoder module, trains a model comparatively using distances.
"""

import os
import pickle
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
                    print(
                        f"Vanishing gradient detected in {name}: {torch.abs(param.grad).max()}"
                    )
                return True
    return False


def check_gradients(
    model, explosion_threshold=1e4, vanishing_threshold=1e-7, suppress=None
):
    suppress = suppress or []
    return {
        "nan": check_grad_nan(model, "nan" in suppress),
        "exploding": check_grad_explosion(
            model, explosion_threshold, "exploding" in suppress
        ),
        "vanishing": check_grad_vanishing(
            model, vanishing_threshold, "vanishing" in suppress
        ),
    }


def _create_save_directory(path):
    os.makedirs(path, exist_ok=True)


def _save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class _NormalizedDistanceLayer(nn.Module):
    """
    Adds a scaling parameter that's set to 1 / average distance on the first iteration.
    Output WILL be normalized.
    """

    def __init__(self, trainable_scaling=True, **kwargs):
        super().__init__(**kwargs)
        self.scaling_param = nn.Parameter(
            torch.ones(1), requires_grad=trainable_scaling
        )

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
        return distances


class ModelTrainer:
    """
    Contains all the necessary code to train a torch model with a nice fit() loop. train_step()
    must be implemented by subclass. Can be reused to train different model types.
    """

    def __init__(
        self,
        model: nn.Module,
        losses=None,
        optimizer=None,
        history=None,
        silence=False,
        properties=None,
        random_seed=None,
        device=None,
        batch_size=128,
        dtype=torch.float64,
        **_,
    ):
        self.properties = properties or {}
        self.properties["silence"] = silence
        self.properties["dtype"] = dtype
        self.properties["device"] = device or self.get_device()
        self.properties["seed"] = random_seed
        self.model = model.to(self.properties["dtype"]).to(self.properties["device"])
        self.history = history or {}
        self.rng = np.random.default_rng(seed=random_seed)
        self.losses = losses
        self.optimizer = optimizer
        if "batch_size" not in self.properties:
            self.properties["batch_size"] = batch_size

    def _print(self, *args, **kwargs):
        if not self.properties["silence"]:
            print(*args, **kwargs)

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            if self.properties["dtype"] == torch.float64:
                self.properties["dtype"] = torch.float32  # MPS uses float32
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

        def to_torch_tensor(data):
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.properties["dtype"])
            if isinstance(data, torch.Tensor):
                return data.to(self.properties["dtype"])
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
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.properties["batch_size"], shuffle=shuffle
        )

    def train_step(self, x, y, suppress_grad_warn=None, clip_grad=True) -> dict:
        """
        Abstract single epoch of training.
        """
        suppress_grad_warn = suppress_grad_warn or []
        suppress_grad_warn = [i.lower() for i in suppress_grad_warn]
        dataloader = self.prepare_torch_dataset(x, y)
        epoch_loss = 0.0
        n = 0
        self.model = self.model.to(self.properties["dtype"]).to(
            self.properties["device"]
        )
        self.model.train()
        if not self.properties["silence"]:
            dataloader = tqdm(
                dataloader, total=len(dataloader), desc="Training model..."
            )

        for batch in dataloader:
            b_x1, b_x2, b_y = map(
                lambda i: i.to(self.properties["dtype"]).to(self.properties["device"]),
                batch,
            )
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(b_x1, b_x2)

            # Compute loss
            loss = self.losses["dist"](outputs, b_y)

            if math.isnan(loss.item()):
                if "nan" in suppress_grad_warn:
                    continue
                print("Diverging to nan")
                return {"loss": math.nan}

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e4)

            errs = check_gradients(self.model, suppress=suppress_grad_warn)
            if errs["nan"]:
                if "nan" in suppress_grad_warn:
                    continue
                return {"loss": math.nan}

            self.optimizer.step()
            epoch_loss += loss.item() * b_x1.shape[0]
            n += b_x1.shape[0]
            running_loss = epoch_loss / n
            if not self.properties["silence"]:
                dataloader.set_description(f"Training model (loss: {running_loss:.4e})")

        if not self.properties["silence"]:
            dataloader.close()
        return {"loss": epoch_loss / n}

    def set_lr(self, lr):
        self.properties["lr"] = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def first_epoch(self, *args, lr=0.1, **kwargs):
        """
        In the first epoch, modify only the scaling parameter with a higher LR.
        """
        orig_lr = self.optimizer.param_groups[0]["lr"]
        self.set_lr(lr)
        this_loss = self.train_step(*args, **kwargs)["loss"]
        # Divergence detection
        if math.isnan(this_loss) or this_loss == 0:
            return False
        self.set_lr(orig_lr)
        return True

    def fit(
        self,
        *args,
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
        patience = patience or epochs + 1  # If patience==0, do not early stop
        if patience < 1:
            raise ValueError("Patience value must be >1.")
        if first_ep_lr:
            self._print("Running fast first epoch...")
            success = self.first_epoch(*args, lr=first_ep_lr, **kwargs)
            if not success:
                self._print(
                    "Stopping due to numerical instability, loss converges (0) or "
                    + "diverges (Nan)"
                )
                return {"loss": [math.nan]}
        if "loss" not in self.history or not isinstance(self.history["loss"], list):
            self.history["loss"] = []

        wait = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for i in range(epochs):  # Iterate over epochs
            start = time.time()
            self._print(f"Epoch {i + 1}:")

            this_history = self.train_step(*args, **kwargs)  # Run the train step
            self._print(f"Epoch time: {time.time() - start}")

            self.history["loss"].append(this_history["loss"])

            this_loss = self.history["loss"][-1]
            prev_best = (
                min(self.history["loss"][:-1])
                if len(self.history["loss"]) > 1
                else this_loss
            )  # Make sure prev_best is the same as this_loss at beginning

            # Divergence detection
            if math.isnan(this_loss) or this_loss == 0:
                self._print(
                    "Stopping due to numerical instability, loss converges (0) or "
                    + "diverges (Nan)"
                )
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

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all values to transform.
        @param batch_size: Batch size for DataLoader.
        @return np.ndarray: Model output for all inputs.
        """
        self.model = self.model.to(self.properties["device"])
        self.model.eval()
        dataset = self.prepare_torch_dataset(data, None, shuffle=False)
        if self.properties["silence"]:
            dataset = tqdm(dataset, total=len(dataset))
        results = []
        with torch.no_grad():
            for batch in dataset:
                results.append(
                    self.model(*[i.to(self.properties["device"]) for i in batch])
                )
        return torch.cat(results, dim=0).detach().cpu().numpy()

    def summary(self):
        return summary(
            self.model.to(self.properties["dtype"]),
            input_shape=(1, *self.properties["input_shape"]),
            dtypes=[self.properties["dtype"]],
        )

    def save(self, path):
        _create_save_directory(path)
        if self.model is not None:
            torch.save(self.model, os.path.join(path, "model.pth"))
        if self.optimizer is not None:
            torch.save(self.optimizer, os.path.join(path, "optimizer.pth"))
        if self.losses is not None:
            _save_object(self.losses, os.path.join(path, "losses.pkl"))
        _save_object(self.history, os.path.join(path, "history.pkl"))
        _save_object(self.properties, os.path.join(path, "properties.pkl"))

    @classmethod
    def load(cls, path):
        trainer = cls.__new__(cls)
        if os.path.exists(p := os.path.join(path, "model.pth")):
            trainer.model = torch.load(p)
        if os.path.exists(p := os.path.join(path, "optimizer.pth")):
            trainer.optimizer = torch.load(p)
        if os.path.exists(p := os.path.join(path, "losses.pkl")):
            trainer.losses = _load_object(p)
        trainer.history = _load_object(os.path.join(path, "history.pkl"))
        trainer.properties = _load_object(os.path.join(path, "properties.pkl"))
        return trainer


class ComparativeEncoder(ModelTrainer):
    """
    Generic comparative encoder that can fit to data and transform sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        properties=None,
        repr_size=None,
        dist=None,
        dtype=torch.float64,
        embed_dist="euclidean",
        **kwargs,
    ):
        """
        @param model: PyTorch model that must support .train() and .eval() at minimum.
        @param properties: Custom properties to override defaults.
        """
        self.properties = properties or {}
        if repr_size is not None:
            self.properties["repr_size"] = repr_size
        elif "repr_size" not in self.properties:
            raise ValueError(
                "repr_size must be provided as argument or in properties dict"
            )
        self.properties["dtype"] = dtype
        self.encoder = model.to(dtype)
        self.distance = dist
        self.embed_dist = embed_dist
        if "embed_dist" not in self.properties:
            self.properties["embed_dist"] = embed_dist
        super().__init__(
            *self.create_model(**kwargs),
            properties=self.properties,
            dtype=dtype,
            **kwargs,
        )
        self.encoder = self.encoder.to(self.properties["device"])

    def create_model(self, loss="corr_coef", lr=0.001, **_):
        self.properties["loss"] = loss
        self.properties["lr"] = lr
        comparative_model = ComparativeLayer(
            self.encoder,
            self.embed_dist,
            self.properties["repr_size"],
        ).to(self.properties["dtype"])

        if loss == "corr_coef":
            losses = {"dist": self.correlation_coefficient_loss}
        elif loss == "r2":
            losses = {"dist": self.r2_loss}
        elif loss == "mse":
            losses = {"dist": nn.MSELoss()}
        else:
            losses = {"dist": loss}

        optimizer = torch.optim.Adam(comparative_model.parameters(), lr=lr)
        return comparative_model, losses, optimizer

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
            repr_size=repr_size, norm_type=norm_type, embed_space=embed_dist
        )
        return cls(encoder, properties=properties, embed_dist=embed_dist, **kwargs)

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
        r2 = r**2 * (r / (torch.abs(r) + 1e-8))
        return 1 - r2

    def random_set(
        self, x: np.ndarray, y: np.ndarray, epoch_factor=1
    ) -> tuple[np.ndarray]:
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

    def train_step(
        self,
        data: np.ndarray,
        distance_on: np.ndarray,
        epoch_factor=1,
        **kwargs,
    ):
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
        y = self.distance.transform_multi(y1, y2)
        return super().train_step((x1, x2), y, **kwargs)

    def fit(self, *args, distance_on=None, **kwargs):
        distance_on = distance_on if distance_on is not None else args[1]
        super().fit(*args, distance_on, **kwargs)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the given data into representations using trained model.
        @param data: np.ndarray containing all sequences to transform.
        @return np.ndarray: Representations for all sequences in data.
        """
        model = self.model
        self.model = self.encoder
        result = super().transform(data)
        self.model = model
        return result

    def comparative_model_summary(self):
        """
        Prints a summary of the comparative encoder.
        """
        super().summary()

    def save(self, path):
        model = self.model
        self.model = None
        super().save(path)
        self.model = model
        torch.save(self.encoder, os.path.join(path, "model.pth"))
        _save_object(self.distance, os.path.join(path, "distance.pkl"))
        _save_object(self.embed_dist, os.path.join(path, "embed_dist.pkl"))

    @classmethod
    def load(cls, path):
        model = super().load(path)
        model.encoder = model.model
        # pylint: disable=no-member
        model.model, model.losses, model.optimizer = model.create_model(
            loss=model.properties["loss"],
            lr=model.properties["lr"],
        )
        model.distance = _load_object(os.path.join(path, "distance.pkl"))
        model.embed_dist = _load_object(os.path.join(path, "embed_dist.pkl"))
        return model

    def random_distance_set(
        self, data: np.ndarray, distance_on: np.ndarray, epoch_factor=1
    ):
        """
        Create a random set of distance data from the inputs.
        """
        self.model.eval()
        x1, x2, y1, y2 = self.random_set(data, distance_on, epoch_factor=epoch_factor)
        self._print("Calculating embedding distances")
        x = super().transform((x1, x2))
        self._print("Calculating true distances")
        y = self.distance.transform_multi(y1, y2)
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
            data, distance_on, epoch_factor=sample_size // len(data) + 1
        )

        r2 = pearsonr(x, y).statistic ** 2
        mse = np.mean((x - y) ** 2)
        self._print(f"Mean squared error of distances: {mse}")
        self._print(f"R-squared correlation coefficient: {r2}")
        return x, y
