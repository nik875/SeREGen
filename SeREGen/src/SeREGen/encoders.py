"""
Module to help build encoders for ComparativeEncoder. It is recommended to use ModelBuilder.
"""

import os
import functools


import torch
from torch import nn
from torchinfo import summary


class IncompatibleDimensionsException(Exception):
    def __init__(self, message=None, prev_shape=None):
        self.message = (message or "Previous layer shape is incompatible with this layer's shape!")
        if prev_shape:
            self.message += f" Previous shape: {prev_shape}"
        super().__init__(self.message)


class LayerCommon(nn.Module):
    """
    Handles saving and loading of layers.
    """

    def __init__(self, norm_type=None, norm_to=1,
                 float_type=torch.float64, residual=False, **config):
        super().__init__()
        self.config = config
        self.config["norm_type"] = norm_type
        self.config["norm_to"] = norm_to
        self.config["float_type"] = float_type
        self.config["residual"] = residual

        if norm_type == "clip":
            self.norm_fn = self.clip_norm
        elif norm_type == "soft_clip":
            self.norm_fn = self.soft_clip_norm
            self.scaling = nn.Parameter(torch.Tensor([1 - 1e-2]), requires_grad=True)
            self.radius = nn.Parameter(torch.Tensor([1 - 1e-2]), requires_grad=True)
        elif norm_type == "scale_down":
            self.norm_fn = self.dynamic_norm_scaling
        elif norm_type == "l2":
            self.norm_fn = self.l2_norm
            self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=False)
        else:
            self.norm_fn = self.empty_layer

    @staticmethod
    def empty_layer(x):
        return x

    def clip_norm(self, x):
        return torch.clamp(x, min=-self.config["norm_to"], max=self.config["norm_to"])

    def soft_clip_norm(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-7)
        return x / norm * torch.tanh(norm * self.scaling) * \
            self.radius.clamp(min=1e-7, max=1 - 1e-7)

    def l2_norm(self, x):
        normalized = nn.functional.normalize(x, p=2, dim=-1, eps=1e-7)
        soft_clamp = self.config["norm_to"] * torch.sigmoid(self.radius)
        return normalized * soft_clamp + 1e-7

    def dynamic_norm_scaling(self, x):
        return x / torch.norm(x, dim=-1).max()

    def forward(self, inputs):
        return self.norm_fn(inputs) + inputs * int(self.config["residual"])

    def __repr__(self):
        kwargs = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({kwargs})"


class Linear(LayerCommon):
    def __init__(self, input_size: int, output_size: int, activation=None, **kwargs):
        super().__init__(input_size=input_size, output_size=output_size, activation=activation,
                         **kwargs)

        layers = nn.ModuleList()
        layers.append(nn.Linear(input_size, output_size, dtype=self.config["float_type"]))
        if activation == "relu":
            layers.append(nn.ReLU())
        self.linear = nn.Sequential(*layers)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return super().forward(linear_out)


class Conv1D(LayerCommon):
    def __init__(self, input_size, filters, kernel_size, **kwargs):
        super().__init__(input_size=input_size, filters=filters, kernel_size=kernel_size, **kwargs,)
        self.conv_unit = nn.Sequential(
            nn.Conv1d(
                input_size,
                filters,
                kernel_size,
                padding="same",
                dtype=self.config["float_type"],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(filters, dtype=self.config["float_type"]),
            nn.Conv1d(filters, input_size, kernel_size=1, dtype=self.config["float_type"]),
        )

    def forward(self, inputs):
        return self.conv_unit(inputs)


class AttentionBlock(LayerCommon):
    """
    Custom AttentionBlock layer that also contains normalization.
    Similar to the Transformer encoder block.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, **kwargs)
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dtype=self.config["float_type"])
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, dtype=self.config["float_type"]),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim, dtype=self.config["float_type"]),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        """
        Calls attention, normalization, feed forward, and second normalization layers.
        """
        attn_output, _ = self.att(inputs, inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)  # Skip connection
        ffn_output = self.ffn(out1)
        return super().forward(self.layernorm2(out1 + ffn_output))


class OneHotEncoding(LayerCommon):
    """
    One hot encoding as a layer. Casts input to integers.
    """

    def __init__(self, depth: int, **kwargs):
        super().__init__(depth=depth, **kwargs)

    def forward(self, inputs):
        encoded = nn.functional.one_hot(inputs.to(torch.int64), num_classes=self.config["depth"])
        return encoded.to(self.config["float_type"])


class Transpose(LayerCommon):
    """
    Transpose the input over given axes.
    """

    def __init__(self, dim0: int, dim1: int, **kwargs):
        super().__init__(dim0=dim0, dim1=dim1, **kwargs)

    def forward(self, inputs):
        return inputs.transpose(self.config["dim0"] + 1, self.config["dim1"] + 1)


class ModelBuilder:
    """
    Class that helps easily build encoders for a ComparativeEncoder model.
    """

    def __init__(self, input_shape: tuple, input_dtype=torch.float64):
        """
        Create a new ModelBuilder object.
        @param input_shape: Shape of model input.
        @param input_dtype: Optional dtype for model input.
        on a single GPU.
        """
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.float_ = input_dtype
        self.layers = nn.ModuleList()
        self.residual_size = 0

    def one_hot_encoding(self, depth: int, **kwargs):
        """
        Add one hot encoding for the input. Input must be ordinally encoded data. Input will be
        casted to int32.
        @param depth: number of categories to encode.
        """
        self.layers.append(OneHotEncoding(depth, float_type=self.float_, **kwargs))

    def embedding(self, input_dim: int, output_dim: int, padding_idx=None, **kwargs):
        """
        Adds an Embedding layer to preprocess ordinally encoded input sequences.
        Arguments are passed directly to Embedding constructor.
        @param input_dim: Each input character must range from [0, input_dim).
        @param output_dim: Size of encoding for each character in the sequences.
        @param padding_idx: Index of padding token. Defaults to None.
        """
        self.input_dtype = torch.long
        self.layers.append(
            nn.Embedding(
                input_dim,
                output_dim,
                padding_idx=padding_idx,
                dtype=self.float_,
                **kwargs,
            ))

    def summary(self, **kwargs):
        """
        Display a summary of the model as it currently stands.
        """
        model = nn.Sequential(*self.layers)
        return summary(model, input_size=(1, *self.input_shape),
                       dtypes=[self.input_dtype], **kwargs,)

    def shape(self) -> tuple:
        """
        Returns the shape of the output layer as a tuple. Excludes the first dimension of batch size
        """
        summary_str = self.summary(verbose=0)
        return tuple(summary_str.summary_list[-1].output_size[1:])

    def compile(self, repr_size=None, embed_space="euclidean",
                norm_type="soft_clip", name="encoder",):
        """
        Create and return an encoder model.
        @param repr_size: Number of dimensions of output point (default 2 for visualization).
        @return nn.Module
        """
        if repr_size:  # Skip this step by setting repr_size to 0
            # Avoid putting points on the very edge of the poincare ball
            norm_to = 1 - 1e-5 if embed_space == "hyperbolic" else 1
            self.flatten()
            self.layers.append(Linear(self.shape()[-1],
                                      repr_size,
                                      activation=None,
                                      norm_type=norm_type,
                                      norm_to=norm_to,
                                      float_type=self.float_))
        if embed_space == "hyperbolic" and norm_type not in [
                "clip", "soft_clip", "scale_down", "l2", ]:
            print("WARN: Empty/invalid norm_type, compiling hyperbolic model without " +
                  "normalization...")

        model = nn.Sequential(*self.layers)
        model.name = name

        # Create properties dict
        properties = {
            "input_shape": self.input_shape,
            "input_dtype": self.float_,
            "repr_size": repr_size if repr_size else self.shape()[-1],
            "depth": len(self.layers),
            "embed_dist": embed_space,
            "norm_type": norm_type,
        }

        return model, properties

    def custom_layer(self, layer: nn.Module):
        """
        Add a layer to the model.
        @param layer: TensorFlow layer to add.
        """
        self.layers.append(layer)

    def reshape(self, new_shape: tuple):
        """
        Add a reshape layer. Additional keyword arguments accepted.
        @param new_shape: tuple new shape.
        """
        self.layers.append(nn.Unflatten(-1, new_shape))

    def transpose(self, a=0, b=1):
        """
        Transposes the input with a Reshape layer over the two given axes (flips them).
        First dimension for batch size is not included.
        @param a: First axis to transpose, defaults to 0.
        @param b: Second axis to transpose, defaults to 1.
        """
        self.layers.append(Transpose(a, b, float_type=self.float_))

    def flatten(self):
        """
        Add a flatten layer. Additional keyword arguments accepted.
        """
        self.layers.append(nn.Flatten(start_dim=1))

    def dropout(self, rate):
        """
        Add a dropout layer. Additional keyword arguments accepted.
        @param rate: rate to drop out inputs.
        """
        self.layers.append(nn.Dropout(p=rate))

    def batch_norm(self, output_size: int):
        """
        Add a batch normalization to the model.
        """
        if len(self.shape()) == 3:
            self.layers.append(nn.BatchNorm2d(output_size, dtype=self.float_))
        elif len(self.shape()) == 2:
            self.layers.append(nn.BatchNorm1d(output_size, dtype=self.float_))
        else:
            raise IncompatibleDimensionsException("Model dims not either 1 or 2")

    def dense(self, output_size: int, depth=1, activation="relu", residual=False):
        """
        Procedurally add dense layers to the model. Input size is inferred.
        @param size: number of nodes per layer.
        @param depth: number of layers to add.
        @param activation: activation function to use (relu by default, can pass in callable/None).
        Additional keyword arguments are passed to TensorFlow Dense layer constructor.
        """
        for _ in range(depth):
            self.layers.append(Linear(self.shape()[-1],
                                      output_size,
                                      activation=activation,
                                      residual=residual,
                                      float_type=self.float_))
            if 1 < len(self.shape()) < 4:
                self.batch_norm(self.shape()[-2])

    def conv1D(self, filters: int, kernel_size: int, residual=False):
        """
        Add a convolutional layer.
        Output passes through feed forward layer with size specified by output_dim.
        @param filters: number of convolution filters to use.
        @param kernel_size: size of convolution kernel. Must be less than the first dimension of
        prior layer's shape.
        @param output_size: output size of the layer.
        @param activation: activation function.
        Additional keyword arguments are passed to TensorFlow Conv1D layer constructor.
        """
        shape = self.shape()
        if len(shape) != 2:
            raise IncompatibleDimensionsException()
        if kernel_size >= shape[1]:
            raise IncompatibleDimensionsException()

        self.layers.append(
            Conv1D(
                shape[0],
                filters,
                kernel_size,
                float_type=self.float_,
                residual=residual))

    def attention(self, num_heads: int, output_size: int, residual=False):
        """
        Add an attention layer. Embeddings must be generated beforehand.
        @param num_heads: Number of attention heads.
        @param output_size: Output size.
        @param rate: Dropout rate for AttentionBlock.
        """
        if len(self.shape()) != 2:
            raise IncompatibleDimensionsException(prev_shape=self.shape())
        self.layers.append(
            AttentionBlock(
                self.shape()[1],
                num_heads,
                output_size,
                residual=residual))

    def save_model(self, path: str):
        """
        Save the current model to a file.

        @param path: The file path where the model should be saved.
        """
        if not self.layers:
            raise ValueError("No model to save. Build the model first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        torch.save({"model": nn.Sequential(*self.layers),
                   "input_shape": self.input_shape, "input_dtype": self.float_}, path)

    @staticmethod
    def load_model(path: str):
        """
        Load a model from a file.

        @param path: The file path from which to load the model.
        @return: A new ModelBuilder instance with the loaded model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")
        return torch.load(path)
