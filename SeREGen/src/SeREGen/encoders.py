"""
Module to help build encoders for ComparativeEncoder. It is recommended to use ModelBuilder.
"""
import string
import random
import os


import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchinfo import summary


class IncompatibleDimensionsException(Exception):
    def __init__(self):
        self.message = "Previous layer shape is incompatible with this layer's shape!"
        super().__init__(self.message)


class _CustomLayer(nn.Module):
    """
    Handles saving and loading of layers.
    """

    def __init__(self, **config):
        self.config = config

    def __repr__(self):
        kwargs = ', '.join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({kwargs})"


class AttentionBlock(_CustomLayer):
    """
    Custom AttentionBlock layer that also contains normalization.
    Similar to the Transformer encoder block.
    """

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
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
        return self.layernorm2(out1 + ffn_output)


class OneHotEncoding(_CustomLayer):
    """
    One hot encoding as a layer. Casts input to integers.
    """

    def __init__(self, depth: int):
        super().__init__(depth=depth)

    def forward(self, inputs):
        return nn.functional.one_hot(inputs.to(torch.int64), num_classes=self.config['depth'])


class TextVectorizer(_CustomLayer):
    """
    Convert input text into ordinally encoded tokens, then vector embeddings.
    """

    def __init__(self, vocab: list[str], embed_dim: int,
                 max_len: int, embeddings=True):
        super().__init__(vocab=str(vocab), embed_dim=embed_dim, max_len=max_len,
                         embeddings=embeddings)
        self.tokenizer = get_tokenizer(list)
        self.vocab = build_vocab_from_iterator(vocab, specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.embedding = nn.Embedding(len(self.vocab), embed_dim,
                                      padding_idx=self.vocab['<pad>']) if embeddings else None

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        # Tokenize all texts at once
        tokens = [self.tokenizer(t) for t in text]
        # Convert tokens to indices
        indices = [torch.tensor([self.vocab[token] for token in t],
                                dtype=torch.long) for t in tokens]
        # Pad sequences
        padded = nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=self.pad_idx)

        # Truncate to max_len if needed
        if padded.size(1) > self.max_len:
            padded = padded[:, :self.max_len]

        if self.embedding is not None:
            padded = self.embedding(padded)
        return padded


class ClipNorm(_CustomLayer):
    def __init__(self, clip_norm):
        super().__init__(clip_norm=clip_norm)

    def forward(self, x):
        return nn.utils.clip_grad_norm(x, max_norm=self.clip_norm)


class SoftClipNorm(_CustomLayer):
    def __init__(self, scale=1.0):
        super().__init__(scale=scale)

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdims=True)
        scaled_norm = self.scale * norm
        soft_clip_factor = torch.tanh(scaled_norm) / scaled_norm
        return x * soft_clip_factor


class L2Normalize(_CustomLayer):
    def __init__(self, dim=-1, eps=1e-8):
        super().__init__(dim=dim, eps=eps)
        self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)

    def forward(self, x):
        min_scale = 1e-7
        max_scale = 1 - 1e3
        normalized = nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)
        scaled = normalized * self.radius.clamp(min=min_scale, max=max_scale)
        return scaled


class DynamicNormScaling(_CustomLayer):
    """
    Scale down the input such that the maximum absolute value is 1.
    """

    def forward(self, x):
        return x / torch.norm(x, dim=-1).max()


class Transpose(_CustomLayer):
    """
    Transpose the input over given axes.
    """

    def __init__(self, dim0: int, dim1: int):
        super().__init__(dim0=dim0, dim1=dim1)

    def forward(self, x):
        return x.transpose(self.config['dim0'] + 1, self.config['dim1'] + 1)


class ModelBuilder:
    """
    Class that helps easily build encoders for a ComparativeEncoder model.
    """

    def __init__(self, input_shape: tuple, input_dtype=None, is_text_input=False):
        """
        Create a new ModelBuilder object.
        @param input_shape: Shape of model input.
        @param input_dtype: Optional dtype for model input.
        on a single GPU.
        """
        self.input_shape = input_shape
        self.input_dtype = input_dtype or torch.float32
        self.is_text_input = is_text_input
        self.layers = nn.ModuleList()

    @classmethod
    def text_input(cls, vocab: list[str], embed_dim: int, max_len: int, embeddings=True, **kwargs):
        """
        Factory function that returns a new ModelBuilder object which can receive text input. Adds a
        TextVectorization and an Embedding layer to preprocess string input data. Split happens
        along characters. Additional keyword arguments are passed to ModelBuilder constructor.

        @param vocab: Vocabulary to adapt TextVectorization layer to. String of characters with no
        duplicates.
        @param embed_dim: Size of embeddings to generate for each character in sequence. If None or
        not passed, defaults to one hot encoding of input sequences.
        @param max_len: Length to trim and pad input sequences to.
        @return ModelBuilder: Newly created object.
        """
        obj = cls((1,), input_dtype=torch.long, is_text_input=True, **kwargs)
        obj.text_vectorization(vocab, embed_dim, max_len, embeddings=embeddings)
        return obj

    def text_vectorization(self, *args, **kwargs):
        """
        Passes arguments directly to TextVectorizer module.
        """
        self.layers.append(TextVectorizer(*args, **kwargs))

    def one_hot_encoding(self, depth: int, **kwargs):
        """
        Add one hot encoding for the input. Input must be ordinally encoded data. Input will be
        casted to int32.
        @param depth: number of categories to encode.
        """
        self.layers.append(OneHotEncoding(depth, **kwargs))

    def embedding(self, input_dim: int, output_dim: int, padding_idx=None, **kwargs):
        """
        Adds an Embedding layer to preprocess ordinally encoded input sequences.
        Arguments are passed directly to Embedding constructor.
        @param input_dim: Each input character must range from [0, input_dim).
        @param output_dim: Size of encoding for each character in the sequences.
        @param padding_idx: Index of padding token. Defaults to None.
        """
        self.layers.append(nn.Embedding(input_dim, output_dim, padding_idx=padding_idx, **kwargs))

    def summary(self):
        """
        Display a summary of the model as it currently stands.
        """
        model = nn.Sequential(*self.layers)
        summary(model, input_size=(-1, *self.input_shape), dtypes=[self.input_dtype])

    def shape(self) -> tuple:
        """
        Returns the shape of the output layer as a tuple. Excludes the first dimension of batch size
        """
        with torch.no_grad():
            if self.is_text_input:
                # For text input, create a random string
                x = [''.join(random.choices(string.ascii_lowercase, k=10))]
            else:
                x = torch.randn(1, *self.input_shape).to(self.input_dtype)

            for layer in self.layers:
                x = layer(x)
        return tuple(x.shape[1:])

    def compile(self, repr_size=None, embed_space='euclidean',
                norm_type='soft_clip', name='encoder'):
        """
        Create and return an encoder model.
        @param repr_size: Number of dimensions of output point (default 2 for visualization).
        @return nn.Module
        """
        if repr_size:
            self.flatten()
            self.dense(repr_size, activation=None)  # Create special output layer
        if embed_space == 'hyperbolic':
            if norm_type == 'clip':
                self.custom_layer(ClipNorm(1))
            elif norm_type == 'soft_clip':
                self.custom_layer(SoftClipNorm(1))
            elif norm_type == 'scale_down':
                self.custom_layer(DynamicNormScaling())
            elif norm_type == 'l2':
                self.custom_layer(L2Normalize())
            else:
                print('WARN: Empty/invalid norm_type, compiling hyperbolic model without ' +
                      'normalization...')

        model = nn.Sequential(*self.layers)
        model.name = name
        return model

    def custom_layer(self, layer: nn.Module):
        """
        Add a custom layer to the model.
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
        self.layers.append(Transpose(a, b))

    def flatten(self):
        """
        Add a flatten layer. Additional keyword arguments accepted.
        """
        self.layers.append(nn.Flatten())

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
        self.layers.append(nn.BatchNorm1d(output_size))

    def dense(self, output_size: int, depth=1, activation='relu'):
        """
        Procedurally add dense layers to the model. Input size is inferred.
        @param size: number of nodes per layer.
        @param depth: number of layers to add.
        @param activation: activation function to use (relu by default, can pass in callable/None).
        Additional keyword arguments are passed to TensorFlow Dense layer constructor.
        """
        for _ in range(depth):
            self.layers.append(nn.Linear(self.shape()[-1], output_size))
            self.batch_norm(output_size)
            if activation is not None:
                self.layers.append(nn.ReLU() if activation == 'relu' else activation)

    def conv1D(self, filters: int, kernel_size: int, output_size: int):
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
        if kernel_size >= shape[0]:
            raise IncompatibleDimensionsException()

        self.layers.append(nn.Conv1D(shape[1], filters, kernel_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(2))
        self.layers.append(nn.Flatten())
        self.batch_norm(filters * ((self.shape()[0] - kernel_size + 1) // 2))
        self.dense(output_size, activation='relu')

    def attention(self, num_heads: int, output_size: int):
        """
        Add an attention layer. Embeddings must be generated beforehand.
        @param num_heads: Number of attention heads.
        @param output_size: Output size.
        @param rate: Dropout rate for AttentionBlock.
        """
        shape = self.shape()
        if len(shape) != 2:
            raise IncompatibleDimensionsException()
        self.layers.append(AttentionBlock(shape[1], num_heads, output_size))

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
        torch.save({
            'model': nn.Sequential(*self.layers),
            'input_shape': self.input_shape,
            'input_dtype': self.input_dtype,
        }, path)

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
