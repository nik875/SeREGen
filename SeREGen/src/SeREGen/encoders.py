"""
Module to help build encoders for ComparativeEncoder. It is recommended to use ModelBuilder.
"""
import tensorflow as tf
from .exceptions import IncompatibleDimensionsException


class AttentionBlock(tf.keras.layers.Layer):
    """
    Custom AttentionBlock layer that also contains normalization.
    Similar to the Transformer encoder block.
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        """
        Calls attention, normalization, feed forward, and second normalization layers.
        """
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


class ModelBuilder:
    """
    Class that helps easily build encoders for a ComparativeEncoder model.
    """
    def __init__(self, input_shape: tuple, input_dtype=None, distribute_strategy=None,
                 v_scope='encoder'):
        """
        Create a new ModelBuilder object.
        @param input_shape: Shape of model input.
        @param input_dtype: Optional dtype for model input.
        @param distribute_strategy: strategy to use for distributed training. Defaults to training
        on a single GPU.
        """
        self.strategy = distribute_strategy or tf.distribute.get_strategy()
        self.v_scope = v_scope
        with tf.name_scope(v_scope):
            with self.strategy.scope():
                self.inputs = tf.keras.layers.Input(input_shape, dtype=input_dtype)
        self.current = self.inputs

    # pylint: disable=no-self-argument,not-callable
    def _apply_scopes(fn):
        def in_scopes(self, *args, **kwargs):
            __doc__ = fn.__doc__  # Set the documentation
            with tf.name_scope(self.v_scope):
                with self.strategy.scope():
                    return fn(self, *args, **kwargs)
        return in_scopes

    @classmethod
    def text_input(cls, vocab: list[str], embed_dim: int, max_len: int, v_scope='encoder',
                   **kwargs):
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
        obj = cls((1,), input_dtype=tf.string, v_scope=v_scope, **kwargs)
        obj.text_vectorization(output_sequence_length=max_len,
                               output_mode='int',
                               vocabulary=vocab,
                               standardize=None,
                               split='character')
        obj.embedding(len(vocab) + 2, embed_dim)
        return obj

    @_apply_scopes
    def text_vectorization(self, *args, **kwargs):
        """
        Passes arguments directly to TextVectorization layer.
        """
        self.current = tf.keras.layers.TextVectorization(*args, **kwargs)(self.current)

    @_apply_scopes
    def embedding(self, input_dim: int, output_dim: int, mask_zero=False, **kwargs):
        """
        Adds an Embedding layer to preprocess ordinally encoded input sequences.
        Arguments are passed directly to Embedding constructor.
        @param input_dim: Each input character must range from [0, input_dim).
        @param output_dim: Size of encoding for each character in the sequences.
        @param mask_zero: Whether to generate a mask for zero values in the input. Defaults to True.
        """
        self.current = tf.keras.layers.Embedding(input_dim, output_dim,
                                                 mask_zero=mask_zero,
                                                 **kwargs)(self.current)

    def summary(self):
        """
        Display a summary of the model as it currently stands.
        """
        tf.keras.Model(inputs=self.inputs, outputs=self.current).summary()

    def shape(self) -> tuple:
        """
        Returns the shape of the output layer as a tuple. Excludes the first dimension of batch size
        """
        return tuple(self.current.shape[1:])

    class DynamicNormScalingLayer(tf.keras.layers.Layer):
        """
        Scale down the input such that the maximum absolute value is 1.
        """
        def __call__(self, inputs):
            return inputs / tf.reduce_max(tf.norm(inputs, axis=-1))

    @_apply_scopes
    def compile(self, repr_size=2, embed_space='euclidean') -> tf.keras.Model:
        """
        Create and return an encoder model.
        @param repr_size: Number of dimensions of output point (default 2 for visualization).
        @return tf.keras.Model
        """
        self.flatten()
        self.dense(repr_size, activation=None)  # Create special output layer
        if embed_space == 'hyperbolic':
            self.custom_layer(self.DynamicNormScalingLayer())
        return tf.keras.Model(inputs=self.inputs, outputs=self.current)

    @_apply_scopes
    def custom_layer(self, layer: tf.keras.layers.Layer):
        """
        Add a custom layer to the model.
        @param layer: TensorFlow layer to add.
        """
        self.current = layer(self.current)

    @_apply_scopes
    def reshape(self, new_shape: tuple, **kwargs):
        """
        Add a reshape layer. Additional keyword arguments accepted.
        @param new_shape: tuple new shape.
        """
        self.current = tf.keras.layers.Reshape(new_shape, **kwargs)(self.current)

    def transpose(self, a=0, b=1, **kwargs):
        """
        Transposes the input with a Reshape layer over the two given axes (flips them).
        First dimension for batch size is not included.
        @param a: First axis to transpose, defaults to 0.
        @param b: Second axis to transpose, defaults to 1.
        """
        shape = list(self.shape())
        tmp = shape[b]
        shape[b] = shape[a]
        shape[a] = tmp
        self.reshape(tuple(shape), **kwargs)

    @_apply_scopes
    def flatten(self, **kwargs):
        """
        Add a flatten layer. Additional keyword arguments accepted.
        """
        self.current = tf.keras.layers.Flatten(**kwargs)(self.current)

    @_apply_scopes
    def dropout(self, rate, **kwargs):
        """
        Add a dropout layer. Additional keyword arguments accepted.
        @param rate: rate to drop out inputs.
        """
        self.current = tf.keras.layers.Dropout(rate=rate, **kwargs)(self.current)

    @_apply_scopes
    def dense(self, size: int, depth=1, activation='relu', **kwargs):
        """
        Procedurally add dense layers to the model.
        @param size: number of nodes per layer.
        @param depth: number of layers to add.
        @param activation: activation function to use (relu by default).
        Additional keyword arguments are passed to TensorFlow Dense layer constructor.
        """
        for _ in range(depth):
            self.current = tf.keras.layers.Dense(size, activation=activation,
                                                 **kwargs)(self.current)
            self.current = tf.keras.layers.BatchNormalization()(self.current)

    @_apply_scopes
    def conv1D(self, filters: int, kernel_size: int, output_size: int, **kwargs):
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
        if len(self.shape()) != 2:
            raise IncompatibleDimensionsException()
        if kernel_size >= self.shape()[0]:
            raise IncompatibleDimensionsException()

        self.current = tf.keras.layers.Conv1D(filters, kernel_size, activation='relu',
                                              **kwargs)(self.current)
        self.current = tf.keras.layers.MaxPooling1D()(self.current)
        # Removes extra dimension from shape
        self.current = tf.keras.layers.Flatten()(self.current)
        self.current = tf.keras.layers.BatchNormalization()(self.current)
        self.dense(output_size, activation='relu')

    @_apply_scopes
    def attention(self, num_heads: int, output_size: int):
        """
        Add an attention layer. Embeddings must be generated beforehand.
        @param num_heads: Number of attention heads.
        @param output_size: Output size.
        @param rate: Dropout rate for AttentionBlock.
        """
        if len(self.shape()) != 2:
            raise IncompatibleDimensionsException()

        self.current = AttentionBlock(self.shape()[1], num_heads, output_size)(self.current)
        self.current = tf.keras.layers.BatchNormalization()(self.current)

