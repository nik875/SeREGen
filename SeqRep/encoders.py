import tensorflow as tf
from .exceptions import IncompatibleDimensionsException


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(AttentionBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ModelBuilder:
    """
    Class that helps easily build encoders for a ComparativeEncoder model.
    """
    def __init__(self, input_shape: tuple):
        """
        Create a new ModelBuilder object.
        @param input_shape: Shape of model input.
        """
        self.inputs = tf.keras.layers.Input(input_shape)
        self.current = self.inputs

    def summary(self):
        """
        Display a summary of the model as it currently stands.
        """
        tf.keras.Model(inputs=self.inputs, outputs=self.current).summary()

    def _shape(self) -> tuple:
        """
        Returns the shape of the output layer as a tuple. Excludes the first dimension of batch size.
        """
        return tuple(self.current.shape[1:])
    
    def compile(self, output_dim=2) -> tf.keras.Model:
        """
        Create and return an encoder model.
        @param output_dim: Number of dimensions of output point (default 2 for visualization).
        @return tf.keras.Model
        """
        self.flatten()
        self.dense(output_dim)  # Create special output layer
        return tf.keras.Model(inputs=self.inputs, outputs=self.current)
    
    def custom_layer(self, layer: tf.keras.layers.Layer):
        """
        Add a custom layer to the model.
        @param layer: TensorFlow layer to add.
        """
        self.current = layer(self.current)
    
    def reshape(self, new_shape: tuple, **kwargs):
        """
        Add a reshape layer. Additional keyword arguments accepted.
        @param new_shape: tuple new shape.
        """
        self.current = tf.keras.layers.Reshape(new_shape, **kwargs)(self.current)
    
    def flatten(self, **kwargs):
        """
        Add a flatten layer. Additional keyword arguments accepted.
        """
        self.current = tf.keras.layers.Flatten(**kwargs)(self.current)

    def dropout(self, rate, **kwargs):
        """
        Add a dropout layer. Additional keyword arguments accepted.
        @param rate: rate to drop out inputs.
        """
        self.current = tf.keras.layers.Dropout(rate=rate, **kwargs)(self.current)
    
    def dense(self, size: int, depth=1, activation='relu', **kwargs):
        """
        Procedurally add dense layers to the model.
        @param size: number of nodes per layer.
        @param depth: number of layers to add.
        @param activation: activation function to use (relu by default).
        Additional keyword arguments are passed to TensorFlow Dense layer constructor.
        """
        for _ in range(depth):
            self.current = tf.keras.layers.Dense(size, activation=activation, **kwargs)(self.current)

    def embeddings(self, size: int, activation='relu', **kwargs):
        """
        Expand dimensions and create `size` embeddings.
        @param size: size of embeddings
        @param activation: activation function to use for dense layers
        Additional keyword arguments are accepted for Dense layer constructor.
        """
        orig_shape = self._shape()
        self.dense(size * self._shape()[-1], activation=activation, **kwargs)
        self.reshape((*orig_shape, size))

    def conv1D(self, filters: int, kernel_size: int, output_size: int, **kwargs):
        """
        Add a convolutional layer. Output passes through feed forward layer with size specified by output_dim.
        @param filters: number of convolution filters to use.
        @param kernel_size: size of convolution kernel. Must be less than the first dimension of prior layer's shape.
        @param output_size: output size of the layer.
        Additional keyword arguments are passed to TensorFlow Conv1D layer constructor.
        """
        if len(self._shape()) != 2:
            raise IncompatibleDimensionsException()
        if kernel_size >= self._shape()[0]:
            raise IncompatibleDimensionsException()

        self.current = tf.keras.layers.Conv1D(filters, kernel_size, **kwargs)(self.current)
        self.current = tf.keras.layers.MaxPooling1D()(self.current)
        self.current = tf.keras.layers.Flatten()(self.current)  # Removes extra dimension from shape
        self.current = tf.keras.layers.BatchNormalization()(self.current)
        self.dense(output_size, activation='relu')
    
    def attention(self, num_heads: int, output_size: int, rate=.1):
        """
        Add an attention layer. Embeddings must be generated beforehand.
        @param num_heads: Number of attention heads.
        @param output_size: Output size.
        @param rate: Learning rate for AttentionBlock.
        """
        if len(self._shape()) != 2:
            raise IncompatibleDimensionsException()

        self.current = AttentionBlock(self._shape()[1], num_heads, output_size, rate=rate)(self.current)
        self.current = tf.keras.layers.BatchNormalization()(self.current)
