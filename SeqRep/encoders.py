import tensorflow as tf
import numpy as np


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
    
    def compile(self, output_dim=2) -> tf.keras.Model:
        """
        Create and return an encoder model.
        @param output_dim: Number of dimensions of output point (default 2 for visualization).
        @return tf.keras.Model
        """
        self.flatten()
        self.dense(output_dim, activation=None)  # Create special output layer
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
    
    def conv1D(self, filters: int, kernel_size: int, divide=None, **kwargs):
        """
        Add a convolutional layer. Output passes through feed forward network with size specified by filters.
        @param filters: output size of the layer.
        @param kernel_size: size of convolution kernel. Must be less than the first dimension of prior layer's shape.
        @param divide: integer value to divide previous size by for dimension expansion. Must be divisible.
        Additional keyword arguments are passed to TensorFlow Conv1D layer constructor.
        """
        orig_shape = tuple(self.current.shape)[1:]
        if divide:
            self.reshape((*orig_shape[:-1], orig_shape[-1] // divide, divide))
        else:
            self.reshape((*orig_shape, 1))
        self.current = tf.keras.layers.Conv1D(filters, kernel_size, **kwargs)(self.current)
        self.current = tf.keras.layers.MaxPooling1D()(self.current)
        self.current = tf.keras.layers.Flatten()(self.current)  # Removes extra dimension from shape
        self.current = tf.keras.layers.BatchNormalization()(self.current)
        self.dense(filters)
    
    def attention(self, embed_dim: int, num_heads: int, output_dim: int, divide=False, rate=.1):
        """
        Add an attention layer.
        @param embed_dim: Desired embedding dimension size. New dimension will be created to accomodate.
        @param num_heads: Number of attention heads.
        @param output_dim: Output size.
        @param divide: whether to divide last dimension of previous layer by embed_dim to generate embeddings.
        @param rate: Learning rate for AttentionBlock.
        """
        shape = tuple(self.current.shape[1:])
        if divide:
            self.reshape((*shape[:-1], shape[-1] // embed_dim, embed_dim))
        else:
            self.dense(shape[-1] * embed_dim)
            self.reshape((*shape[:-1], shape[-1], embed_dim))
        self.current = AttentionBlock(embed_dim, num_heads, output_dim, rate=rate)(self.current)
        self.current = tf.keras.layers.BatchNormalization()(self.current)
