import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

# Initialize weights with normal distribution
weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

def Dense(units, activation=None):
    """
    Fully connected (Dense) layer.
    Params:
        units: Number of output neurons.
        activation: Activation function.
    Returns:
        A Dense layer with specified units and activation.
    """
    return tf.keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_initializer=weights_initializer,
        bias_initializer='zeros'
    )

def Conv1D(filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=True):
    """
    1D Convolutional layer.
    Params:
        filters: Number of output filters.
        kernel_size: Size of the convolutional kernel.
        strides: Stride of the convolution.
        padding: Padding method ('valid' or 'same').
        activation: Activation function.
    Returns:
        A Conv2D layer configured for 1D convolution.
    """
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding=padding,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=weights_initializer
    )

Conv2D = tf.keras.layers.Conv2D  # Alias for 2D convolution

def DeConv1D(filters, kernel_size, strides=1, padding='valid', use_bias=True):
    """
    1D Deconvolutional (Transposed Convolution) layer.
    Params:
        filters: Number of output filters.
        kernel_size: Size of the deconvolutional kernel.
        strides: Stride of the deconvolution.
        padding: Padding method.
    Returns:
        A Conv2DTranspose layer configured for 1D deconvolution.
    """
    return tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=weights_initializer
    )

def BatchNormalization(trainable=True, virtual_batch_size=None):
    """
    Batch Normalization layer.
    Params:
        trainable: If True, the layer is updated during training.
        virtual_batch_size: Optional size for virtual batching.
    Returns:
        A BatchNormalization layer with default settings.
    """
    return tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size
    )

def Activation(x, activation):
    """
    Applies an activation function.
    Params:
        x: Input tensor.
        activation: Type of activation ('relu', 'leaky_relu', 'sigmoid', 'softmax', or 'tanh').
    Returns:
        Tensor with the specified activation applied.
    """
    activations = {
        'relu': lambda: tf.keras.activations.relu(x),
        'leaky_relu': lambda: tf.keras.activations.relu(x, alpha=0.2),
        'sigmoid': lambda: tf.keras.activations.sigmoid(x),
        'softmax': lambda: tf.keras.activations.softmax(x, axis=-1),
        'tanh': lambda: tf.keras.activations.tanh(x)
    }
    return activations.get(activation, lambda: ValueError("Unknown activation type"))()

def Dropout(rate):
    """
    Dropout layer to reduce overfitting.
    Params:
        rate: Dropout rate (0-1).
    Returns:
        A Dropout layer with the specified rate.
    """
    return tf.keras.layers.Dropout(rate=rate)

def flatten():
    """
    Flatten layer to reshape inputs to a single vector.
    Returns:
        A Flatten layer.
    """
    return tf.keras.layers.Flatten()

def normalization(name):
    """
    Returns a normalization layer.
    Params:
        name: Type of normalization ('none', 'batch_norm', 'instance_norm', or 'layer_norm').
    Returns:
        A normalization layer or identity function based on input.
    """
    normalizations = {
        'none': lambda: lambda x: x,
        'batch_norm': lambda: keras.layers.BatchNormalization(),
        'instance_norm': lambda: tfa.layers.InstanceNormalization(),
        'layer_norm': lambda: keras.layers.LayerNormalization()
    }
    return normalizations.get(name, lambda: ValueError("Unknown normalization type"))()

def attention_block_1d(curr_layer, conn_layer):
    """
    Attention mechanism for 1D data, enhancing spatial relationship learning.
    Params:
        curr_layer: The current input layer.
        conn_layer: The connected layer.
    Returns:
        Layer with applied attention weights.
    """
    inter_channel = curr_layer.get_shape().as_list()[3]

    # Attention components
    theta_x = Conv1D(inter_channel, 1, 1)(conn_layer)
    phi_g = Conv1D(inter_channel, 1, 1)(curr_layer)
    f = Activation(keras.layers.add([theta_x, phi_g]), 'relu')
    psi_f = Conv1D(1, 1, 1)(f)
    rate = Activation(psi_f, 'sigmoid')

    return keras.layers.multiply([conn_layer, rate])
