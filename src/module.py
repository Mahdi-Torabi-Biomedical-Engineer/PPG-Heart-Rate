import tensorflow as tf
import tensorflow.keras as keras
import layers

def generator_attention(
    input_shape=512, 
    filter_size=[64, 128, 256, 512, 512, 512],
    kernel_size=[16, 16, 16, 16, 16, 16],
    n_downsample=6,
    norm='layer_norm', 
    skip_connection=True
):
    """
    Generator model with attention mechanism.
    Converts PPG signal to ECG using a series of downsampling and upsampling layers.
    Parameters:
        input_shape: Input signal length (default: 512).
        filter_size: List of filters for each downsampling level.
        kernel_size: Kernel sizes for each layer in the generator.
        n_downsample: Number of downsampling steps.
        norm: Type of normalization (e.g., 'layer_norm').
        skip_connection: If True, applies skip connections with attention.
    Returns:
        Model object for generating ECG signals from PPG input.
    """

    def _downsample(ip, filter_size, kernel_size, norm, stride_size=2):
        """
        Downsampling layer with convolution, normalization, and activation.
        Params:
            ip: Input tensor.
            filter_size: Number of filters.
            kernel_size: Size of the convolution kernel.
            norm: Normalization type ('layer_norm', 'batch_norm', etc.).
            stride_size: Stride size for convolution.
        Returns:
            Downsampled tensor.
        """
        ip = layers.Conv1D(filters=filter_size, kernel_size=kernel_size, strides=stride_size, padding='same', use_bias=False)(ip)
        if norm != 'none':
            ip = layers.normalization(norm)(ip)
        ip = layers.Activation(ip, activation='leaky_relu')
        return ip

    def _upsample(ip, filter_size, kernel_size, norm, stride_size=2, drop_rate=0.5, apply_dropout=False):
        """
        Upsampling layer with transposed convolution, normalization, and optional dropout.
        Params:
            ip: Input tensor.
            filter_size: Number of filters.
            kernel_size: Size of the convolution kernel.
            norm: Normalization type.
            stride_size: Stride size for upsampling.
            drop_rate: Dropout rate.
            apply_dropout: Whether to apply dropout.
        Returns:
            Upsampled tensor.
        """
        ip = layers.DeConv1D(filters=filter_size, kernel_size=kernel_size, strides=stride_size, padding='same', use_bias=False)(ip)
        if norm != 'none':
            ip = layers.normalization(norm)(ip)
        if apply_dropout:
            ip = layers.Dropout(rate=drop_rate)
        ip = layers.Activation(ip, activation='relu')
        return ip

    # Input and Reshaping
    h = inputs = keras.Input(shape=input_shape)  # Input tensor with specified shape
    h = tf.expand_dims(h, axis=1)  # Expand to 4D for compatibility with Conv layers
    h = tf.expand_dims(h, axis=3)

    # Downsampling with connections for skip connections
    connections = []
    for k in range(n_downsample):
        if k == 0:
            h = _downsample(h, filter_size[k], kernel_size[k], 'none')
        else:
            h = _downsample(h, filter_size[k], kernel_size[k], norm)
        connections.append(h)

    # First Upsampling
    h = _upsample(h, filter_size[-1], kernel_size[-1], norm, stride_size=1)
    if skip_connection:
        _h = layers.attention_block_1d(curr_layer=h, conn_layer=connections[-1])   
        h = keras.layers.add([h, _h])

    # Remaining Upsampling Layers with Skip Connections
    for l in range(1, n_downsample):
        h = _upsample(h, filter_size[-1 - l], kernel_size[-1 - l], norm)
        if skip_connection:
            _h = layers.attention_block_1d(curr_layer=h, conn_layer=connections[-1 - l])
            h = keras.layers.add([h, _h])

    # Output Layer
    h = layers.DeConv1D(filters=1, kernel_size=kernel_size[0], strides=2, padding='same')(h)
    h = layers.Activation(h, activation='tanh')
    h = tf.squeeze(h, axis=1)
    h = tf.squeeze(h, axis=2)

    return keras.Model(inputs=inputs, outputs=h)
