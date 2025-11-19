import tensorflow as tf
from tensorflow.keras.layers import (
    GlobalAveragePooling1D, Conv1D, Multiply, Reshape, Activation,
    Input, Permute, Concatenate, BatchNormalization, Add, Flatten, Dropout, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import cupy as cp


def ECA_Module(input_tensor, gamma=2, b=1):
    """
    Efficient Channel Attention (ECA)
    input_tensor: shape (B, L, C) or (B, C), typically output of a Conv1D
    returns: same shape, with channel-wise attention applied
    """
    channel = input_tensor.shape[-1]

    # Step 1: Global Average Pooling
    gap = GlobalAveragePooling1D()(input_tensor)  # shape: (B, C)

    # Step 2: Adaptive kernel size
    t = int(abs((tf.math.log(tf.cast(channel, tf.float32)) / tf.math.log(2.0)) / gamma + b))
    k_size = t if t % 2 == 1 else t + 1  # ensure odd kernel size

    # Step 3: Conv1D across channels (temporal domain)
    x = Reshape((channel, 1))(gap)  # shape: (B, C, 1)
    x = Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(x)  # shape: (B, C, 1)
    x = Activation('sigmoid')(x)

    # Step 4: Reweight input_tensor
    x = Reshape((1, channel))(x)  # shape: (B, 1, C)
    output = Multiply()([input_tensor, x])  # broadcasting

    return output

def make_model_inception(pairs: int = 2, plain_bits: int = 32):
    """
    Inception-based Neural Distinguisher (generic across ciphers)

    Input encoding per sample: concatenate [ΔC || C || C*] for `pairs` pairs.
    - Vector length per pair: 3 * plain_bits
    - Total input_dim = pairs * 3 * plain_bits
    We reshape to (pairs, 6, word_size) with word_size = plain_bits // 2 (since 6 * word_size = 3 * plain_bits).

    Requirements:
    - plain_bits must be divisible by 2.
    """
    if plain_bits % 2 != 0:
        raise ValueError("plain_bits must be divisible by 2 for model reshaping (got %d)" % plain_bits)

    word_size = plain_bits // 2
    input_dim = pairs * 3 * plain_bits
    num_filters = 32
    d1, d2 = 64, 64
    depth = 5
    ks = 3
    reg_param = 1e-4

    inp = Input(shape=(input_dim,))
    x = Reshape((pairs, 6, word_size))(inp)     # (None, pairs, 6, word_size)
    x = Permute((1, 3, 2))(x)                   # (None, pairs, word_size, 6)
    x = Reshape((pairs * word_size, 6))(x)      # (None, pairs*word_size, 6)

    # Inception block
    conv01 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(x)
    conv02 = Conv1D(num_filters, kernel_size=3, padding='same', kernel_regularizer=l2(reg_param))(x)
    conv03 = Conv1D(num_filters, kernel_size=5, padding='same', kernel_regularizer=l2(reg_param))(x)
    conv04 = Conv1D(num_filters, kernel_size=7, padding='same', kernel_regularizer=l2(reg_param))(x)
    x = Concatenate(axis=-1)([conv01, conv02, conv03, conv04])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ECA Attention after Inception block
    x = ECA_Module(x)

    shortcut = x

    # Residual tower
    for _ in range(depth):
        conv1 = Conv1D(num_filters * 4, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters * 4, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2

    # Prediction head
    x = Flatten()(shortcut)
    x = Dropout(0.3)(x)
    x = Dense(512, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(d1, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(d2, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(x)

    return Model(inputs=inp, outputs=out)

