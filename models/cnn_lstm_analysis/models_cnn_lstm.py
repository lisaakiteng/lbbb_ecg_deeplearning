"""
===========================================================
models_cnn_lstm.py
===========================================================

CNN-LSTM ECG classifier for CRT prediction.

CNN feature extraction followed by LSTM for time-dependencies,
finished by a Dense classifier.

Input:
- 12-lead ECG
- Shape: (12, 5000)
- Internally permuted to (5000, 12)

Output:
- Single binary logit
- Use BinaryCrossentropy(from_logits=True)
===========================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AdaptiveConcatPoolRNN(layers.Layer):
    def call(self, x):
        last = x[:, -1, :]
        max_pool = tf.reduce_max(x, axis=1)
        avg_pool = tf.reduce_mean(x, axis=1)
        return tf.concat([last, max_pool, avg_pool], axis=1)


def cnn_feature_block(x, filters, kernel_size=7, stride=2, dropout=0.1, name=None):
    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv"
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_relu")(x)

    if dropout > 0:
        x = layers.SpatialDropout1D(
            dropout,
            name=None if name is None else f"{name}_spdrop"
        )(x)

    return x


def build_cnn_lstm_ecg(
    input_shape=(12, 5000),
    filters=(32, 64, 128),
    kernel_size=7,
    lstm_units=128,
    cnn_dropout=0.1,
    head_dropout=0.5,
    bidirectional=False,
    name="cnn_lstm_ecg"
):
    inp = keras.Input(shape=input_shape, name="ecg_input")

    # (12, 5000) -> (5000, 12)
    x = layers.Permute((2, 1), name="to_time_channels")(inp)

    # CNN feature extraction + temporal downsampling
    for i, f in enumerate(filters):
        x = cnn_feature_block(
            x,
            filters=f,
            kernel_size=kernel_size,
            stride=2,
            dropout=cnn_dropout,
            name=f"cnn_block{i+1}"
        )

    # LSTM for temporal dependencies
    if bidirectional:
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True),
            name="bilstm"
        )(x)
    else:
        x = layers.LSTM(
            lstm_units,
            return_sequences=True,
            name="lstm"
        )(x)

    x = AdaptiveConcatPoolRNN(name="adaptive_concat_pool_rnn")(x)

    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(head_dropout, name="head_dropout")(x)

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(inp, out, name=name)


def cnn_lstm_ecg():
    return build_cnn_lstm_ecg(
        filters=(32, 64, 128),
        kernel_size=7,
        lstm_units=128,
        cnn_dropout=0.1,
        head_dropout=0.5,
        bidirectional=False,
        name="cnn_lstm_ecg"
    )


def cnn_bilstm_ecg():
    return build_cnn_lstm_ecg(
        filters=(32, 64, 128),
        kernel_size=7,
        lstm_units=128,
        cnn_dropout=0.1,
        head_dropout=0.5,
        bidirectional=True,
        name="cnn_bilstm_ecg"
    )