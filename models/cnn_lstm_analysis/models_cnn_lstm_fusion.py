"""
===========================================================
models_cnn_lstm_fusion.py
===========================================================

CNN-LSTM fusion model:
- ECG CNN-LSTM backbone
- Tabular tower
- Fusion classifier
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


def cnn_block(x, filters, kernel_size=7, stride=2, dropout=0.1):

    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if dropout > 0:
        x = layers.SpatialDropout1D(dropout)(x)

    return x


def cnn_lstm_backbone(
    input_len=5000,
    filters=(32, 64, 128),
    kernel_size=7,
    lstm_units=128
):

    ecg_in = keras.Input(shape=(12, input_len), name="ecg")

    x = layers.Permute((2, 1))(ecg_in)

    for f in filters:

        x = cnn_block(
            x,
            filters=f,
            kernel_size=kernel_size,
            stride=2,
            dropout=0.1
        )

    x = layers.LSTM(
        lstm_units,
        return_sequences=True
    )(x)

    x = AdaptiveConcatPoolRNN()(x)

    return keras.Model(ecg_in, x, name="cnn_lstm_backbone")


def fusion_cnn_lstm(
    n_tab,
    input_len=5000
):

    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = cnn_lstm_backbone(input_len=input_len)(ecg_in)

    # tabular tower
    t = layers.Dense(64)(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    t = layers.Dense(32)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)

    # fusion
    x = layers.Concatenate()([ecg_feat, t])

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1)(x)

    return keras.Model(
        [ecg_in, tab_in],
        out,
        name="fusion_cnn_lstm"
    )