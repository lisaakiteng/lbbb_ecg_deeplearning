"""
models_lstm_fusion.py

ECG branch:
- 2-layer LSTM
- hidden_dim=256
- AdaptiveConcatPoolRNN

Fusion:
- ECG embedding + tabular tower
- single binary CRT logit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AdaptiveConcatPoolRNN(layers.Layer):
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=1)
        max_pool = tf.reduce_max(x, axis=1)
        last_features = x[:, -1, :]

        return tf.concat([avg_pool, max_pool, last_features], axis=-1)


def lstm_backbone(input_len=5000, hidden_dim=256, ps_head=0.5):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")

    x = layers.Permute((2, 1), name="to_time_channels")(ecg_in)

    x = layers.LSTM(
        hidden_dim,
        return_sequences=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        name="lstm_1",
    )(x)

    x = layers.LSTM(
        hidden_dim,
        return_sequences=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        name="lstm_2",
    )(x)

    x = AdaptiveConcatPoolRNN(name="adaptive_concat_pool_rnn")(x)

    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(ps_head, name="head_dropout")(x)

    return keras.Model(ecg_in, x, name="lstm_backbone")


def fusion_lstm(n_tab, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = lstm_backbone(input_len=input_len)(ecg_in)

    t = layers.Dense(64, name="tab_dense1")(tab_in)
    t = layers.BatchNormalization(name="tab_bn1")(t)
    t = layers.ReLU(name="tab_relu1")(t)
    t = layers.Dropout(0.3, name="tab_dropout1")(t)

    t = layers.Dense(32, name="tab_dense2")(t)
    t = layers.BatchNormalization(name="tab_bn2")(t)
    t = layers.ReLU(name="tab_relu2")(t)

    x = layers.Concatenate(name="fusion_concat")([ecg_feat, t])

    x = layers.Dense(64, activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(0.3, name="fusion_dropout1")(x)

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model([ecg_in, tab_in], out, name="fusion_lstm")