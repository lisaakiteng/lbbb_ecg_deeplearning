"""
models_lstm.py

Keras adaptation of PTB-XL benchmarking LSTM.

Original idea:
- LSTM
- hidden_dim = 256
- num_layers = 2
- bidirectional = False
- AdaptiveConcatPoolRNN
- classifier head
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

        return tf.concat(
            [avg_pool, max_pool, last_features],
            axis=-1,
        )


def lstm_ecg(
    input_shape=(12, 5000),
    hidden_dim=256,
    ps_head=0.5,
):
    inp = keras.Input(shape=input_shape, name="ecg_input")

    # (12, 5000) -> (5000, 12)
    x = layers.Permute((2, 1), name="to_time_channels")(inp)

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

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(inp, out, name="lstm_ecg")