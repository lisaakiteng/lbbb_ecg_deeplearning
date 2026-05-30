"""
models_lstm_bidir.py

Original repo idea:
- RNN1d
- LSTM
- hidden_dim = 256
- num_layers = 2
- bidirectional = True
- AdaptiveConcatPoolRNN:
    avg pool + max pool + final forward/backward features
- classifier head
- output layer

Adapted for:
- input shape: (12, 5000)
- binary CRT prediction
- output: single logit
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AdaptiveConcatPoolRNN(layers.Layer):
    """
    Keras version of PTB-XL AdaptiveConcatPoolRNN for bidirectional LSTM.

    Input:
        x shape = (batch, time, channels)

    For bidirectional LSTM:
        channels = 2 * hidden_dim
        first half = forward direction
        second half = backward direction

    Output:
        concat(avg_pool, max_pool, last_forward + first_backward)
    """

    def __init__(self, bidirectional=True, **kwargs):
        super().__init__(**kwargs)
        self.bidirectional = bidirectional

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=1)
        max_pool = tf.reduce_max(x, axis=1)

        if self.bidirectional:
            channels = tf.shape(x)[-1]
            half = channels // 2

            forward_last = x[:, -1, :half]
            backward_first = x[:, 0, half:]

            last_features = tf.concat(
                [forward_last, backward_first],
                axis=-1,
            )
        else:
            last_features = x[:, -1, :]

        return tf.concat(
            [avg_pool, max_pool, last_features],
            axis=-1,
        )


def lstm_bidir_ecg(
    input_shape=(12, 5000),
    hidden_dim=256,
    num_layers=2,
    ps_head=0.5,
):
    inp = keras.Input(shape=input_shape, name="ecg_input")

    # Original repo expects (batch, channels, time), then transposes to (time, batch, channels).
    # In Keras we use (batch, time, channels).
    x = layers.Permute((2, 1), name="to_time_channels")(inp)

    # 2-layer bidirectional LSTM.
    # First layer must return sequences so the second LSTM receives a sequence.
    x = layers.Bidirectional(
        layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=0.0,
            recurrent_dropout=0.0,
        ),
        name="bidir_lstm_1",
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=0.0,
            recurrent_dropout=0.0,
        ),
        name="bidir_lstm_2",
    )(x)

    # AdaptiveConcatPoolRNN
    x = AdaptiveConcatPoolRNN(
        bidirectional=True,
        name="adaptive_concat_pool_rnn",
    )(x)

    # PTB-XL default head for only output layer:
    # lin_ftrs_head = [nf, num_classes]
    # ps_head = 0.5
    # bn_drop_lin applies BN + Dropout + Linear
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(ps_head, name="head_dropout")(x)

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(inp, out, name="lstm_bidir_ecg")