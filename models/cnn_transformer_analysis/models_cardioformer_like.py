"""
===========================================================
models_cardioformer.py
===========================================================

Based on the Cardioformer architecture idea:
ECG -> Cross-channel multi-granularity patch embeddings
    -> Intra-granularity attention
    -> Inter-granularity attention via router tokens
    -> ResNet-style feed-forward refinement
    -> Flatten
    -> Dense binary CRT logit

Input:
    ECG shape: (12, 5000)

Output:
    Single binary logit
===========================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# Replication padding along time axis
# =========================================================

class ReplicationPadTime(layers.Layer):
    def __init__(self, pad_right, **kwargs):
        super().__init__(**kwargs)
        self.pad_right = pad_right

    def call(self, x):
        # x: (B, C, L)
        if self.pad_right == 0:
            return x

        last = x[:, :, -1:]
        pad = tf.repeat(last, repeats=self.pad_right, axis=2)
        return tf.concat([x, pad], axis=2)


# =========================================================
# Sinusoidal positional embedding
# =========================================================

class SinusoidalPositionalEmbedding(layers.Layer):
    def __init__(self, d_model, max_len=20000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / d_model)
        )

        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)

        pe = tf.concat([pe_sin, pe_cos], axis=-1)
        pe = pe[:, :d_model]

        self.pe = tf.expand_dims(pe, axis=0)

    def call(self, x):
        length = tf.shape(x)[1]
        return self.pe[:, :length, :]


# =========================================================
# Cross-channel patch embedding
# =========================================================

class CrossChannelTokenEmbedding(layers.Layer):
    def __init__(self, enc_in, patch_len, d_model, stride=None, name=None):
        super().__init__(name=name)

        if stride is None:
            stride = patch_len

        self.enc_in = enc_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.stride = stride

        self.conv = layers.Conv2D(
            filters=d_model,
            kernel_size=(enc_in, patch_len),
            strides=(1, stride),
            padding="valid",
            use_bias=False,
            name="token_conv2d"
        )

    def call(self, x):
        # x: (B, C, L)
        x = tf.expand_dims(x, axis=-1)      # (B, C, L, 1)
        x = self.conv(x)                    # (B, 1, patch_num, d_model)
        x = tf.squeeze(x, axis=1)           # (B, patch_num, d_model)
        return x


# =========================================================
# Multi-granularity patch embedding
# =========================================================

class ListPatchEmbedding(layers.Layer):
    def __init__(
        self,
        enc_in,
        d_model,
        patch_len_list,
        stride_list,
        dropout=0.1,
        name="list_patch_embedding"
    ):
        super().__init__(name=name)

        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list

        self.paddings = [
            ReplicationPadTime(stride, name=f"rep_pad_{patch_len}")
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        self.value_embeddings = [
            CrossChannelTokenEmbedding(
                enc_in=enc_in,
                patch_len=patch_len,
                d_model=d_model,
                stride=stride,
                name=f"cross_channel_patch_{patch_len}"
            )
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        self.position_embedding = SinusoidalPositionalEmbedding(d_model=d_model)
        self.dropout = layers.Dropout(dropout)

        self.context_vectors = []

    def build(self, input_shape):
        for i, _ in enumerate(self.patch_len_list):
            self.context_vectors.append(
                self.add_weight(
                    name=f"learnable_context_{i}",
                    shape=(1, 1, self.d_model),
                    initializer="random_normal",
                    trainable=True
                )
            )

    def call(self, x, training=False):
        # input x: (B, L, C)
        # convert to (B, C, L)
        x = tf.transpose(x, perm=[0, 2, 1])

        x_list = []

        for i, (padding, embedding) in enumerate(
            zip(self.paddings, self.value_embeddings)
        ):
            x_new = padding(x)
            x_new = embedding(x_new)

            pos = self.position_embedding(x_new)
            ctx = self.context_vectors[i]

            x_new = x_new + pos + ctx
            x_new = self.dropout(x_new, training=training)

            x_list.append(x_new)

        return x_list


# =========================================================
# Attention layer
# =========================================================

class AttentionLayer(layers.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, name=None):
        super().__init__(name=name)

        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
            output_shape=d_model,
            name="mha"
        )

    def call(self, queries, keys, values, training=False):
        return self.mha(
            query=queries,
            key=keys,
            value=values,
            training=training
        )


# =========================================================
# Cardioformer attention layer
# intra-granularity + inter-granularity attention
# =========================================================

class CardioformerLayer(layers.Layer):
    def __init__(
        self,
        num_blocks,
        d_model,
        n_heads,
        dropout=0.1,
        no_inter=False,
        name="cardioformer_layer"
    ):
        super().__init__(name=name)

        self.num_blocks = num_blocks
        self.no_inter = no_inter

        self.intra_attentions = [
            AttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                name=f"intra_attention_{i}"
            )
            for i in range(num_blocks)
        ]

        if no_inter or num_blocks <= 1:
            self.inter_attention = None
        else:
            self.inter_attention = AttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                name="inter_attention"
            )

    def call(self, x_list, training=False):
        x_intra = []

        # Intra-granularity attention
        for x, attn in zip(x_list, self.intra_attentions):
            x_out = attn(x, x, x, training=training)
            x_intra.append(x_out)

        # Inter-granularity attention using router tokens
        if self.inter_attention is not None:
            routers = tf.concat(
                [x[:, -1:, :] for x in x_intra],
                axis=1
            )

            x_inter = self.inter_attention(
                routers,
                routers,
                routers,
                training=training
            )

            x_out = []

            for i, x in enumerate(x_intra):
                x_without_last = x[:, :-1, :]
                router_i = x_inter[:, i:i + 1, :]
                x_new = tf.concat([x_without_last, router_i], axis=1)
                x_out.append(x_new)

            return x_out

        return x_intra


# =========================================================
# ResNet-style feed-forward block
# =========================================================

class ResNetBlockType1(layers.Layer):
    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1,
        activation="gelu",
        identity=True,
        name=None
    ):
        super().__init__(name=name)

        self.identity = identity

        self.conv1 = layers.Conv1D(
            filters=d_ff,
            kernel_size=1,
            padding="same",
            name="conv1"
        )

        self.conv2 = layers.Conv1D(
            filters=d_model,
            kernel_size=1,
            padding="same",
            name="conv2"
        )

        if not identity:
            self.conv3 = layers.Conv1D(
                filters=d_model,
                kernel_size=1,
                padding="same",
                name="conv3"
            )
        else:
            self.conv3 = None

        if activation == "gelu":
            self.activation = tf.nn.gelu
        else:
            self.activation = tf.nn.relu

        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization()

    def call(self, x, training=False):
        residual = x

        y = self.conv1(x)
        y = self.activation(y)
        y = self.dropout(y, training=training)

        y = self.conv2(y)
        y = self.dropout(y, training=training)

        if self.conv3 is not None:
            residual = self.conv3(residual)
            residual = self.dropout(residual, training=training)

        return self.norm(residual + y)


# =========================================================
# Encoder layer
# =========================================================

class EncoderLayer(layers.Layer):
    def __init__(
        self,
        num_blocks,
        d_model,
        d_ff,
        n_heads,
        dropout=0.1,
        activation="gelu",
        no_inter=False,
        name="encoder_layer"
    ):
        super().__init__(name=name)

        self.attention = CardioformerLayer(
            num_blocks=num_blocks,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            no_inter=no_inter,
            name="cardioformer_attention"
        )

        self.resblock1 = ResNetBlockType1(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            identity=True,
            name="resblock1"
        )

        self.resblock2 = ResNetBlockType1(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            identity=True,
            name="resblock2"
        )

        self.resblock3 = ResNetBlockType1(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            identity=True,
            name="resblock3"
        )

        self.norm1 = layers.LayerNormalization(name="norm1")
        self.norm2 = layers.LayerNormalization(name="norm2")
        self.dropout = layers.Dropout(dropout)

    def call(self, x_list, training=False):
        new_x = self.attention(x_list, training=training)

        x = [
            self.norm1(old + self.dropout(new, training=training))
            for old, new in zip(x_list, new_x)
        ]

        y = x
        y = [self.resblock1(_y, training=training) for _y in y]
        y = [self.resblock2(_y, training=training) for _y in y]
        y = [self.resblock3(_y, training=training) for _y in y]

        out = [
            self.norm2(_x + _y)
            for _x, _y in zip(x, y)
        ]

        return out


# =========================================================
# Encoder
# =========================================================

class Encoder(layers.Layer):
    def __init__(
        self,
        e_layers,
        num_blocks,
        d_model,
        d_ff,
        n_heads,
        dropout=0.1,
        activation="gelu",
        no_inter=False,
        name="encoder"
    ):
        super().__init__(name=name)

        self.layers_ = [
            EncoderLayer(
                num_blocks=num_blocks,
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                activation=activation,
                no_inter=no_inter,
                name=f"encoder_layer_{i}"
            )
            for i in range(e_layers)
        ]

        self.norm = layers.LayerNormalization(name="encoder_norm")

    def call(self, x_list, training=False):
        x = x_list

        for layer in self.layers_:
            x = layer(x, training=training)

        x = tf.concat(x, axis=1)
        x = self.norm(x)

        return x


# =========================================================
# Full model
# =========================================================

def build_cardioformer_like_ecg(
    input_shape=(12, 5000),
    enc_in=12,
    patch_len_list=(25, 50, 100),
    d_model=128,
    n_heads=4,
    d_ff=256,
    e_layers=2,
    dropout=0.1,
    activation="gelu",
    no_inter=False,
    name="cardioformer_like_ecg"
):
    inp = keras.Input(shape=input_shape, name="ecg_input")

    # Your ECG: (B, 12, 5000)
    # Cardioformer-style input: (B, 5000, 12)
    x = layers.Permute((2, 1), name="to_time_channels")(inp)

    stride_list = patch_len_list

    x_list = ListPatchEmbedding(
        enc_in=enc_in,
        d_model=d_model,
        patch_len_list=patch_len_list,
        stride_list=stride_list,
        dropout=dropout,
        name="enc_embedding"
    )(x)

    enc_out = Encoder(
        e_layers=e_layers,
        num_blocks=len(patch_len_list),
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        activation=activation,
        no_inter=no_inter,
        name="encoder"
    )(x_list)

    x = layers.Activation(tf.nn.gelu, name="classification_gelu")(enc_out)
    x = layers.Dropout(dropout, name="classification_dropout")(x)
    x = layers.Flatten(name="flatten_tokens")(x)

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(inp, out, name=name)


def cardioformer_like_ecg():
    return build_cardioformer_like_ecg(
        input_shape=(12, 5000),
        enc_in=12,
        patch_len_list=(25, 50, 100),
        d_model=128,
        n_heads=4,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
        activation="gelu",
        no_inter=False,
        name="cardioformer_like_ecg"
    )