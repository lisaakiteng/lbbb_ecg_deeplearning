"""
===========================================================
models_cardioformer_like_fusion.py
===========================================================

Cardioformer-like fusion model for CRT prediction.

ECG branch:
- Cross-channel multi-granularity patching
- Intra-granularity attention
- Residual token refinement
- Inter-granularity attention

Tabular branch:
- AgeAtECG, GENDER, DIAGN_*, RISK_*

Fusion:
- Concatenate ECG + tabular features
- Dense classifier

Output:
- Single binary logit
===========================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.1, name=None):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=None if name is None else f"{name}_mha"
    )(x, x)

    x = layers.Add(name=None if name is None else f"{name}_attn_add")([x, attn])
    x = layers.LayerNormalization(name=None if name is None else f"{name}_attn_ln")(x)

    ff = layers.Dense(ff_dim, activation="relu", name=None if name is None else f"{name}_ff1")(x)
    ff = layers.Dropout(dropout, name=None if name is None else f"{name}_ff_drop")(ff)
    ff = layers.Dense(x.shape[-1], name=None if name is None else f"{name}_ff2")(ff)

    x = layers.Add(name=None if name is None else f"{name}_ff_add")([x, ff])
    x = layers.LayerNormalization(name=None if name is None else f"{name}_ff_ln")(x)

    return x


def residual_token_block(x, filters, kernel_size=3, dropout=0.1, name=None):
    shortcut = x

    y = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv1"
    )(x)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(y)
    y = layers.ReLU(name=None if name is None else f"{name}_relu1")(y)

    if dropout > 0:
        y = layers.SpatialDropout1D(
            dropout,
            name=None if name is None else f"{name}_spdrop"
        )(y)

    y = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv2"
    )(y)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            1,
            padding="same",
            use_bias=False,
            name=None if name is None else f"{name}_shortcut_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(
            name=None if name is None else f"{name}_shortcut_bn"
        )(shortcut)

    y = layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])
    y = layers.ReLU(name=None if name is None else f"{name}_out")(y)

    return y


def cross_channel_patch_embedding(x, patch_size, d_model=128, name=None):
    time_len = x.shape[1]

    if time_len is not None:
        pad_len = (patch_size - (time_len % patch_size)) % patch_size
    else:
        pad_len = 0

    if pad_len > 0:
        x = layers.ZeroPadding1D(
            padding=(0, pad_len),
            name=None if name is None else f"{name}_pad"
        )(x)

    x = layers.Conv1D(
        filters=d_model,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name=None if name is None else f"{name}_patch_projection"
    )(x)

    x = layers.LayerNormalization(name=None if name is None else f"{name}_ln")(x)

    return x


def granularity_branch(
    x,
    patch_size,
    d_model=128,
    num_heads=4,
    key_dim=32,
    ff_dim=256,
    transformer_depth=1,
    dropout=0.1,
    name=None
):
    tokens = cross_channel_patch_embedding(
        x,
        patch_size=patch_size,
        d_model=d_model,
        name=None if name is None else f"{name}_patch"
    )

    for i in range(transformer_depth):
        tokens = transformer_block(
            tokens,
            num_heads=num_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            name=None if name is None else f"{name}_intra_attn_{i+1}"
        )

    tokens = residual_token_block(
        tokens,
        filters=d_model,
        kernel_size=3,
        dropout=dropout,
        name=None if name is None else f"{name}_res_token"
    )

    avg_pool = layers.GlobalAveragePooling1D(
        name=None if name is None else f"{name}_avg_pool"
    )(tokens)

    max_pool = layers.GlobalMaxPooling1D(
        name=None if name is None else f"{name}_max_pool"
    )(tokens)

    summary = layers.Concatenate(
        name=None if name is None else f"{name}_summary_concat"
    )([avg_pool, max_pool])

    summary = layers.Dense(
        d_model,
        activation="relu",
        name=None if name is None else f"{name}_summary_dense"
    )(summary)

    return summary


def cardioformer_like_backbone(
    input_len=5000,
    patch_sizes=(25, 50, 100),
    d_model=128,
    num_heads=4,
    key_dim=32,
    ff_dim=256,
    transformer_depth=1,
    dropout=0.1
):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")

    x = layers.Permute((2, 1), name="to_time_channels")(ecg_in)

    branch_outputs = []

    for ps in patch_sizes:
        branch = granularity_branch(
            x,
            patch_size=ps,
            d_model=d_model,
            num_heads=num_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            transformer_depth=transformer_depth,
            dropout=dropout,
            name=f"granularity_ps{ps}"
        )
        branch_outputs.append(branch)

    g = layers.Lambda(
        lambda tensors: tf.stack(tensors, axis=1),
        name="stack_granularity_tokens"
    )(branch_outputs)

    g = transformer_block(
        g,
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=ff_dim,
        dropout=dropout,
        name="inter_granularity_attention"
    )

    avg_pool = layers.GlobalAveragePooling1D(name="inter_avg_pool")(g)
    max_pool = layers.GlobalMaxPooling1D(name="inter_max_pool")(g)

    feat = layers.Concatenate(name="cardioformer_feature")([avg_pool, max_pool])

    return keras.Model(ecg_in, feat, name="cardioformer_like_backbone")


def fusion_cardioformer_like(
    n_tab,
    input_len=5000
):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = cardioformer_like_backbone(input_len=input_len)(ecg_in)

    t = layers.Dense(64)(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    t = layers.Dense(32)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)

    x = layers.Concatenate()([ecg_feat, t])

    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1)(x)

    return keras.Model(
        [ecg_in, tab_in],
        out,
        name="fusion_cardioformer_like"
    )