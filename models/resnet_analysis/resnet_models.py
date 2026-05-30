"""
===========================================================
resnet_models.py
===========================================================

ResNet-style 1D CNN architectures for ECG-based CRT prediction.

Goal:
Test whether stronger residual ECG architectures improve over
the previous simple CNN models.

Input:
- 12-lead ECG
- Shape: (12, 5000)
- Internally permuted to (5000, 12)

Output:
- Single binary logit for CRT prediction
- Use BinaryCrossentropy(from_logits=True)
===========================================================
"""

from tensorflow import keras
from tensorflow.keras import layers


def residual_block(x, filters, kernel_size=7, stride=1, dropout=0.0, name=None):
    shortcut = x

    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(x)
    x = layers.ReLU(name=None if name is None else f"{name}_relu1")(x)

    if dropout > 0:
        x = layers.SpatialDropout1D(dropout, name=None if name is None else f"{name}_spdrop")(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=None if name is None else f"{name}_shortcut_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=None if name is None else f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=None if name is None else f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=None if name is None else f"{name}_out")(x)

    return x


def build_resnet1d(
    input_shape=(12, 5000),
    filters=(32, 64, 128, 256),
    blocks_per_stage=2,
    kernel_size=7,
    dropout=0.1,
    head_dropout=0.5,
    name="resnet1d_deep"):
    inp = keras.Input(shape=input_shape, name="ecg_input")

    # Convert from (leads, time) to (time, channels)
    x = layers.Permute((2, 1), name="to_time_channels")(inp)

    # Stem
    x = layers.Conv1D(32, 15, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Residual stages
    for stage_idx, f in enumerate(filters):
        for block_idx in range(blocks_per_stage):
            stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
            x = residual_block(
                x,
                filters=f,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
                name=f"stage{stage_idx+1}_block{block_idx+1}"
            )

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(head_dropout, name="head_dropout")(x)

    # Logit output, no sigmoid
    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(inp, out, name=name)


def resnet1d_light():
    return build_resnet1d(
        filters=(32, 64, 128),
        blocks_per_stage=1,
        kernel_size=15,
        dropout=0.0,
        head_dropout=0.3,
        name="resnet1d_light")


def resnet1d_medium_k7():
    return build_resnet1d(
        filters=(32, 64, 128, 256),
        blocks_per_stage=2,
        kernel_size=7,
        dropout=0.1,
        head_dropout=0.5,
        name="resnet1d_medium_k7")


def resnet1d_medium_k15():
    return build_resnet1d(
        filters=(32, 64, 128, 256),
        blocks_per_stage=2,
        kernel_size=15,
        dropout=0.1,
        head_dropout=0.5,
        name="resnet1d_medium_k15")


def resnet1d_deep_k7():
    return build_resnet1d(
        filters=(32, 64, 128, 256),
        blocks_per_stage=3,
        kernel_size=7,
        dropout=0.1,
        head_dropout=0.5,
        name="resnet1d_deep_k7")