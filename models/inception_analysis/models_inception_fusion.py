"""
models_inception_fusion.py

Inception-style ECG + tabular fusion model for CRT prediction.
Output is a single logit.
"""

from tensorflow import keras
from tensorflow.keras import layers


def inception_block(x, filters=32, bottleneck=32, name="inception"):
    shortcut = x

    if bottleneck > 0:
        x = layers.Conv1D(bottleneck, 1, padding="same", use_bias=False, name=f"{name}_bottleneck_conv")(x)
        x = layers.BatchNormalization(name=f"{name}_bottleneck_bn")(x)
        x = layers.ReLU(name=f"{name}_bottleneck_relu")(x)

    b1 = layers.Conv1D(filters, 9, padding="same", use_bias=False, name=f"{name}_conv9")(x)
    b2 = layers.Conv1D(filters, 19, padding="same", use_bias=False, name=f"{name}_conv19")(x)
    b3 = layers.Conv1D(filters, 39, padding="same", use_bias=False, name=f"{name}_conv39")(x)

    b4 = layers.MaxPooling1D(pool_size=3, strides=1, padding="same", name=f"{name}_maxpool")(x)
    b4 = layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_pool_conv")(b4)

    x = layers.Concatenate(name=f"{name}_concat")([b1, b2, b3, b4])
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv1D(int(x.shape[-1]), 1, padding="same", use_bias=False, name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out")(x)

    return x


def inception_light_backbone(input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")

    x = layers.Permute((2, 1), name="to_time_channels")(ecg_in)

    x = layers.Conv1D(32, 15, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = inception_block(x, filters=32, bottleneck=32, name="inception_block1")
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = inception_block(x, filters=32, bottleneck=32, name="inception_block2")
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = inception_block(x, filters=64, bottleneck=32, name="inception_block3")
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    return keras.Model(ecg_in, x, name="inception_light_backbone")


def fusion_inception_light(n_tab, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = inception_light_backbone(input_len=input_len)(ecg_in)

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

    return keras.Model([ecg_in, tab_in], out, name="fusion_inception_light")