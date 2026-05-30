"""
models_resnet_fusion.py

ResNet + tabular fusion model for CRT prediction.
Uses same input/data setup as previous fusion experiments.
Output is a single logit.
"""

from tensorflow import keras
from tensorflow.keras import layers


def res_block(x, filters, k=7, stride=1, dropout=0.1):
    shortcut = x

    x = layers.Conv1D(filters, k, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if dropout > 0:
        x = layers.SpatialDropout1D(dropout)(x)

    x = layers.Conv1D(filters, k, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def resnet_medium_backbone(k=7, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    x = layers.Permute((2, 1))(ecg_in)

    x = layers.Conv1D(32, 15, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for i, filters in enumerate([32, 64, 128, 256]):
        stride = 1 if i == 0 else 2
        x = res_block(x, filters, k=k, stride=stride, dropout=0.1)
        x = res_block(x, filters, k=k, stride=1, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(ecg_in, x, name="resnet_medium_backbone")


def fusion_resnet_medium(n_tab, k=7, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = resnet_medium_backbone(k=k, input_len=input_len)(ecg_in)

    t = layers.Dense(64)(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    t = layers.Dense(32)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)

    x = layers.Concatenate()([ecg_feat, t])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1)(x)
    return keras.Model([ecg_in, tab_in], out, name="fusion_resnet_medium")