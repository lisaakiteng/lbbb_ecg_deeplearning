"""
models_fusion.py

Multimodal ECG + tabular models for CRT prediction.

This module contains:
- CNN backbone for ECG feature extraction
- Residual (ResNet-style) ECG backbone
- Separable convolution backbone
- Fusion architectures combining ECG + tabular features

All models output logits (no sigmoid).
"""

from tensorflow import keras
from tensorflow.keras import layers


# =========================================
# BASIC CNN BACKBONE (ECG only)
# =========================================
def cnn_backbone(k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    x = layers.Permute((2, 1))(ecg_in)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, k, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(ecg_in, x, name="cnn_backbone")


# =========================================
# RESNET BACKBONE
# =========================================
def res_block(x, filters, k=7, stride=1):
    shortcut = x

    x = layers.Conv1D(filters, k, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters, k, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def resnet_backbone(k=7, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    x = layers.Permute((2, 1))(ecg_in)

    x = layers.Conv1D(32, k, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = res_block(x, 32, k=k)
    x = res_block(x, 64, k=k, stride=2)
    x = res_block(x, 128, k=k, stride=2)

    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(ecg_in, x, name="resnet_backbone")
    
# =========================================
# FUSION WITHOUT RESIDUALS (BEST MODEL) this is the final model considered
# =========================================

def conv_block(x, filters, k=15, stride=1):
    x = layers.Conv1D(filters, k, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def ecg_backbone_no_residual(k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    x = layers.Permute((2, 1))(ecg_in)

    for filters in [32, 32, 64, 64, 128, 128]:
        stride = 2 if filters in [32, 64, 128] else 1
        x = conv_block(x, filters, k=k, stride=stride)

    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(ecg_in, x, name="ecg_no_residual")


def fusion_no_residual(n_tab, k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = ecg_backbone_no_residual(k=k, input_len=input_len)(ecg_in)

    t = layers.Dense(64)(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    t = layers.Dense(32)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    x = layers.Concatenate()([ecg_feat, t])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1)(x)
    return keras.Model([ecg_in, tab_in], out, name="fusion_no_residual")

# =========================================
# IMPROVED FUSION MODEL (V2)
# - Kernel diversity
# - Balanced fusion
# - Stronger interaction layers
# =========================================

def ecg_backbone_no_residual_v2(input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    x = layers.Permute((2, 1))(ecg_in)

    filters_list = [32, 32, 64, 64, 128, 128]
    kernels_list = [15, 15, 11, 11, 7, 7]

    for filters, k in zip(filters_list, kernels_list):
        stride = 2 if filters in [32, 64, 128] else 1
        x = conv_block(x, filters, k=k, stride=stride)

    x = layers.GlobalAveragePooling1D()(x)

    return keras.Model(ecg_in, x, name="ecg_no_residual_v2")


def fusion_no_residual_v2(n_tab, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    # ECG branch (improved backbone)
    ecg_feat = ecg_backbone_no_residual_v2(input_len=input_len)(ecg_in)
    ecg_feat = layers.Dense(128, activation="relu")(ecg_feat)

    # Tabular branch
    t = layers.Dense(64)(tab_in)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    t = layers.Dense(32)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Dropout(0.3)(t)

    # Balance feature scale
    t = layers.Dense(128, activation="relu")(t)

    # Fusion
    x = layers.Concatenate()([ecg_feat, t])

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1)(x)

    return keras.Model([ecg_in, tab_in], out, name="fusion_no_residual_v2")

# =========================================
# SEPARABLE CNN BACKBONE
# =========================================
def cnn_backbone_sep(k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    x = layers.Permute((2, 1))(ecg_in)

    for filters in [32, 64, 128]:
        x = layers.SeparableConv1D(filters, k, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(ecg_in, x, name="cnn_backbone_sep")


# =========================================
# FUSION MODEL (BASELINE)
# =========================================
def fusion_cnn(n_tab, k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = cnn_backbone(k=k, input_len=input_len)(ecg_in)

    t = layers.Dense(64, activation="relu")(tab_in)
    t = layers.Dense(32, activation="relu")(t)

    x = layers.Concatenate()([ecg_feat, t])
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model([ecg_in, tab_in], out, name="fusion_cnn")


# =========================================
# FUSION (IMPROVED TABULAR BRANCH)
# =========================================
def fusion_cnn_improved(n_tab, k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = cnn_backbone(k=k, input_len=input_len)(ecg_in)

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
    return keras.Model([ecg_in, tab_in], out, name="fusion_cnn_improved")


# =========================================
# FUSION WITH RESNET BACKBONE
# =========================================
def fusion_resnet(n_tab, k=7, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = resnet_backbone(k=k, input_len=input_len)(ecg_in)

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
    return keras.Model([ecg_in, tab_in], out, name="fusion_resnet")


# =========================================
# FUSION WITH SEPARABLE CNN
# =========================================
def fusion_sep(n_tab, k=15, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = cnn_backbone_sep(k=k, input_len=input_len)(ecg_in)

    t = layers.Dense(64, activation="relu")(tab_in)
    t = layers.Dense(32, activation="relu")(t)

    x = layers.Concatenate()([ecg_feat, t])
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model([ecg_in, tab_in], out, name="fusion_sep")