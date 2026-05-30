"""
===========================================================
05_train_cnn.py
===========================================================

This script trains and evaluates multiple 1D CNN architectures
for predicting dyssynchrony-related heart failure (CRT proxy)
from 12-lead ECG signals.

All models are trained and evaluated under identical conditions:
- Input: 12-lead ECG (shape: 12 x 5000)
- Patient-wise train/validation split (no leakage)
- Loss: Binary cross-entropy (from logits)
- Metric: ROC AUC (ECG-level and patient-level)
- Patient-level aggregation: mean probability per patient



-----------------------------------------------------------
MODEL VARIANTS OVERVIEW
-----------------------------------------------------------

1. Baseline models (cnn0_avg, cnn0_max)
   - No convolution layers
   - Only global pooling across time
   - Purpose: test whether simple signal averaging already
     contains predictive information

2. Shallow CNN (cnn2)
   - Single convolution layer
   - Learns basic local waveform patterns
   - Tests: do simple features improve over baseline?

3. Deeper CNN (cnn3)
   - Two convolution layers with downsampling (stride=2)
   - Captures hierarchical temporal patterns
   - Tests: does depth improve feature extraction?

4. Standard CNN (cnn5)
   - Three convolution blocks (32 → 64 → 128 filters)
   - Batch normalization + dropout
   - Strong baseline architecture for ECG classification

5. Pooling variants (cnn_with_pool)
   - Adds local pooling (max or average) between layers
   - Tests: effect of local temporal aggregation vs global only

6. Kernel size variants (cnn_k5, cnn_k7, cnn_k32)
   - Different receptive fields:
       small kernel → fine details
       large kernel → broader patterns
   - Tests: what temporal scale is most informative?

7. Dropout variants
   - Standard dropout vs spatial dropout
   - Tests: robustness and overfitting control

8. Residual model (resnet1d)
   - Skip connections (ResNet-style)
   - Helps training deeper networks
   - Tests: can residual learning improve performance?

-----------------------------------------------------------
HYPERPARAMETER SEARCH
-----------------------------------------------------------

Selected models are trained with multiple learning rates:
[1e-2, 1e-3, 3e-4, 1e-4]

-----------------------------------------------------------
OUTPUTS
-----------------------------------------------------------

- cnn_results.csv → summary of all experiments
- ROC curves per model
- Training/validation AUC curves

-----------------------------------------------------------
NOTES
-----------------------------------------------------------

- All results are evaluated at patient level to reflect
  clinical decision-making
- ECG-level AUC is also reported for reference
- Differences in performance reflect architecture design,
  not data leakage or split differences

===========================================================
"""


from tensorflow import keras
from tensorflow.keras import layers


# =========================
# BASELINE MODELS
# =========================

def cnn0_avg():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="cnn0_avg")


def cnn0_max():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)
    x = layers.GlobalMaxPooling1D()(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="cnn0_max")


# =========================
# SIMPLE CNN
# =========================

def cnn2_avg():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    x = layers.Conv1D(16, 15, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)

    return keras.Model(inp, out, name="cnn2_avg")


def cnn2_max():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    x = layers.Conv1D(16, 15, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.GlobalMaxPooling1D()(x)
    out = layers.Dense(1)(x)

    return keras.Model(inp, out, name="cnn2_max")


# =========================
# DEEPER CNN
# =========================

def cnn3_avg():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    x = layers.Conv1D(32, 15, strides=2, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, 15, strides=2, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)

    return keras.Model(inp, out, name="cnn3_avg")


def cnn3_max():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    x = layers.Conv1D(32, 15, strides=2, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, 15, strides=2, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.GlobalMaxPooling1D()(x)
    out = layers.Dense(1)(x)

    return keras.Model(inp, out, name="cnn3_max")


# =========================
# BEST BASELINE (cnn 4)
# =========================

def cnn4_avg():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="cnn5_avg")


def cnn4_max():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="cnn4_max")


# =========================
# POOLING VARIANTS
# =========================

def cnn_with_pool(pool="max"):
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        if pool == "max":
            x = layers.MaxPooling1D(2)(x)
        else:
            x = layers.AveragePooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name=f"cnn_pool_{pool}")


# =========================
# KERNEL VARIANTS
# =========================

def cnn_k5():
    return cnn_kernel(5)

def cnn_k7():
    return cnn_kernel(7)

def cnn_k32():
    return cnn_kernel(32)


def cnn_kernel(k):
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, k, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name=f"cnn_k{k}")


# =========================
# DROPOUT VARIANTS
# =========================

def cnn_dropout_std():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="cnn_dropout_std")


def cnn_dropout_spatial():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SpatialDropout1D(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="cnn_dropout_spatial")


# =========================
# RESNET
# =========================

def res_block(x, filters, k=15, stride=1):
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


def resnet1d():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    x = layers.Conv1D(32, 15, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = res_block(x, 32)
    x = res_block(x, 64, stride=2)
    x = res_block(x, 128, stride=2)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="resnet1d")