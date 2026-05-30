import sys
import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIG
# =========================
BASE_DATA = "/data/leuven/376/vsc37666/lbbb_data/"

experiment = "exp4"

# =========================
# MODEL DEFINITIONS
# =========================
def cnn_simple():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    x = layers.Conv1D(32, 15, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, 15, padding="same")(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_simple")


def cnn_bn_dropout_v1():  # no strides
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_bn_dropout_v1")


def cnn_bn_dropout_v2():  # stride = 2
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_bn_dropout_v2")
    
def cnn_4layers_v1():                          # strides = 2
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_4layers_v1")
    
def cnn_4layers_v2():                           # no strides
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_4layers_v1")
    
def cnn_4layers_v3():
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128, 256, 256]:
        x = layers.Conv1D(filters, 15, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_4layers_v3")
    
def cnn_5layers_v1():  
    inp = keras.Input(shape=(12, 5000))
    x = layers.Permute((2,1))(inp)

    for filters in [32, 64, 128, 256, 512]:
        x = layers.Conv1D(filters, 15, strides = 2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inp, out, name="cnn_5layers_v1")


# =========================
# SELECT MODEL
# =========================
models = {
    "cnn_simple": cnn_simple,
    "cnn_bn_dropout_v1": cnn_bn_dropout_v1,
    "cnn_bn_dropout_v2": cnn_bn_dropout_v2,
    "cnn_4layers_v1": cnn_4layers_v1,
    "cnn_4layers_v2": cnn_4layers_v2, 
    "cnn_4layers_v3": cnn_4layers_v3, #no strides
    "cnn_5layers_v1": cnn_5layers_v1}

if len(sys.argv) < 2:
    raise ValueError("Use: python train_ptbxl.py <model_name>")

model_name = sys.argv[1]

if model_name not in models:
    raise ValueError("Unknown model name")

build_model = models[model_name]

# =========================
# LOAD METADATA
# =========================
print("Loading metadata...")
df = pd.read_csv(BASE_DATA + "ptbxl_database.csv")

def get_label(x):
    return 0 if 'NORM' in x else 1

df['label'] = df['scp_codes'].apply(get_label)

# =========================
# SPLIT USING FOLDS
# =========================
train_df = df[df['strat_fold'] <= 8]
val_df   = df[df['strat_fold'] == 9]

# Reduce size for speed
#train_df = train_df.sample(2000, random_state=42)
#val_df   = val_df.sample(500, random_state=42)

# =========================
# LOAD ECG DATA
# =========================
print("Loading training data...")
X_train, y_train = [], []

for i in range(len(train_df)):
    path = BASE_DATA + train_df['filename_hr'].iloc[i]
    signal, _ = wfdb.rdsamp(path)
    X_train.append(signal.T)
    y_train.append(train_df['label'].iloc[i])

print("Loading validation data...")
X_val, y_val = [], []

for i in range(len(val_df)):
    path = BASE_DATA + val_df['filename_hr'].iloc[i]
    signal, _ = wfdb.rdsamp(path)
    X_val.append(signal.T)
    y_val.append(val_df['label'].iloc[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

print("\n" + "="*40)
description = f"{model_name}"
print(f"\nExperiment: {experiment}")
print(f"Description: {description}")
print("="*40)

# =========================
# TRAINING LOOP (MULTI-LR)
# =========================
learning_rates = [1e-3]

results = []
best_global_auc = 0
best_model_path = f"{experiment}_{model_name}_best.h5"

for lr in learning_rates:
    print(f"\nRunning LR = {lr}...")

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")])

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max", patience=5,
        restore_best_weights=True)

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80, 
        batch_size=32,
        callbacks=[early_stop], verbose=1)

    end_time = time.time()
    elapsed = end_time - start_time

    best_val_auc = max(history.history['val_auc'])
    final_train_auc = history.history['auc'][-1]
    final_val_auc = history.history['val_auc'][-1]

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    results.append((lr, best_val_auc, final_val_auc, minutes, seconds))

    # Save best overall model
    if best_val_auc > best_global_auc:
        best_global_auc = best_val_auc
        model.save(best_model_path)

    # Save per-LR model
    model.save(f"{experiment}_{model_name}_lr{lr}.h5")

    # Save plots
    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.legend()
    plt.title(f"AUC (LR={lr})")
    plt.savefig(f"{experiment}_auc_{model_name}_lr{lr}.png")

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f"Loss (LR={lr})")
    plt.savefig(f"{experiment}_loss_{model_name}_lr{lr}.png")


# =========================
# FINAL SUMMARY
# =========================

print("\nFINAL RESULTS:")
print("-" * 60)

best_lr = None
best_auc = 0

for lr, best_auc_lr, final_auc_lr, m, s in results:
    print(f"LR: {lr} → Best Val AUC: {best_auc_lr:.4f} | Final Val AUC: {final_auc_lr:.4f} | Time: {m}m {s}s")

    if best_auc_lr > best_auc:
        best_auc = best_auc_lr
        best_lr = lr

print("-" * 60)
print(f"BEST LR: {best_lr} (AUC: {best_auc:.4f})")
print(f"Best model saved to: {best_model_path}")
print("All plots saved.")

print("\n" + "="*40)
description = f"{model_name} 80 epochs"
print(f"\nExperiment: {experiment}")
print(f"Description: {description}")
print("="*40)
