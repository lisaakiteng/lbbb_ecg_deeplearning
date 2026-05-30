import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import pack_leads, make_ds


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"
PRETRAINED_MODEL_PATH = "/user/leuven/376/vsc37666/lbbb_analysis/cnn_analysis/exp2_cnn_4layers_v1_best.h5"

EXPERIMENT = "tl_step1_frozen"
BATCH_SIZE = 32


print("\n===== CONFIG =====")
print("Experiment:", EXPERIMENT)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================
print("\nLoading pretrained PTB-XL model...")
pretrained = keras.models.load_model(PRETRAINED_MODEL_PATH)

# Remove classifier and keep backbone
backbone = keras.Model(
    inputs=pretrained.input,
    outputs=pretrained.layers[-2].output,
    name="ptbxl_backbone")

backbone.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================
ecg_in = keras.Input(shape=(12, 5000), name="ecg")

x = backbone(ecg_in)
x = layers.Dense(64, activation="relu", name="crt_dense")(x)
x = layers.Dropout(0.5, name="crt_dropout")(x)
out = layers.Dense(1, activation="sigmoid", name="crt_output")(x)

model = keras.Model(ecg_in, out, name="crt_frozen_model")

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")])

model.summary()


# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")

train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df   = pd.read_parquet(DATA_DIR + "val.parquet")

lead_cols = [c for c in train_df.columns if c.lower().startswith("lead_")][:12]

train_packed, y_train, pat_train = pack_leads(train_df, lead_cols)
val_packed, y_val, pat_val       = pack_leads(val_df, lead_cols)

train_ds = make_ds(train_packed, y_train, batch_size=BATCH_SIZE, training=True)
val_ds   = make_ds(val_packed, y_val, batch_size=BATCH_SIZE, training=False)


# ============================================================
# MODEL CHECKS
# ============================================================
print("\n===== Model checks =====")
for ecg_batch, y_batch in train_ds.take(1):
    print("ECG batch:", ecg_batch.shape)
    print("Labels:", y_batch.shape)
    print("Backbone output:", backbone(ecg_batch).shape)
    print("Model output:", model(ecg_batch).shape)


# ============================================================
# TRAIN
# ============================================================
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
    ModelCheckpoint(f"{EXPERIMENT}_best.keras", monitor="val_auc", 
    mode="max", save_best_only=True)]

print("\nTraining frozen model...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks)

print("Best Val AUC:", max(history.history["val_auc"]))

model.save(f"{EXPERIMENT}_final.keras")


# ============================================================
# PLOTS
# ============================================================
plt.figure()
plt.plot(history.history["auc"], label="train_auc")
plt.plot(history.history["val_auc"], label="val_auc")
plt.legend()
plt.title(EXPERIMENT + " AUC")
plt.savefig(f"{EXPERIMENT}_auc.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title(EXPERIMENT + " Loss")
plt.savefig(f"{EXPERIMENT}_loss.png")
plt.close()

print("Plots saved.")