import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import pack_leads, make_ds


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"
FROZEN_MODEL_PATH = "tl_step1_frozen_best.keras"

EXPERIMENT = "tl_step2_finetune_last4"
BATCH_SIZE = 32


print("\n===== CONFIG =====")
print("Experiment:", EXPERIMENT)
print("Unfreezing: last 4 layers (non-BN)")


# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading frozen model...")
model = keras.models.load_model(FROZEN_MODEL_PATH)

backbone = model.get_layer("ptbxl_backbone")


# ============================================================
# PARTIAL FINE-TUNING
# ============================================================
backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[-8:]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True

print("\nTrainable layers:")
for l in backbone.layers:
    if l.trainable:
        print("  ", l.name)


model.compile(
    optimizer=keras.optimizers.Adam(1e-6),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")])


# ============================================================
# LOAD DATA
# ============================================================
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
print("\n===== MODEL SUMMARY =====")
model.summary()

print("\n===== BACKBONE SUMMARY =====")
backbone.summary()

print("\n===== TRAINABLE STATUS =====")
for layer in backbone.layers:
    status = "trainable" if layer.trainable else "frozen"
    print(f"{layer.name:30s} -> {status}")

print("\n===== PARAMETER COUNT =====")
print("Total params:", model.count_params())
print("Trainable params:", np.sum([np.prod(v.shape) for v in model.trainable_weights]))


# ============================================================
# TRAIN
# ============================================================
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
    ModelCheckpoint(f"{EXPERIMENT}_best.keras", monitor="val_auc", 
    mode="max", save_best_only=True)]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks)

print("Best Val AUC:", max(history.history["val_auc"]))


# ============================================================
# ROC
# ============================================================
probs = model.predict(val_ds).ravel()

auc = roc_auc_score(y_val[:len(probs)], probs)
fpr, tpr, _ = roc_curve(y_val[:len(probs)], probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title(EXPERIMENT + " ROC")
plt.savefig(f"{EXPERIMENT}_roc.png")
plt.close()

print("ROC saved.")