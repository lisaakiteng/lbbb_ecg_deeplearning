# =========================================
# train_cnn_lstm_debug.py
# =========================================

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from dataset import pack_leads, make_ds
from models_cnn_lstm import cnn_lstm_ecg

from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# =========================================
# PATHS
# =========================================

DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"

OUTPUT_DIR = "/data/leuven/376/vsc37666/lbbb_outputs/outputs_cnn_lstm_debug"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

for d in [
    MODEL_DIR,
    PLOT_DIR,
    PRED_DIR,
    RESULT_DIR,
    SUMMARY_DIR,
    DEBUG_DIR,
]:
    os.makedirs(d, exist_ok=True)


# =========================================
# EXPERIMENT SETTINGS
# =========================================

EXPERIMENT_NAME = "cnn_lstm_debug"

model_name = "cnn_lstm_ecg"
model_fn = cnn_lstm_ecg

lr = 1e-3
epochs = 60
batch_size = 32
seed = 42

tf.keras.utils.set_random_seed(seed)

print("======================================")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("Model:", model_name)
print("Learning rate:", lr)
print("Epochs:", epochs)
print("Batch size:", batch_size)
print("Seed:", seed)
print("======================================")


# =========================================
# LOAD DATA
# =========================================

print("\n======================================")
print("Loading data")
print("======================================")

train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df = pd.read_parquet(DATA_DIR + "val.parquet")
test_df = pd.read_parquet(DATA_DIR + "test.parquet")

print("\nData shapes:")
print("train:", train_df.shape)
print("val:  ", val_df.shape)
print("test: ", test_df.shape)

print("\nECG-level label counts:")

for split_name, df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    print(f"\n{split_name}:")
    print(df["y"].value_counts())

print("\nPatient-level label counts:")

for split_name, df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    print(f"\n{split_name}:")
    print(df.groupby("ID_STUDY_PART")["y"].max().value_counts())


# =========================================
# ECG LEADS
# =========================================

lead_cols = [
    c for c in train_df.columns
    if c.lower().startswith("lead_")
]

lead_cols = lead_cols[:12]

print("\nLead columns:")

for i, c in enumerate(lead_cols):
    print(i, c)


# =========================================
# PACK ECGs
# =========================================

print("\n======================================")
print("Packing ECGs")
print("======================================")

train_packed, y_train, pat_train = pack_leads(train_df, lead_cols)
val_packed, y_val, pat_val = pack_leads(val_df, lead_cols)
test_packed, y_test, pat_test = pack_leads(test_df, lead_cols)

print("\nPacked lengths:")
print("train:", len(train_packed))
print("val:  ", len(val_packed))
print("test: ", len(test_packed))

print("\nLabel arrays:")
print(
    "y_train:",
    y_train.shape,
    "pos:",
    int(np.sum(y_train)),
    "neg:",
    int(len(y_train) - np.sum(y_train)),
)

print(
    "y_val:",
    y_val.shape,
    "pos:",
    int(np.sum(y_val)),
    "neg:",
    int(len(y_val) - np.sum(y_val)),
)

print(
    "y_test:",
    y_test.shape,
    "pos:",
    int(np.sum(y_test)),
    "neg:",
    int(len(y_test) - np.sum(y_test)),
)


# =========================================
# DATASETS
# =========================================

print("\n======================================")
print("Creating tf.data datasets")
print("======================================")

train_ds = make_ds(
    train_packed,
    y_train,
    batch_size=batch_size,
    training=True,
)

val_ds = make_ds(
    val_packed,
    y_val,
    batch_size=batch_size,
)

test_ds = make_ds(
    test_packed,
    y_test,
    batch_size=batch_size,
)

for batch_x, batch_y in train_ds.take(1):

    print("\nOne training batch:")
    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)

    print("batch_x dtype:", batch_x.dtype)
    print("batch_y dtype:", batch_y.dtype)

    print("batch_y first 10:", batch_y.numpy().flatten()[:10])

    print("batch_x min:", float(tf.reduce_min(batch_x).numpy()))
    print("batch_x max:", float(tf.reduce_max(batch_x).numpy()))
    print("batch_x mean:", float(tf.reduce_mean(batch_x).numpy()))
    print("batch_x std:", float(tf.math.reduce_std(batch_x).numpy()))


# =========================================
# HELPERS
# =========================================

def get_probs(model, ds):

    y_true, y_pred = [], []

    for ecg, y in ds:

        logits = model(ecg, training=False)
        probs = tf.math.sigmoid(logits).numpy().flatten()

        y_true.extend(y.numpy().flatten())
        y_pred.extend(probs)

    return np.array(y_true), np.array(y_pred)


def patient_predictions(pat_ids, y_true, y_pred, agg="mean"):

    df = pd.DataFrame({
        "pat": pat_ids,
        "y": y_true,
        "pred": y_pred
    })

    if agg == "mean":

        return df.groupby("pat").agg(
            y=("y", "max"),
            pred=("pred", "mean")
        ).reset_index()

    elif agg == "max":

        return df.groupby("pat").agg(
            y=("y", "max"),
            pred=("pred", "max")
        ).reset_index()

    else:
        raise ValueError("agg must be mean or max")


def save_training_curves(history, run_name):

    plt.figure()
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.legend()
    plt.title(f"{run_name} AUC Curve")

    plt.savefig(
        os.path.join(PLOT_DIR, f"{run_name}_auc_curve.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title(f"{run_name} Loss Curve")

    plt.savefig(
        os.path.join(PLOT_DIR, f"{run_name}_loss_curve.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def save_roc_curve(dfp, auc_value, run_name, split):

    fpr, tpr, _ = roc_curve(dfp["y"], dfp["pred"])

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.title(f"{split.upper()} ROC - {run_name}")

    plt.legend()

    plt.savefig(
        os.path.join(PLOT_DIR, f"{run_name}_{split}_roc.png"),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def save_layer_shapes(model, run_name):

    path = os.path.join(
        DEBUG_DIR,
        f"{run_name}_layer_shapes.txt"
    )

    print("\nLayer shapes:")

    with open(path, "w") as f:

        for i, layer in enumerate(model.layers):

            try:
                out_shape = layer.output.shape

            except Exception:
                out_shape = "unknown"

            line = (
                f"{i:03d} | "
                f"{layer.name:35s} | "
                f"{layer.__class__.__name__:25s} | "
                f"output: {out_shape}"
            )

            print(line)
            f.write(line + "\n")


def forward_debug(model, ds, run_name):

    for bx, by in ds.take(1):

        logits = model(bx, training=False)
        probs = tf.math.sigmoid(logits)

        print("\nForward-pass debug before training:")
        print("Input:", bx.shape)
        print("Logits:", logits.shape)
        print("Probabilities:", probs.shape)

        print(
            "First 10 probabilities:",
            probs.numpy().flatten()[:10]
        )

        path = os.path.join(
            DEBUG_DIR,
            f"{run_name}_forward_debug.txt"
        )

        with open(path, "w") as f:

            f.write(f"Input: {bx.shape}\n")
            f.write(f"Logits: {logits.shape}\n")
            f.write(f"Probabilities: {probs.shape}\n")

            f.write(
                f"First 10 probabilities: "
                f"{probs.numpy().flatten()[:10]}\n"
            )

        break


# =========================================
# TRAINING
# =========================================

all_results = []

run_name = f"{EXPERIMENT_NAME}_{model_name}_lr{lr:g}"

print("\n======================================")
print(f"TRAINING {run_name}")
print("======================================")

start_time = time.time()

model = model_fn()

print("\nModel summary:")
model.summary()

summary_path = os.path.join(
    SUMMARY_DIR,
    f"{run_name}_summary.txt"
)

with open(summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

save_layer_shapes(model, run_name)
forward_debug(model, val_ds, run_name)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=lr
    ),

    loss=keras.losses.BinaryCrossentropy(
        from_logits=True
    ),

    metrics=[
        keras.metrics.AUC(
            name="auc",
            from_logits=True
        )
    ]
)

checkpoint_path = os.path.join(
    MODEL_DIR,
    f"{run_name}_best.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_auc",
        patience=6,
        mode="max",
        restore_best_weights=True
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

runtime_sec = time.time() - start_time


# =========================================
# EVALUATION
# =========================================

print("\n======================================")
print("Evaluating")
print("======================================")

y_val_true, y_val_pred = get_probs(model, val_ds)
y_test_true, y_test_pred = get_probs(model, test_ds)

val_ecg_auc = roc_auc_score(
    y_val_true,
    y_val_pred
)

test_ecg_auc = roc_auc_score(
    y_test_true,
    y_test_pred
)

df_val_mean = patient_predictions(
    pat_val,
    y_val_true,
    y_val_pred,
    "mean"
)

df_val_max = patient_predictions(
    pat_val,
    y_val_true,
    y_val_pred,
    "max"
)

df_test_mean = patient_predictions(
    pat_test,
    y_test_true,
    y_test_pred,
    "mean"
)

df_test_max = patient_predictions(
    pat_test,
    y_test_true,
    y_test_pred,
    "max"
)

val_auc_mean = roc_auc_score(
    df_val_mean["y"],
    df_val_mean["pred"]
)

val_auc_max = roc_auc_score(
    df_val_max["y"],
    df_val_max["pred"]
)

test_auc_mean = roc_auc_score(
    df_test_mean["y"],
    df_test_mean["pred"]
)

test_auc_max = roc_auc_score(
    df_test_max["y"],
    df_test_max["pred"]
)

best_val_auc_training = max(
    history.history["val_auc"]
)

final_val_auc_training = history.history["val_auc"][-1]

print("\n======================================")
print(f"RESULTS: {run_name}")

print(f"Val ECG AUC: {val_ecg_auc:.4f}")
print(f"Val patient AUC mean: {val_auc_mean:.4f}")
print(f"Val patient AUC max: {val_auc_max:.4f}")

print(f"Test ECG AUC: {test_ecg_auc:.4f}")
print(f"Test patient AUC mean: {test_auc_mean:.4f}")
print(f"Test patient AUC max: {test_auc_max:.4f}")

print(
    f"Best val_auc during training: "
    f"{best_val_auc_training:.4f}"
)

print(
    f"Final val_auc during training: "
    f"{final_val_auc_training:.4f}"
)

print(f"Epochs ran: {len(history.history['loss'])}")

print(f"Runtime: {runtime_sec / 60:.2f} min")
print("======================================")