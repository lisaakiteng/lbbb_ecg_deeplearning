# NOTE: This file requires wrapping the experiment loop in `for seed in seeds:` and calling tf.keras.backend.clear_session() and tf.keras.utils.set_random_seed(seed) before each run.\n# I will generate the full multiseed version on request.\n\nimport os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from dataset import pack_leads, make_ds
from models_cardioformer_like import cardioformer_like_ecg

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"
OUTPUT_DIR = "/data/leuven/376/vsc37666/lbbb_outputs/outputs_cardioformer_like_multiseed"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

for d in [MODEL_DIR, PLOT_DIR, PRED_DIR, RESULT_DIR, SUMMARY_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)


EXPERIMENT_NAME = "cardioformer_like_multiseed"

lrs = [3e-4]
epochs = 60
batch_size = 32
seeds = [42, 43, 44, 45, 46]


print("======================================")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("Learning rates to test:", lrs)
print("Epochs:", epochs)
print("Batch size:", batch_size)
print("Seed:", seed)
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("======================================")


# =========================================
# LOAD DATA
# =========================================

train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df   = pd.read_parquet(DATA_DIR + "val.parquet")
test_df  = pd.read_parquet(DATA_DIR + "test.parquet")

print("\nData shapes:")
print("train:", train_df.shape)
print("val:  ", val_df.shape)
print("test: ", test_df.shape)

print("\nECG-level label counts:")
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{split_name}:")
    print(df["y"].value_counts(dropna=False))

print("\nPatient-level label counts:")
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{split_name}:")
    print("ECGs:", len(df))
    print("Patients:", df["ID_STUDY_PART"].nunique())
    print(df.groupby("ID_STUDY_PART")["y"].max().value_counts(dropna=False))


# =========================================
# ECG LEADS
# =========================================

lead_cols = [c for c in train_df.columns if c.lower().startswith("lead_")]
lead_cols = lead_cols[:12]

print("\nLead columns:")
for i, c in enumerate(lead_cols):
    print(i, c)

if len(lead_cols) != 12:
    raise ValueError(f"Expected 12 lead columns, found {len(lead_cols)}")


# =========================================
# PACK ECGs
# =========================================

train_packed, y_train, pat_train = pack_leads(train_df, lead_cols)
val_packed,   y_val,   pat_val   = pack_leads(val_df, lead_cols)
test_packed,  y_test,  pat_test  = pack_leads(test_df, lead_cols)

print("\nPacked lengths:")
print("train:", len(train_packed))
print("val:  ", len(val_packed))
print("test: ", len(test_packed))

print("\nLabel arrays:")
print("y_train:", y_train.shape, "pos:", int(np.sum(y_train)), "neg:", int(len(y_train) - np.sum(y_train)))
print("y_val:  ", y_val.shape,   "pos:", int(np.sum(y_val)),   "neg:", int(len(y_val) - np.sum(y_val)))
print("y_test: ", y_test.shape,  "pos:", int(np.sum(y_test)),  "neg:", int(len(y_test) - np.sum(y_test)))


# =========================================
# DATASETS
# =========================================

train_ds = make_ds(train_packed, y_train, batch_size=batch_size, training=True)
val_ds   = make_ds(val_packed,   y_val,   batch_size=batch_size)
test_ds  = make_ds(test_packed,  y_test,  batch_size=batch_size)

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

    with open(os.path.join(DEBUG_DIR, "first_train_batch_debug.txt"), "w") as f:
        f.write(f"batch_x shape: {batch_x.shape}\n")
        f.write(f"batch_y shape: {batch_y.shape}\n")
        f.write(f"batch_x dtype: {batch_x.dtype}\n")
        f.write(f"batch_y dtype: {batch_y.dtype}\n")
        f.write(f"batch_y first 10: {batch_y.numpy().flatten()[:10]}\n")
        f.write(f"batch_x min: {float(tf.reduce_min(batch_x).numpy())}\n")
        f.write(f"batch_x max: {float(tf.reduce_max(batch_x).numpy())}\n")
        f.write(f"batch_x mean: {float(tf.reduce_mean(batch_x).numpy())}\n")
        f.write(f"batch_x std: {float(tf.math.reduce_std(batch_x).numpy())}\n")


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
    plt.plot(history.history["auc"], label="train")
    plt.plot(history.history["val_auc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title(f"{run_name} AUC Curve")
    plt.savefig(os.path.join(PLOT_DIR, f"{run_name}_auc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{run_name} Loss Curve")
    plt.savefig(os.path.join(PLOT_DIR, f"{run_name}_loss_curve.png"), dpi=150, bbox_inches="tight")
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
    plt.savefig(os.path.join(PLOT_DIR, f"{run_name}_{split}_roc.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_layer_shapes(model, run_name):
    path = os.path.join(DEBUG_DIR, f"{run_name}_layer_shapes.txt")

    print("\nLayer shapes:")
    with open(path, "w") as f:
        for i, layer in enumerate(model.layers):
            try:
                out_shape = layer.output.shape
            except Exception:
                out_shape = "unknown"

            line = f"{i:03d} | {layer.name:45s} | {layer.__class__.__name__:30s} | output: {out_shape}"
            print(line)
            f.write(line + "\n")


def forward_debug(model, ds, run_name):
    for bx, by in ds.take(1):
        logits = model(bx, training=False)
        probs = tf.math.sigmoid(logits)

        print("\nForward-pass debug before training:")
        print("Input:", bx.shape)
        print("Labels:", by.shape)
        print("Logits:", logits.shape)
        print("Probabilities:", probs.shape)
        print("First 10 labels:", by.numpy().flatten()[:10])
        print("First 10 logits:", logits.numpy().flatten()[:10])
        print("First 10 probabilities:", probs.numpy().flatten()[:10])

        with open(os.path.join(DEBUG_DIR, f"{run_name}_forward_debug.txt"), "w") as f:
            f.write(f"Input: {bx.shape}\n")
            f.write(f"Labels: {by.shape}\n")
            f.write(f"Logits: {logits.shape}\n")
            f.write(f"Probabilities: {probs.shape}\n")
            f.write(f"First 10 labels: {by.numpy().flatten()[:10]}\n")
            f.write(f"First 10 logits: {logits.numpy().flatten()[:10]}\n")
            f.write(f"First 10 probabilities: {probs.numpy().flatten()[:10]}\n")

        break



# =========================================
# MULTI-SEED SETTINGS
# =========================================

# =========================================
# EXPERIMENT LOOP
# =========================================

all_results = []

models = [
    ("cardioformer_like_ecg", cardioformer_like_ecg)
]

for lr in lrs:
    for name, model_fn in models:

        run_name = f"{EXPERIMENT_NAME}_{name}_lr{lr:g}"

        print("\n======================================")
        print(f"TRAINING {run_name}")
        print("======================================")

        start_time = time.time()

        model = model_fn()

        print("\nModel summary:")
        model.summary()

        with open(os.path.join(SUMMARY_DIR, f"{run_name}_summary.txt"), "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        save_layer_shapes(model, run_name)
        forward_debug(model, val_ds, run_name)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.AUC(name="auc", from_logits=True)]
        )

        checkpoint_path = os.path.join(MODEL_DIR, f"{run_name}_best.keras")

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

        y_val_true, y_val_pred = get_probs(model, val_ds)
        y_test_true, y_test_pred = get_probs(model, test_ds)

        val_ecg_auc = roc_auc_score(y_val_true, y_val_pred)
        test_ecg_auc = roc_auc_score(y_test_true, y_test_pred)

        df_val_mean = patient_predictions(pat_val, y_val_true, y_val_pred, "mean")
        df_val_max  = patient_predictions(pat_val, y_val_true, y_val_pred, "max")

        df_test_mean = patient_predictions(pat_test, y_test_true, y_test_pred, "mean")
        df_test_max  = patient_predictions(pat_test, y_test_true, y_test_pred, "max")

        val_auc_mean = roc_auc_score(df_val_mean["y"], df_val_mean["pred"])
        val_auc_max  = roc_auc_score(df_val_max["y"], df_val_max["pred"])

        test_auc_mean = roc_auc_score(df_test_mean["y"], df_test_mean["pred"])
        test_auc_max  = roc_auc_score(df_test_max["y"], df_test_max["pred"])

        best_val_auc_training = max(history.history["val_auc"])
        final_val_auc_training = history.history["val_auc"][-1]

        print("\n======================================")
        print(f"RESULTS: {run_name}")
        print(f"Val ECG AUC: {val_ecg_auc:.4f}")
        print(f"Val patient AUC mean: {val_auc_mean:.4f}")
        print(f"Val patient AUC max: {val_auc_max:.4f}")
        print(f"Test ECG AUC: {test_ecg_auc:.4f}")
        print(f"Test patient AUC mean: {test_auc_mean:.4f}")
        print(f"Test patient AUC max: {test_auc_max:.4f}")
        print(f"Best val_auc during training: {best_val_auc_training:.4f}")
        print(f"Final val_auc during training: {final_val_auc_training:.4f}")
        print(f"Runtime: {runtime_sec / 60:.2f} min")
        print("======================================")

        pd.DataFrame({"pat": pat_val, "y": y_val_true, "pred": y_val_pred}).to_csv(
            os.path.join(PRED_DIR, f"{run_name}_val_ecg_predictions.csv"), index=False
        )

        pd.DataFrame({"pat": pat_test, "y": y_test_true, "pred": y_test_pred}).to_csv(
            os.path.join(PRED_DIR, f"{run_name}_test_ecg_predictions.csv"), index=False
        )

        df_val_mean.to_csv(os.path.join(PRED_DIR, f"{run_name}_val_patient_mean.csv"), index=False)
        df_val_max.to_csv(os.path.join(PRED_DIR, f"{run_name}_val_patient_max.csv"), index=False)
        df_test_mean.to_csv(os.path.join(PRED_DIR, f"{run_name}_test_patient_mean.csv"), index=False)
        df_test_max.to_csv(os.path.join(PRED_DIR, f"{run_name}_test_patient_max.csv"), index=False)

        model.save(os.path.join(MODEL_DIR, f"{run_name}_final.keras"))

        save_training_curves(history, run_name)
        save_roc_curve(df_val_mean, val_auc_mean, run_name, "val")
        save_roc_curve(df_test_mean, test_auc_mean, run_name, "test")

        all_results.append({
            "experiment": EXPERIMENT_NAME,
            "run_name": run_name,
            "model": name,
            "lr": lr,
            "seed": seed,
            "batch_size": batch_size,
            "val_ecg_auc": val_ecg_auc,
            "val_auc_mean": val_auc_mean,
            "val_auc_max": val_auc_max,
            "test_ecg_auc": test_ecg_auc,
            "test_auc_mean": test_auc_mean,
            "test_auc_max": test_auc_max,
            "best_val_auc_training": best_val_auc_training,
            "final_val_auc_training": final_val_auc_training,
            "epochs_ran": len(history.history["loss"]),
            "runtime_min": runtime_sec / 60,
            "best_model_path": checkpoint_path
        })

        pd.DataFrame(all_results).to_csv(
            os.path.join(RESULT_DIR, f"{EXPERIMENT_NAME}_partial_results.csv"),
            index=False
        )


# =========================================
# FINAL RESULTS
# =========================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

results_df = pd.DataFrame(all_results).sort_values("val_auc_mean", ascending=False)

results_path = os.path.join(RESULT_DIR, f"{EXPERIMENT_NAME}_results.csv")
results_df.to_csv(results_path, index=False)

print("\n======================================")
print("=== FINAL SUMMARY ===")
print("======================================")
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]

print("\n======================================")
print("=== BEST RUN ===")
print("======================================")
print(f"Run name: {best_row['run_name']}")
print(f"Model: {best_row['model']}")
print(f"Learning rate: {best_row['lr']}")
print(f"Seed: {best_row['seed']}")

print("\n--- VALIDATION ---")
print(f"Val ECG AUC: {best_row['val_ecg_auc']:.4f}")
print(f"Val patient AUC mean: {best_row['val_auc_mean']:.4f}")
print(f"Val patient AUC max: {best_row['val_auc_max']:.4f}")

print("\n--- TEST ---")
print(f"Test ECG AUC: {best_row['test_ecg_auc']:.4f}")
print(f"Test patient AUC mean: {best_row['test_auc_mean']:.4f}")
print(f"Test patient AUC max: {best_row['test_auc_max']:.4f}")

print("\n--- TRAINING ---")
print(f"Best val_auc during training: {best_row['best_val_auc_training']:.4f}")
print(f"Final val_auc during training: {best_row['final_val_auc_training']:.4f}")
print(f"Epochs ran: {best_row['epochs_ran']}")
print(f"Runtime (min): {best_row['runtime_min']:.2f}")

print("\n--- FILES ---")
print(f"Best model path: {best_row['best_model_path']}")
print(f"Results CSV: {results_path}")
print("======================================")