# =========================
# train_fusion_multiseed.py
# =========================

import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset import pack_leads
from dataset_multi import make_ds_multi
from models_fusion import fusion_no_residual


DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"
OUTPUT_DIR = "/data/leuven/376/vsc37666/lbbb_outputs/outputs_fusion_6mo_multiseed"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

for d in [MODEL_DIR, PLOT_DIR, PRED_DIR, RESULT_DIR, SUMMARY_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)


EXPERIMENT_NAME = "fusion_6mo_multiseed"
model_name = "fusion_no_residual"

seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
epochs = 80
batch_size = 32
lrs = [1e-3]


print("======================================")
print("Loading corrected 6-month fusion data")
print("======================================")

train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df = pd.read_parquet(DATA_DIR + "val.parquet")
test_df = pd.read_parquet(DATA_DIR + "test.parquet")
patient_df = pd.read_parquet(DATA_DIR + "patient_df.parquet")

print("\nData shapes:")
print("train:", train_df.shape)
print("val:  ", val_df.shape)
print("test: ", test_df.shape)
print("patient_df:", patient_df.shape)

print("\nECG-level label counts:")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{name}:")
    print(df["y"].value_counts())

print("\nPatient-level label counts:")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{name}:")
    print(df.groupby("ID_STUDY_PART")["y"].max().value_counts())

if "y_old_ever_crt" in train_df.columns:
    print("\nOld vs corrected label check:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}:")
        print("old ever-CRT ECG positives:", int(df["y_old_ever_crt"].sum()))
        print("new 6mo ECG positives:", int(df["y"].sum()))
        print("positive patients:", df[df["y"] == 1]["ID_STUDY_PART"].nunique())


# =========================
# TABULAR FEATURES
# =========================

print("\n======================================")
print("Preparing tabular features")
print("======================================")

rf_cols = [
    c for c in patient_df.columns
    if c.startswith("DIAGN_") or c.startswith("RISK_")
]

feature_cols = ["AgeAtECG", "GENDER"] + rf_cols

print("Number of tabular features:", len(feature_cols))
print("Feature columns:")
for i, c in enumerate(feature_cols):
    print(i, c)

patient_df["ID_STUDY_PART"] = patient_df["ID_STUDY_PART"].astype(str)
train_df["ID_STUDY_PART"] = train_df["ID_STUDY_PART"].astype(str)
val_df["ID_STUDY_PART"] = val_df["ID_STUDY_PART"].astype(str)
test_df["ID_STUDY_PART"] = test_df["ID_STUDY_PART"].astype(str)

patient_features = patient_df.set_index("ID_STUDY_PART")[feature_cols]

Xtab_train = patient_features.loc[train_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)
Xtab_val = patient_features.loc[val_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)
Xtab_test = patient_features.loc[test_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)

n_tab = Xtab_train.shape[1]

print("\nTabular shapes:")
print("Xtab_train:", Xtab_train.shape)
print("Xtab_val:  ", Xtab_val.shape)
print("Xtab_test: ", Xtab_test.shape)

print("\nTabular debug:")
print("train min:", np.nanmin(Xtab_train), "max:", np.nanmax(Xtab_train), "mean:", np.nanmean(Xtab_train))
print("val min:  ", np.nanmin(Xtab_val), "max:", np.nanmax(Xtab_val), "mean:", np.nanmean(Xtab_val))
print("test min: ", np.nanmin(Xtab_test), "max:", np.nanmax(Xtab_test), "mean:", np.nanmean(Xtab_test))


# =========================
# ECG FEATURES
# =========================

lead_cols = [c for c in train_df.columns if c.lower().startswith("lead_")]
lead_cols = lead_cols[:12]

if len(lead_cols) != 12:
    raise ValueError(f"Expected 12 lead columns, found {len(lead_cols)}")

print("\nLead columns:")
for i, c in enumerate(lead_cols):
    print(i, c)

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
print("y_train:", y_train.shape, "pos:", int(np.sum(y_train)), "neg:", int(len(y_train) - np.sum(y_train)))
print("y_val:  ", y_val.shape, "pos:", int(np.sum(y_val)), "neg:", int(len(y_val) - np.sum(y_val)))
print("y_test: ", y_test.shape, "pos:", int(np.sum(y_test)), "neg:", int(len(y_test) - np.sum(y_test)))


# =========================
# DATASETS
# =========================

print("\n======================================")
print("Creating validation and test tf.data datasets")
print("======================================")

val_ds = make_ds_multi(
    val_packed,
    Xtab_val,
    y_val,
    batch_size=batch_size,
)

test_ds = make_ds_multi(
    test_packed,
    Xtab_test,
    y_test,
    batch_size=batch_size,
)

for (batch_ecg, batch_tab), batch_y in val_ds.take(1):
    print("\nOne validation batch:")
    print("batch_ecg shape:", batch_ecg.shape)
    print("batch_tab shape:", batch_tab.shape)
    print("batch_y shape:", batch_y.shape)
    print("batch_ecg dtype:", batch_ecg.dtype)
    print("batch_tab dtype:", batch_tab.dtype)
    print("batch_y dtype:", batch_y.dtype)
    print("batch_y first 10:", batch_y.numpy().flatten()[:10])
    print("batch_ecg min:", float(tf.reduce_min(batch_ecg).numpy()))
    print("batch_ecg max:", float(tf.reduce_max(batch_ecg).numpy()))
    print("batch_ecg mean:", float(tf.reduce_mean(batch_ecg).numpy()))
    print("batch_ecg std:", float(tf.math.reduce_std(batch_ecg).numpy()))
    print("batch_tab first row:", batch_tab.numpy()[0])
    break


print("======================================")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("Model:", model_name)
print("Learning rates:", lrs)
print("Seeds:", seeds)
print("Epochs:", epochs)
print("Batch size:", batch_size)
print("Tabular features:", n_tab)
print("Train ECGs:", len(train_packed), "| Patients:", train_df["ID_STUDY_PART"].nunique(), "| Pos ECGs:", int(np.sum(y_train)))
print("Val ECGs:  ", len(val_packed), "| Patients:", val_df["ID_STUDY_PART"].nunique(), "| Pos ECGs:", int(np.sum(y_val)))
print("Test ECGs: ", len(test_packed), "| Patients:", test_df["ID_STUDY_PART"].nunique(), "| Pos ECGs:", int(np.sum(y_test)))
print("Output directory:", OUTPUT_DIR)
print("======================================")


# =========================
# HELPERS
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_probs(model, ds):
    y_true, y_pred = [], []

    for (ecg, tab), y in ds:
        logits = model([ecg, tab], training=False)
        probs = tf.math.sigmoid(logits).numpy().flatten()

        y_true.extend(y.numpy().flatten())
        y_pred.extend(probs)

    return np.array(y_true), np.array(y_pred)


def patient_predictions(pat_ids, y_true, y_pred, agg="mean"):
    dfp = pd.DataFrame({"pat": pat_ids, "y": y_true, "pred": y_pred})

    if agg == "mean":
        return dfp.groupby("pat").agg(
            y=("y", "max"),
            pred=("pred", "mean"),
        ).reset_index()

    if agg == "max":
        return dfp.groupby("pat").agg(
            y=("y", "max"),
            pred=("pred", "max"),
        ).reset_index()

    raise ValueError("agg must be mean or max")


def save_training_curves(history, run_name):
    plt.figure()
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title(f"{run_name} AUC Curve")
    plt.savefig(os.path.join(PLOT_DIR, f"{run_name}_auc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
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

    with open(path, "w") as f:
        for i, layer in enumerate(model.layers):
            try:
                out_shape = layer.output.shape
            except Exception:
                out_shape = "unknown"

            line = f"{i:03d} | {layer.name:35s} | {layer.__class__.__name__:25s} | output: {out_shape}"
            f.write(line + "\n")


def forward_debug(model, ds, run_name):
    for (bx_ecg, bx_tab), by in ds.take(1):
        logits = model([bx_ecg, bx_tab], training=False)
        probs = tf.math.sigmoid(logits)

        path = os.path.join(DEBUG_DIR, f"{run_name}_forward_debug.txt")
        with open(path, "w") as f:
            f.write(f"ECG input: {bx_ecg.shape}\n")
            f.write(f"Tab input: {bx_tab.shape}\n")
            f.write(f"Logits: {logits.shape}\n")
            f.write(f"Probabilities: {probs.shape}\n")
            f.write(f"First 10 probabilities: {probs.numpy().flatten()[:10]}\n")
        break


# =========================
# TRAINING LOOP
# =========================

all_results = []

print("\n======================================")
print("STARTING FUSION MULTI-SEED TRAINING")
print("======================================")

for lr in lrs:
    for seed in seeds:
        set_seed(seed)

        train_ds = make_ds_multi(
            train_packed,
            Xtab_train,
            y_train,
            batch_size=batch_size,
            training=True,
        )

        run_name = f"{EXPERIMENT_NAME}_{model_name}_lr{lr:g}_seed{seed}"

        print("\n======================================")
        print(f"RUN: {run_name}")
        print("======================================")

        start_time = time.time()

        model = fusion_no_residual(n_tab)

        summary_path = os.path.join(SUMMARY_DIR, f"{run_name}_summary.txt")
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        save_layer_shapes(model, run_name)
        forward_debug(model, val_ds, run_name)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.AUC(name="auc", from_logits=True)],
        )

        checkpoint_path = os.path.join(MODEL_DIR, f"{run_name}_best.keras")

        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                patience=8,
                mode="max",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=3,
                min_lr=1e-6,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )

        runtime_sec = time.time() - start_time

        y_val_true, y_val_pred = get_probs(model, val_ds)
        y_test_true, y_test_pred = get_probs(model, test_ds)

        val_ecg_auc = roc_auc_score(y_val_true, y_val_pred)
        test_ecg_auc = roc_auc_score(y_test_true, y_test_pred)

        df_val_mean = patient_predictions(pat_val, y_val_true, y_val_pred, "mean")
        df_val_max = patient_predictions(pat_val, y_val_true, y_val_pred, "max")
        df_test_mean = patient_predictions(pat_test, y_test_true, y_test_pred, "mean")
        df_test_max = patient_predictions(pat_test, y_test_true, y_test_pred, "max")

        val_auc_mean = roc_auc_score(df_val_mean["y"], df_val_mean["pred"])
        val_auc_max = roc_auc_score(df_val_max["y"], df_val_max["pred"])
        test_auc_mean = roc_auc_score(df_test_mean["y"], df_test_mean["pred"])
        test_auc_max = roc_auc_score(df_test_max["y"], df_test_max["pred"])

        best_val_auc_training = max(history.history["val_auc"])
        final_val_auc_training = history.history["val_auc"][-1]
        epochs_ran = len(history.history["loss"])

        pd.DataFrame({"pat": pat_val, "y": y_val_true, "pred": y_val_pred}).to_csv(
            os.path.join(PRED_DIR, f"{run_name}_val_ecg_predictions.csv"),
            index=False,
        )

        pd.DataFrame({"pat": pat_test, "y": y_test_true, "pred": y_test_pred}).to_csv(
            os.path.join(PRED_DIR, f"{run_name}_test_ecg_predictions.csv"),
            index=False,
        )

        df_val_mean.to_csv(os.path.join(PRED_DIR, f"{run_name}_val_patient_mean.csv"), index=False)
        df_val_max.to_csv(os.path.join(PRED_DIR, f"{run_name}_val_patient_max.csv"), index=False)
        df_test_mean.to_csv(os.path.join(PRED_DIR, f"{run_name}_test_patient_mean.csv"), index=False)
        df_test_max.to_csv(os.path.join(PRED_DIR, f"{run_name}_test_patient_max.csv"), index=False)

        model.save(os.path.join(MODEL_DIR, f"{run_name}_final.keras"))

        save_training_curves(history, run_name)
        save_roc_curve(df_val_mean, val_auc_mean, run_name, "val")
        save_roc_curve(df_test_mean, test_auc_mean, run_name, "test")

        row = {
            "experiment": EXPERIMENT_NAME,
            "run_name": run_name,
            "model": model_name,
            "lr": lr,
            "seed": seed,
            "n_tab_features": n_tab,
            "val_ecg_auc": val_ecg_auc,
            "val_auc_mean": val_auc_mean,
            "val_auc_max": val_auc_max,
            "test_ecg_auc": test_ecg_auc,
            "test_auc_mean": test_auc_mean,
            "test_auc_max": test_auc_max,
            "best_val_auc_training": best_val_auc_training,
            "final_val_auc_training": final_val_auc_training,
            "epochs_ran": epochs_ran,
            "runtime_min": runtime_sec / 60,
            "total_params": model.count_params(),
            "best_model_path": checkpoint_path,
        }

        all_results.append(row)

        pd.DataFrame(all_results).to_csv(
            os.path.join(RESULT_DIR, f"{EXPERIMENT_NAME}_partial_results.csv"),
            index=False,
        )

        print(f"Val ECG AUC:           {val_ecg_auc:.4f}")
        print(f"Val patient AUC mean:  {val_auc_mean:.4f}")
        print(f"Val patient AUC max:   {val_auc_max:.4f}")
        print(f"Test ECG AUC:          {test_ecg_auc:.4f}")
        print(f"Test patient AUC mean: {test_auc_mean:.4f}")
        print(f"Test patient AUC max:  {test_auc_max:.4f}")
        print(f"Best val_auc training: {best_val_auc_training:.4f}")
        print(f"Epochs ran:            {epochs_ran}")
        print(f"Runtime:               {runtime_sec / 60:.2f} min")

        keras.backend.clear_session()


# =========================
# FINAL SUMMARY
# =========================

results_df = pd.DataFrame(all_results)

results_path = os.path.join(RESULT_DIR, f"{EXPERIMENT_NAME}_all_seed_results.csv")
results_df.to_csv(results_path, index=False)

summary_df = (
    results_df
    .groupby(["experiment", "model", "lr"])
    .agg(
        n_runs=("seed", "count"),
        val_ecg_auc_mean=("val_ecg_auc", "mean"),
        val_ecg_auc_std=("val_ecg_auc", "std"),
        val_auc_mean_mean=("val_auc_mean", "mean"),
        val_auc_mean_std=("val_auc_mean", "std"),
        val_auc_max_mean=("val_auc_max", "mean"),
        val_auc_max_std=("val_auc_max", "std"),
        test_ecg_auc_mean=("test_ecg_auc", "mean"),
        test_ecg_auc_std=("test_ecg_auc", "std"),
        test_auc_mean_mean=("test_auc_mean", "mean"),
        test_auc_mean_std=("test_auc_mean", "std"),
        test_auc_max_mean=("test_auc_max", "mean"),
        test_auc_max_std=("test_auc_max", "std"),
        epochs_ran_mean=("epochs_ran", "mean"),
        runtime_min_mean=("runtime_min", "mean"),
        runtime_min_std=("runtime_min", "std"),
    )
    .reset_index()
)

summary_path = os.path.join(RESULT_DIR, f"{EXPERIMENT_NAME}_mean_std_summary.csv")
summary_df.to_csv(summary_path, index=False)

best_individual = results_df.sort_values("val_auc_mean", ascending=False).iloc[0]
best_summary = summary_df.sort_values("val_auc_mean_mean", ascending=False).iloc[0]

print("\n======================================")
print("=== MEAN ± SD SUMMARY ACROSS SEEDS ===")
print("======================================")
print(summary_df.to_string(index=False))

print("\n======================================")
print("=== BEST INDIVIDUAL RUN BY VAL PATIENT MEAN AUC ===")
print("======================================")
print(f"Run name: {best_individual['run_name']}")
print(f"Model: {best_individual['model']}")
print(f"Learning rate: {best_individual['lr']}")
print(f"Seed: {best_individual['seed']}")

print("\n--- VALIDATION ---")
print(f"Val ECG AUC: {best_individual['val_ecg_auc']:.4f}")
print(f"Val patient AUC mean: {best_individual['val_auc_mean']:.4f}")
print(f"Val patient AUC max: {best_individual['val_auc_max']:.4f}")

print("\n--- TEST ---")
print(f"Test ECG AUC: {best_individual['test_ecg_auc']:.4f}")
print(f"Test patient AUC mean: {best_individual['test_auc_mean']:.4f}")
print(f"Test patient AUC max: {best_individual['test_auc_max']:.4f}")

print("\n======================================")
print("=== BEST MEAN RUN GROUP BY VAL PATIENT MEAN AUC ===")
print("======================================")
print(f"Model: {best_summary['model']}")
print(f"Learning rate: {best_summary['lr']}")
print(f"Number of runs: {int(best_summary['n_runs'])}")
print(f"Val patient AUC mean: {best_summary['val_auc_mean_mean']:.4f} ± {best_summary['val_auc_mean_std']:.4f}")
print(f"Test patient AUC mean: {best_summary['test_auc_mean_mean']:.4f} ± {best_summary['test_auc_mean_std']:.4f}")

print("\n--- FILES ---")
print(f"All seed results CSV: {results_path}")
print(f"Mean/std summary CSV: {summary_path}")
print("======================================")
