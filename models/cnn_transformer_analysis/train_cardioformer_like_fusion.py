# train_cardioformer_like_fusion.py

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from dataset import pack_leads
from dataset_multi import make_ds_multi
from models_cardioformer_like_fusion import fusion_cardioformer_like

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =========================================
# PATHS
# =========================================

DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"
OUTPUT_DIR = "/data/leuven/376/vsc37666/lbbb_outputs/outputs_cardioformer_like_fusion"

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")

for d in [MODEL_DIR, PLOT_DIR, PRED_DIR, RESULT_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================
# EXPERIMENT SETTINGS
# =========================================

EXPERIMENT_NAME = "exp1_cardioformer_like_fusion"

lrs = [3e-4, 1e-4]
epochs = 60
seed = 42

tf.keras.utils.set_random_seed(seed)

print("======================================")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print("Learning rates to test:", lrs)
print("Seed:", seed)
print("======================================")


# =========================================
# LOAD DATA
# =========================================

train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df   = pd.read_parquet(DATA_DIR + "val.parquet")
test_df  = pd.read_parquet(DATA_DIR + "test.parquet")
patient_df = pd.read_parquet(DATA_DIR + "patient_df.parquet")


# =========================================
# TABULAR FEATURES
# =========================================

rf_cols = [
    c for c in patient_df.columns
    if c.startswith("DIAGN_") or c.startswith("RISK_")
]

feature_cols = ["AgeAtECG", "GENDER"] + rf_cols
patient_features = patient_df.set_index("ID_STUDY_PART")[feature_cols]

Xtab_train = patient_features.loc[train_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)
Xtab_val   = patient_features.loc[val_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)
Xtab_test  = patient_features.loc[test_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)

n_tab = Xtab_train.shape[1]

print("Number of tabular features:", n_tab)
print("Tabular features:", feature_cols)


# =========================================
# ECG LEADS
# =========================================

lead_cols = [c for c in train_df.columns if c.lower().startswith("lead_")]
lead_cols = lead_cols[:12]

print("Lead columns:", lead_cols)

train_packed, y_train, pat_train = pack_leads(train_df, lead_cols)
val_packed,   y_val,   pat_val   = pack_leads(val_df, lead_cols)
test_packed,  y_test,  pat_test  = pack_leads(test_df, lead_cols)

train_ds = make_ds_multi(train_packed, Xtab_train, y_train, training=True)
val_ds   = make_ds_multi(val_packed, Xtab_val, y_val)
test_ds  = make_ds_multi(test_packed, Xtab_test, y_test)


# =========================================
# HELPERS
# =========================================

def get_probs(model, ds):
    y_true, y_pred = [], []

    for (ecg, tab), y in ds:
        logits = model([ecg, tab], training=False)
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

    if agg == "max":
        return df.groupby("pat").agg(
            y=("y", "max"),
            pred=("pred", "max")
        ).reset_index()

    raise ValueError("agg must be mean or max")


def save_training_curves(history, run_name):
    plt.figure()
    plt.plot(history.history["auc"], label="train")
    plt.plot(history.history["val_auc"], label="val")
    plt.legend()
    plt.title(f"{run_name} AUC Curve")
    plt.savefig(os.path.join(PLOT_DIR, f"{run_name}_auc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
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


# =========================================
# TRAINING
# =========================================

all_results = []

models = [
    ("fusion_cardioformer_like", lambda: fusion_cardioformer_like(n_tab=n_tab))
]

for lr in lrs:
    for name, model_fn in models:

        run_name = f"{EXPERIMENT_NAME}_{name}_lr{lr:g}"

        print("\n======================================")
        print(f"TRAINING {run_name}")
        print("======================================")

        start_time = time.time()

        model = model_fn()

        # =========================================
        # FULL EXPANDED MODEL SUMMARY
        # =========================================

        print("\n======================================")
        print("FULL MODEL SUMMARY — EXPANDED")
        print("======================================")

        model.summary(expand_nested=True)

        full_summary_path = os.path.join(SUMMARY_DIR, f"{run_name}_expanded_summary.txt")

        with open(full_summary_path, "w") as f:
            model.summary(
                expand_nested=True,
                print_fn=lambda x: f.write(x + "\n")
            )

        print(f"Saved expanded full model summary to: {full_summary_path}")

        # =========================================
        # ECG BACKBONE SUMMARY ONLY
        # =========================================

        print("\n======================================")
        print("ECG BACKBONE SUMMARY ONLY")
        print("======================================")

        try:
            backbone = model.get_layer("cardioformer_like_backbone")
            backbone.summary(expand_nested=True)

            backbone_summary_path = os.path.join(SUMMARY_DIR, f"{run_name}_backbone_summary.txt")

            with open(backbone_summary_path, "w") as f:
                backbone.summary(
                    expand_nested=True,
                    print_fn=lambda x: f.write(x + "\n")
                )

            print(f"Saved ECG backbone summary to: {backbone_summary_path}")

        except ValueError:
            print("Could not find layer named 'cardioformer_like_backbone'.")
            print("Available layers:")
            for layer in model.layers:
                print(layer.name)

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

        # =========================================
        # EVALUATION
        # =========================================

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

        # =========================================
        # SAVE OUTPUTS
        # =========================================

        pd.DataFrame({
            "pat": pat_val,
            "y": y_val_true,
            "pred": y_val_pred
        }).to_csv(
            os.path.join(PRED_DIR, f"{run_name}_val_ecg_predictions.csv"),
            index=False
        )

        pd.DataFrame({
            "pat": pat_test,
            "y": y_test_true,
            "pred": y_test_pred
        }).to_csv(
            os.path.join(PRED_DIR, f"{run_name}_test_ecg_predictions.csv"),
            index=False
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
            "n_tab": n_tab,
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