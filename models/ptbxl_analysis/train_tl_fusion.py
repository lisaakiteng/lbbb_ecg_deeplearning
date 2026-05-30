import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import pack_leads
from dataset_multi import make_ds_multi


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"

TRANSFER_MODEL_PATH = "tl_step2_partial_finetune_lr1e6_2_best.keras"

EXPERIMENT = "fusion_exp4"

BATCH_SIZE = 32

FROZEN_LR = 1e-4
FINETUNE_LR = 1e-6

FROZEN_EPOCHS = 80
FINETUNE_EPOCHS = 50


# ============================================================
# PLOT FUNCTION
# ============================================================
def save_training_plots(history, prefix, title):
    # AUC curve
    plt.figure()
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{title} - AUC")
    plt.legend()
    plt.savefig(f"{prefix}_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.savefig(f"{prefix}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots: {prefix}_auc.png and {prefix}_loss.png")


# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")

train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df   = pd.read_parquet(DATA_DIR + "val.parquet")
test_df  = pd.read_parquet(DATA_DIR + "test.parquet")

patient_df = pd.read_parquet(DATA_DIR + "patient_df.parquet")

lead_cols = [c for c in train_df.columns if c.lower().startswith("lead_")][:12]


# ============================================================
# TABULAR FEATURES
# ============================================================
rf_cols = [
    c for c in patient_df.columns
    if c.startswith("DIAGN_") or c.startswith("RISK_")
]

feature_cols = ["AgeAtECG", "GENDER"] + rf_cols

print("Number of tabular features:", len(feature_cols))
print("Tabular features:")
print(feature_cols)

patient_features = patient_df.set_index("ID_STUDY_PART")[feature_cols]

Xtab_train = patient_features.loc[train_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)
Xtab_val   = patient_features.loc[val_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)
Xtab_test  = patient_features.loc[test_df["ID_STUDY_PART"]].fillna(0).to_numpy(np.float32)

n_tab = Xtab_train.shape[1]


# ============================================================
# ECG FEATURES
# ============================================================
train_packed, y_train, pat_train = pack_leads(train_df, lead_cols)
val_packed,   y_val,   pat_val   = pack_leads(val_df,   lead_cols)
test_packed,  y_test,  pat_test  = pack_leads(test_df,  lead_cols)

train_ds = make_ds_multi(
    train_packed, Xtab_train, y_train,
    batch_size=BATCH_SIZE,
    training=True
)

val_ds = make_ds_multi(
    val_packed, Xtab_val, y_val,
    batch_size=BATCH_SIZE,
    training=False
)

test_ds = make_ds_multi(
    test_packed, Xtab_test, y_test,
    batch_size=BATCH_SIZE,
    training=False
)


# ============================================================
# LOAD TRANSFER MODEL AND EXTRACT ECG BACKBONE
# ============================================================
print("\nLoading ECG transfer model...")
transfer_model = keras.models.load_model(TRANSFER_MODEL_PATH)

print("Loaded model:")
transfer_model.summary()

print("\nExtracting PTB-XL backbone...")
ptbxl_backbone = transfer_model.get_layer("ptbxl_backbone")

ptbxl_backbone.trainable = False


# ============================================================
# BUILD FUSION MODEL
# ============================================================
def build_tl_fusion_model(ptbxl_backbone, n_tab, input_len=5000):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    # ECG branch
    ecg_feat = ptbxl_backbone(ecg_in)
    ecg_feat = layers.Dense(128, activation="relu", name="ecg_projection")(ecg_feat)
    ecg_feat = layers.Dropout(0.3, name="ecg_projection_dropout")(ecg_feat)

    # Tabular branch
    t = layers.Dense(64, name="tab_dense_1")(tab_in)
    t = layers.BatchNormalization(name="tab_bn_1")(t)
    t = layers.ReLU(name="tab_relu_1")(t)
    t = layers.Dropout(0.3, name="tab_dropout_1")(t)

    t = layers.Dense(32, name="tab_dense_2")(t)
    t = layers.BatchNormalization(name="tab_bn_2")(t)
    t = layers.ReLU(name="tab_relu_2")(t)
    t = layers.Dropout(0.3, name="tab_dropout_2")(t)

    # Fusion
    x = layers.Concatenate(name="fusion_concat")([ecg_feat, t])
    x = layers.Dense(64, activation="relu", name="fusion_dense_1")(x)
    x = layers.Dropout(0.3, name="fusion_dropout_1")(x)

    # Logit output
    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model([ecg_in, tab_in], out, name="tl_fusion_ptbxl_backbone")


model = build_tl_fusion_model(ptbxl_backbone, n_tab)

print("\nFusion model:")
model.summary()


# ============================================================
# DEBUG SHAPES
# ============================================================
print("\nDEBUG: Checking one batch...")

for (ecg_batch, tab_batch), y_batch in train_ds.take(1):
    print("ECG batch shape:", ecg_batch.shape)
    print("Tabular batch shape:", tab_batch.shape)
    print("Label batch shape:", y_batch.shape)

    backbone_out = ptbxl_backbone(ecg_batch)
    print("Backbone output shape:", backbone_out.shape)

    model_out = model([ecg_batch, tab_batch])
    print("Model output/logit shape:", model_out.shape)
    print("First 5 logits:", model_out.numpy().ravel()[:5])


# ============================================================
# HELPERS
# ============================================================
def get_probs(model, ds):
    y_true, y_pred = [], []

    for (ecg, tab), y in ds:
        logits = model([ecg, tab], training=False)
        probs = tf.math.sigmoid(logits).numpy().ravel()

        y_true.extend(y.numpy().ravel())
        y_pred.extend(probs)

    return np.array(y_true), np.array(y_pred)


def compute_patient_auc(pat_ids, y_true, y_pred, agg="mean"):
    df = pd.DataFrame({
        "patient": pat_ids[:len(y_pred)],
        "y": y_true,
        "prob": y_pred
    })

    if agg == "mean":
        dfp = df.groupby("patient").agg(
            y=("y", "max"),
            prob=("prob", "mean")
        ).reset_index()
    elif agg == "max":
        dfp = df.groupby("patient").agg(
            y=("y", "max"),
            prob=("prob", "max")
        ).reset_index()
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    return roc_auc_score(dfp["y"], dfp["prob"])


def evaluate_model(model, ds, pat_ids, split_name):
    y_true, y_pred = get_probs(model, ds)

    ecg_auc = roc_auc_score(y_true, y_pred)
    patient_auc_mean = compute_patient_auc(pat_ids, y_true, y_pred, agg="mean")
    patient_auc_max  = compute_patient_auc(pat_ids, y_true, y_pred, agg="max")

    print(f"\n{split_name} RESULTS")
    print("=" * 60)
    print(f"ECG-level AUC:          {ecg_auc:.4f}")
    print(f"Patient-level AUC mean: {patient_auc_mean:.4f}")
    print(f"Patient-level AUC max:  {patient_auc_max:.4f}")
    print("=" * 60)

    print("First 10 labels:", y_true[:10])
    print("First 10 probabilities:", y_pred[:10])

    return {
        f"{split_name.lower()}_ecg_auc": ecg_auc,
        f"{split_name.lower()}_patient_auc_mean": patient_auc_mean,
        f"{split_name.lower()}_patient_auc_max": patient_auc_max,
    }


# ============================================================
# STEP 1: FROZEN BACKBONE TRAINING
# ============================================================
print("\n" + "=" * 80)
print("STEP 1: TRAIN FUSION HEAD WITH PTB-XL BACKBONE FROZEN")
print("=" * 80)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FROZEN_LR),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.AUC(name="auc", from_logits=True)]
)

callbacks_frozen = [
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        f"{EXPERIMENT}_step1_frozen_best.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FROZEN_EPOCHS,
    callbacks=callbacks_frozen,
    verbose=1
)

best_frozen_val_auc = max(history_frozen.history["val_auc"])
print(f"\nBest frozen fusion Keras Val AUC: {best_frozen_val_auc:.4f}")

save_training_plots(
    history_frozen,
    prefix=f"{EXPERIMENT}_step1_frozen",
    title="TL Fusion Step 1: Frozen PTB-XL Backbone"
)

model.save(f"{EXPERIMENT}_step1_frozen_final.keras")


# ============================================================
# EVALUATE FROZEN MODEL
# ============================================================
frozen_val_results = evaluate_model(model, val_ds, pat_val, "VAL_FROZEN")
frozen_test_results = evaluate_model(model, test_ds, pat_test, "TEST_FROZEN")


# ============================================================
# STEP 2: PARTIAL FINE-TUNING
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: PARTIAL FINE-TUNING OF ECG BACKBONE")
print("=" * 80)

ptbxl_backbone.trainable = True

for layer in ptbxl_backbone.layers:
    layer.trainable = False

for layer in ptbxl_backbone.layers[-8:]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True

print("\nTrainable layers in PTB-XL backbone:")
for layer in ptbxl_backbone.layers:
    if layer.trainable:
        print("  ", layer.name)

print("Total trainable weights:", len(model.trainable_weights))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.AUC(name="auc", from_logits=True)])

callbacks_finetune = [
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True),
    ModelCheckpoint(
        f"{EXPERIMENT}_step2_finetuned_best.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True)
]

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks_finetune,
    verbose=1
)

best_finetune_val_auc = max(history_finetune.history["val_auc"])
print(f"\nBest fine-tuned fusion Keras Val AUC: {best_finetune_val_auc:.4f}")

save_training_plots(
    history_finetune,
    prefix=f"{EXPERIMENT}_step2_finetuned",
    title="TL Fusion Step 2: Partial Fine-Tuning"
)

model.save(f"{EXPERIMENT}_step2_finetuned_final.keras")


# ============================================================
# FINAL EVALUATION
# ============================================================
finetune_val_results = evaluate_model(model, val_ds, pat_val, "VAL_FINETUNED")
finetune_test_results = evaluate_model(model, test_ds, pat_test, "TEST_FINETUNED")


# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "experiment": EXPERIMENT,
    "transfer_model_path": TRANSFER_MODEL_PATH,
    "n_tabular_features": n_tab,
    "frozen_lr": FROZEN_LR,
    "finetune_lr": FINETUNE_LR,
    "best_frozen_keras_val_auc": best_frozen_val_auc,
    "best_finetune_keras_val_auc": best_finetune_val_auc,
}

results.update(frozen_val_results)
results.update(frozen_test_results)
results.update(finetune_val_results)
results.update(finetune_test_results)

results_df = pd.DataFrame([results])
results_path = f"{EXPERIMENT}_results.csv"
results_df.to_csv(results_path, index=False)

print("\nSaved results to:", results_path)
print(results_df.T)