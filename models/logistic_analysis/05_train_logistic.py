import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from models_tabular import build_logistic_basic, build_logistic_full


DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"
OUT_DIR = "/data/leuven/376/vsc37666/lbbb_outputs/logistic_outputs_6mo"

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("LOGISTIC REGRESSION BASELINE - 6 MONTH LABEL")
print("=" * 60)

patient_df = pd.read_parquet(DATA_DIR + "patient_df.parquet")
train_df = pd.read_parquet(DATA_DIR + "train.parquet")
val_df   = pd.read_parquet(DATA_DIR + "val.parquet")
test_df  = pd.read_parquet(DATA_DIR + "test.parquet")

patient_df["ID_STUDY_PART"] = patient_df["ID_STUDY_PART"].astype(str)
train_df["ID_STUDY_PART"] = train_df["ID_STUDY_PART"].astype(str)
val_df["ID_STUDY_PART"] = val_df["ID_STUDY_PART"].astype(str)
test_df["ID_STUDY_PART"] = test_df["ID_STUDY_PART"].astype(str)

train_p = train_df["ID_STUDY_PART"].unique()
val_p = val_df["ID_STUDY_PART"].unique()
test_p = test_df["ID_STUDY_PART"].unique()

print("\nDataset checks:")
print("Total patients in patient_df:", len(patient_df))
print("Train patients:", len(train_p))
print("Val patients:", len(val_p))
print("Test patients:", len(test_p))

rf_cols = [
    c for c in patient_df.columns
    if c.startswith("DIAGN_") or c.startswith("RISK_")
]
rf_cols = [c for c in rf_cols if "DATE" not in c.upper()]

feature_basic = ["AgeAtECG", "GENDER"]
feature_full = ["AgeAtECG", "GENDER"] + rf_cols

print("\nBasic features:", feature_basic)
print("Number full features:", len(feature_full))

train_mask = patient_df["ID_STUDY_PART"].isin(train_p)
val_mask = patient_df["ID_STUDY_PART"].isin(val_p)
test_mask = patient_df["ID_STUDY_PART"].isin(test_p)

print("\nPatient-level label distribution:")
for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
    print(f"\n{split_name}:")
    print(patient_df.loc[mask, "y"].value_counts())

def run_model(model_name, features, model_builder):
    print("\n" + "=" * 60)
    print(f"Running {model_name}")
    print("=" * 60)

    X_train = patient_df.loc[train_mask, features].copy().fillna(0)
    y_train = patient_df.loc[train_mask, "y"].copy()

    X_val = patient_df.loc[val_mask, features].copy().fillna(0)
    y_val = patient_df.loc[val_mask, "y"].copy()

    X_test = patient_df.loc[test_mask, features].copy().fillna(0)
    y_test = patient_df.loc[test_mask, "y"].copy()

    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("n_features:", len(features))

    print("\nMissing values after fill:")
    print("Train:", X_train.isna().sum().sum())
    print("Val:  ", X_val.isna().sum().sum())
    print("Test: ", X_test.isna().sum().sum())

    model = make_pipeline(
        StandardScaler(),
        model_builder()
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_pred = (val_probs >= 0.5).astype(int)
    test_pred = (test_probs >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)

    print(f"\n{model_name} Val AUC:  {val_auc:.4f}")
    print(f"{model_name} Test AUC: {test_auc:.4f}")

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, digits=4))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, digits=4))

    for split_name, y_true, probs, preds in [
        ("val", y_val, val_probs, val_pred),
        ("test", y_test, test_probs, test_pred),
    ]:
        auc = roc_auc_score(y_true, probs)

        fpr, tpr, _ = roc_curve(y_true, probs)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{split_name.upper()} ROC - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{model_name}_{split_name}_roc.png"), dpi=300)
        plt.close()

        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{split_name.upper()} Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{model_name}_{split_name}_confusion_matrix.png"), dpi=300)
        plt.close()

    pd.DataFrame({
        "ID_STUDY_PART": patient_df.loc[val_mask, "ID_STUDY_PART"].values,
        "y_true": y_val.values,
        "y_prob": val_probs,
        "y_pred_0.5": val_pred,
    }).to_csv(os.path.join(OUT_DIR, f"{model_name}_val_predictions.csv"), index=False)

    pd.DataFrame({
        "ID_STUDY_PART": patient_df.loc[test_mask, "ID_STUDY_PART"].values,
        "y_true": y_test.values,
        "y_prob": test_probs,
        "y_pred_0.5": test_pred,
    }).to_csv(os.path.join(OUT_DIR, f"{model_name}_test_predictions.csv"), index=False)

    logreg = model.named_steps["logisticregression"]
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": logreg.coef_[0],
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(os.path.join(OUT_DIR, f"{model_name}_coefficients.csv"), index=False)

    return {
        "model": model_name,
        "n_features": len(features),
        "val_patient_auc": val_auc,
        "test_patient_auc": test_auc,
    }

results = []

results.append(
    run_model("logreg_basic_6mo", feature_basic, build_logistic_basic)
)

results.append(
    run_model("logreg_full_6mo", feature_full, build_logistic_full)
)

results_df = pd.DataFrame(results)
results_path = os.path.join(OUT_DIR, "results_logistic_6mo.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(results_df)
print("\nSaved summary to:", results_path)
print("Done.")