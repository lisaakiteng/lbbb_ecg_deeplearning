import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ============================================================
# CONFIG
# ============================================================

models = {
    "Fusion Inception": {
        "pred_dir": "/data/leuven/376/vsc37666/lbbb_outputs/outputs_inception_fusion_multiseed/predictions"
    },
    "Fusion Cardioformer": {
        "pred_dir": "/data/leuven/376/vsc37666/lbbb_outputs/outputs_cardioformer_like_fusion_multiseed/predictions"
    },
}

SAVE_DIR = "calibration_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# PLOT CALIBRATION CURVES
# ============================================================

plt.figure(figsize=(8, 8))

results = []

for model_name, cfg in models.items():

    files = sorted(
        glob.glob(
            os.path.join(
                cfg["pred_dir"],
                "*val_patient_mean*.csv"
            )
        )
    )

    if len(files) == 0:
        print(f"No prediction files found for {model_name}")
        continue

    # Use first matching prediction file
    df = pd.read_csv(files[0])

    y_true = df["y"].values
    y_prob = df["pred"].values

    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    brier = brier_score_loss(y_true, y_prob)

    plt.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        label=f"{model_name} (Brier={brier:.3f})"
    )

    results.append({
        "Model": model_name,
        "Prediction file": files[0],
        "Brier Score": brier
    })


plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed fraction of positives")
plt.title("Calibration Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, "calibration_curves_inception_cardioformer.png")

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE BRIER SCORES
# ============================================================

results_df = pd.DataFrame(results)

brier_path = os.path.join(
    SAVE_DIR,
    "brier_scores_inception_cardioformer.csv"
)

results_df.to_csv(
    brier_path,
    index=False
)

print(results_df)
print(f"\nSaved plot to: {plot_path}")
print(f"Saved Brier scores to: {brier_path}")