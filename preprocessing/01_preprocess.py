# =========================
# 01_preprocess.py
# =========================

import numpy as np
import pandas as pd

# -------- Paths --------
DATA_PATH = "/data/leuven/376/vsc37666/lbbb_data/ecg_lbbb.xlsx"
OUTPUT_PATH = "/data/leuven/376/vsc37666/lbbb_data/df500_clean_6mo.parquet"
EXTREME_OUTPUT_PATH = "/data/leuven/376/vsc37666/lbbb_data/extreme_ecgs_removed_6mo.csv"

# -------- Load Data --------
print("Loading data...")
xls = pd.ExcelFile(DATA_PATH)
df = pd.read_excel(xls, "ECG_Waves_Clincical")

print(f"Initial shape: {df.shape}")

# -------- Lead Columns --------
lead_cols = [c for c in df.columns if c.startswith("Lead_")]

# -------- Helper Functions --------
def parse_lead_str(s, sep="^", dtype=np.float32):
    return np.array(str(s).split(sep), dtype=dtype)

def parse_ecg_row(row, lead_cols, sep="^", dtype=np.float32):
    leads = [parse_lead_str(row[c], sep=sep, dtype=dtype) for c in lead_cols]
    return np.vstack(leads)

# -------- Remove Extreme ECGs --------
print("Removing extreme ECGs...")

TH = 2000
hits = []

for idx, row in df.iterrows():
    ecg = parse_ecg_row(row, lead_cols)
    ext = ((ecg > TH) | (ecg < -TH)).any(axis=1)

    if ext.any():
        hits.append({
            "row_idx": idx,
            "patient_id": row["ID_STUDY_PART"] if "ID_STUDY_PART" in df.columns else idx,
            "n_extreme_leads": int(ext.sum()),
            "extreme_lead_ids": np.where(ext)[0].tolist(),
            "max_val": float(ecg.max()),
            "min_val": float(ecg.min()),
        })

extreme_df = pd.DataFrame(hits)

print(f"Total ECGs removed (>|2000|): {len(extreme_df)}")

if len(extreme_df) > 0:
    extreme_df.to_csv(EXTREME_OUTPUT_PATH, index=False)
    df = df.drop(index=extreme_df["row_idx"].to_numpy()).copy()
else:
    df = df.copy()

print(f"After extreme removal: {df.shape}")

# -------- Filter to 500 Hz --------
print("Filtering 500 Hz ECGs...")
df500 = df[df["ECG_Frequency_Hz"] == 500].copy()

# -------- Dates and CRT Columns --------
df500["DATE_ECG"] = pd.to_datetime(df500["DATE_ECG"], errors="coerce")
df500["OPER_CRT_Date_First"] = pd.to_datetime(df500["OPER_CRT_Date_First"], errors="coerce")

df500["OPER_CRT"] = (
    pd.to_numeric(df500["OPER_CRT"], errors="coerce")
    .fillna(0)
    .astype(int)
)

# -------- Ensure Correct Waveform Length --------
print("Filtering ECGs with correct length 5000...")

lead_cols = [c for c in df500.columns if c.lower().startswith("lead_")]
lens = df500[lead_cols[0]].astype(str).str.split("^").str.len()
df500 = df500[lens == 5000].copy()

# -------- Old Label for Checking Only --------
df500["y_old_ever_crt"] = (
    df500.groupby("ID_STUDY_PART")["OPER_CRT"]
    .transform("max")
    .fillna(0)
    .astype(int)
)

# -------- Final 6-Month ECG-Level Label --------
df500["days_ecg_to_crt"] = (
    df500["OPER_CRT_Date_First"] - df500["DATE_ECG"]
).dt.days

df500["y"] = (
    (df500["OPER_CRT"] == 1)
    & (df500["days_ecg_to_crt"] >= 0)
    & (df500["days_ecg_to_crt"] <= 180)
).astype(int)

# -------- Checks --------
print("\nFinal dataset summary:")
print(f"ECGs: {df500['ID_STUDY_ECG'].nunique()}")
print(f"Patients: {df500['ID_STUDY_PART'].nunique()}")

print("\nOld ever-CRT label counts:")
print(df500["y_old_ever_crt"].value_counts())

print("\nNew 6-month ECG-level label counts:")
print(df500["y"].value_counts())

old_positive = (df500["y_old_ever_crt"] == 1).sum()
changed = df500[(df500["y_old_ever_crt"] == 1) & (df500["y"] == 0)]

print("\nECGs previously positive but now negative:")
print(len(changed))

if old_positive > 0:
    print("\nPercentage of old positives changed:")
    print(len(changed) / old_positive * 100)

print("\nMissing CRT dates among old positives:")
print(df500.loc[df500["y_old_ever_crt"] == 1, "OPER_CRT_Date_First"].isna().sum())

# -------- Save --------
print("\nSaving cleaned 6-month dataset...")
df500.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved to: {OUTPUT_PATH}")
print("Done.")