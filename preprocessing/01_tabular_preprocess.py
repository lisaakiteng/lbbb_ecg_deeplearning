import pandas as pd
import numpy as np

DATA_DIR = "/data/leuven/376/vsc37666/lbbb_data/"

print("Loading corrected df500_clean_6mo...")
df = pd.read_parquet(DATA_DIR + "df500_clean_6mo.parquet")

df["ID_STUDY_PART"] = df["ID_STUDY_PART"].astype(str)

diagn_cols = df.filter(regex=r"(?i)^diagn_").columns.tolist()
risk_cols  = df.filter(regex=r"(?i)^risk_").columns.tolist()
rf_cols = sorted(set(diagn_cols + risk_cols))
rf_cols = [c for c in rf_cols if "DATE" not in c.upper()]

def dutch_yes_no_to_int(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x == "ja":
        return 1
    if x == "nee":
        return 0
    return np.nan

df[rf_cols] = df[rf_cols].apply(lambda col: col.map(dutch_yes_no_to_int))

agg_dict = {
    "AgeAtECG": "first",
    "GENDER": "first",
    "y": "max",
}

for c in rf_cols:
    agg_dict[c] = "max"

patient_df = df.groupby("ID_STUDY_PART").agg(agg_dict).reset_index()

patient_df[rf_cols] = patient_df[rf_cols].fillna(0)
patient_df["GENDER"] = patient_df["GENDER"].map({"M": 1, "F": 0})
patient_df["y"] = pd.to_numeric(patient_df["y"], errors="coerce").fillna(0).astype(int)

print("\nPatient-level label counts:")
print(patient_df["y"].value_counts())

print("\nSaving corrected patient_df.parquet...")
patient_df.to_parquet(DATA_DIR + "patient_df.parquet", index=False)
print("Done.")