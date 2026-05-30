# =========================
# 02_split_6mo.py
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split

# -------- Paths --------
INPUT_PATH = "/data/leuven/376/vsc37666/lbbb_data/df500_clean_6mo.parquet"

TRAIN_PATH = "/data/leuven/376/vsc37666/lbbb_data/train.parquet"
VAL_PATH   = "/data/leuven/376/vsc37666/lbbb_data/val.parquet"
TEST_PATH  = "/data/leuven/376/vsc37666/lbbb_data/test.parquet"

# -------- Load Corrected Dataset --------
print("Loading corrected 6-month dataset...")
df500 = pd.read_parquet(INPUT_PATH)

# Fix Arrow/string dtype issue
df500["ID_STUDY_PART"] = df500["ID_STUDY_PART"].astype(str)

print(f"Total ECG rows: {len(df500)}")
print(f"Total ECGs: {df500['ID_STUDY_ECG'].nunique()}")
print(f"Total patients: {df500['ID_STUDY_PART'].nunique()}")

print("\nOverall label distribution:")
print(df500["y"].value_counts())

# -------- Patient-Level Split --------
print("\nPerforming patient-level split 60/20/20...")

patients = df500["ID_STUDY_PART"].drop_duplicates().to_numpy()

train_p, test_p = train_test_split(
    patients,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

train_p, val_p = train_test_split(
    train_p,
    test_size=0.25,
    random_state=0,
    shuffle=True,
)

# -------- Create Split DataFrames --------
train_df = df500[df500["ID_STUDY_PART"].isin(train_p)].reset_index(drop=True)
val_df   = df500[df500["ID_STUDY_PART"].isin(val_p)].reset_index(drop=True)
test_df  = df500[df500["ID_STUDY_PART"].isin(test_p)].reset_index(drop=True)

# -------- Leakage Check --------
print("\nChecking patient leakage...")

assert set(train_df["ID_STUDY_PART"]).isdisjoint(val_df["ID_STUDY_PART"])
assert set(train_df["ID_STUDY_PART"]).isdisjoint(test_df["ID_STUDY_PART"])
assert set(val_df["ID_STUDY_PART"]).isdisjoint(test_df["ID_STUDY_PART"])

print("No patient leakage detected.")

# -------- Summary --------
print("\nSplit summary:")

for name, split_df in [
    ("Train", train_df),
    ("Val", val_df),
    ("Test", test_df),
]:
    print(f"\n{name}:")
    print(f"Rows: {len(split_df)}")
    print(f"ECGs: {split_df['ID_STUDY_ECG'].nunique()}")
    print(f"Patients: {split_df['ID_STUDY_PART'].nunique()}")

    print("ECG-level label counts:")
    print(split_df["y"].value_counts())

    patient_labels = (
        split_df.groupby("ID_STUDY_PART")["y"]
        .max()
        .value_counts()
    )

    print("Patient-level label counts:")
    print(patient_labels)

# -------- Save --------
print("\nSaving corrected splits...")

train_df.to_parquet(TRAIN_PATH, index=False)
val_df.to_parquet(VAL_PATH, index=False)
test_df.to_parquet(TEST_PATH, index=False)

print("Saved to:")
print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print("\nDone.")