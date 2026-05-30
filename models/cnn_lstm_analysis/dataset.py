# =========================
# 03_dataset.py
# =========================

# -------- Imports --------
import numpy as np
import tensorflow as tf


# -------- Pack ECG Leads --------
def pack_leads(df_split, lead_cols):
    """
    Combine 12 lead columns into one packed string per ECG.
    """
    packed = df_split[lead_cols].astype(str).agg("|".join, axis=1).values
    y = df_split["y"].astype(np.float32).values
    pat = df_split["ID_STUDY_PART"].values  # for patient-level evaluation
    return packed, y, pat


# -------- Parse Packed ECG --------
def parse_packed_ecg(packed_str, y):
    """
    Convert packed string → (12, 5000) tensor
    """

    # Split into 12 lead strings
    leads = tf.strings.split(packed_str, sep="|")
    leads = leads[:12]

    # Split each lead into samples
    samples = tf.strings.split(leads, sep="^")

    # Convert to fixed tensor shape
    samples = samples.to_tensor(default_value="0", shape=[12, 5000])

    # Convert to float
    x = tf.strings.to_number(samples, out_type=tf.float32)
    x.set_shape([12, 5000])

    return x, tf.cast(y, tf.float32)


# -------- Create TF Dataset --------
def make_ds(X, y, batch_size=32, training=False, cache=True, cache_path=None):
    """
    Create TensorFlow dataset from packed ECG strings.
    """

    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        ds = ds.shuffle(5000, seed=0, reshuffle_each_iteration=True)

        # Speed optimization (non-deterministic order)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

    # Parse ECG strings → tensors
    ds = ds.map(parse_packed_ecg, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache (important for performance)
    if cache:
        ds = ds.cache(cache_path) if cache_path else ds.cache()

    # Batch + prefetch
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds