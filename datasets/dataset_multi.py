import tensorflow as tf
from dataset import parse_packed_ecg  # reuse existing logic

def make_ds_multi(packed, tab, y, batch_size=32, training=False):

    def parse_ecg_and_tab(p, t, y):
        ecg, y = parse_packed_ecg(p, y)
        return (ecg, tf.cast(t, tf.float32)), y

    ds = tf.data.Dataset.from_tensor_slices((packed, tab, y))

    if training:
        ds = ds.shuffle(5000, seed=0)

    ds = ds.map(parse_ecg_and_tab, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds