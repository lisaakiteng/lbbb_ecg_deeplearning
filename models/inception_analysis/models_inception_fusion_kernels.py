from tensorflow import keras
from tensorflow.keras import layers


def inception_block(x, filters=32, bottleneck=32, kernel_sizes=(9, 19, 39), name="inception"):
    shortcut = x

    if bottleneck > 0:
        x = layers.Conv1D(bottleneck, 1, padding="same", use_bias=False, name=f"{name}_bottleneck_conv")(x)
        x = layers.BatchNormalization(name=f"{name}_bottleneck_bn")(x)
        x = layers.ReLU(name=f"{name}_bottleneck_relu")(x)

    branches = []
    for k in kernel_sizes:
        b = layers.Conv1D(filters, k, padding="same", use_bias=False, name=f"{name}_conv{k}")(x)
        branches.append(b)

    b_pool = layers.MaxPooling1D(pool_size=3, strides=1, padding="same", name=f"{name}_maxpool")(x)
    b_pool = layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_pool_conv")(b_pool)

    x = layers.Concatenate(name=f"{name}_concat")(branches + [b_pool])
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv1D(int(x.shape[-1]), 1, padding="same", use_bias=False, name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out")(x)

    return x


def inception_light_backbone_kernels(input_len=5000, kernel_sizes=(9, 19, 39)):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")

    x = layers.Permute((2, 1), name="to_time_channels")(ecg_in)

    x = layers.Conv1D(32, 15, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = inception_block(x, filters=32, bottleneck=32, kernel_sizes=kernel_sizes, name="inception_block1")
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = inception_block(x, filters=32, bottleneck=32, kernel_sizes=kernel_sizes, name="inception_block2")
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = inception_block(x, filters=64, bottleneck=32, kernel_sizes=kernel_sizes, name="inception_block3")
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    x = layers.Concatenate(name="concat_avg_max_pool")([avg_pool, max_pool])

    return keras.Model(
        ecg_in,
        x,
        name=f"inception_light_backbone_kernels_{'_'.join(map(str, kernel_sizes))}"
    )


def fusion_inception_light_kernels(n_tab, input_len=5000, kernel_sizes=(9, 19, 39), dropout=0.3):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = inception_light_backbone_kernels(
        input_len=input_len,
        kernel_sizes=kernel_sizes
    )(ecg_in)

    t = layers.Dense(64, name="tab_dense1")(tab_in)
    t = layers.BatchNormalization(name="tab_bn1")(t)
    t = layers.ReLU(name="tab_relu1")(t)
    t = layers.Dropout(0.3, name="tab_dropout1")(t)

    t = layers.Dense(32, name="tab_dense2")(t)
    t = layers.BatchNormalization(name="tab_bn2")(t)
    t = layers.ReLU(name="tab_relu2")(t)

    x = layers.Concatenate(name="fusion_concat")([ecg_feat, t])

    x = layers.Dense(64, activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(dropout, name="fusion_dropout1")(x)

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(
        [ecg_in, tab_in],
        out,
        name=f"fusion_inception_light_kernels_{'_'.join(map(str, kernel_sizes))}"
    )


def fusion_inception_light_original(n_tab):
    return fusion_inception_light_kernels(n_tab=n_tab, kernel_sizes=(9, 19, 39), dropout=0.3)


def fusion_inception_light_small(n_tab):
    return fusion_inception_light_kernels(n_tab=n_tab, kernel_sizes=(5, 11, 21), dropout=0.3)


def fusion_inception_light_wide(n_tab):
    return fusion_inception_light_kernels(n_tab=n_tab, kernel_sizes=(15, 31, 63), dropout=0.3)
    
def fusion_inception_light_tiny(n_tab):
    return fusion_inception_light_kernels(
        n_tab=n_tab,
        kernel_sizes=(3, 7, 15),dropout=0.3,)


def fusion_inception_light_mid_small(n_tab):
    return fusion_inception_light_kernels(
        n_tab=n_tab,
        kernel_sizes=(5, 9, 19),
        dropout=0.3,
    )


def fusion_inception_light_mixed_small(n_tab):
    return fusion_inception_light_kernels(
        n_tab=n_tab,
        kernel_sizes=(3, 9, 21),
        dropout=0.3,
    )
    
def inception_light_backbone_5blocks_kernels(
    input_len=5000,
    kernel_sizes=(5, 11, 21),
):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")

    x = layers.Permute((2, 1), name="to_time_channels")(ecg_in)

    x = layers.Conv1D(
        32,
        15,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = inception_block(
        x,
        filters=32,
        bottleneck=32,
        kernel_sizes=kernel_sizes,
        name="inception_block1",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = inception_block(
        x,
        filters=32,
        bottleneck=32,
        kernel_sizes=kernel_sizes,
        name="inception_block2",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = inception_block(
        x,
        filters=64,
        bottleneck=32,
        kernel_sizes=kernel_sizes,
        name="inception_block3",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    x = inception_block(
        x,
        filters=64,
        bottleneck=32,
        kernel_sizes=kernel_sizes,
        name="inception_block4",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool4")(x)

    x = inception_block(
        x,
        filters=128,
        bottleneck=32,
        kernel_sizes=kernel_sizes,
        name="inception_block5",
    )

    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    x = layers.Concatenate(name="concat_avg_max_pool")([avg_pool, max_pool])

    return keras.Model(
        ecg_in,
        x,
        name="inception_light_5blocks_backbone_k5_11_21",
    )


def fusion_inception_light_5blocks_small(n_tab, input_len=5000, dropout=0.3):
    ecg_in = keras.Input(shape=(12, input_len), name="ecg")
    tab_in = keras.Input(shape=(n_tab,), name="tab")

    ecg_feat = inception_light_backbone_5blocks_kernels(
        input_len=input_len,
        kernel_sizes=(5, 11, 21),
    )(ecg_in)

    t = layers.Dense(64, name="tab_dense1")(tab_in)
    t = layers.BatchNormalization(name="tab_bn1")(t)
    t = layers.ReLU(name="tab_relu1")(t)
    t = layers.Dropout(0.3, name="tab_dropout1")(t)

    t = layers.Dense(32, name="tab_dense2")(t)
    t = layers.BatchNormalization(name="tab_bn2")(t)
    t = layers.ReLU(name="tab_relu2")(t)

    x = layers.Concatenate(name="fusion_concat")([ecg_feat, t])

    x = layers.Dense(64, activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(dropout, name="fusion_dropout1")(x)

    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(
        [ecg_in, tab_in],
        out,
        name="fusion_inception_light_5blocks_k5_11_21",
    )