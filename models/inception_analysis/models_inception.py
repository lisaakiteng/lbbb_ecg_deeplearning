"""
models_inception.py

"""

from tensorflow import keras
from tensorflow.keras import layers


def conv1d_same(x, filters, kernel_size, stride=1, name=None):
    return layers.Conv1D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=name,
    )(x)


def inception_block_1d(
    x,
    nb_filters=32,
    kernel_sizes=(39, 19, 9),
    bottleneck_size=32,
    name="inception_block",
):
    """
    a version like PTB-XL InceptionBlock1d

    Branches:
    - bottleneck 1x1 conv -> Conv1D k=39
    - bottleneck 1x1 conv -> Conv1D k=19
    - bottleneck 1x1 conv -> Conv1D k=9
    - MaxPool1D -> 1x1 Conv1D
    """

    input_tensor = x
    in_channels = x.shape[-1]

    if bottleneck_size > 0:
        bottled = conv1d_same(
            x,
            bottleneck_size,
            kernel_size=1,
            name=f"{name}_bottleneck",
        )
    else:
        bottled = x

    conv_branches = []

    for i, k in enumerate(kernel_sizes):
        branch = conv1d_same(
            bottled,
            nb_filters,
            kernel_size=k,
            name=f"{name}_conv_k{k}",
        )
        conv_branches.append(branch)

    pool_branch = layers.MaxPooling1D(
        pool_size=3,
        strides=1,
        padding="same",
        name=f"{name}_maxpool",
    )(input_tensor)

    pool_branch = conv1d_same(
        pool_branch,
        nb_filters,
        kernel_size=1,
        name=f"{name}_pool_conv",
    )

    x = layers.Concatenate(name=f"{name}_concat")(conv_branches + [pool_branch])
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)

    return x


def shortcut_1d(input_tensor, output_tensor, name="shortcut"):
    """
    Keras version of PTB-XL Shortcut1d.
    Projects input channels to match output channels, then adds.
    """

    out_channels = output_tensor.shape[-1]

    shortcut = conv1d_same(
        input_tensor,
        out_channels,
        kernel_size=1,
        name=f"{name}_conv",
    )
    shortcut = layers.BatchNormalization(name=f"{name}_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([output_tensor, shortcut])
    x = layers.ReLU(name=f"{name}_relu")(x)

    return x


def inception_backbone_1d(
    input_shape=(12, 5000),
    kernel_size=40,
    depth=6,
    bottleneck_size=32,
    nb_filters=32,
    use_residual=True,
    name="inception_backbone",
):
    """
    Keras adaptation of PTB-XL InceptionBackbone.

    PTB-XL code:
    kernel_size=40 -> [39, 19, 9]
    depth=6
    residual every 3 blocks
    """

    assert depth % 3 == 0, "depth must be divisible by 3"

    inp = keras.Input(shape=input_shape, name="ecg_input")

    # Your ECGs are (leads, time), Keras Conv1D expects (time, channels)
    x = layers.Permute((2, 1), name="to_time_channels")(inp)

    kernel_sizes = [
        k - 1 if k % 2 == 0 else k
        for k in [kernel_size, kernel_size // 2, kernel_size // 4]
    ]

    input_res = x

    for d in range(depth):
        x = inception_block_1d(
            x,
            nb_filters=nb_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_size=bottleneck_size,
            name=f"inception_block_{d+1}",
        )

        if use_residual and d % 3 == 2:
            x = shortcut_1d(
                input_res,
                x,
                name=f"shortcut_{(d // 3) + 1}",
            )
            input_res = x

    return keras.Model(inp, x, name=name)


def inception1d_ptbxl_ecg(
    input_shape=(12, 5000),
    kernel_size=40,
    depth=6,
    bottleneck_size=32,
    nb_filters=32,
    use_residual=True,
    head_dropout=0.5,
):
    """
    ECG-only Inception1D model.

    This keeps the PTB-XL Inception1D backbone design,
    but changes the classifier output to one binary CRT logit.
    """

    inp = keras.Input(shape=input_shape, name="ecg")

    backbone = inception_backbone_1d(
        input_shape=input_shape,
        kernel_size=kernel_size,
        depth=depth,
        bottleneck_size=bottleneck_size,
        nb_filters=nb_filters,
        use_residual=use_residual,
        name="inception1d_ptbxl_backbone",
    )

    x = backbone(inp)

    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = layers.Concatenate(name="concat_pool")([avg_pool, max_pool])

    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(head_dropout, name="head_dropout")(x)

    # Single binary logit
    out = layers.Dense(1, name="crt_logit")(x)

    return keras.Model(inp, out, name="inception1d_ptbxl_ecg")