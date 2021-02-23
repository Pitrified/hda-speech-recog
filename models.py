import tensorflow as tf  # type: ignore
from tensorflow.keras import models  # type: ignore
from tensorflow.keras import layers as L  # type: ignore
from tensorflow.keras import backend as K  # type: ignore
from tensorflow.keras import utils as U  # type: ignore

# from tensorflow.keras.applications import Xception  # type: ignore
from tensorflow.keras import applications  # type: ignore

from pathlib import Path


def CNNmodel(
    num_labels,
    input_shape,
    base_filters,
    kernel_sizes,
    pool_sizes,
    base_dense_width,
    dropouts,
):
    inputs = L.Input(shape=input_shape)

    x = L.BatchNormalization()(inputs)

    ks = kernel_sizes[0]
    ps = pool_sizes[0]
    nf = base_filters * 1
    dr = dropouts[0]
    x = L.Conv2D(nf, kernel_size=ks, activation="relu", padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D(pool_size=ps)(x)
    x = L.Dropout(dr)(x)

    ks = kernel_sizes[1]
    ps = pool_sizes[1]
    nf = base_filters * 2
    dr = dropouts[1]
    x = L.Conv2D(nf, kernel_size=ks, activation="relu", padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D(pool_size=ps)(x)
    x = L.Dropout(dr)(x)

    ks = kernel_sizes[2]
    ps = pool_sizes[2]
    nf = base_filters * 3
    x = L.Conv2D(nf, kernel_size=ks, activation="relu", padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D(pool_size=ps)(x)

    x = L.Flatten()(x)
    x = L.Dense(base_dense_width * 2, activation="relu")(x)
    x = L.Dense(base_dense_width * 1, activation="relu")(x)

    output = L.Dense(num_labels, activation="softmax")(x)

    model = models.Model(inputs=[inputs], outputs=[output], name="CNNmodel")

    return model


def AttentionModel(
    num_labels,
    input_shape,
    conv_sizes,
    dropout,
    kernel_sizes,
    lstm_units,
    query_style,
    dense_width,
):
    inputs = L.Input(shape=input_shape, name="input")
    # (?, mel_dim, time_steps, 1)

    x = L.Permute((2, 1, 3))(inputs)
    # (?, time_steps, mel_dim, 1)

    x = L.BatchNormalization()(x)

    for i, (cs, ks) in enumerate(zip(conv_sizes, kernel_sizes)):
        # a cs==0 skips the layers
        if cs > 0:
            x = L.Conv2D(
                filters=cs,
                kernel_size=ks,
                activation="relu",
                padding="same",
                name=f"conv2d_feature_{i}",
            )(x)
            x = L.BatchNormalization(name=f"batchnorm_feature_{i}")(x)
            # (?, time_steps, mel_dim, cs)
            if dropout > 0:
                x = L.Dropout(dropout, name=f"dropout_feature_{i}")(x)

    # the last conv_sizes is always 1 so now the shape is again
    # (?, time_steps, mel_dim, 1)

    # remove the last dimension
    x = L.Lambda(lambda q: K.squeeze(q, axis=-1), name="squeeze_last_dim")(x)
    # (?, time_steps, mel_dim)

    for units in lstm_units:
        if units > 0:
            x = L.Bidirectional(L.LSTM(units, return_sequences=True))(x)
            # (?, time_steps, units * 2)

            # save the dimension of the last LSTM layer
            last_lstm_dim = units * 2

    # the query starts from (?, time_steps, units * 2)
    # and ends with (?, units * 2)

    if query_style.startswith("dense"):

        # the params for the extraction
        if query_style == "dense01":
            sample_index = -1
        elif query_style == "dense02":
            sample_index = input_shape[1] // 2

        x_first = L.Lambda(lambda q: q[:, sample_index], name="x_first")(x)
        # (?, units * 2)
        query = L.Dense(last_lstm_dim, name="query")(x_first)
        # (?, units * 2)

    elif query_style.startswith("conv"):

        # the parameters for the conv net
        if query_style == "conv01":
            num_fil = [10, 10, 10]
            kern_sz = [(4, 4), (4, 4), (4, 4)]
            pool_sz = [(3, 3), (3, 3), (3, 3)]
            x_q = K.expand_dims(x, axis=-1)
            # (?, time_steps, units * 2, 1) conv needs an 'image'

        elif query_style == "conv02":
            num_fil = [5, 5, 5, 5]
            kern_sz = [(4, 4), (4, 4), (4, 4), (4, 4)]
            pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2)]
            x_q = K.expand_dims(x, axis=-1)
            # (?, time_steps, units * 2, 1) conv needs an 'image'

        elif query_style == "conv03":
            num_fil = [5, 5, 5, 5]
            kern_sz = [(4, 4), (4, 4), (4, 4), (4, 4)]
            pool_sz = [(2, 2), (2, 2), (2, 2), (2, 2)]
            x_q = inputs

        # build the CNN
        for i, (nf, ks, ps) in enumerate(zip(num_fil, kern_sz, pool_sz)):
            x_q = L.Conv2D(
                nf,
                kernel_size=ks,
                activation="relu",
                padding="same",
                name=f"conv2d_query_{i}",
            )(x_q)
            x_q = L.BatchNormalization(name=f"batchnorm_query_{i}")(x_q)
            x_q = L.MaxPooling2D(pool_size=ps, name=f"maxpool_query_{i}")(x_q)
            x_q = L.Dropout(dropout, name=f"dropout_query_{i}")(x_q)

        x_q = L.Flatten()(x_q)
        query = L.Dense(last_lstm_dim, name="query")(x_q)

    att_scores = L.Dot(axes=[1, 2], name="att_scores_dot")([query, x])
    att_scores = L.Softmax(name="att_softmax")(att_scores)
    # (?, time_steps)

    att_vector = L.Dot(axes=[1, 1], name="att_vector_dot")([att_scores, x])
    # (?, units * 2)

    x = L.Dense(dense_width * 2, activation="relu")(att_vector)
    x = L.Dense(dense_width)(x)

    output = L.Dense(num_labels, activation="softmax", name="output")(x)
    model = models.Model(inputs=[inputs], outputs=[output])

    return model


def TRAmodel(
    num_labels, input_shape, net_type, dense_widths, dropout, data_mean, data_variance
) -> tf.keras.models.Model:
    """"""

    if net_type == "TRA":
        pretrained_model = applications.Xception
    elif net_type == "TB0":
        pretrained_model = applications.EfficientNetB0
    elif net_type == "TB4":
        pretrained_model = applications.EfficientNetB4
    elif net_type == "TB7":
        pretrained_model = applications.EfficientNetB7
    elif net_type == "TD1":
        pretrained_model = applications.DenseNet121

    # load weights pre-trained on ImageNet
    # do not include the ImageNet classifier at the top
    base_model = pretrained_model(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False,
    )

    # freeze the base_model
    base_model.trainable = False

    # create new model on top
    inputs = tf.keras.Input(shape=input_shape)

    # normalize the data for xception
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Normalization
    norm_layer = L.experimental.preprocessing.Normalization(
        mean=data_mean, variance=data_variance
    )
    # norm_layer.adapt(data)

    x = norm_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = L.GlobalAveragePooling2D()(x)

    if dropout > 0:
        x = L.Dropout(dropout)(x)  # Regularize with dropout

    if dense_widths[0] > 0:
        x = L.Dense(dense_widths[0], activation="relu")(x)

    if dense_widths[1] > 0:
        x = L.Dense(dense_widths[1], activation="relu")(x)

    outputs = L.Dense(num_labels, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs], name="TRAmodel")

    return model, base_model


def test_attention_model():
    """"""
    mp = {}
    # mp["num_labels"] = 4
    # mp["input_shape"] = (20, 30, 1)
    # mp["conv_sizes"] = [10, 0, 1]
    # mp["dropout"] = 0
    # mp["kernel_sizes"] = [(5, 1), (5, 1), (5, 1)]
    # mp["lstm_units"] = [64, 64]
    # mp["att_sample"] = "last"
    # mp["query_style"] = "dense01"
    # mp["dense_width"] = 32

    # query_style = "conv03"
    query_style = "dense01"

    mp["num_labels"] = 8
    # mp["input_shape"] = (128, 32, 1)
    mp["input_shape"] = (80, 120, 1)
    # mp["input_shape"] = (128, 128, 1)
    # mp["input_shape"] = (20, 30, 1)
    # mp["conv_sizes"] = [10, 0, 1]
    mp["conv_sizes"] = [10, 10, 1]
    mp["dropout"] = 0.2
    mp["kernel_sizes"] = [(5, 1), (5, 1), (5, 1)]
    mp["lstm_units"] = [64, 64]
    # mp["att_sample"] = "mid"
    # mp["query_style"] = "dense01"
    mp["query_style"] = query_style
    mp["dense_width"] = 32

    attention_model = AttentionModel(**mp)
    attention_model.summary()

    model_folder = Path("plot_models")
    model_pic_name = model_folder / f"attention_model_{query_style}.png"
    U.plot_model(attention_model, model_pic_name, show_shapes=True, dpi=400)


def test_tra_model() -> None:
    """MAKEDOC: what is test_tra_model doing?"""

    net_type = "TD1"
    input_shape = (128, 128, 3)
    num_labels = 10
    dense_widths = [0, 0]
    dropout = 0.2
    data_mean = 0
    data_variance = 1

    model, base_model = TRAmodel(
        num_labels,
        input_shape,
        net_type,
        dense_widths,
        dropout,
        data_mean,
        data_variance,
    )
    model.summary(line_length=120)
    base_model.summary(line_length=120)


if __name__ == "__main__":
    test_attention_model()
    # test_tra_model()
