import tensorflow as tf  # type: ignore
from tensorflow.keras import models  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import backend  # type: ignore
from tensorflow.keras import utils  # type: ignore
from tensorflow.keras.applications import Xception  # type: ignore


def CNNmodel(
    num_labels,
    input_shape,
    base_filters,
    kernel_sizes,
    pool_sizes,
    base_dense_width,
    dropouts,
):
    inputs = layers.Input(shape=input_shape)

    x = layers.BatchNormalization()(inputs)

    ks = kernel_sizes[0]
    ps = pool_sizes[0]
    nf = base_filters * 1
    dr = dropouts[0]
    x = layers.Conv2D(nf, kernel_size=ks, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=ps)(x)
    x = layers.Dropout(dr)(x)

    ks = kernel_sizes[1]
    ps = pool_sizes[1]
    nf = base_filters * 2
    dr = dropouts[1]
    x = layers.Conv2D(nf, kernel_size=ks, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=ps)(x)
    x = layers.Dropout(dr)(x)

    ks = kernel_sizes[2]
    ps = pool_sizes[2]
    nf = base_filters * 3
    x = layers.Conv2D(nf, kernel_size=ks, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=ps)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(base_dense_width * 2, activation="relu")(x)
    x = layers.Dense(base_dense_width * 1, activation="relu")(x)

    output = layers.Dense(num_labels, activation="softmax")(x)

    model = models.Model(inputs=[inputs], outputs=[output], name="CNNmodel")

    return model


def AttRNNmodel(num_labels, input_shape, rnn_func=layers.LSTM):

    inputs = layers.Input(shape=input_shape, name="input")

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs
    # x = layers.Permute((2, 1, 3))(x)

    x = layers.BatchNormalization()(inputs)

    # x = layers.Conv2D(10, (5, 1), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (5, 1), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    # x = layers.Conv2D(1, (5, 1), activation="relu", padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(1, (5, 1), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = layers.Lambda(lambda q: backend.squeeze(q, -1), name="squeeze_last_dim")(x)

    x = layers.Bidirectional(rnn_func(64, return_sequences=True))(x)
    # [b_s, seq_len, vec_dim]
    x = layers.Bidirectional(rnn_func(64, return_sequences=True))(x)
    # [b_s, seq_len, vec_dim]

    xFirst = layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = layers.Dense(128)(xFirst)

    # dot product attention
    attScores = layers.Dot(axes=[1, 2])([query, x])
    attScores = layers.Softmax(name="attSoftmax")(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = layers.Dense(64, activation="relu")(attVector)
    x = layers.Dense(32)(x)

    output = layers.Dense(num_labels, activation="softmax", name="output")(x)

    model = models.Model(inputs=[inputs], outputs=[output])

    return model


def AttentionModel(
    num_labels,
    input_shape,
    conv_sizes,
    dropout,
    kernel_sizes,
    lstm_units,
    att_sample,
    query_style,
    dense_width,
):
    inputs = layers.Input(shape=input_shape, name="input")
    # (?, mel_dim, time_steps, 1)

    x = layers.Permute((2, 1, 3))(inputs)

    # x = layers.BatchNormalization()(inputs)
    x = layers.BatchNormalization()(x)

    for cs, ks in zip(conv_sizes, kernel_sizes):
        # a cs==0 removes the layer
        if cs > 0:
            x = layers.Conv2D(cs, ks, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            # (?, mel_dim, time_steps, cs)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)

    # the last conv_sizes is always 1 so now the shape is
    # (?, mel_dim, time_steps, 1)

    # remove the last dimension
    x = layers.Lambda(lambda q: backend.squeeze(q, axis=-1), name="squeeze_last_dim")(x)
    # (?, mel_dim, time_steps)

    for units in lstm_units:
        if units > 0:
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
            # (?, mel_dim, units * 2)

            # save the dimension of the last LSTM layer
            last_lstm_dim = units * 2

    if att_sample == "last":
        sample_index = -1
    elif att_sample == "mid":
        sample_index = last_lstm_dim // 2

    xFirst = layers.Lambda(lambda q: q[:, sample_index], name="x_first")(x)
    # (?, units * 2)

    if query_style == "dense01":
        query = layers.Dense(last_lstm_dim, name="query")(xFirst)
        # (?, units * 2)

    attScores = layers.Dot(axes=[1, 2], name="att_scores_dot")([query, x])
    # (?, mel_dim)
    attScores = layers.Softmax(name="att_softmax")(attScores)
    # (?, mel_dim)

    attVector = layers.Dot(axes=[1, 1], name="att_vector_dot")([attScores, x])
    # (?, units * 2)

    x = layers.Dense(dense_width * 2, activation="relu")(attVector)
    x = layers.Dense(dense_width)(x)

    output = layers.Dense(num_labels, activation="softmax", name="output")(x)
    model = models.Model(inputs=[inputs], outputs=[output])

    return model


def TRAmodel(num_labels, input_shape, dense_widths, dropout, data):
    """"""

    # load weights pre-trained on ImageNet
    # do not include the ImageNet classifier at the top
    base_model = Xception(
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
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    norm_layer.adapt(data["training"])

    x = norm_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)  # Regularize with dropout

    if dense_widths[0] > 0:
        x = tf.keras.layers.Dense(dense_widths[0], activation="relu")(x)

    if dense_widths[1] > 0:
        x = tf.keras.layers.Dense(dense_widths[1], activation="relu")(x)

    outputs = tf.keras.layers.Dense(num_labels, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs], name="TRAmodel")

    return model, base_model


def test_attention_model():
    """"""
    mp = {}
    mp["num_labels"] = 4
    mp["input_shape"] = (20, 30, 1)
    mp["conv_sizes"] = [10, 0, 1]
    mp["dropout"] = 0
    mp["kernel_sizes"] = [(5, 1), (5, 1), (5, 1)]
    mp["lstm_units"] = [64, 64]
    mp["att_sample"] = "last"
    mp["query_style"] = "dense01"
    mp["dense_width"] = 32

    attention_model = AttentionModel(**mp)
    attention_model.summary()

    model_pic_name = "plot_models/attention_model_01.png"
    utils.plot_model(attention_model, model_pic_name, show_shapes=True, dpi=400)


if __name__ == "__main__":
    test_attention_model()
