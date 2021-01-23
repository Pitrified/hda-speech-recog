from tensorflow.keras import models  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import backend  # type: ignore


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

    # 1st layer
    # x = layers.Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same")(x)
    # x = layers.Conv2D(20, kernel_size=(5, 1), activation="relu", padding="same")(x)
    # x = layers.Conv2D(20, kernel_size=(8, 2), activation="relu", padding="same")(x)
    # x = layers.Conv2D(64, kernel_size=(5, 1), activation="relu", padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    # x = layers.Dropout(0.03)(x)

    # 2nd layer
    # x = layers.Conv2D(128, kernel_size=(2, 2), activation="relu", padding="same")(x)
    # x = layers.Conv2D(40, kernel_size=(3, 3), activation="relu", padding="same")(x)
    # x = layers.Conv2D(40, kernel_size=(4, 4), activation="relu", padding="same")(x)
    # x = layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Dropout(0.01)(x)

    # 3rd layer
    # x = layers.Conv2D(256, kernel_size=(2, 2), activation="relu", padding="same")(x)
    # x = layers.Conv2D(80, kernel_size=(3, 3), activation="relu", padding="same")(x)
    # x = layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(base_dense_width * 2, activation="relu")(x)
    x = layers.Dense(base_dense_width * 1, activation="relu")(x)

    # x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dense(32, activation="relu")(x)

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
