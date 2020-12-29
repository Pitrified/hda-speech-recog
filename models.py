from tensorflow.keras import models  # type: ignore
from tensorflow.keras import layers  # type: ignore


def CNNmodel(num_labels, input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.BatchNormalization()(inputs)

    x = layers.Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.03)(x)

    x = layers.Conv2D(128, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.01)(x)

    x = layers.Conv2D(256, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    output = layers.Dense(num_labels, activation="softmax")(x)

    model = models.Model(inputs=[inputs], outputs=[output], name="CNNmodel")

    return model
