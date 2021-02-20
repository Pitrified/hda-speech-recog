from pathlib import Path
import logging

from tensorflow.keras import layers as L  # type: ignore
from tensorflow.keras import utils as U  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import tensorflow as tf  # type: ignore

from keras_diagram import ascii_model


class AreaNet:
    @staticmethod
    def conv_module(x, num_filters, filters_shape, stride, padding="same"):
        # define a CONV => BN => RELU pattern
        x = L.Conv2D(num_filters, filters_shape, strides=stride, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)

        # return the block
        return x

    @staticmethod
    def build(input_shape, num_classes):
        inputs = L.Input(shape=input_shape)

        aa = inputs

        ## area conv

        # extract the areas of interest
        aa = AreaNet.conv_module(aa, 20, (3, 3), (1, 1))
        aa = AreaNet.conv_module(aa, 20, (3, 3), (1, 1))
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        aa = L.Dropout(0.2)(aa)

        aa = AreaNet.conv_module(aa, 30, (3, 3), (1, 1))
        aa = AreaNet.conv_module(aa, 30, (3, 3), (1, 1))
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        aa = L.Dropout(0.2)(aa)

        aa = AreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = AreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        aa = L.Dropout(0.2)(aa)

        aa = AreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = AreaNet.conv_module(aa, 80, (3, 3), (1, 1))
        aa = L.Dropout(0.2)(aa)

        aa = L.Lambda(lambda q: tf.math.reduce_mean(q, axis=-1, keepdims=True))(aa)

        # THIS IS WRONG AND CREATES A WHOLE LOT OF ONES
        # SO THE NET WAS WORKING WELL BUT ONLY USING THE TAIL
        aa = tf.nn.softmax(aa, axis=-1, name="area_values_softmax")

        aa = L.UpSampling2D(size=(8, 8), interpolation="nearest", name="area_values")(
            aa
        )

        x = L.Multiply()([aa, inputs])

        ## feature conv

        x = AreaNet.conv_module(x, 40, (3, 3), (1, 1))
        x = AreaNet.conv_module(x, 40, (3, 3), (1, 1))
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        x = L.Dropout(0.2)(x)

        x = AreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = AreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        x = L.Dropout(0.2)(x)

        x = AreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = AreaNet.conv_module(x, 160, (3, 3), (1, 1))
        x = L.Dropout(0.2)(x)

        ## classifier

        # softmax classifier
        x = L.GlobalAvgPool2D()(x)
        # x = L.Flatten()(x)
        x = L.Dense(num_classes)(x)
        x = L.Activation("softmax", name="output")(x)

        # create the model
        model = Model(inputs, x, name="area_net")

        # return the constructed network architecture
        return model


class SimpleNet:
    @staticmethod
    def conv_module(x, num_filters, filters_shape, stride, padding="same"):
        # define a CONV => BN => RELU pattern
        x = L.Conv2D(num_filters, filters_shape, strides=stride, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)

        # return the block
        return x

    @staticmethod
    def build(input_shape, num_classes, sim_type="1"):
        inputs = L.Input(shape=input_shape)

        x = inputs

        block_param_list = []

        if sim_type == "1":
            conv_params = []
            conv_params.append({"nf": 40, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 40, "fs": (3, 3), "st": (1, 1)})
            block_param_list.append({"cvp": conv_params, "ps": (2, 2), "dr": 0.2})

            conv_params = []
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            block_param_list.append({"cvp": conv_params, "ps": (2, 2), "dr": 0.2})

            conv_params = []
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 160, "fs": (3, 3), "st": (1, 1)})
            block_param_list.append({"cvp": conv_params, "ps": None, "dr": 0.2})

        elif sim_type == "2":
            conv_params = []
            conv_params.append({"nf": 40, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 40, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 40, "fs": (3, 3), "st": (1, 1)})
            block_param_list.append({"cvp": conv_params, "ps": (2, 2), "dr": 0.2})

            conv_params = []
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            block_param_list.append({"cvp": conv_params, "ps": (2, 2), "dr": 0.2})

            conv_params = []
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 80, "fs": (3, 3), "st": (1, 1)})
            conv_params.append({"nf": 160, "fs": (3, 3), "st": (1, 1)})
            block_param_list.append({"cvp": conv_params, "ps": None, "dr": 0.2})

        for block_params in block_param_list:

            # add convolutional modules
            for conv_params in block_params["cvp"]:
                num_filters = conv_params["nf"]
                filters_shape = conv_params["fs"]
                stride = conv_params["st"]
                x = SimpleNet.conv_module(x, num_filters, filters_shape, stride)

            # add pooling layer
            pool_size = block_params["ps"]
            if pool_size is not None:
                x = L.MaxPooling2D(pool_size=pool_size)(x)

            # add dropout layer
            dropout = block_params["dr"]
            if dropout > 0:
                x = L.Dropout(dropout)(x)

        # x = SimpleNet.conv_module(x, 40, (3, 3), (1, 1))
        # x = SimpleNet.conv_module(x, 40, (3, 3), (1, 1))
        # x = L.MaxPooling2D(pool_size=(2, 2))(x)
        # x = L.Dropout(0.2)(x)

        # x = SimpleNet.conv_module(x, 80, (3, 3), (1, 1))
        # x = SimpleNet.conv_module(x, 80, (3, 3), (1, 1))
        # x = L.MaxPooling2D(pool_size=(2, 2))(x)
        # x = L.Dropout(0.2)(x)

        # x = SimpleNet.conv_module(x, 80, (3, 3), (1, 1))
        # x = SimpleNet.conv_module(x, 160, (3, 3), (1, 1))
        # x = L.Dropout(0.2)(x)

        ## classifier

        # softmax classifier
        x = L.GlobalAvgPool2D()(x)
        # x = L.Flatten()(x)
        x = L.Dense(num_classes)(x)
        x = L.Activation("softmax", name="output")(x)

        # create the model
        model = Model(inputs, x, name="area_net")

        # return the constructed network architecture
        return model


class ActualAreaNet:
    @staticmethod
    def conv_module(x, num_filters, filters_shape, stride, padding="same"):
        # define a CONV => BN => RELU pattern
        x = L.Conv2D(num_filters, filters_shape, strides=stride, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)

        # return the block
        return x

    @staticmethod
    def build(input_shape, num_classes):
        inputs = L.Input(shape=input_shape)

        aa = inputs

        ## area conv

        # TODO: models can be layers, so split the query in separate part

        # extract the areas of interest
        aa = ActualAreaNet.conv_module(aa, 20, (3, 3), (1, 1))
        aa = ActualAreaNet.conv_module(aa, 20, (3, 3), (1, 1))
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        aa = L.Dropout(0.2)(aa)

        aa = ActualAreaNet.conv_module(aa, 30, (3, 3), (1, 1))
        aa = ActualAreaNet.conv_module(aa, 30, (3, 3), (1, 1))
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        aa = L.Dropout(0.2)(aa)

        aa = ActualAreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = ActualAreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        aa = L.Dropout(0.2)(aa)

        aa = ActualAreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = ActualAreaNet.conv_module(aa, 80, (3, 3), (1, 1))
        aa = L.Dropout(0.2)(aa)

        aa = L.Lambda(
            lambda q: tf.math.reduce_mean(q, axis=-1, keepdims=True),
            name="area_values_small",
        )(aa)

        # aa = tf.nn.softmax(aa, axis=-1, name="area_values_softmax")
        aa = L.Lambda(
            lambda q: tf.exp(q) / tf.reduce_sum(tf.exp(q)), name="area_values_softmax"
        )(aa)

        aa = L.UpSampling2D(size=(8, 8), interpolation="nearest", name="area_values")(
            aa
        )

        x = L.Multiply()([aa, inputs])

        ## feature conv

        x = ActualAreaNet.conv_module(x, 40, (3, 3), (1, 1))
        x = ActualAreaNet.conv_module(x, 40, (3, 3), (1, 1))
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        x = L.Dropout(0.2)(x)

        x = ActualAreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = ActualAreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        x = L.Dropout(0.2)(x)

        x = ActualAreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = ActualAreaNet.conv_module(x, 160, (3, 3), (1, 1))
        x = L.Dropout(0.2)(x)

        ## classifier

        # softmax classifier
        x = L.GlobalAvgPool2D()(x)
        # x = L.Flatten()(x)
        x = L.Dense(num_classes)(x)
        x = L.Activation("softmax", name="output")(x)

        # create the model
        model = Model(inputs, x, name="area_net")

        # return the constructed network architecture
        return model


class VerticalAreaNet:
    @staticmethod
    def conv_module(x, num_filters, filters_shape, stride, padding="same"):
        # define a CONV => BN => RELU pattern
        x = L.Conv2D(num_filters, filters_shape, strides=stride, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)

        # return the block
        return x

    @staticmethod
    def build(input_shape, num_classes):
        inputs = L.Input(shape=input_shape)

        aa = inputs

        ## area conv

        # TODO: models can be layers, so split the query in separate part

        # extract the areas of interest
        aa = VerticalAreaNet.conv_module(aa, 20, (3, 3), (1, 1))
        aa = VerticalAreaNet.conv_module(aa, 20, (3, 3), (1, 1))
        print(f"aa.shape: {aa.shape}")
        aa = L.MaxPooling2D(pool_size=(2, 1))(aa)
        print(f"aa.shape: {aa.shape}")
        aa = L.Dropout(0.2)(aa)

        aa = VerticalAreaNet.conv_module(aa, 30, (3, 3), (1, 1))
        aa = VerticalAreaNet.conv_module(aa, 30, (3, 3), (1, 1))
        print(f"aa.shape: {aa.shape}")
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        print(f"aa.shape: {aa.shape}")
        aa = L.Dropout(0.2)(aa)

        aa = VerticalAreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = VerticalAreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        print(f"aa.shape: {aa.shape}")
        aa = L.MaxPooling2D(pool_size=(2, 2))(aa)
        print(f"aa.shape: {aa.shape}")
        aa = L.Dropout(0.2)(aa)

        aa = VerticalAreaNet.conv_module(aa, 40, (3, 3), (1, 1))
        aa = VerticalAreaNet.conv_module(aa, 80, (3, 3), (1, 1))
        aa = L.Dropout(0.2)(aa)

        print(f"aa.shape: {aa.shape}")

        aa = L.Lambda(
            lambda q: tf.math.reduce_mean(q, axis=-1, keepdims=True),
            name="area_values_small_mean_feat",
        )(aa)
        print(f"aa.shape: {aa.shape}")

        aa = L.Lambda(
            lambda q: tf.math.reduce_mean(q, axis=1, keepdims=True),
            name="area_values_small_mean_freq",
        )(aa)
        print(f"aa.shape: {aa.shape}")

        # aa = tf.nn.softmax(aa, axis=-1, name="area_values_softmax")
        aa = L.Lambda(
            lambda q: tf.exp(q) / tf.reduce_sum(tf.exp(q)), name="area_values_softmax"
        )(aa)

        vert_upsample = input_shape[0]
        aa = L.UpSampling2D(
            size=(vert_upsample, 4), interpolation="nearest", name="area_values"
        )(aa)

        x = L.Multiply()([aa, inputs])

        ## feature conv

        x = VerticalAreaNet.conv_module(x, 40, (3, 3), (1, 1))
        x = VerticalAreaNet.conv_module(x, 40, (3, 3), (1, 1))
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        x = L.Dropout(0.2)(x)

        x = VerticalAreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = VerticalAreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = L.MaxPooling2D(pool_size=(2, 2))(x)
        x = L.Dropout(0.2)(x)

        x = VerticalAreaNet.conv_module(x, 80, (3, 3), (1, 1))
        x = VerticalAreaNet.conv_module(x, 160, (3, 3), (1, 1))
        x = L.Dropout(0.2)(x)

        ## classifier

        # softmax classifier
        x = L.GlobalAvgPool2D()(x)
        # x = L.Flatten()(x)
        x = L.Dense(num_classes)(x)
        x = L.Activation("softmax", name="output")(x)

        # create the model
        model = Model(inputs, x, name="area_net")

        # return the constructed network architecture
        return model


def test_build_aa() -> None:
    """MAKEDOC: what is test_build_aa doing?"""
    logg = logging.getLogger(f"c.{__name__}.test_build_aa")
    # logg.setLevel("INFO")
    logg.debug("Start test_build_aa")

    # input_shape = (64, 64, 1)
    input_shape = (256, 256, 3)
    # input_shape = (80, 128, 1)
    num_classes = 4

    # area_model = AreaNet.build(input_shape, num_classes)
    area_model = ActualAreaNet.build(input_shape, num_classes)
    # area_model = VerticalAreaNet.build(input_shape, num_classes)
    # area_model = SimpleNet.build(input_shape, num_classes, sim_type="2")

    area_model.summary(line_length=120)

    model_folder = Path("plot_models")
    model_pic_name = model_folder / "actual_area_model_04.png"
    U.plot_model(area_model, model_pic_name, show_shapes=True, dpi=400)

    asc = ascii_model(area_model)
    print(f"asc:\n{asc}")


if __name__ == "__main__":
    test_build_aa()
