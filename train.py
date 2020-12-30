import argparse
import logging
from pathlib import Path

# import numpy as np  # type: ignore
# from tensorflow.data.Dataset import from_tensor_slices  # type: ignore
# from tensorflow import data as tfdata  # type: ignore
import tensorflow as tf
from tensorflow.data import Dataset  # type: ignore

from models import CNNmodel
from preprocess_data import load_processed
from utils import setup_logger
from utils import ALL_WORDS


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        "--path_input",
        type=str,
        default="hp.jpg",
        help="path to input image to use",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env():
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 train.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def setup_gpus():
    """TODO: what is setup_gpus doing?

    https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    https://github.com/tensorflow/tensorflow/issues/25138
    """
    logg = logging.getLogger(f"c.{__name__}.setup_gpus")
    logg.debug(f"Start setup_gpus")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        logg.debug(f"Broski non ho trovato le GPU")


def run_train(args):
    """TODO: What is train doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train")
    logg.debug("Starting run_train")

    setup_gpus()

    processed_path = Path("data_proc/mfcc")
    words = ALL_WORDS
    words = ["happy", "learn", "wow", "visual"]
    data, labels = load_processed(processed_path, words)

    model_folder = Path("models")
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / "CNNmodel_002.h5"
    logg.debug(f"model_path: {model_path}")

    model = CNNmodel(len(words), input_shape=data["training"][0].shape)
    model.compile(
        optimizer="adam",
        loss=["categorical_crossentropy"],
        metrics=["categorical_accuracy"],
    )
    model.summary()

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    datasets = {}
    for which in ["validation", "training", "testing"]:
        logg.debug(f"data[{which}].shape: {data[which].shape}")
        datasets[which] = Dataset.from_tensor_slices((data[which], labels[which]))
        datasets[which] = datasets[which].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    model.fit(
        datasets["training"],
        validation_data=datasets["validation"],
        batch_size=BATCH_SIZE,
        epochs=60,
    )
    model.save(model_path)

    logg.debug(f"\nmodel.metrics_names: {model.metrics_names}")
    for which in ["validation", "training", "testing"]:
        model_eval = model.evaluate(datasets[which])
        logg.debug(f"{which}: model_eval: {model_eval}")


if __name__ == "__main__":
    args = setup_env()
    run_train(args)
