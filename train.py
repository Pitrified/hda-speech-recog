import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore

# import numpy as np  # type: ignore
# from tensorflow.data.Dataset import from_tensor_slices  # type: ignore
# from tensorflow import data as tfdata  # type: ignore
# import tensorflow as tf  # type: ignore
from tensorflow.data import Dataset  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from models import CNNmodel
from preprocess_data import load_processed
from utils import setup_logger
from utils import WORDS_ALL, WORDS_NUMBERS, WORDS_DIRECTION
from utils import pred_hot_2_cm
from utils import setup_gpus
from plot_utils import plot_loss
from plot_utils import plot_cat_acc
from plot_utils import plot_confusion_matrix


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


def run_train(args):
    """TODO: What is train doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train")
    logg.debug("Starting run_train")

    setup_gpus()

    # input data
    processed_path = Path("data_proc/mfcc")
    words = WORDS_DIRECTION
    words = WORDS_NUMBERS
    words = WORDS_ALL
    # words = ["happy", "learn", "wow", "visual"]
    data, labels = load_processed(processed_path, words)

    model_name = "CNNmodel_002"

    model_folder = Path("models")
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / f"{model_name}.h5"
    logg.debug(f"model_path: {model_path}")

    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

    # create the model
    model = CNNmodel(len(words), input_shape=data["training"][0].shape)
    model.compile(
        optimizer="adam",
        loss=["categorical_crossentropy"],
        metrics=["categorical_accuracy"],
    )
    model.summary()

    # training parameters
    BATCH_SIZE = 128
    # BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 200
    EPOCH_NUM = 30

    # load the datasets
    datasets = {}
    for which in ["validation", "training", "testing"]:
        logg.debug(f"data[{which}].shape: {data[which].shape}")
        datasets[which] = Dataset.from_tensor_slices((data[which], labels[which]))
        datasets[which] = datasets[which].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    # setup early stopping
    early_stop = EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=10,
        verbose=1,
        restore_best_weights=True,
    )

    # train the model
    results = model.fit(
        # datasets["training"],
        data["training"],
        labels["training"],
        validation_data=datasets["validation"],
        batch_size=BATCH_SIZE,
        epochs=EPOCH_NUM,
        verbose=1,
        callbacks=[early_stop],
    )
    model.save(model_path)

    # quickly evaluate the results
    logg.debug(f"\nmodel.metrics_names: {model.metrics_names}")
    for which in ["validation", "training", "testing"]:
        model_eval = model.evaluate(datasets[which])
        logg.debug(f"{which}: model_eval: {model_eval}")

    # save plots about training
    # loss
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_loss(results.history["loss"], results.history["val_loss"], ax, model_name)
    plot_loss_path = info_folder / "train_loss.png"
    fig.savefig(plot_loss_path)

    # categorical accuracy
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_cat_acc(
        results.history["categorical_accuracy"],
        results.history["val_categorical_accuracy"],
        ax,
        model_name,
    )
    plot_cat_acc_path = info_folder / "train_cat_acc.png"
    fig.savefig(plot_cat_acc_path)

    # a dict to recreate this training
    recap = {}
    recap["words"] = words

    # compute the confusion matrix
    y_pred = model.predict(datasets["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, words)
    logg.debug(f"cm: {cm}")
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words)
    plot_cm_path = info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)

    plt.show()


if __name__ == "__main__":
    args = setup_env()
    run_train(args)
