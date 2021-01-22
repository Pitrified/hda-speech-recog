import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import json

# import numpy as np  # type: ignore
# from tensorflow.data.Dataset import from_tensor_slices  # type: ignore
# from tensorflow import data as tfdata  # type: ignore
# import tensorflow as tf  # type: ignore
from tensorflow.data import Dataset  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from models import CNNmodel
from models import AttRNNmodel
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


def hyper_train():
    """TODO: what is hyper_train doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train")
    logg.debug("Start hyper_train")


def run_train(args):
    """TODO: What is train doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train")
    logg.debug("Starting run_train")

    # magic to fix the GPUs
    setup_gpus()

    # setup the hyperparameters (WILL ARRIVE FROM GRID)
    hypa = {}
    # hypa["base_filters"] = 20
    hypa["base_filters"] = 64
    hypa["kernel_size_type"] = "02"
    hypa["pool_size_type"] = "02"
    hypa["base_dense_width"] = 32
    hypa["dropout_type"] = "01"
    hypa["batch_size"] = 64
    hypa["epoch_num"] = 30
    hypa["dataset"] = "mfcc03"
    hypa["words"] = "f1"

    # get the words
    words_types = {
        "all": WORDS_ALL,
        "dir": WORDS_DIRECTION,
        "num": WORDS_NUMBERS,
        "f1": ["happy", "learn", "wow", "visual"],
        "f2": ["backward", "eight", "go", "yes"],
    }
    words = words_types[hypa["words"]]

    # input data
    processed_path = Path(f"data_proc/{hypa['dataset']}")
    data, labels = load_processed(processed_path, words)

    # from hypa extract model param
    model_param = {}
    model_param["num_labels"] = len(words)
    model_param["input_shape"] = data["training"][0].shape
    model_param["base_filters"] = hypa["base_filters"]
    model_param["base_dense_width"] = hypa["base_dense_width"]

    # translate types to actual values
    kernel_size_types = {"01": [(2, 2), (2, 2), (2, 2)], "02": [(5, 1), (3, 3), (3, 3)]}
    model_param["kernel_sizes"] = kernel_size_types[hypa["kernel_size_type"]]

    pool_size_types = {"01": [(2, 2), (2, 2), (2, 2)], "02": [(2, 1), (2, 2), (2, 2)]}
    model_param["pool_sizes"] = pool_size_types[hypa["pool_size_type"]]

    dropout_types = {"01": [0.03, 0.01]}
    model_param["dropouts"] = dropout_types[hypa["dropout_type"]]

    # create the model
    model = CNNmodel(**model_param)

    # name the model
    model_name = "CNN"
    model_name += f"_nf{hypa['base_filters']}"
    model_name += f"_ks{hypa['kernel_size_type']}"
    model_name += f"_ps{hypa['pool_size_type']}"
    model_name += f"_dw{hypa['base_dense_width']}"
    model_name += f"_dr{hypa['dropout_type']}"
    model_name += f"_ds{hypa['dataset']}"
    model_name += f"_bs{hypa['batch_size']}"
    model_name += f"_en{hypa['epoch_num']}"
    logg.debug(f"model_name: {model_name}")

    # save the trained model here
    model_folder = Path("trained_models")
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / f"{model_name}.h5"
    logg.debug(f"model_path: {model_path}")

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

    # a dict to recreate this training
    recap = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["model_name"] = model_name
    logg.debug(f"recap: {recap}")
    recap_path = info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    model.compile(
        optimizer="adam",
        loss=["categorical_crossentropy"],
        metrics=["categorical_accuracy"],
    )
    model.summary()

    # training parameters
    # BATCH_SIZE = 128
    # BATCH_SIZE = 64
    # EPOCH_NUM = 30
    BATCH_SIZE = hypa["batch_size"]
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE
    EPOCH_NUM = hypa["epoch_num"]

    # load the datasets
    datasets = {}
    for which in ["training", "validation", "testing"]:
        logg.debug(f"data[{which}].shape: {data[which].shape}")
        datasets[which] = Dataset.from_tensor_slices((data[which], labels[which]))
        datasets[which] = datasets[which].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    # setup early stopping
    early_stop = EarlyStopping(
        # monitor="val_categorical_accuracy",
        monitor="val_loss",
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
    for which in ["training", "validation", "testing"]:
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
