from pathlib import Path
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore

from plot_utils import plot_confusion_matrix
from preprocess_data import load_processed
from utils import WORDS_ALL, WORDS_NUMBERS, WORDS_DIRECTION
from utils import pred_hot_2_cm
from utils import setup_gpus
from utils import setup_logger


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
    setup_logger()

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 evaluate.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def analyze_confusion(confusion, true_labels):
    """Compute the F-score from the confusion matrix, and print the intermediate results

    Precision: TP / ( TP + FP)
    Recall: TP / ( TP + FN)
    F-score: 2 (PxR) / (P+R)
    """
    logg = logging.getLogger(f"c.{__name__}.analyze_confusion")
    logg.setLevel("INFO")
    logg.debug("Start analyze_confusion")

    logg.debug("Confusion matrix:")
    logg.debug(row_fmt("Pre\\Tru", true_labels))

    for line, label in zip(confusion, true_labels):
        logg.debug(row_fmt(f"{label}", line))

    TP = confusion.diagonal()
    FN = np.sum(confusion, axis=0) - TP
    FP = np.sum(confusion, axis=1) - TP

    logg.debug("")
    logg.debug(row_fmt("TP", TP))
    logg.debug(row_fmt("FP", FP))
    logg.debug(row_fmt("FN", FN))

    # https://stackoverflow.com/a/37977222
    #  P = TP / ( TP + FP)
    #  R = TP / ( TP + FN)
    dP = TP + FP
    P = np.divide(TP, dP, out=np.zeros_like(TP, dtype=float), where=dP != 0)
    dR = TP + FN
    R = np.divide(TP, dR, out=np.zeros_like(TP, dtype=float), where=dR != 0)

    logg.debug("\nPrecision = TP / ( TP + FP)\tRecall = TP / ( TP + FN)")
    logg.debug(row_fmt("Prec", P, ":.4f"))
    logg.debug(row_fmt("Recall", R, ":.4f"))

    avgP = np.sum(P) / len(true_labels)
    avgR = np.sum(R) / len(true_labels)
    logg.debug(f"Average P: {avgP:.4f}\tR: {avgR:.4f}")

    logg.debug("F-score = 2 (PxR) / (P+R)")
    #  F = 2 (PxR) / (P+R)
    PdR = 2 * P * R
    PpR = P + R
    F = np.divide(PdR, PpR, out=np.zeros_like(TP, dtype=float), where=PpR != 0)
    logg.debug(row_fmt("F-score", F, ":.4f"))

    avgF = np.sum(F) / len(true_labels)
    logg.debug(f"Average F-score {avgF}")

    return avgF


def row_fmt(header, iterable, formatter=""):
    row = header
    for item in iterable:
        #  row += f'\t{item{formatter}}'
        row += "\t{{i{f}}}".format(f=formatter).format(i=item)
    return row


def evaluate_results_recap():
    """TODO: what is evaluate_results_recap doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_recap")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_recap")

    info_folder = Path("info")

    pandito = {
        "base_dense_width": [],
        "base_filters": [],
        "batch_size": [],
        "dataset": [],
        "dropout_type": [],
        "epoch_num": [],
        "kernel_size_type": [],
        "pool_size_type": [],
        "words": [],
        "fscore": [],
        "loss": [],
        "categorical_accuracy": [],
        "model_name": [],
    }

    for model_folder in info_folder.iterdir():
        logg.debug(f"model_folder: {model_folder}")

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue

        results_recap = json.loads(res_recap_path.read_text())
        logg.debug(f"results_recap['cm']: {results_recap['cm']}")

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())
        logg.debug(f"recap['words']: {recap['words']}")

        cm = np.array(results_recap["cm"])
        fscore = analyze_confusion(cm, recap["words"])

        categorical_accuracy = results_recap["categorical_accuracy"]
        logg.debug(f"categorical_accuracy: {categorical_accuracy}")

        pandito["base_dense_width"].append(recap["hypa"]["base_dense_width"])
        pandito["base_filters"].append(recap["hypa"]["base_filters"])
        pandito["batch_size"].append(recap["hypa"]["batch_size"])
        pandito["dataset"].append(recap["hypa"]["dataset"])
        pandito["dropout_type"].append(recap["hypa"]["dropout_type"])
        pandito["epoch_num"].append(recap["hypa"]["epoch_num"])
        pandito["kernel_size_type"].append(recap["hypa"]["kernel_size_type"])
        pandito["pool_size_type"].append(recap["hypa"]["pool_size_type"])
        pandito["words"].append(recap["hypa"]["words"])
        pandito["categorical_accuracy"].append(results_recap["categorical_accuracy"])
        pandito["loss"].append(results_recap["loss"])
        pandito["model_name"].append(results_recap["model_name"])
        pandito["fscore"].append(fscore)

    df = pd.DataFrame(pandito)
    logg.info(f"{df.sort_values('fscore', ascending=False)[:10]}")
    # logg.info(f"{df.sort_values('categorical_accuracy', ascending=False)[:10]}")


def evaluate_model():
    """TODO: what is evaluate_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model")
    logg.debug("Start evaluate_model")

    # magic to fix the GPUs
    setup_gpus()

    # setup the parameters
    base_filters = 20
    kernel_size_type = "01"
    pool_size_type = "01"
    base_dense_width = 32
    dropout_type = "02"
    batch_size = 32
    epoch_num = 60
    dataset = "mfcc01"
    words = "f1"

    # get the words
    words_types = {
        "all": WORDS_ALL,
        "dir": WORDS_DIRECTION,
        "num": WORDS_NUMBERS,
        "f1": ["happy", "learn", "wow", "visual"],
        "f2": ["backward", "eight", "go", "yes"],
    }
    sel_words = words_types[words]

    # name the model
    model_name = "CNN"
    model_name += f"_nf{base_filters}"
    model_name += f"_ks{kernel_size_type}"
    model_name += f"_ps{pool_size_type}"
    model_name += f"_dw{base_dense_width}"
    model_name += f"_dr{dropout_type}"
    model_name += f"_ds{dataset}"
    model_name += f"_bs{batch_size}"
    model_name += f"_en{epoch_num}"
    model_name += f"_w{words}"
    logg.debug(f"model_name: {model_name}")

    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    model.summary()

    # input data
    processed_path = Path(f"data_proc/{dataset}")
    data, labels = load_processed(processed_path, sel_words)

    # evaluate on the words you trained on
    logg.debug("Evaluate on test data:")
    model.evaluate(data["testing"], labels["testing"])
    # model.evaluate(data["validation"], labels["validation"])

    # predict labels
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, sel_words)
    # y_pred = model.predict(data["validation"])
    # cm = pred_hot_2_cm(labels["validation"], y_pred, sel_words)
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, sel_words)

    fscore = analyze_confusion(cm, sel_words)
    logg.debug(f"fscore: {fscore}")

    plt.show()


def run_evaluate(args):
    """TODO: What is evaluate doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate")
    logg.debug("Starting run_evaluate")

    evaluate_results_recap()
    # evaluate_model()


if __name__ == "__main__":
    args = setup_env()
    run_evaluate(args)
