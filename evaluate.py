from pathlib import Path
from time import sleep
import argparse
import json
import logging

from scipy.io.wavfile import write  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore

from tensorflow.keras.models import Model  # type: ignore

from plot_utils import plot_confusion_matrix
from plot_utils import plot_pred
from plot_utils import plot_spec
from plot_utils import plot_waveform
from preprocess_data import get_spec_dict
from preprocess_data import load_processed
from preprocess_data import wav2mel
from utils import pred_hot_2_cm
from utils import setup_gpus
from utils import setup_logger
from utils import words_types

from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="results",
        choices=[
            "results",
            "results_transfer",
            "model",
            "audio",
            "delete",
            "delete_transfer",
        ],
        help="Which evaluation to perform",
    )

    parser.add_argument(
        "-tw",
        "--train_words_type",
        type=str,
        default="f2",
        choices=["all", "dir", "num", "f1", "f2"],
        help="Words the dataset was trained on",
    )

    parser.add_argument(
        "-rw",
        "--rec_words_type",
        type=str,
        default="dataset",
        choices=["all", "dir", "num", "f1", "f2", "dataset"],
        help="Words to record and test",
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


def evaluate_results_recap(args):
    """TODO: what is evaluate_results_recap doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_recap")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_recap")

    info_folder = Path("info")

    pandito = {
        "dense_width": [],
        "filters": [],
        "batch_size": [],
        "dataset": [],
        "dropout": [],
        "epoch_num": [],
        "kernel_size": [],
        "pool_size": [],
        "lr": [],
        "opt": [],
        "words": [],
        "fscore": [],
        "loss": [],
        "cat_acc": [],
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

        pandito["dense_width"].append(recap["hypa"]["base_dense_width"])
        pandito["filters"].append(recap["hypa"]["base_filters"])
        pandito["batch_size"].append(recap["hypa"]["batch_size"])
        pandito["dataset"].append(recap["hypa"]["dataset"])
        pandito["dropout"].append(recap["hypa"]["dropout_type"])
        pandito["epoch_num"].append(recap["hypa"]["epoch_num"])
        pandito["kernel_size"].append(recap["hypa"]["kernel_size_type"])
        pandito["pool_size"].append(recap["hypa"]["pool_size_type"])
        pandito["words"].append(recap["hypa"]["words"])
        pandito["cat_acc"].append(results_recap["categorical_accuracy"])
        pandito["loss"].append(results_recap["loss"])
        pandito["model_name"].append(results_recap["model_name"])
        pandito["fscore"].append(fscore)

        if "version" in recap:
            if recap["version"] == "001":
                pandito["lr"].append(recap["hypa"]["learning_rate_type"])
                pandito["opt"].append(recap["hypa"]["optimizer_type"])
        else:
            pandito["lr"].append("default")
            pandito["opt"].append("adam")

    pd.set_option("max_colwidth", 100)
    df = pd.DataFrame(pandito)
    logg.info(f"{df.sort_values('fscore', ascending=False)[:30]}")
    # logg.info(f"{df.sort_values('categorical_accuracy', ascending=False)[:10]}")


def load_trained_model(
    base_filters,
    kernel_size_type,
    pool_size_type,
    base_dense_width,
    dropout_type,
    batch_size,
    epoch_num,
    dataset,
    words,
    learning_rate_type,
    optimizer_type,
):
    """TODO: what is load_trained_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model")
    logg.debug("Start load_trained_model")

    # name the model
    model_name = "CNN"
    model_name += f"_nf{base_filters}"
    model_name += f"_ks{kernel_size_type}"
    model_name += f"_ps{pool_size_type}"
    model_name += f"_dw{base_dense_width}"
    model_name += f"_dr{dropout_type}"
    if learning_rate_type != "default":
        model_name += f"_lr{learning_rate_type}"
    if optimizer_type != "adam":
        model_name += f"_op{optimizer_type}"
    model_name += f"_ds{dataset}"
    model_name += f"_bs{batch_size}"
    model_name += f"_en{epoch_num}"
    model_name += f"_w{words}"

    logg.debug(f"model_name: {model_name}")

    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        raise FileNotFoundError

    model = tf.keras.models.load_model(model_path)
    return model, model_name


def evaluate_model(args):
    """TODO: what is evaluate_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_model")

    # magic to fix the GPUs
    setup_gpus()

    # setup the parameters
    dataset = "mel01"
    words = args.train_words_type

    base_dense_width = 32
    base_filters = 20
    batch_size = 32
    dropout_type = "01"
    epoch_num = 16
    kernel_size_type = "02"
    pool_size_type = "02"
    learning_rate_type = "02"
    optimizer_type = "a1"

    # get the words
    train_words = words_types[words]

    model, model_name = load_trained_model(
        base_filters,
        kernel_size_type,
        pool_size_type,
        base_dense_width,
        dropout_type,
        batch_size,
        epoch_num,
        dataset,
        words,
        learning_rate_type,
        optimizer_type,
    )
    model.summary()

    # input data
    processed_path = Path(f"data_proc/{dataset}")
    data, labels = load_processed(processed_path, train_words)
    logg.debug(f"data['testing'].shape: {data['testing'].shape}")

    # evaluate on the words you trained on
    logg.debug("Evaluate on test data:")
    model.evaluate(data["testing"], labels["testing"])
    # model.evaluate(data["validation"], labels["validation"])

    # predict labels/cm/fscore
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, train_words)
    # y_pred = model.predict(data["validation"])
    # cm = pred_hot_2_cm(labels["validation"], y_pred, train_words)
    fscore = analyze_confusion(cm, train_words)
    logg.debug(f"fscore: {fscore}")

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, train_words, fscore)

    plt.show()


def evaluate_audio(args):
    """TODO: what is evaluate_audio doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_audio")
    logg.debug("Start evaluate_audio")

    # magic to fix the GPUs
    setup_gpus()

    # importing sounddevice takes time, only do it if needed
    import sounddevice as sd  # type: ignore

    # where to save the audios
    audio_folder = Path("recorded_audio")
    if not audio_folder.exists():
        audio_folder.mkdir(parents=True, exist_ok=True)

    # need to know on which dataset the model was trained to compute specs
    dataset_name = "mel01"

    # words that the dataset was trained on
    train_words_type = args.train_words_type
    train_words = words_types[train_words_type]

    rec_words_type = args.rec_words_type
    if rec_words_type == "dataset":
        rec_words = train_words
    else:
        rec_words = words_types[rec_words_type]
    num_rec_words = len(rec_words)

    # input parameters
    fs = 16000  # Sample rate
    seconds = 1  # Duration of recording

    audios = []
    specs = []

    timeout = 0  # time to get ready to talk for each word
    if timeout == 0:  # if there is no time give it at least in the beginning
        logg.debug(f"Get ready to start recording {rec_words[0]}")
        sleep(1)
    for word in rec_words:
        for i in range(timeout, 0, -1):
            logg.debug(f"Start recording {word} in {i}s")
            sleep(1)
        logg.debug(f"Start recording {word} NOW!")

        # record
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        logg.debug("Stop recording")

        # save the audio
        audio_path = audio_folder / f"{word}_01.wav"
        write(audio_path, fs, myrecording)  # Save as WAV file

        # convert it to mel
        p2d_kwargs = {"ref": np.max}
        spec_dict = get_spec_dict()
        spec_kwargs = spec_dict[dataset_name]
        log_spec = wav2mel(audio_path, spec_kwargs, p2d_kwargs)
        img_spec = log_spec.reshape((*log_spec.shape, 1))
        logg.debug(f"img_spec.shape: {img_spec.shape}")  # img_spec.shape: (128, 32, 1)

        audios.append(myrecording)
        specs.append(log_spec)

    # the data needs to look like this data['testing'].shape: (735, 128, 32, 1)
    # data = log_spec.reshape((1, *log_spec.shape, 1))
    data = np.stack(specs)
    logg.debug(f"data.shape: {data.shape}")

    # parameters of the model
    base_dense_width = 32
    base_filters = 20
    batch_size = 32
    dropout_type = "01"
    epoch_num = 16
    kernel_size_type = "02"
    pool_size_type = "02"
    learning_rate_type = "02"
    optimizer_type = "a1"

    model, model_name = load_trained_model(
        base_filters,
        kernel_size_type,
        pool_size_type,
        base_dense_width,
        dropout_type,
        batch_size,
        epoch_num,
        dataset_name,
        train_words_type,
        learning_rate_type,
        optimizer_type,
    )

    pred = model.predict(data)
    # logg.debug(f"pred: {pred}")

    # plot the thing
    plot_size = 5
    fw = plot_size * 3
    fh = plot_size * num_rec_words
    fig, axes = plt.subplots(nrows=num_rec_words, ncols=3, figsize=(fw, fh))
    fig.suptitle("Recorded audios", fontsize=18)

    for i, word in enumerate(rec_words):
        plot_waveform(audios[i], axes[i][0])
        plot_spec(specs[i], axes[i][1])
        plot_pred(pred[i], train_words, axes[i][2], f"Prediction for {rec_words[i]}", i)

    # https://stackoverflow.com/q/8248467
    # https://stackoverflow.com/q/2418125
    fig.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.97])

    results_path = audio_folder / f"results_{train_words_type}_{rec_words_type}.png"
    fig.savefig(results_path)

    if num_rec_words <= 6:
        plt.show()


def delete_bad_models(args) -> None:
    """TODO: what is delete_bad_models doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models")
    # logg.setLevel("INFO")
    logg.debug("Start delete_bad_models")

    info_folder = Path("info")
    trained_folder = Path("trained_models")
    f_tresh = 0.87
    ca_tresh = 0.95
    deleted = 0
    recreated = 0

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue

        results_recap = json.loads(res_recap_path.read_text())
        # logg.debug(f"results_recap['cm']: {results_recap['cm']}")

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())
        # logg.debug(f"recap['words']: {recap['words']}")

        cm = np.array(results_recap["cm"])
        fscore = analyze_confusion(cm, recap["words"])
        # logg.debug(f"fscore: {fscore}")

        categorical_accuracy = results_recap["categorical_accuracy"]
        # logg.debug(f"categorical_accuracy: {categorical_accuracy}")

        if fscore < f_tresh and categorical_accuracy < ca_tresh:
            model_name = model_folder.name
            model_path = trained_folder / f"{model_name}.h5"

            if model_path.exists():
                # manually uncomment when ready to delete to be safe
                # model_path.unlink()
                deleted += 1
                logg.debug(f"Deleting model_path: {model_path}")
                logg.debug(f"fscore: {fscore}")
                logg.debug(f"categorical_accuracy: {categorical_accuracy}")
                model_path.write_text("Deleted")

            # you gone goofed and deleted a model
            else:
                model_path.write_text("Deleted")
                logg.debug(f"Recreating model_path: {model_path}")
                recreated += 1

    logg.debug(f"deleted: {deleted}")
    logg.debug(f"recreated: {recreated}")


def load_transfer_model(
    hypa: Dict[str, str], use_validation: bool, do_load: bool
) -> Tuple[Optional[Model], str]:
    """TODO: what is load_transfer_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_transfer_model")
    # logg.setLevel("INFO")
    logg.debug("Start load_transfer_model")

    model_name = "TRA"
    model_name += f"_dw{hypa['dense_width_type']}"
    model_name += f"_dr{hypa['dropout_type']}"
    model_name += f"_bs{hypa['batch_size_type']}"
    model_name += f"_en{hypa['epoch_num_type']}"
    model_name += f"_lr{hypa['learning_rate_type']}"
    model_name += f"_op{hypa['optimizer_type']}"
    model_name += f"_ds{hypa['datasets_type']}"
    model_name += f"_w{hypa['words_type']}"
    if not use_validation:
        model_name += "_noval"
    logg.debug(f"model_name: {model_name}")

    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"
    if do_load and model_path.exists():
        model = tf.keras.models.load_model(model_path)
    else:
        model = None

    return model, model_name


def evaluate_results_transfer(args: argparse.Namespace) -> None:
    """TODO: what is evaluate_results_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_transfer")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_transfer")

    pandito: Dict[str, List[str]] = {
        "dense_width": [],
        "dropout": [],
        "batch_size": [],
        "epoch_num": [],
        "learning_rate": [],
        "optimizer": [],
        "datasets": [],
        "words": [],
        "use_val": [],
        "loss": [],
        "cat_acc": [],
        "precision": [],
        "recall": [],
        "fscore": [],
        "model_name": [],
    }
    info_folder = Path("info")

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")
        model_name = model_folder.name
        if not model_name.startswith("TRA"):
            continue
        logg.debug(f"model_name: {model_name}")

        # res_freeze_path = model_folder / "results_freeze_recap.json"
        # res_freeze = json.loads(res_freeze_path.read_text())

        res_full_path = model_folder / "results_full_recap.json"
        res_full = json.loads(res_full_path.read_text())
        logg.debug(f"res_full['fscore']: {res_full['fscore']}")

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())
        logg.debug(f"recap['words']: {recap['words']}")

        hypa = recap["hypa"]
        pandito["dense_width"].append(hypa["dense_width_type"])
        pandito["dropout"].append(hypa["dropout_type"])
        pandito["batch_size"].append(hypa["batch_size_type"])
        pandito["epoch_num"].append(hypa["epoch_num_type"])
        pandito["learning_rate"].append(hypa["learning_rate_type"])
        pandito["optimizer"].append(hypa["optimizer_type"])
        pandito["datasets"].append(hypa["datasets_type"])
        pandito["words"].append(hypa["words_type"])
        if recap["version"] == "001":
            pandito["use_val"].append("True")
        elif recap["version"] == "002":
            pandito["use_val"].append(recap["use_validation"])
        pandito["loss"].append(res_full["loss"])
        pandito["cat_acc"].append(res_full["categorical_accuracy"])
        pandito["precision"].append(res_full["precision"])
        pandito["recall"].append(res_full["recall"])
        pandito["fscore"].append(res_full["fscore"])

        pandito["model_name"].append(res_full["model_name"])

    pd.set_option("max_colwidth", 100)
    df = pd.DataFrame(pandito)
    logg.info(f"{df.sort_values('fscore', ascending=False)[:30]}")


def delete_bad_transfer(args: argparse.Namespace) -> None:
    """TODO: what is delete_bad_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start delete_bad_transfer")

    info_folder = Path("info")
    trained_folder = Path("trained_models")
    f_tresh = 0.92
    deleted = 0
    recreated = 0

    for model_folder in info_folder.iterdir():

        model_name = model_folder.name
        if not model_name.startswith("TRA"):
            continue

        res_full_path = model_folder / "results_full_recap.json"
        res_full = json.loads(res_full_path.read_text())
        fscore = res_full["fscore"]

        if fscore < f_tresh:
            model_path = trained_folder / f"{model_name}.h5"

            if model_path.exists():
                # manually uncomment when ready to delete to be safe
                # model_path.unlink()
                deleted += 1
                logg.debug(f"Deleting model_path: {model_path}")
                logg.debug(f"fscore: {fscore}")
                model_path.write_text("Deleted")

            # you gone goofed and deleted a model
            else:
                model_path.write_text("Deleted")
                logg.debug(f"Recreating model_path: {model_path}")
                recreated += 1

    logg.debug(f"deleted: {deleted}")
    logg.debug(f"recreated: {recreated}")


def run_evaluate(args) -> None:
    """TODO: What is evaluate doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate")
    logg.debug("Starting run_evaluate")

    if args.evaluation_type == "results":
        evaluate_results_recap(args)
    elif args.evaluation_type == "results_transfer":
        evaluate_results_transfer(args)
    elif args.evaluation_type == "model":
        evaluate_model(args)
    elif args.evaluation_type == "audio":
        evaluate_audio(args)
    elif args.evaluation_type == "delete":
        delete_bad_models(args)
    elif args.evaluation_type == "delete_transfer":
        delete_bad_transfer(args)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate(args)
