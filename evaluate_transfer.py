from pathlib import Path
import argparse
import json
import logging

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore

from plot_utils import plot_pred
from plot_utils import plot_spec
from plot_utils import plot_waveform
from preprocess_data import get_spec_dict
from preprocess_data import wav2mel
from train import build_transfer_name
from utils import compute_permutation
from utils import record_audios
from utils import setup_gpus
from utils import setup_logger
from utils import words_types

# import typing as ty
from typing import Dict
from typing import List


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="results",
        choices=[
            "results",
            "audio",
            "delete_bad_models",
        ],
        help="Which evaluation to perform",
    )

    tra_types = [w for w in words_types.keys() if not w.startswith("_")]
    parser.add_argument(
        "-tw",
        "--train_words_type",
        type=str,
        default="f2",
        choices=tra_types,
        help="Words the dataset was trained on",
    )

    rec_types = [w for w in words_types.keys() if not w.startswith("_")]
    rec_types.append("train")
    parser.add_argument(
        "-rw",
        "--rec_words_type",
        type=str,
        default="train",
        choices=rec_types,
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


def build_tra_results_df() -> pd.DataFrame:
    """TODO: what is build_tra_results_df doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_tra_results_df")
    # logg.setLevel("INFO")
    logg.debug("Start build_tra_results_df")

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
        if not res_full_path.exists():
            logg.info(f"Skipping res_full_path: {res_full_path}, not found")
            continue
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

    return df


def evaluate_results_transfer(args: argparse.Namespace) -> None:
    """TODO: what is evaluate_results_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_transfer")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_transfer")

    results_df = build_tra_results_df()
    fscore_df = results_df.sort_values("fscore", ascending=False).head(30)
    logg.info(f"{fscore_df}")


def delete_bad_models_transfer(args: argparse.Namespace) -> None:
    """TODO: what is delete_bad_models_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models_transfer")
    logg.setLevel("INFO")
    logg.debug("Start delete_bad_models_transfer")

    info_folder = Path("info")
    trained_folder = Path("trained_models")
    f_tresh = 0.93
    deleted = 0
    recreated = 0

    for model_folder in info_folder.iterdir():

        model_name = model_folder.name
        if not model_name.startswith("TRA"):
            continue

        res_full_path = model_folder / "results_full_recap.json"
        if not res_full_path.exists():
            logg.info(f"Skipping res_full_path: {res_full_path}")
            continue
        res_full = json.loads(res_full_path.read_text())
        fscore = res_full["fscore"]

        if fscore < f_tresh:
            model_path = trained_folder / f"{model_name}.h5"

            if model_path.exists():
                # model_path.unlink()
                deleted += 1
                logg.debug(f"Deleting model_path: {model_path}")
                logg.debug(f"fscore: {fscore}")
                # model_path.write_text("Deleted")

            # you gone goofed and deleted a model
            else:
                # model_path.write_text("Deleted")
                logg.debug(f"Recreating model_path: {model_path}")
                recreated += 1

    logg.info(f"deleted: {deleted}")
    logg.info(f"recreated: {recreated}")


def evaluate_audio_transfer(train_words_type: str, rec_words_type: str) -> None:
    """TODO: what is evaluate_audio_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_audio_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_audio_transfer")

    # magic to fix the GPUs
    setup_gpus()

    datasets_type = "01"
    datasets_types = {
        "01": ["mel05", "mel09", "mel10"],
        "02": ["mel05", "mel10", "mfcc07"],
        "03": ["mfcc06", "mfcc07", "mfcc08"],
        "04": ["mel05", "mfcc06", "melc1"],
        "05": ["melc1", "melc2", "melc4"],
    }
    dataset_names = datasets_types[datasets_type]

    # we do not support composed datasets for now
    for dn in dataset_names:
        if dn.startswith("melc"):
            logg.error(f"not supported: {dataset_names}")
            return

    # words that the dataset was trained on
    train_words_type = args.train_words_type
    train_words = words_types[train_words_type]

    # the model predicts sorted words
    perm_pred = compute_permutation(train_words)

    if rec_words_type == "train":
        rec_words = train_words
    else:
        rec_words = words_types[rec_words_type]
    num_rec_words = len(rec_words)

    # where to save the audios
    audio_folder = Path("recorded_audio")
    if not audio_folder.exists():
        audio_folder.mkdir(parents=True, exist_ok=True)

    # record the audios and save them in audio_folder
    audio_path_fmt = "{}_02.wav"
    audios = record_audios(rec_words, audio_folder, audio_path_fmt, timeout=0)

    # compute the spectrograms and build the dataset of correct shape
    specs_3ch: List[np.ndarray] = []
    # params for the mel conversion
    p2d_kwargs = {"ref": np.max}
    spec_dict = get_spec_dict()
    for word in rec_words:
        # get the name
        audio_path = audio_folder / audio_path_fmt.format(word)

        # convert it to mel for each type of dataset
        specs: List[np.ndarray] = []
        for dataset_name in dataset_names:
            spec_kwargs = spec_dict[dataset_name]
            log_spec = wav2mel(audio_path, spec_kwargs, p2d_kwargs)
            specs.append(log_spec)
        img_spec = np.stack(specs, axis=2)
        # logg.debug(f"img_spec.shape: {img_spec.shape}")  # (128, 128, 3)

        specs_3ch.append(img_spec)

    data = np.stack(specs_3ch)
    logg.debug(f"data.shape: {data.shape}")

    hypa: Dict[str, str] = {}
    hypa["dense_width_type"] = "03"
    hypa["dropout_type"] = "01"
    hypa["batch_size_type"] = "02"
    hypa["epoch_num_type"] = "01"
    hypa["learning_rate_type"] = "01"
    hypa["optimizer_type"] = "a1"
    hypa["datasets_type"] = datasets_type
    hypa["words_type"] = train_words_type
    use_validation = False

    # hypa: Dict[str, str] = {}
    # hypa["dense_width_type"] = "02"
    # hypa["dropout_type"] = "01"
    # hypa["batch_size_type"] = "01"
    # hypa["epoch_num_type"] = "01"
    # hypa["learning_rate_type"] = "01"
    # hypa["optimizer_type"] = "a1"
    # hypa["datasets_type"] = datasets_type
    # hypa["words_type"] = train_words_type
    # use_validation = True

    # get the model name
    model_name = build_transfer_name(hypa, use_validation)

    # load the model
    # model_folder = Path("trained_models")
    model_folder = Path("saved_models")
    model_path = model_folder / f"{model_name}.h5"
    model = tf.keras.models.load_model(model_path)

    # predict!
    pred = model.predict(data)

    # plot everything
    plot_size = 5
    fw = plot_size * 5
    fh = plot_size * num_rec_words
    fig, axes = plt.subplots(nrows=num_rec_words, ncols=5, figsize=(fw, fh))
    fig.suptitle("Recorded audios", fontsize=18)

    for i, word in enumerate(rec_words):
        plot_waveform(audios[i], axes[i][0])
        img_spec = specs_3ch[i]
        plot_spec(img_spec[:, :, 0], axes[i][1])
        plot_spec(img_spec[:, :, 1], axes[i][2])
        plot_spec(img_spec[:, :, 2], axes[i][3])
        plot_pred(
            pred[i][perm_pred],
            train_words,
            axes[i][4],
            f"Prediction for {rec_words[i]}",
            train_words.index(word),
        )

    # https://stackoverflow.com/q/8248467
    # https://stackoverflow.com/q/2418125
    fig.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.97])

    fig_name = f"{model_name}_{train_words_type}_{rec_words_type}.png"
    results_path = audio_folder / fig_name
    fig.savefig(results_path)

    if num_rec_words <= 6:
        plt.show()


def run_evaluate_transfer(args: argparse.Namespace) -> None:
    """TODO: What is evaluate_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_transfer")
    logg.debug("Starting run_evaluate_transfer")

    evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type
    rec_words_type = args.rec_words_type

    pd.set_option("max_colwidth", 100)

    if evaluation_type == "results":
        evaluate_results_transfer(args)
    elif evaluation_type == "audio":
        evaluate_audio_transfer(train_words_type, rec_words_type)
    elif evaluation_type == "delete_bad_models":
        delete_bad_models_transfer(args)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_transfer(args)