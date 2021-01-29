from itertools import combinations
from pathlib import Path
from tqdm import tqdm  # type: ignore
import argparse
import json
import logging

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore

from plot_utils import plot_att_weights
from plot_utils import plot_confusion_matrix
from plot_utils import plot_double_data
from plot_utils import plot_pred
from plot_utils import plot_spec
from plot_utils import plot_triple_data
from plot_utils import plot_waveform
from preprocess_data import get_spec_dict
from preprocess_data import load_processed
from preprocess_data import wav2mel
from train import build_attention_name
from train import build_cnn_name
from train import build_transfer_name
from utils import analyze_confusion
from utils import compute_permutation
from utils import find_rowcol
from utils import pred_hot_2_cm
from utils import record_audios
from utils import setup_gpus
from utils import setup_logger
from utils import words_types

import typing as ty
from typing import Dict
from typing import List
from typing import Any
from typing import Union


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="results",
        choices=[
            "results_cnn",
            "model_cnn",
            "audio_cnn",
            "delete_cnn",
            "make_plots_cnn",
            "results_transfer",
            "audio_transfer",
            "delete_transfer",
            "results_attention",
            "attention_weights",
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


#####################################################################
#             CNN model
#####################################################################


def build_cnn_results_df() -> pd.DataFrame:
    """TODO: what is build_cnn_results_df doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_cnn_results_df")
    logg.setLevel("INFO")
    logg.debug("Start build_cnn_results_df")

    info_folder = Path("info")

    pandito: ty.Dict[str, ty.List[str]] = {
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

        model_name = model_folder.name
        if not model_name.startswith("CNN"):
            continue

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

    df = pd.DataFrame(pandito)
    return df


def evaluate_results_cnn(args):
    """TODO: what is evaluate_results_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_cnn")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_cnn")

    results_df = build_cnn_results_df()
    fscore_df = results_df.sort_values("fscore", ascending=False).head(30)
    logg.info(f"{fscore_df}")


def old_make_plots_cnn() -> None:
    """TODO: what is old_make_plots_cnn doing?

    z grafici (uno per dataset)
    y gruppi di colonne (epoch_num)
    x colonne per gruppo (batch_size)
    ciascuna colonna ha le errorbar della std_dev di tutti i risultati per quella
    combinazione di (dataset, epoch_num, batch_size)
    """
    logg = logging.getLogger(f"c.{__name__}.old_make_plots_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start old_make_plots_cnn")

    dataset_name = "mel01"
    train_words_type = "f1"

    results_df = build_cnn_results_df()

    # setup the parameters
    hypa: Dict[str, Union[str, int]] = {}
    hypa["dense_width"] = 32
    hypa["filters"] = 20
    hypa["batch_size"] = 32
    hypa["dropout"] = "01"
    # hypa["epoch_num"] = 16
    hypa["epoch_num"] = 15
    hypa["kernel_size"] = "02"
    hypa["pool_size"] = "02"
    hypa["lr"] = "02"
    hypa["opt"] = "a1"
    hypa["dataset"] = dataset_name
    hypa["words"] = train_words_type

    for hp_name in hypa:
        logg.debug(f"results_df[{hp_name}].unique(): {results_df[hp_name].unique()}")

    fixed_hp = [
        # "dense_width",
        # "filters",
        "batch_size",
        # "dropout",
        "epoch_num",
        # "kernel_size",
        # "pool_size",
        # "lr",
        # "opt",
        "dataset",
        # "words",
    ]

    q_str = "1==1 "
    for hp_name in fixed_hp:
        q_str += f" and ({hp_name} == '{hypa[hp_name]}')"
    logg.debug(f"q_str: {q_str}")

    q_df = results_df.query(q_str)
    logg.debug(f"q_df:\n{q_df.sort_values('fscore', ascending=False).head(30)}")
    logg.debug(f"q_df:\n{q_df.sort_values('fscore', ascending=True).head(30)}")

    logg.debug(f"len(q_df): {len(q_df)}")

    logg.debug(f"q_df.fscore.std(): {q_df.fscore.std()}")
    logg.debug(f"q_df.fscore.var(): {q_df.fscore.var()}")
    logg.debug(f"q_df.fscore.min(): {q_df.fscore.min()}")
    logg.debug(f"q_df.fscore.max(): {q_df.fscore.max()}")
    logg.debug(f"q_df.fscore.mean(): {q_df.fscore.mean()}")
    q_df.fscore.plot.hist()

    ####################################

    plot_folder = Path("plot_results")

    hypa_grid: Dict[str, Any] = {}
    # hypa_grid["batch_size"] = [16, 32, 64, 128]
    # hypa_grid["batch_size"] = [32, 64, 128]
    # hypa_grid["epoch_num"] = [15, 30, 60]

    hypa_grid["filters"] = [10, 20, 30, 32, 64, 128]
    hypa_grid["kernel_size"] = ["01", "02", "03"]
    hypa_grid["pool_size"] = ["01", "02"]
    hypa_grid["dense_width"] = [16, 32, 64, 128]
    hypa_grid["dropout"] = ["01", "02"]
    hypa_grid["batch_size"] = [16, 32, 64]
    hypa_grid["epoch_num"] = [15, 30, 60]
    # hypa_grid["lr"] = ["01", "02", "03"]
    hypa_grid["lr"] = ["01", "02"]
    hypa_grid["opt"] = ["a1", "r1"]

    ds = ["mel01", "mel02", "mel03", "mel04"]
    # ds.extend(["mfcc01", "mfcc02", "mfcc03", "mfcc04"])
    hypa_grid["dataset"] = ds
    # hypa_grid["dataset"] = ["mfcc01", "mfcc02", "mfcc03", "mfcc04"]
    hypa_grid["words"] = ["f2", "f1", "num", "dir", "k1", "w2", "all"]
    # hypa_grid["words"] = ["num", "f1"]

    a1l_hp_to_plot = list(combinations(hypa_grid.keys(), 3))
    logg.debug(f"len(a1l_hp_to_plot): {len(a1l_hp_to_plot)}")

    # all_hp_to_plot = [["batch_size", "dense_width", "dropout"]]
    # all_hp_to_plot = [["kernel_size", "dense_width", "dropout"]]
    # all_hp_to_plot = [["epoch_num", "dataset", "words"]]
    all_hp_to_plot = [["epoch_num", "dataset", "lr"]]

    for hp_to_plot in tqdm(all_hp_to_plot[:1]):
        lab_values = [hypa_grid[hptp] for hptp in hp_to_plot]

        labels_dim = [len(lab) for lab in lab_values]

        f_mean = np.zeros(labels_dim)
        # logg.debug(f"f_mean.shape: {f_mean.shape}")
        f_min = np.zeros(labels_dim)
        f_max = np.zeros(labels_dim)
        f_std = np.zeros(labels_dim)

        for iz, vz in enumerate(lab_values[2]):
            # logg.debug(f"\nvz: {vz}")

            fig, ax = plt.subplots(figsize=(8, 8))

            for iy, vy in enumerate(lab_values[1]):
                for ix, vx in enumerate(lab_values[0]):
                    # logg.debug(f"\nvx: {vx} vy: {vy} vz: {vz}")

                    q_str = f"({hp_to_plot[0]} == '{vx}')"
                    q_str += f" and ({hp_to_plot[1]} == '{vy}')"
                    q_str += f" and ({hp_to_plot[2]} == '{vz}')"
                    # logg.debug(f"q_str: {q_str}")

                    q_df = results_df.query(q_str)
                    # logg.debug(f"len(q_df): {len(q_df)}")
                    if len(q_df) == 0:
                        continue

                    # logg.debug(f"q_df:\n{q_df.sort_values('fscore').head(10)}")
                    # logg.debug(f"q_df:\n{q_df.sort_values('fscore').tail(10)}")

                    f_mean[ix, iy, iz] = q_df.fscore.mean()
                    f_min[ix, iy, iz] = q_df.fscore.min()
                    f_max[ix, iy, iz] = q_df.fscore.max()
                    f_std[ix, iy, iz] = q_df.fscore.std()

            # logg.debug(f"f_mean:\n{f_mean}")
            plot_double_data(
                lab_values[:2],
                hp_to_plot,
                f_mean[:, :, iz],
                f_min[:, :, iz],
                f_max[:, :, iz],
                f_std[:, :, iz],
                vz,
                ax,
            )

            fig_name = "Fscore"
            fig_name += f"_{hp_to_plot[2]}_{vz}"
            fig_name += f"_{hp_to_plot[1]}_{vy}"
            fig_name += f"_{hp_to_plot[0]}_{vx}"
            fig_path = plot_folder / fig_name
            fig.savefig(fig_path)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_triple_data(ax, lab_values, hp_to_plot, f_mean, f_min, f_max, f_std)

    plt.show()


def make_plots_cnn() -> None:
    """TODO: what is make_plots_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.make_plots_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start make_plots_cnn")

    results_df = build_cnn_results_df()

    plot_folder = Path("plot_results")

    hypa_grid: Dict[str, Any] = {}
    hypa_grid["filters"] = [10, 20, 30, 32, 64, 128]
    hypa_grid["kernel_size"] = ["01", "02", "03"]
    hypa_grid["pool_size"] = ["01", "02"]
    hypa_grid["dense_width"] = [16, 32, 64, 128]
    hypa_grid["dropout"] = ["01", "02"]
    hypa_grid["batch_size"] = [16, 32, 64]
    hypa_grid["epoch_num"] = [15, 30, 60]
    hypa_grid["lr"] = ["01", "02", "03"]
    hypa_grid["opt"] = ["a1", "r1"]

    ds = ["mel01", "mel02", "mel03", "mel04"]
    ds.extend(["mfcc01", "mfcc02", "mfcc03", "mfcc04"])
    hypa_grid["dataset"] = ds
    # hypa_grid["dataset"] = ["mfcc01", "mfcc02", "mfcc03", "mfcc04"]
    hypa_grid["words"] = ["f2", "f1", "num", "dir", "k1", "w2", "all"]

    all_hp_to_plot = [["epoch_num", "dataset", "lr", "filters"]]

    for hp_to_plot in tqdm(all_hp_to_plot[:1]):
        outer_hp = hp_to_plot[-1]
        inner_hp = hp_to_plot[:-1]
        logg.debug(f"outer_hp: {outer_hp} inner_hp: {inner_hp}")

        outer_values = hypa_grid[outer_hp]
        outer_dim = len(outer_values)

        nrows, ncols = find_rowcol(outer_dim)
        base_figsize = 10
        figsize = (ncols * base_figsize, nrows * base_figsize)
        logg.debug(f"figsize: {figsize}")
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        # make flat_ax always iterable
        flat_ax = list(axes.flat) if outer_dim > 1 else [axes]

        for iv, outer_value in enumerate(outer_values):
            lab_values = [hypa_grid[hptp] for hptp in inner_hp]
            labels_dim = [len(lab) for lab in lab_values]

            f_mean = np.zeros(labels_dim)
            f_min = np.zeros(labels_dim)
            f_max = np.zeros(labels_dim)
            f_std = np.zeros(labels_dim)

            for iz, vz in enumerate(lab_values[2]):
                for iy, vy in enumerate(lab_values[1]):
                    for ix, vx in enumerate(lab_values[0]):
                        q_str = f"({hp_to_plot[0]} == '{vx}')"
                        q_str += f" and ({hp_to_plot[1]} == '{vy}')"
                        q_str += f" and ({hp_to_plot[2]} == '{vz}')"
                        q_df = results_df.query(q_str)

                        f_mean[ix, iy, iz] = q_df.fscore.mean()
                        f_min[ix, iy, iz] = q_df.fscore.min()
                        f_max[ix, iy, iz] = q_df.fscore.max()
                        f_std[ix, iy, iz] = q_df.fscore.std()

            plot_triple_data(
                flat_ax[iv],
                lab_values,
                hp_to_plot,
                f_mean,
                f_min,
                f_max,
                f_std,
                outer_hp,
                outer_value,
            )

        fig_name = "Fscore"
        fig_name += f"_{hp_to_plot[2]}_{vz}"
        fig_name += f"_{hp_to_plot[1]}_{vy}"
        fig_name += f"_{hp_to_plot[0]}_{vx}"
        fig_name += f"_{outer_hp}.pdf"
        fig_path = plot_folder / fig_name
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)


def evaluate_model_cnn(args):
    """TODO: what is evaluate_model_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_model_cnn")

    train_words_type = args.train_words_type
    dataset = "mel01"

    # magic to fix the GPUs
    setup_gpus()

    # setup the parameters
    hypa: Dict[str, Union[str, int]] = {}
    hypa["base_dense_width"] = 32
    hypa["base_filters"] = 20
    hypa["batch_size"] = 32
    hypa["dropout_type"] = "01"
    hypa["epoch_num"] = 16
    hypa["kernel_size_type"] = "02"
    hypa["pool_size_type"] = "02"
    hypa["learning_rate_type"] = "02"
    hypa["optimizer_type"] = "a1"
    hypa["dataset"] = dataset
    hypa["words"] = train_words_type

    # get the words
    train_words = words_types[train_words_type]

    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        raise FileNotFoundError

    model = tf.keras.models.load_model(model_path)
    model.summary()

    # input data
    processed_path = Path("data_proc") / f"{dataset}"
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


def evaluate_audio_cnn(args):
    """TODO: what is evaluate_audio_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_audio_cnn")
    logg.debug("Start evaluate_audio_cnn")

    # magic to fix the GPUs
    setup_gpus()

    # need to know on which dataset the model was trained to compute specs
    dataset_name = "mel01"

    # words that the dataset was trained on
    train_words_type = args.train_words_type
    train_words = words_types[train_words_type]

    # permutation from sorted to your wor(l)d order
    perm_pred = compute_permutation(train_words)

    rec_words_type = args.rec_words_type
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
    specs = []
    spec_dict = get_spec_dict()
    spec_kwargs = spec_dict[dataset_name]
    p2d_kwargs = {"ref": np.max}
    for word in rec_words:
        # get the name
        audio_path = audio_folder / audio_path_fmt.format(word)

        # convert it to mel
        log_spec = wav2mel(audio_path, spec_kwargs, p2d_kwargs)
        img_spec = log_spec.reshape((*log_spec.shape, 1))
        logg.debug(f"img_spec.shape: {img_spec.shape}")  # img_spec.shape: (128, 32, 1)

        specs.append(log_spec)

    # the data needs to look like this data['testing'].shape: (735, 128, 32, 1)
    # data = log_spec.reshape((1, *log_spec.shape, 1))
    data = np.stack(specs)
    logg.debug(f"data.shape: {data.shape}")

    hypa: Dict[str, Union[str, int]] = {}
    hypa["base_dense_width"] = 32
    hypa["base_filters"] = 20
    hypa["batch_size"] = 32
    hypa["dropout_type"] = "01"
    hypa["epoch_num"] = 16
    hypa["kernel_size_type"] = "02"
    hypa["pool_size_type"] = "02"
    hypa["learning_rate_type"] = "02"
    hypa["optimizer_type"] = "a1"
    hypa["dataset"] = dataset_name
    hypa["words"] = train_words_type

    # get the words
    train_words = words_types[train_words_type]

    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    # model_folder = Path("trained_models")
    model_folder = Path("saved_models")
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        raise FileNotFoundError

    model = tf.keras.models.load_model(model_path)
    model.summary()

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
        plot_pred(
            pred[i][perm_pred],
            train_words,
            axes[i][2],
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


def delete_bad_models_cnn(args) -> None:
    """TODO: what is delete_bad_models_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start delete_bad_models_cnn")

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
                model_path.unlink()
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


#####################################################################
#             TRA model
#####################################################################


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
    logg.info(f"{df.sort_values('fscore', ascending=False)[:30]}")


def delete_bad_models_transfer(args: argparse.Namespace) -> None:
    """TODO: what is delete_bad_models_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models_transfer")
    # logg.setLevel("INFO")
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
        res_full = json.loads(res_full_path.read_text())
        fscore = res_full["fscore"]

        if fscore < f_tresh:
            model_path = trained_folder / f"{model_name}.h5"

            if model_path.exists():
                model_path.unlink()
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


#####################################################################
#             ATT model
#####################################################################


def evaluate_results_attention() -> None:
    """TODO: what is evaluate_results_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_attention")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_results_attention")

    pandito: Dict[str, List[str]] = {
        "words": [],
        "dataset": [],
        "conv": [],
        "dropout": [],
        "kernel": [],
        "lstm": [],
        "att": [],
        "query": [],
        "dense": [],
        "lr": [],
        "optimizer": [],
        "batch": [],
        "epoch": [],
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
        if not model_name.startswith("ATT"):
            continue
        logg.debug(f"model_name: {model_name}")

        res_path = model_folder / "results_recap.json"
        if not res_path.exists():
            logg.info(f"Skipping res_path: {res_path}, not found")
            continue
        res = json.loads(res_path.read_text())

        recap_path = model_folder / "recap.json"
        if not recap_path.exists():
            logg.info(f"Skipping recap_path: {recap_path}, not found")
            continue
        recap = json.loads(recap_path.read_text())

        if res["results_recap_version"] == "001":
            logg.debug("\nWHAT ARE YOU DOING using this version\n")

        hypa: Dict[str, Any] = recap["hypa"]

        pandito["words"].append(hypa["words_type"])
        pandito["dataset"].append(hypa["dataset_name"])
        pandito["conv"].append(hypa["conv_size_type"])
        pandito["dropout"].append(hypa["dropout_type"])
        pandito["kernel"].append(hypa["kernel_size_type"])
        pandito["lstm"].append(hypa["lstm_units_type"])
        pandito["att"].append(hypa["att_sample_type"])
        pandito["query"].append(hypa["query_style_type"])
        pandito["dense"].append(hypa["dense_width_type"])

        pandito["lr"].append(hypa["learning_rate_type"])
        pandito["optimizer"].append(hypa["optimizer_type"])
        pandito["batch"].append(hypa["batch_size_type"])
        pandito["epoch"].append(hypa["epoch_num_type"])

        pandito["use_val"].append(recap["use_validation"])
        pandito["loss"].append(res["loss"])
        pandito["cat_acc"].append(res["categorical_accuracy"])
        pandito["precision"].append(res["precision"])
        pandito["recall"].append(res["recall"])
        pandito["fscore"].append(res["fscore"])

        pandito["model_name"].append(res["model_name"])

    pd.set_option("max_colwidth", 100)
    df = pd.DataFrame(pandito)
    filtered = df.sort_values("fscore", ascending=False).head(30)
    logg.info(f"{filtered}")

    # produce a dict for recreating the training
    for index, row in filtered.iterrows():
        hp = "    {"
        hp += f"        'words_type': '{row['words']}',"
        hp += f"        'dataset_name': '{row['dataset']}',"
        hp += f"        'conv_size_type': '{row['conv']}',"
        hp += f"        'dropout_type': '{row['dropout']}',"
        hp += f"        'kernel_size_type': '{row['kernel']}',"
        hp += f"        'lstm_units_type': '{row['lstm']}',"
        hp += f"        'att_sample_type': '{row['att']}',"
        hp += f"        'query_style_type': '{row['query']}',"
        hp += f"        'dense_width_type': '{row['dense']}',"
        hp += f"        'learning_rate_type': '{row['lr']}',"
        hp += f"        'optimizer_type': '{row['optimizer']}',"
        hp += f"        'batch_size_type': '{row['batch']}',"
        hp += f"        'epoch_num_type': '{row['epoch']}',"
        hp += "    },"
        # logg.debug(f"hp: {hp}")

    # compare the rankings with and without validation
    best_val = df.query("use_val==True").sort_values("fscore", ascending=False)
    best_noval = df.query("use_val==False").sort_values("fscore", ascending=False)

    rank: Dict[str, List[int]] = {}
    num_head = 30
    for i, (_, row) in enumerate(best_val.head(num_head).iterrows()):
        model_name = row["model_name"]
        hypa_str = model_name[4:]  # chop off ATT_ at the beginning
        # logg.debug(f"hypa_str: {hypa_str}")
        rank[hypa_str] = [i, 0]

    for i, (_, row) in enumerate(best_noval.head(num_head).iterrows()):
        model_name = row["model_name"]
        hypa_str = model_name[4:]  # chop off ATT_ at the beginning
        hypa_str = hypa_str[:-6]  # chop off _noval at the end
        # logg.debug(f"hypa_str: {hypa_str}")
        rank[hypa_str][1] = i

    rank_pd: Dict[str, Any] = {
        "hypa_str": list(rank.keys()),
        "rank_val": [rank[hs][0] for hs in rank],
        "rank_noval": [rank[hs][1] for hs in rank],
    }
    rank_df = pd.DataFrame(rank_pd)
    ranked = rank_df.sort_values("rank_val", ascending=True).head(30)
    logg.debug(f"ranked:\n{ranked}")


def evaluate_attention_weights(train_words_type: str) -> None:
    """TODO: what is evaluate_attention_weights doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_attention_weights")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_attention_weights")

    # magic to fix the GPUs
    setup_gpus()

    dataset_name = "mela1"

    hypa: Dict[str, str] = {}

    hypa["conv_size_type"] = "02"
    hypa["dropout_type"] = "01"
    hypa["kernel_size_type"] = "02"
    hypa["lstm_units_type"] = "01"
    hypa["att_sample_type"] = "01"
    hypa["query_style_type"] = "01"
    hypa["dense_width_type"] = "01"
    hypa["optimizer_type"] = "a1"
    hypa["learning_rate_type"] = "01"
    hypa["batch_size_type"] = "01"
    hypa["epoch_num_type"] = "01"

    hypa["dataset_name"] = dataset_name
    hypa["words_type"] = train_words_type

    use_validation = True

    model_name = build_attention_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # load the model
    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"

    # model = tf.keras.models.load_model(model_path)
    # https://github.com/keras-team/keras/issues/5088#issuecomment-401498334
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "backend": tf.keras.backend,
        },
    )

    att_weight_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer("output").output,
            model.get_layer("att_softmax").output,
        ],
    )
    att_weight_model.summary()
    logg.debug(f"att_weight_model.outputs: {att_weight_model.outputs}")

    # get the words
    train_words = words_types[train_words_type]
    logg.debug(f"train_words: {train_words}")
    perm_pred = compute_permutation(train_words)

    # select a word
    correct_index = 0
    word = train_words[correct_index]

    # input data
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{dataset_name}"
    data, labels = load_processed(processed_path, [word])

    # get prediction and attention weights
    pred, att_weights = att_weight_model.predict(data["testing"])
    logg.debug(f"att_weights.shape: {att_weights.shape}")
    logg.debug(f"att_weights[0].shape: {att_weights[0].shape}")

    # plot the spectrogram and the weights
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    fig.suptitle(f"Attention weights and prediction for {word}")

    # which word in the dataset to plot
    word_id = 0

    # extract the spectrogram, data shape (?, x, y, 1)
    word_data = data["testing"][word_id][:, :, -1]
    logg.debug(f"word_data.shape: {word_data.shape}")
    title = f"Spectrogram for {word}"
    plot_spec(word_data, axes[0], title=title)

    # plot the weights
    word_att_weights = att_weights[word_id]
    title = f"Attention weights for {word}"
    plot_att_weights(word_att_weights, axes[1], title)

    # plot the predictions
    word_pred = pred[word_id]
    # permute the prediction from sorted to the order you have
    word_pred = word_pred[perm_pred]
    pred_index = np.argmax(word_pred)
    title = f"Predictions for {word}"
    plot_pred(word_pred, train_words, axes[2], title, pred_index)

    fig.tight_layout()

    plt.show()


def run_evaluate(args) -> None:
    """TODO: What is evaluate doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate")
    logg.debug("Starting run_evaluate")

    evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type
    rec_words_type = args.rec_words_type

    pd.set_option("max_colwidth", 100)

    if evaluation_type == "results_cnn":
        evaluate_results_cnn(args)
    elif evaluation_type == "make_plots_cnn":
        make_plots_cnn()
    elif evaluation_type == "model_cnn":
        evaluate_model_cnn(args)
    elif evaluation_type == "audio_cnn":
        evaluate_audio_cnn(args)
    elif evaluation_type == "delete_cnn":
        delete_bad_models_cnn(args)

    elif evaluation_type == "results_transfer":
        evaluate_results_transfer(args)
    elif evaluation_type == "audio_transfer":
        evaluate_audio_transfer(train_words_type, rec_words_type)
    elif evaluation_type == "delete_transfer":
        delete_bad_models_transfer(args)

    elif evaluation_type == "results_attention":
        evaluate_results_attention()
    elif evaluation_type == "attention_weights":
        evaluate_attention_weights(train_words_type)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate(args)
