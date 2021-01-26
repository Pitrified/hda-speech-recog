import argparse
import logging

import numpy as np  # type: ignore

# from random import seed as rseed
# from timeit import default_timer as timer

import matplotlib.pyplot as plt  # type: ignore
import librosa  # type: ignore
from pathlib import Path

from plot_utils import plot_spec
from plot_utils import plot_waveform
from utils import setup_logger
from utils import words_types


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

    parser.add_argument(
        "-s", "--rand_seed", type=int, default=-1, help="random seed to use"
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env():
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 visualize.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def visualize_spec():
    """TODO: what is visualize_spec doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_spec")
    logg.debug("Start visualize_spec")

    dataset_path = Path("data_raw")
    logg.debug(f"dataset_path: {dataset_path}")

    sample_path = dataset_path / "happy" / "0a2b400e_nohash_0.wav"
    logg.debug(f"sample_path: {sample_path}")
    sample_sig, sample_rate = librosa.load(sample_path, sr=None)
    logg.debug(f"sample_sig.shape: {sample_sig.shape}")

    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    plot_waveform(sample_sig, ax[0], sample_rate=sample_rate)

    sample_melspec = librosa.feature.melspectrogram(sample_sig, sr=sample_rate)
    logg.debug(f"sample_melspec.shape: {sample_melspec.shape}")
    sample_log_melspec = librosa.power_to_db(sample_melspec, ref=np.max)
    logg.debug(f"sample_log_melspec.shape: {sample_log_melspec.shape}")
    plot_spec(sample_log_melspec, ax[1])

    sample_mfcc = librosa.feature.mfcc(sample_sig, sr=sample_rate)
    logg.debug(f"sample_mfcc.shape: {sample_mfcc.shape}")
    sample_log_mfcc = librosa.power_to_db(sample_mfcc, ref=np.max)
    logg.debug(f"sample_log_mfcc.shape: {sample_log_mfcc.shape}")
    plot_spec(sample_log_mfcc, ax[2])

    plt.tight_layout()


def visualize_datasets():
    """TODO: what is visualize_datasets doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_datasets")
    logg.debug("Start visualize_datasets")

    # show different datasets

    # datasets = ["mfcc01", "mfcc02", "mfcc03", "mfcc04", "mfcc05", "mfcc06"]
    datasets = [
        "mel01",
        "mel02",
        "mel03",
        "mel04",
        "mel05",
        "mel06",
        "mel07",
        "mel08",
        "mel09",
        "mel10",
        "mel11",
        "melc1",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(12, 12))
    fig.suptitle("happy")

    for i, ax in enumerate(axes.flat[: len(datasets)]):
        dataset_name = datasets[i]
        processed_path = Path(f"data_proc/{dataset_name}")
        word_path = processed_path / "happy_training.npy"
        word_data = np.load(word_path, allow_pickle=True)
        logg.debug(f"{dataset_name} word shape: {word_data[0].shape}")
        title = f"{dataset_name} shape {word_data[0].shape}"
        plot_spec(word_data[0], ax, title=title)

    fig.tight_layout()

    # show different words

    for dataset_name in datasets:
        processed_path = Path(f"data_proc/{dataset_name}")
        words = words_types["f1"]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
        title = f"{dataset_name} shape {word_data[0].shape}"
        fig.suptitle(title)
        for i, ax in enumerate(axes.flat):
            word_path = processed_path / f"{words[i]}_training.npy"
            word_data = np.load(word_path, allow_pickle=True)
            # logg.debug(f"{dataset_name} word shape: {word_data[0].shape}")
            plot_spec(word_data[0], ax, title=words[i])

        fig.tight_layout()


def run_visualize(args):
    """TODO: What is visualize doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_visualize")
    logg.debug("Starting run_visualize")

    # visualize_spec()
    visualize_datasets()

    plt.show()


if __name__ == "__main__":
    args = setup_env()
    run_visualize(args)
