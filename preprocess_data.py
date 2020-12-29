import argparse
import logging

from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore

# from random import seed as rseed
# from timeit import default_timer as timer

import librosa  # type: ignore
from pathlib import Path

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
    recap = "python3 preprocess_data.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def wav2mfcc(wav_path, mfcc_kwargs):
    """TODO: what is wav2mfcc doing?"""
    sig, sample_rate = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(sig, sr=sample_rate)
    log_mfcc = librosa.power_to_db(mfcc, **mfcc_kwargs)

    # the shape is not consistent, pad it
    pad_needed = 32 - log_mfcc.shape[1]
    # number of values padded to the edges of each axis.
    pad_width = ((0, 0), (0, pad_needed))
    padded_log_mfcc = np.pad(log_mfcc, pad_width=pad_width)
    return padded_log_mfcc


def preprocess_mfcc():
    """TODO: what is preprocess_mfcc doing?"""
    logg = logging.getLogger(f"c.{__name__}.preprocess_mfcc")
    logg.debug("Start preprocess_mfcc")

    # args for the mfcc spec
    mfcc_kwargs = {"ref": np.max}

    # original / processed dataset base locations
    dataset_path = Path("data_raw")
    processed_path = Path("data_proc/mfcc")
    if not processed_path.exists():
        processed_path.mkdir(parents=True, exist_ok=True)

    # list of file names for validation
    validation_path = dataset_path / "validation_list.txt"
    validation_names = []
    with validation_path.open() as fvp:
        for line in fvp:
            validation_names.append(line.strip())
    # logg.debug(f"validation_names: {validation_names[:10]}")

    # list of file names for testing
    testing_path = dataset_path / "testing_list.txt"
    testing_names = []
    with testing_path.open() as fvp:
        for line in fvp:
            testing_names.append(line.strip())
    # logg.debug(f"testing_names: {testing_names[:10]}")

    words = ALL_WORDS
    # words = ["happy", "learn"]
    for word in words:
        word_in_path = dataset_path / word
        logg.debug(f"Processing folder: {word_in_path}")

        word_mfcc = {"validation": [], "training": [], "testing": []}

        all_wavs = list(word_in_path.iterdir())
        for wav_path in tqdm(all_wavs):
            # logg.debug(f"wav_path: {wav_path}")
            wav_name = f"{word}/{wav_path.name}"
            # logg.debug(f"wav_name: {wav_name}")
            log_mfcc = wav2mfcc(wav_path, mfcc_kwargs)

            if wav_name in validation_names:
                which = "validation"
            elif wav_name in testing_names:
                which = "testing"
            else:
                which = "training"

            word_mfcc[which].append(log_mfcc)

        for which in ["validation", "training", "testing"]:
            np_mfcc = np.array(word_mfcc[which], dtype=object)
            logg.debug(f"{which} np_mfcc.shape: {np_mfcc.shape}")
            word_out_path = processed_path / f"{word}_{which}.npy"
            np.save(word_out_path, np_mfcc)


def load_processed(processed_path, words):
    """TODO: what is load_processed doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_processed")
    logg.debug("Start load_processed")

    loaded = {"validation": [], "training": [], "testing": []}
    for word in words:
        loaded[word] = {}
        for which in ["validation", "training", "testing"]:
            word_path = processed_path / f"{word}_{which}.npy"
            loaded[which].append(np.load(word_path, allow_pickle=True))

    data = {}
    for which in ["validation", "training", "testing"]:
        data[which] = np.vstack(loaded[which])
        logg.debug(f"data[{which}].shape: {data[which].shape}")

    return data


def test_load_processed():
    """TODO: what is test_load_processed doing?"""
    logg = logging.getLogger(f"c.{__name__}.test_load_processed")
    logg.debug("Start test_load_processed")
    processed_path = Path("data_proc/mfcc")
    words = ["happy", "learn"]
    words = ALL_WORDS
    load_processed(processed_path, words)


def run_preprocess_data(args):
    """TODO: What is preprocess_data doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_preprocess_data")
    logg.debug("Starting run_preprocess_data")

    # preprocess_mfcc()
    test_load_processed()


if __name__ == "__main__":
    args = setup_env()
    run_preprocess_data(args)
