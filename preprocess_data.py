import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm  # type: ignore
import librosa  # type: ignore

from sklearn.preprocessing import LabelEncoder  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import numpy as np  # type: ignore

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


def wav2mfcc(wav_path, mfcc_kwargs, p2d_kwargs):
    """TODO: what is wav2mfcc doing?"""
    sig, sample_rate = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(sig, sr=sample_rate, **mfcc_kwargs)
    log_mfcc = librosa.power_to_db(mfcc, **p2d_kwargs)

    # print(f"sig.shape: {sig.shape}")
    # print(f"log_mfcc.shape: {log_mfcc.shape}")

    # the shape is not consistent, pad it
    pad_needed = 16384 // mfcc_kwargs["hop_length"] - log_mfcc.shape[1]
    # print(f"pad_needed: {pad_needed}")
    # number of values padded to the edges of each axis.
    pad_width = ((0, 0), (0, pad_needed))
    padded_log_mfcc = np.pad(log_mfcc, pad_width=pad_width)
    return padded_log_mfcc


def wav2mel(wav_path, mel_kwargs, p2d_kwargs):
    """TODO: what is wav2mel doing?"""
    sig, sample_rate = librosa.load(wav_path, sr=None)
    mel = librosa.feature.melspectrogram(sig, sr=sample_rate, **mel_kwargs)
    log_mel = librosa.power_to_db(mel, **p2d_kwargs)

    # the shape is not consistent, pad it
    pad_needed = 16384 // mel_kwargs["hop_length"] - log_mel.shape[1]
    # print(f"pad_needed: {pad_needed}")
    # number of values padded to the edges of each axis.
    pad_width = ((0, 0), (0, pad_needed))
    padded_log_mel = np.pad(log_mel, pad_width=pad_width)
    return padded_log_mel


def get_spec_dict():
    """TODO: what is get_spec_dict doing?"""

    spec_dict = {
        "mfcc01": {"n_mfcc": 20, "n_fft": 2048, "hop_length": 512},
        "mfcc02": {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512},
        "mfcc03": {"n_mfcc": 40, "n_fft": 2048, "hop_length": 256},
        "mfcc04": {"n_mfcc": 80, "n_fft": 1024, "hop_length": 128},
        "mfcc05": {"n_mfcc": 10, "n_fft": 4096, "hop_length": 1024},
        "mel01": {"n_mels": 128, "n_fft": 2048, "hop_length": 512},
        "mel02": {"n_mels": 64, "n_fft": 4096, "hop_length": 1024},
        "mel03": {"n_mels": 64, "n_fft": 2048, "hop_length": 512},
        "mel04": {"n_mels": 64, "n_fft": 1024, "hop_length": 256},
    }
    return spec_dict


def preprocess_spec():
    """TODO: what is preprocess_spec doing?"""
    logg = logging.getLogger(f"c.{__name__}.preprocess_spec")
    logg.debug("Start preprocess_spec")

    dataset_name = "mel01"
    logg.debug(f"dataset_name: {dataset_name}")

    # args for the power_to_db function
    p2d_kwargs = {"ref": np.max}

    # args for the mfcc spec
    spec_dict = get_spec_dict()
    spec_kwargs = spec_dict[dataset_name]

    # original / processed dataset base locations
    dataset_path = Path("data_raw")
    processed_path = Path(f"data_proc/{dataset_name}")
    if not processed_path.exists():
        processed_path.mkdir(parents=True, exist_ok=True)

    # write info regarding the dataset generation
    recap = {}
    recap["spec_kwargs"] = spec_kwargs
    logg.debug(f"recap: {recap}")
    recap_path = processed_path / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

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

    words = words_types["all"]
    # words = words_types["f2"]
    # words = words_types["num"]
    # words = words_types["dir"]
    for word in words:
        word_in_path = dataset_path / word
        logg.debug(f"Processing folder: {word_in_path}")

        word_spec = {"validation": [], "training": [], "testing": []}

        all_wavs = list(word_in_path.iterdir())
        for wav_path in tqdm(all_wavs):
            # logg.debug(f"wav_path: {wav_path}")
            wav_name = f"{word}/{wav_path.name}"
            # logg.debug(f"wav_name: {wav_name}")

            if dataset_name.startswith("mfcc"):
                log_spec = wav2mfcc(wav_path, spec_kwargs, p2d_kwargs)
            elif dataset_name.startswith("mel"):
                log_spec = wav2mel(wav_path, spec_kwargs, p2d_kwargs)

            if wav_name in validation_names:
                which = "validation"
            elif wav_name in testing_names:
                which = "testing"
            else:
                which = "training"

            word_spec[which].append(log_spec)

        for which in ["training", "validation", "testing"]:
            # logg.debug(f"word_spec[{which}][0].shape: {word_spec[which][0].shape}")
            np_spec = np.stack(word_spec[which])
            # logg.debug(f"{which} np_spec.shape: {np_spec.shape}")
            word_out_path = processed_path / f"{word}_{which}.npy"
            np.save(word_out_path, np_spec)


def load_processed(processed_path, words):
    """TODO: what is load_processed doing?"""
    # logg = logging.getLogger(f"c.{__name__}.load_processed")
    # logg.debug("Start load_processed")

    loaded_words = {"validation": [], "training": [], "testing": []}
    loaded_labels = {"validation": [], "training": [], "testing": []}
    for word in words:
        loaded_words[word] = {}
        for which in ["training", "validation", "testing"]:
            word_path = processed_path / f"{word}_{which}.npy"
            word_data = np.load(word_path, allow_pickle=True)
            loaded_words[which].append(word_data)
            word_label = np.full(word_data.shape[0], fill_value=word)
            loaded_labels[which].append(word_label)

    data = {}
    labels = {}
    for which in ["training", "validation", "testing"]:
        # data have shape (*, 20, 32) so we use vstack
        data[which] = np.vstack(loaded_words[which])
        # labels have shape (*, ) so we use hstack
        labels[which] = np.hstack(loaded_labels[which])

        # logg.debug(f"data[{which}].shape: {data[which].shape}")
        # logg.debug(f"labels[{which}].shape: {labels[which].shape}")

    for which in ["training", "validation", "testing"]:
        data[which] = np.reshape(data[which], (*data[which].shape, 1))
        y = LabelEncoder().fit_transform(labels[which])
        labels[which] = to_categorical(y)
        # logg.debug(f"data[{which}].shape: {data[which].shape}")
        # logg.debug(f"labels[{which}].shape: {labels[which].shape}")

    return data, labels


def test_load_processed():
    """TODO: what is test_load_processed doing?"""
    logg = logging.getLogger(f"c.{__name__}.test_load_processed")
    logg.debug("Start test_load_processed")
    processed_path = Path("data_proc/mfcc01")
    words = ["happy", "learn"]
    # words = WORDS_ALL
    load_processed(processed_path, words)


def run_preprocess_data(args):
    """TODO: What is preprocess_data doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_preprocess_data")
    logg.debug("Starting run_preprocess_data")

    preprocess_spec()
    # test_load_processed()


if __name__ == "__main__":
    args = setup_env()
    run_preprocess_data(args)
