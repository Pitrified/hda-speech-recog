from pathlib import Path
from sklearn.preprocessing import LabelEncoder  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tqdm import tqdm  # type: ignore
import argparse
import json
import librosa  # type: ignore
import logging
import multiprocessing as mp
import numpy as np  # type: ignore
import typing as ty

from utils import get_val_test_list
from utils import setup_logger
from utils import words_types


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-pt",
        "--preprocess_type",
        type=str,
        default="preprocess_spec",
        choices=["preprocess_spec", "compose_spec", "preprocess_split"],
        help="Which preprocess to perform",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="mel01",
        help="Name of the output dataset folder",
    )

    parser.add_argument(
        "-wt",
        "--words_type",
        type=str,
        default="f2",
        choices=words_types.keys(),
        help="Words to preprocess",
    )

    parser.add_argument(
        "-fp",
        "--force_preprocess",
        dest="force_preprocess",
        action="store_true",
        help="Force the preprocess and overwrite the previous results",
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


##########################################################
#   Utils
##########################################################


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


def wav2mel(wav_path: Path, mel_kwargs, p2d_kwargs):
    """TODO: what is wav2mel doing?"""
    sig, sample_rate = librosa.load(wav_path, sr=None)
    mel = librosa.feature.melspectrogram(sig, sr=sample_rate, **mel_kwargs)
    log_mel = librosa.power_to_db(mel, **p2d_kwargs)

    # the shape is not consistent, pad it

    word_name = wav_path.parent.name

    # the bodgiest bodge that ever bodged
    if word_name.startswith("loudest") or word_name.endswith("loud"):
        pad_needed = 16384 // 2 // mel_kwargs["hop_length"] - log_mel.shape[1]
    else:
        pad_needed = 16384 // mel_kwargs["hop_length"] - log_mel.shape[1]

    pad_needed = max(0, pad_needed)

    # number of values padded to the edges of each axis.
    pad_width = ((0, 0), (0, pad_needed))
    padded_log_mel = np.pad(log_mel, pad_width=pad_width)

    # recap = f"len(sig) {len(sig)}"
    # recap += f" pad_needed {pad_needed}"
    # recap += f" padded_log_mel.shape {padded_log_mel.shape}"
    # print(recap)

    return padded_log_mel


##########################################################
#   Spec by fold
##########################################################


def get_spec_dict():
    """TODO: what is get_spec_dict doing?"""

    spec_dict = {
        "mfcc01": {"n_mfcc": 20, "n_fft": 2048, "hop_length": 512},  # (20, 32)
        "mfcc02": {"n_mfcc": 40, "n_fft": 2048, "hop_length": 512},  # (40, 32)
        "mfcc03": {"n_mfcc": 40, "n_fft": 2048, "hop_length": 256},  # (40, 64)
        "mfcc04": {"n_mfcc": 80, "n_fft": 1024, "hop_length": 128},  # (80, 128)
        "mfcc05": {"n_mfcc": 10, "n_fft": 4096, "hop_length": 1024},  # (10, 16)
        "mfcc06": {"n_mfcc": 128, "n_fft": 1024, "hop_length": 128},  # (128, 128)
        "mfcc07": {"n_mfcc": 128, "n_fft": 512, "hop_length": 128},  # (128, 128)
        "mfcc08": {"n_mfcc": 128, "n_fft": 2048, "hop_length": 128},  # (128, 128)
        "mel01": {"n_mels": 128, "n_fft": 2048, "hop_length": 512},  # (128, 32)
        "mel02": {"n_mels": 64, "n_fft": 4096, "hop_length": 1024},  # (64, 16)
        "mel03": {"n_mels": 64, "n_fft": 2048, "hop_length": 512},  # (64, 32)
        "mel04": {"n_mels": 64, "n_fft": 1024, "hop_length": 256},  # (64, 64)
        "mel05": {"n_mels": 128, "n_fft": 1024, "hop_length": 128},  # (128, 128)
        "mel06": {"n_mels": 128, "n_fft": 1024, "hop_length": 256},  # (128, 64)
        "mel07": {"n_mels": 128, "n_fft": 2048, "hop_length": 256},  # (128, 64)
        "mel08": {"n_mels": 128, "n_fft": 512, "hop_length": 256},  # (128, 64)
        "mel09": {"n_mels": 128, "n_fft": 512, "hop_length": 128},  # (128, 128)
        "mel10": {"n_mels": 128, "n_fft": 2048, "hop_length": 128},  # (128, 128)
        "mel11": {"n_mels": 128, "n_fft": 256, "hop_length": 128},  # (128, 128)
        "mel12": {"n_mels": 128, "n_fft": 4096, "hop_length": 256},  # (128, 64)
        "mel13": {"n_mels": 128, "n_fft": 512, "hop_length": 256},  # (128, 64)
        "mel14": {"n_mels": 128, "n_fft": 256, "hop_length": 256},  # (128, 64)
        "mel15": {"n_mels": 128, "n_fft": 3072, "hop_length": 256},  # (128, 64)
        "mela1": {"n_mels": 80, "n_fft": 1024, "hop_length": 128, "fmin": 40},
        "Lmel04": {"n_mels": 64, "n_fft": 512, "hop_length": 128},  # (64, 64)
    }

    return spec_dict


def preprocess_spec(
    which_dataset: str, words_type: str, force_preprocess: bool = False
) -> None:
    """TODO: what is preprocess_spec doing?"""
    logg = logging.getLogger(f"c.{__name__}.preprocess_spec")
    logg.debug("Start preprocess_spec")

    # args for the power_to_db function
    p2d_kwargs = {"ref": np.max}

    # args for the mfcc spec
    spec_dict = get_spec_dict()

    if which_dataset == "all":
        dataset_names = spec_dict.keys()
    elif which_dataset == "ch3":
        dataset_names = [
            "mel05",
            "mel09",
            "mel10",
            "mfcc06",
            "mfcc07",
            "mfcc08",
        ]
    else:
        dataset_names = [which_dataset]

    for dataset_name in dataset_names:
        logg.debug(f"\ndataset_name: {dataset_name}")

        spec_kwargs = spec_dict[dataset_name]

        # original / processed dataset base locations
        dataset_path = Path("data_raw")
        processed_path = Path("data_proc") / f"{dataset_name}"
        if not processed_path.exists():
            processed_path.mkdir(parents=True, exist_ok=True)

        # write info regarding the dataset generation
        recap = {}
        recap["spec_kwargs"] = spec_kwargs
        logg.debug(f"recap: {recap}")
        recap_path = processed_path / "recap.json"
        recap_path.write_text(json.dumps(recap, indent=4))

        validation_names, testing_names = get_val_test_list(dataset_path)

        words = words_types[words_type]
        for word in words:
            word_in_path = dataset_path / word
            logg.debug(f"Processing folder: {word_in_path}")

            word_spec: ty.Dict[str, ty.List[np.ndarray]] = {
                "training": [],
                "validation": [],
                "testing": [],
            }

            word_out_path = processed_path / f"{word}_testing.npy"
            if word_out_path.exists():
                logg.info(f"word_out_path {word_out_path} already preprocessed")
                if force_preprocess:
                    logg.debug("OVERWRITING the previous results")
                else:
                    continue

            all_wavs = list(word_in_path.iterdir())
            for wav_path in tqdm(all_wavs):
                # logg.debug(f"wav_path: {wav_path}")
                wav_name = f"{word}/{wav_path.name}"
                # logg.debug(f"wav_name: {wav_name}")

                if dataset_name.startswith("mfcc"):
                    log_spec = wav2mfcc(wav_path, spec_kwargs, p2d_kwargs)
                elif dataset_name.startswith("mel"):
                    log_spec = wav2mel(wav_path, spec_kwargs, p2d_kwargs)
                elif dataset_name.startswith("Lmel"):
                    log_spec = wav2mel(wav_path, spec_kwargs, p2d_kwargs)

                if wav_name in validation_names:
                    which = "validation"
                elif wav_name in testing_names:
                    which = "testing"
                else:
                    which = "training"

                word_spec[which].append(log_spec)

            for which in ["training", "validation", "testing"]:
                if len(word_spec[which]) == 0:
                    logg.debug(f"No specs for {which}")
                    continue

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
            if not word_path.exists():
                continue

            word_data = np.load(word_path, allow_pickle=True)
            loaded_words[which].append(word_data)
            word_label = np.full(word_data.shape[0], fill_value=word)
            loaded_labels[which].append(word_label)

    data = {}
    labels = {}
    for which in ["training", "validation", "testing"]:
        if len(loaded_words[which]) == 0:
            # FIXME this is ugly, put for which in folds as outermost cycle
            continue
        # data have shape (*, 20, 32) so we use vstack
        data[which] = np.vstack(loaded_words[which])
        # labels have shape (*, ) so we use hstack
        labels[which] = np.hstack(loaded_labels[which])

        # logg.debug(f"data[{which}].shape: {data[which].shape}")
        # logg.debug(f"labels[{which}].shape: {labels[which].shape}")

    for which in ["training", "validation", "testing"]:
        if len(loaded_words[which]) == 0:
            continue
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
    processed_path = Path("data_proc") / "mfcc01"
    words = ["happy", "learn"]
    # words = WORDS_ALL
    load_processed(processed_path, words)


##########################################################
#   Composed spec by fold
##########################################################


def get_compose_types():
    """TODO: what is get_compose_types doing?"""

    compose_types = {
        "melc1": ["mel06", "mel08"],
        "melc2": ["mel07", "mel12"],
        "melc3": ["mel13", "mel14"],
        "melc4": ["mel13", "mel15"],
    }

    return compose_types


def compose_spec(
    which_dataset: str, words_type: str, force_preprocess: bool = False
) -> None:
    """TODO: what is compose_spec doing?"""
    logg = logging.getLogger(f"c.{__name__}.compose_spec")
    # logg.setLevel("INFO")
    logg.debug("Start compose_spec")

    dataset_name_out = which_dataset

    compose_types = get_compose_types()

    # get the two dataset to compose
    dataset_name_1, dataset_name_2 = compose_types[dataset_name_out]

    # be sure that the datasets to compose are available
    preprocess_spec(dataset_name_1, words_type, force_preprocess)
    preprocess_spec(dataset_name_2, words_type, force_preprocess)

    processed_path_1 = Path("data_proc") / f"{dataset_name_1}"
    processed_path_2 = Path("data_proc") / f"{dataset_name_2}"

    processed_path_out = Path("data_proc") / f"{dataset_name_out}"
    if not processed_path_out.exists():
        processed_path_out.mkdir(parents=True, exist_ok=True)

    # write info regarding the dataset generation
    recap = {}
    recap["dataset_name_1"] = dataset_name_1
    recap["dataset_name_2"] = dataset_name_2
    recap["dataset_name_out"] = dataset_name_out
    logg.debug(f"recap: {recap}")
    recap_path = processed_path_out / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    words = words_types[words_type]

    for which in ["training", "validation", "testing"]:
        for word in words:
            word_path_1 = processed_path_1 / f"{word}_{which}.npy"
            word_path_2 = processed_path_2 / f"{word}_{which}.npy"

            word_data_1 = np.load(word_path_1, allow_pickle=True)
            word_data_2 = np.load(word_path_2, allow_pickle=True)
            # logg.debug(f"word_data_1.shape: {word_data_1.shape}")
            # logg.debug(f"word_data_2.shape: {word_data_2.shape}")

            word_comp = np.dstack((word_data_1, word_data_2))
            # logg.debug(f"word_comp.shape: {word_comp.shape}")

            word_path_out = processed_path_out / f"{word}_{which}.npy"
            np.save(word_path_out, word_comp)


def load_triple(
    data_paths: ty.Iterable[Path], words: ty.Iterable[str]
) -> ty.Tuple[ty.Dict[str, np.ndarray], ty.Dict[str, np.ndarray]]:
    """TODO: what is load_triple doing?"""
    # logg = logging.getLogger(f"c.{__name__}.load_triple")
    # logg.setLevel("INFO")
    # logg.debug("Start load_triple")

    all_loaded_words: ty.Dict[str, np.ndarray] = {}
    all_loaded_labels: ty.Dict[str, np.ndarray] = {}

    for which in ["training", "validation", "testing"]:
        # logg.debug(f"\nwhich: {which}")

        data: ty.List[np.ndarray] = []
        labels: ty.List[np.ndarray] = []

        for this_path in data_paths:
            # this_name = this_path.name
            # logg.debug(f"this_name: {this_name}")

            loaded_words = []
            loaded_labels = []
            for word in words:
                word_path = this_path / f"{word}_{which}.npy"
                word_data = np.load(word_path, allow_pickle=True)
                loaded_words.append(word_data)
                word_label = np.full(word_data.shape[0], fill_value=word)
                loaded_labels.append(word_label)

            data.append(np.vstack(loaded_words))
            labels.append(np.hstack(loaded_labels))
            # logg.debug(f"data[-1].shape: {data[-1].shape}")

        # stack the three specs as a 3 channel "image"
        data_3ch = np.stack(data, axis=-1)
        # logg.debug(f"data_3ch.shape: {data_3ch.shape}")
        # logg.debug(f"data_3ch[0].shape: {data_3ch[0].shape}")
        all_loaded_words[which] = data_3ch

        # the labels should be the same for all datasets
        # logg.debug(f"np.sum(labels[0]!=labels[1]): {np.sum(labels[0]!=labels[1])}")
        # logg.debug(f"np.sum(labels[1]!=labels[2]): {np.sum(labels[1]!=labels[2])}")

        # we transform with to_categorical into one hot encoded
        y = LabelEncoder().fit_transform(labels[0])
        all_loaded_labels[which] = to_categorical(y)
        # logg.debug(f"labels[0].shape: {labels[0].shape}")
        # logg.debug(f"all_loaded_labels[which].shape: {all_loaded_labels[which].shape}")
        # logg.debug(f"all_loaded_labels[which][0]: {all_loaded_labels[which][0]}")
        # logg.debug(f"all_loaded_labels[which][-1]: {all_loaded_labels[which][-1]}")

    return all_loaded_words, all_loaded_labels


def test_load_triple(args: argparse.Namespace) -> None:
    """TODO: what is test_load_triple doing?"""
    logg = logging.getLogger(f"c.{__name__}.test_load_triple")
    # logg.setLevel("INFO")
    logg.debug("Start test_load_triple")

    datasets_types = {
        "01": ["mel05", "mel09", "mel10"],
        "02": ["mel05", "mel10", "mfcc07"],
        "03": ["mfcc06", "mfcc07", "mfcc08"],
        "04": ["mel05", "mfcc06", "melc1"],
    }
    # dataset_names = ["mel05", "mel09", "mel10"]
    dataset_names = datasets_types["04"]

    processed_folder = Path("data_proc")
    data_paths = [processed_folder / f"{dn}" for dn in dataset_names]
    logg.debug(f"data_paths: {data_paths}")

    words = ["happy", "learn"]
    load_triple(data_paths, words)


##########################################################
#   Split spec
##########################################################


def compute_folder_spec(
    word_in_folder,
    word_out_folder,
    dataset_name,
    force_preprocess,
    spec_kwargs,
    p2d_kwargs,
) -> None:
    """TODO: what is compute_folder_spec doing?"""
    # extract all the wavs
    all_wavs = list(word_in_folder.iterdir())

    for wav_path in all_wavs:
        wav_stem = wav_path.stem

        spec_path = word_out_folder / f"{wav_stem}.npy"
        if spec_path.exists():
            if force_preprocess:
                pass
            else:
                continue

        if dataset_name.startswith("mfcc"):
            log_spec = wav2mfcc(wav_path, spec_kwargs, p2d_kwargs)
        elif dataset_name.startswith("mel"):
            log_spec = wav2mel(wav_path, spec_kwargs, p2d_kwargs)

        # img_spec = log_spec.reshape((*log_spec.shape, 1))
        # np.save(spec_path, img_spec)
        np.save(spec_path, log_spec)


def preprocess_split(
    dataset_name: str, words_type: str, force_preprocess: bool = False
) -> None:
    """TODO: what is preprocess_split doing?"""
    logg = logging.getLogger(f"c.{__name__}.preprocess_split")
    # logg.setLevel("INFO")
    # logg.debug("Start preprocess_split")

    # original / processed dataset base locations
    data_raw_path = Path("data_raw")
    processed_folder = Path("data_split") / f"{dataset_name}"
    if not processed_folder.exists():
        processed_folder.mkdir(parents=True, exist_ok=True)

    # args for the power_to_db function
    p2d_kwargs = {"ref": np.max}

    # args for the mfcc spec
    spec_dict = get_spec_dict()
    spec_kwargs = spec_dict[dataset_name]

    arguments = []

    words = words_types[words_type]
    for word in words:
        # logg.debug(f"Processing {word}")

        # the output folder path
        word_out_folder = processed_folder / word
        if not word_out_folder.exists():
            word_out_folder.mkdir(parents=True, exist_ok=True)

        # the input folder path
        word_in_folder = data_raw_path / word

        # build the arguments of the functions you will run
        arguments.append(
            (
                word_in_folder,
                word_out_folder,
                dataset_name,
                force_preprocess,
                spec_kwargs,
                p2d_kwargs,
            )
        )

    logg.info(f"dataset_name: {dataset_name}")
    pool = mp.Pool(processes=mp.cpu_count() * 2)
    results = [pool.apply_async(compute_folder_spec, args=a) for a in arguments]
    for p in tqdm(results):
        p.get()


def prepare_partitions(
    words_type: str,
) -> ty.Tuple[ty.Dict[str, ty.List[str]], ty.Dict[str, ty.Tuple[str, str]]]:
    """TODO: what is prepare_partitions doing?

    TODO: add unknown, silence and things
    """
    logg = logging.getLogger(f"c.{__name__}.prepare_partitions")
    # logg.setLevel("INFO")
    logg.debug("Start prepare_partitions")

    data_raw_path = Path("data_raw")

    validation_names, testing_names = get_val_test_list(data_raw_path)

    partition: ty.Dict[str, ty.List[str]] = {
        "training": [],
        "validation": [],
        "testing": [],
    }
    ids2labels: ty.Dict[str, ty.Tuple[str, str]] = {}

    words = words_types[words_type]
    logg.debug(f"Partinioning words: {words}")
    for word in tqdm(words):

        # the input folder path
        word_in_folder = data_raw_path / word

        # go through all the words in this folder
        for wav_path in word_in_folder.iterdir():
            wav_stem = wav_path.stem
            wav_name = f"{word}/{wav_path.name}"

            if wav_name in validation_names:
                which = "validation"
            elif wav_name in testing_names:
                which = "testing"
            else:
                which = "training"

            ID = wav_name
            partition[which].append(ID)
            ids2labels[ID] = (word, wav_stem)

    return partition, ids2labels


def run_preprocess_data(args) -> None:
    """TODO: What is preprocess_data doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_preprocess_data")
    logg.debug("Starting run_preprocess_data")

    which_dataset = args.dataset_name
    words_type = args.words_type
    force_preprocess = args.force_preprocess
    preprocess_type = args.preprocess_type

    if preprocess_type == "preprocess_spec":
        preprocess_spec(which_dataset, words_type, force_preprocess)
    elif preprocess_type == "compose_spec":
        compose_spec(which_dataset, words_type, force_preprocess)
    elif preprocess_type == "preprocess_split":
        preprocess_split(which_dataset, words_type, force_preprocess)
    # test_load_processed()
    # test_load_triple(args)


if __name__ == "__main__":
    args = setup_env()
    run_preprocess_data(args)
