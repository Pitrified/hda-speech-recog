from pathlib import Path
from tensorflow.keras import models  # type: ignore
import argparse
import librosa  # type: ignore
import logging
import numpy as np  # type: ignore
import re
import typing as ty
import matplotlib.pyplot as plt  # type: ignore

from augment_data import augment_signals
from train_cnn import build_cnn_name
from utils import setup_logger
from utils import words_types
from utils import setup_gpus


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="ltts",
        choices=["ltts"],
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

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="mel01",
        help="Name of the dataset folder",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 evaluate_stream.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def build_ltts_sentence_list(
    train_words_type: str,
) -> ty.Tuple[ty.Dict[str, Path], ty.Dict[str, str]]:
    """MAKEDOC: what is build_ltts_sentence_list doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_ltts_sentence_list")
    # logg.setLevel("INFO")
    logg.debug("Start build_ltts_sentence_list")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    # the location of the original dataset
    ltts_base_folder = Path.home() / "audiodatasets" / "LibriTTS" / "dev-clean"
    logg.debug(f"ltts_base_folder: {ltts_base_folder}")

    # get the words
    # train_words = words_types["all"]
    train_words = words_types[train_words_type]

    train_words_bound = [fr"\b{w}\b" for w in train_words]
    # logg.debug(f"train_words_bound: {train_words_bound}")
    train_words_re = re.compile("|".join(train_words_bound))
    # logg.debug(f"train_words_re: {train_words_re}")

    # build good_wav_paths dict
    # from wav_ID to orig_wav_path
    good_wav_paths: ty.Dict[str, Path] = {}

    # build good_sentences dict
    # from wav_ID to norm_tra
    good_sentences: ty.Dict[str, str] = {}

    for reader_ID_path in ltts_base_folder.iterdir():
        for chapter_ID_path in reader_ID_path.iterdir():
            for file_path in chapter_ID_path.iterdir():

                # extract the name of the file
                file_name = file_path.name

                # only process normalized files
                if "normalized" not in file_name:
                    continue
                # logg.debug(f"file_path: {file_path}")

                # read the normalized transcription
                norm_tra = file_path.read_text()

                # select this row if one of the training words is in the sentence
                match = train_words_re.search(norm_tra)
                if match is None:
                    continue

                # build the wav path
                #    file_path /path/to/file/wav_ID.normalized.txt
                #    with stem extract wav_ID.normalized
                #    then remove .normalized
                wav_ID = file_path.stem[:-11]
                # logg.debug(f"wav_ID: {wav_ID}")
                orig_wav_path = chapter_ID_path / f"{wav_ID}.wav"

                # save the file path
                good_wav_paths[wav_ID] = orig_wav_path
                good_sentences[wav_ID] = norm_tra

    return good_wav_paths, good_sentences


def split_sentence(
    sentence_sig: np.ndarray,
    spec_type: str,
    sentence_hop_length: int,
    new_sr: int = 16000,
) -> ty.List[np.ndarray]:
    """MAKEDOC: what is split_sentence doing?"""
    logg = logging.getLogger(f"c.{__name__}.split_sentence")
    logg.setLevel("INFO")
    logg.debug("Start split_sentence")

    # split the sentence here
    splits: ty.List[np.ndarray] = []

    # count the splits
    num_split = (len(sentence_sig) - new_sr) // sentence_hop_length
    logg.debug(f"num_split: {num_split}")

    for split_index in range(num_split):
        start_index = split_index * sentence_hop_length
        end_index = start_index + new_sr
        split = sentence_sig[start_index:end_index]
        logg.debug(f"split.shape: {split.shape}")
        splits.append(split)

    return splits


def load_trained_model_cnn(override_hypa) -> models.Model:
    """MAKEDOC: what is load_trained_model_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model_cnn")

    # default values for the hypas
    hypa: ty.Dict[str, ty.Union[str, int]] = {
        "base_dense_width": 32,
        "base_filters": 32,
        "batch_size": 32,
        "dataset": "aug07",
        "dropout_type": "01",
        "epoch_num": 15,
        "kernel_size_type": "02",
        "learning_rate_type": "04",
        "optimizer_type": "a1",
        "pool_size_type": "01",
        "words": "all",
    }

    # override the values
    for hypa_name in override_hypa:
        hypa[hypa_name] = override_hypa[hypa_name]

    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    model_folder = Path("trained_models") / "cnn"
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        raise FileNotFoundError

    model = models.load_model(model_path)

    return model


def load_trained_model(
    model_type: str, datasets_type: str, train_words_type: str
) -> models.Model:
    """MAKEDOC: what is load_trained_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model")

    if model_type == "cnn":
        override_hypa = {"dataset": datasets_type, "words": train_words_type}
        model = load_trained_model_cnn(override_hypa)

    return model


def plot_sentence_pred(
    sentence_sig: np.ndarray,
    y_pred: np.ndarray,
    norm_tra: str,
    train_words: ty.List[str],
) -> None:
    """MAKEDOC: what is plot_sentence_pred doing?"""
    logg = logging.getLogger(f"c.{__name__}.plot_sentence_pred")
    # logg.setLevel("INFO")
    logg.debug("Start plot_sentence_pred")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    ax[0].plot(sentence_sig)
    ax[1].imshow(y_pred.T, cmap=plt.cm.viridis, aspect="auto")

    ax[0].set_title(norm_tra)
    # norm_tra_list = norm_tra.split()
    # len_sentence_words = len(norm_tra_list)
    # x_tickpos = np.arange(len_sentence_words) * len(sentence_sig) / len_sentence_words
    # ax[0].set_xticks(x_tickpos)
    # ax[0].set_xticklabels(norm_tra_list, rotation=40)

    y_tickpos = np.arange(len(train_words))
    ax[1].set_yticks(y_tickpos)
    ax[1].set_yticklabels(train_words)

    fig.tight_layout()


def evaluate_stream(
    evaluation_type: str, datasets_type: str, train_words_type: str
) -> None:
    """MAKEDOC: what is evaluate_stream doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_stream")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_stream")

    # magic to fix the GPUs
    setup_gpus()

    # a random number generator to use
    rng = np.random.default_rng(12345)

    model_type = "cnn"
    model = load_trained_model(model_type, datasets_type, train_words_type)
    # model.summary()

    if evaluation_type == "ltts":
        sentence_wav_paths, sentence_norm_tra = build_ltts_sentence_list(
            train_words_type
        )

    # get info for one sentence
    wav_ID = list(sentence_wav_paths.keys())[40]
    orig_wav_path = sentence_wav_paths[wav_ID]
    logg.debug(f"sentence_wav_paths[{wav_ID}]: {orig_wav_path}")
    norm_tra = sentence_norm_tra[wav_ID]
    logg.debug(f"sentence_norm_tra[{wav_ID}]: {norm_tra}")

    # the sample rate to use
    new_sr = 16000

    # load the sentence and resample it
    sentence_sig, sentence_sr = librosa.load(orig_wav_path, sr=None)
    sentence_sig = librosa.resample(sentence_sig, sentence_sr, new_sr)

    # split the sentence in chunks every sentence_hop_length
    sentence_hop_length = new_sr // 16
    splits = split_sentence(sentence_sig, datasets_type, sentence_hop_length)

    # compute spectrograms / augment / compose
    if datasets_type.startswith("aug"):
        specs = augment_signals(splits, datasets_type, rng, which_fold="testing")
        logg.debug(f"specs.shape: {specs.shape}")
        specs_img = np.expand_dims(specs, axis=-1)
        logg.debug(f"specs_img.shape: {specs_img.shape}")

    words = sorted(words_types[train_words_type])
    logg.debug(f"words: {words}")
    y_pred = model.predict(specs_img)
    # logg.debug(f"y_pred: {y_pred}")
    y_index = np.argmax(y_pred, axis=1)
    # logg.debug(f"y_index: {y_index}")
    y_pred_labels = [words[i] for i in y_index]
    logg.debug(f"y_pred_labels: {y_pred_labels}")

    plot_sentence_pred(sentence_sig, y_pred, norm_tra, words)
    plt.show()


def run_evaluate_stream(args: argparse.Namespace) -> None:
    """TODO: What is evaluate_stream doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_stream")
    logg.debug("Starting run_evaluate_stream")

    evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type
    which_dataset = args.dataset_name

    evaluate_stream(evaluation_type, which_dataset, train_words_type)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_stream(args)
