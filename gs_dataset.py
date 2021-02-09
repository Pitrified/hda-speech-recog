from pathlib import Path
from scipy.io.wavfile import write as write_wav  # type: ignore
import argparse
import librosa  # type: ignore
import logging
import numpy as np  # type: ignore
import typing as ty

from utils import get_val_test_list
from utils import words_types


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel: str = "DEBUG") -> None:
    """Setup logger that outputs to console for the module"""
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)
    module_console_handler = logging.StreamHandler()
    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = '%(name)s: %(message)s'
    log_format_module = "%(message)s"
    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)
    logroot.addHandler(module_console_handler)
    logging.addLevelName(5, "TRACE")


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 gs_dataset.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def extract_loudest_section(orig_sig, sr) -> np.ndarray:
    """MAKEDOC: what is extract_loudest_section doing?"""
    logg = logging.getLogger(f"c.{__name__}.extract_loudest_section")
    logg.setLevel("INFO")
    logg.debug("Start extract_loudest_section")

    section_lenght = sr // 2
    hop_len = section_lenght // 10
    num_split = (len(orig_sig) - section_lenght) // hop_len
    recap = f"num_split: {num_split}"
    recap += f" hop_len {hop_len}"
    recap += f" section_lenght {section_lenght}"
    logg.debug(recap)

    squared_sig = orig_sig ** 2

    best_sum = 0
    best_index = 0

    for split_index in range(num_split):
        start_index = split_index * hop_len
        end_index = start_index + section_lenght
        split = squared_sig[start_index:end_index]
        # logg.debug(f"split.shape: {split.shape}")

        sum_ = np.sum(split)

        if sum_ > best_sum:
            best_sum = sum_
            best_index = split_index
            logg.debug(f"best_index {best_index} best_sum: {best_sum}")

    start_index = best_index * hop_len
    end_index = start_index + section_lenght
    split = orig_sig[start_index:end_index]
    return split


def shorten_gs_utterances() -> None:
    """MAKEDOC: what is shorten_gs_utterances doing?"""
    logg = logging.getLogger(f"c.{__name__}.shorten_gs_utterances")
    # logg.setLevel("INFO")
    logg.debug("Start shorten_gs_utterances")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    # the location of the original dataset
    orig_base_folder = this_file_folder / "data_raw"
    logg.debug(f"orig_base_folder: {orig_base_folder}")

    # which words to elaborate
    # words = words_types["all"]
    words = words_types["num"]

    loudest_template = "loudest_{}"

    for word in words:

        # the folder where the original wavs are
        orig_word_folder = orig_base_folder / word
        logg.debug(f"orig_word_folder: {orig_word_folder}")

        # where to save the shortened wavs
        loudest_word = loudest_template.format(word)
        logg.debug(f"loudest_word: {loudest_word}")
        loudest_word_folder = orig_base_folder / loudest_word
        if not loudest_word_folder.exists():
            loudest_word_folder.mkdir(parents=True, exist_ok=True)

        for orig_wav_path in orig_word_folder.iterdir():

            # load the original signal
            orig_sig, orig_sr = librosa.load(orig_wav_path, sr=None)

            # extract the loudes section
            loudest_split = extract_loudest_section(orig_sig, orig_sr)

            # create the name for the new wav
            wav_name = orig_wav_path.name
            loudest_path = loudest_word_folder / wav_name
            # logg.debug(f"loudest_path: {loudest_path}")

            write_wav(loudest_path, orig_sr, loudest_split)

    # the file to list the file names for the testing fold
    val_list_path = orig_base_folder / "validation_list_loudest.txt"
    word_val_list: ty.List[str] = []
    test_list_path = orig_base_folder / "testing_list_loudest.txt"
    word_test_list: ty.List[str] = []

    # the validation and testing lists of the original GS dataset
    validation_names, testing_names = get_val_test_list(orig_base_folder)

    for val_ID in validation_names:
        loud_val_ID = loudest_template.format(val_ID)
        word_val_list.append(loud_val_ID)
    for test_ID in testing_names:
        loud_test_ID = loudest_template.format(test_ID)
        word_test_list.append(loud_test_ID)

    # save the IDs
    word_val_str = "\n".join(word_val_list)
    val_list_path.write_text(word_val_str)
    word_test_str = "\n".join(word_test_list)
    test_list_path.write_text(word_test_str)


def run_gs_dataset(args: argparse.Namespace) -> None:
    """TODO: What is gs_dataset doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_gs_dataset")
    logg.debug("Starting run_gs_dataset")

    shorten_gs_utterances()


if __name__ == "__main__":
    args = setup_env()
    run_gs_dataset(args)
