from pathlib import Path
from scipy.io.wavfile import write as write_wav  # type: ignore
from tqdm import tqdm  # type: ignore
import argparse
import librosa  # type: ignore
import logging
import numpy as np  # type: ignore

from augment_data import pad_signal
from utils import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 fsdd_preprocess.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def preprocess_fsdd() -> None:
    """TODO: what is preprocess_fsdd doing?

    Get the dataset with:
    git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git ~/free_spoken_digit_dataset
    """
    logg = logging.getLogger(f"c.{__name__}.preprocess_fsdd")
    # logg.setLevel("INFO")
    logg.debug("Start preprocess_fsdd")

    # the location of the original dataset
    fsdd_base_folder = Path("~").expanduser().absolute()
    fsdd_rec_folder = fsdd_base_folder / "free_spoken_digit_dataset" / "recordings"
    logg.debug(f"fsdd_rec_folder: {fsdd_rec_folder}")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    # where to save the results
    # fsdd_proc_folder = this_file_folder / "data_fsdd_raw"
    fsdd_proc_folder = this_file_folder / "data_raw"
    logg.debug(f"fsdd_proc_folder: {fsdd_proc_folder}")
    if not fsdd_proc_folder.exists():
        fsdd_proc_folder.mkdir(parents=True, exist_ok=True)

    # the file to list the file names for the testing fold
    test_list_path = fsdd_proc_folder / "testing_list_fsdd.txt"
    word_test_list = []

    num2name = {
        0: "fsdd_zero",
        1: "fsdd_one",
        2: "fsdd_two",
        3: "fsdd_three",
        4: "fsdd_four",
        5: "fsdd_five",
        6: "fsdd_six",
        7: "fsdd_seven",
        8: "fsdd_eight",
        9: "fsdd_nine",
    }

    for num, name in num2name.items():
        name_folder = fsdd_proc_folder / name
        if not name_folder.exists():
            name_folder.mkdir(parents=True, exist_ok=True)

    # a random number generator to use
    rng = np.random.default_rng(12345)

    # the target sr
    new_sr = 16000

    for wav_path in tqdm(fsdd_rec_folder.iterdir()):
        # logg.debug(f"wav_path: {wav_path}")

        wav_name = wav_path.name
        # logg.debug(f"wav_name: {wav_name}")

        orig_sig, orig_sr = librosa.load(wav_path, sr=None)
        # logg.debug(f"orig_sig.shape: {orig_sig.shape} orig_sr: {orig_sr}")

        new_sig = librosa.resample(orig_sig, orig_sr, new_sr)
        # logg.debug(f"new_sig.shape: {new_sig.shape}")

        padded_sig = pad_signal(new_sig, new_sr)
        # logg.debug(f"padded_sig.shape: {padded_sig.shape}")

        # if there is space to move the signal around, roll it a bit
        max_roll = (new_sr - new_sig.shape[0] - 50) // 2
        if max_roll > 50:
            sample_shift = rng.integers(-max_roll, max_roll)
            rolled_sig = np.roll(padded_sig, sample_shift)
        else:
            rolled_sig = padded_sig

        # the label of this word
        word_label = num2name[int(wav_name[0])]

        # save the processed file
        name_folder = fsdd_proc_folder / word_label
        new_path = name_folder / wav_name
        write_wav(new_path, new_sr, rolled_sig)

        # create the id for this word
        word_id = f"{word_label}/{wav_name}"
        word_test_list.append(word_id)

    # save the IDs
    word_test_str = "\n".join(word_test_list)
    test_list_path.write_text(word_test_str)


def run_fsdd_preprocess(args: argparse.Namespace) -> None:
    """TODO: What is fsdd_preprocess doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_fsdd_preprocess")
    logg.debug("Starting run_fsdd_preprocess")

    preprocess_fsdd()


if __name__ == "__main__":
    args = setup_env()
    run_fsdd_preprocess(args)
