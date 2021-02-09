from pathlib import Path
from scipy.io.wavfile import write as write_wav  # type: ignore
import argparse
import librosa  # type: ignore
import logging
import numpy as np  # type: ignore
import pandas  # type: ignore
import typing as ty

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
    recap = "python3 ljspeech_preprocess.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def split_ljspeech() -> None:
    """MAKEDOC: what is split_ljspeech doing?"""
    logg = logging.getLogger(f"c.{__name__}.split_ljspeech")
    # logg.setLevel("INFO")
    logg.debug("Start split_ljspeech")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    ####################################################################################
    #   Find the list of sentences without training words inside                       #
    ####################################################################################

    # the location of the original dataset
    ljs_base_folder = Path("~").expanduser() / "audiodatasets" / "LJSpeech-1.1"
    logg.debug(f"ljs_base_folder: {ljs_base_folder}")
    ljs_wav_folder = ljs_base_folder / "wavs"
    logg.debug(f"ljs_wav_folder: {ljs_wav_folder}")

    # column names in the metadata file
    column_names = ["wav_ID", "tra", "norm_tra"]

    # load the metadata file
    metadata_path = ljs_base_folder / "metadata.csv"
    meta_df = pandas.read_csv(
        metadata_path, sep="|", header=0, names=column_names, index_col=False
    )

    # shuffle it
    meta_df = meta_df.sample(frac=1)

    # remove NaN
    meta_df = meta_df.dropna()

    # show some rows to be sure
    logg.debug(f"meta_df.head():\n{meta_df.head()}")

    # get the words
    train_words = words_types["all"]
    # train_words = words_types["num"]

    # count good sentences
    good_sent = 0
    skip_sent = 0

    # find good sentences
    good_IDs: ty.List[str] = []
    for index_row, row in meta_df.iterrows():
        norm_tra = row["norm_tra"]

        # skip this row if one of the training words is in the sentence
        found_train_word = False
        for word in train_words:
            if word in norm_tra:
                # logg.debug(f"Found '{word}' in {norm_tra}")
                found_train_word = True
                skip_sent += 1
                break
        if found_train_word:
            continue
        good_sent += 1

        # save the file ID
        good_IDs.append(row["wav_ID"])

    logg.debug(f"good_sent: {good_sent}")
    logg.debug(f"skip_sent: {skip_sent}")

    ####################################################################################
    #   Create the samples                                                             #
    ####################################################################################

    # where to save the results
    # ljs_proc_folder = this_file_folder / "data_ljs_raw"
    ljs_proc_folder = this_file_folder / "data_raw"

    ljs_label = "_other"
    ljs_label_folder = ljs_proc_folder / ljs_label

    logg.debug(f"ljs_label_folder: {ljs_label_folder}")
    if not ljs_label_folder.exists():
        ljs_label_folder.mkdir(parents=True, exist_ok=True)

    # the file to list the file names for the testing fold
    val_list_path = ljs_proc_folder / "validation_list_ljspeech.txt"
    word_val_list: ty.List[str] = []
    test_list_path = ljs_proc_folder / "testing_list_ljspeech.txt"
    word_test_list: ty.List[str] = []

    # the sizes of the test and validation folds
    val_fold_size = 0.1
    test_fold_size = 0.1

    # the target sr
    new_sr = 16000

    # how many samples to generate
    max_num_samples = 5000
    saved_samples = 0

    # a random number generator to use
    rng = np.random.default_rng(12345)

    for wav_ID in good_IDs[:]:
        wav_name = f"{wav_ID}.wav"

        # load the original signal
        orig_wav_path = ljs_wav_folder / wav_name
        orig_sig, orig_sr = librosa.load(orig_wav_path, sr=None)
        # logg.debug(f"orig_sig.shape: {orig_sig.shape} orig_sr: {orig_sr}")

        # resample it to 16000 Hz
        new_sig = librosa.resample(orig_sig, orig_sr, new_sr)
        # logg.debug(f"new_sig.shape: {new_sig.shape}")

        # split it in 1 second samples
        len_new_sig = new_sig.shape[0]
        num_samples = len_new_sig // new_sr

        sample_wav_template = f"ljs_{wav_ID}_{{:03d}}.wav"
        for i_sample in range(num_samples):

            # cut the sample
            sample_sig = new_sig[i_sample * new_sr : (i_sample + 1) * new_sr]

            # get the name of the wav file
            sample_name = sample_wav_template.format(i_sample)

            # save the sample
            sample_path = ljs_label_folder / sample_name
            write_wav(sample_path, new_sr, sample_sig)
            saved_samples += 1

            # the ID of the sample
            sample_id = f"{ljs_label}/{sample_name}"

            # split in trainin/validation/testing
            x = rng.random()
            if x < test_fold_size:
                word_test_list.append(sample_id)
            elif x < test_fold_size + val_fold_size:
                word_val_list.append(sample_id)

        if saved_samples > max_num_samples:
            break

    # save the IDs
    word_val_str = "\n".join(word_val_list)
    val_list_path.write_text(word_val_str)
    word_test_str = "\n".join(word_test_list)
    test_list_path.write_text(word_test_str)


def run_ljspeech_preprocess(args: argparse.Namespace) -> None:
    """TODO: What is ljspeech_preprocess doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_ljspeech_preprocess")
    logg.debug("Starting run_ljspeech_preprocess")

    split_ljspeech()


if __name__ == "__main__":
    args = setup_env()
    run_ljspeech_preprocess(args)
