from pathlib import Path
from scipy.io.wavfile import write as write_wav  # type: ignore
import librosa  # type: ignore
import numpy as np  # type: ignore
import typing as ty
import argparse
import logging

from utils import words_types


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-dl",
        "--do_loud",
        dest="do_loud",
        action="store_true",
        help="Extract half a second samples",
    )

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
    recap = "python3 ltts_preprocess.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def split_ltts(do_loud: bool) -> None:
    """MAKEDOC: what is split_ltts doing?"""
    logg = logging.getLogger(f"c.{__name__}.split_ltts")
    # logg.setLevel("INFO")
    logg.debug("Start split_ltts")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    ####################################################################################
    #   Build the dict of sentences without training words inside                      #
    ####################################################################################

    # the location of the original dataset
    ltts_base_folder = Path.home() / "audiodatasets" / "LibriTTS" / "dev-clean"
    logg.debug(f"ltts_base_folder: {ltts_base_folder}")

    # build good_IDs dict
    # from wav_ID to orig_wav_path
    good_IDs: ty.Dict[str, Path] = {}

    # get the words
    train_words = words_types["all"]
    # train_words = words_types["num"]

    # count good sentences
    good_sent = 0
    skip_sent = 0

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

                # build the wav path
                #    file_path /path/to/file/wav_ID.normalized.txt
                #    with stem extract wav_ID.normalized
                #    then remove .normalized
                wav_ID = file_path.stem[:-11]
                # logg.debug(f"wav_ID: {wav_ID}")
                orig_wav_path = chapter_ID_path / f"{wav_ID}.wav"

                # save the file path
                good_IDs[wav_ID] = orig_wav_path

    logg.debug(f"good_sent: {good_sent}")
    logg.debug(f"skip_sent: {skip_sent}")

    ####################################################################################
    #   Create the samples                                                             #
    ####################################################################################

    # where to save the results
    ltts_proc_folder = this_file_folder / "data_raw"

    if do_loud:
        loud_tag = "_loud"
    else:
        loud_tag = ""

    ltts_label = f"_other_ltts{loud_tag}"
    ltts_label_folder = ltts_proc_folder / ltts_label

    logg.debug(f"ltts_label_folder: {ltts_label_folder}")
    if not ltts_label_folder.exists():
        ltts_label_folder.mkdir(parents=True, exist_ok=True)

    # the file to list the file names for the testing fold
    val_list_path = ltts_proc_folder / f"validation_list_lttspeech{loud_tag}.txt"
    word_val_list: ty.List[str] = []
    test_list_path = ltts_proc_folder / f"testing_list_lttspeech{loud_tag}.txt"
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

    for wav_ID in good_IDs:
        orig_wav_path = good_IDs[wav_ID]
        logg.debug(f"good_IDs[{wav_ID}]: {good_IDs[wav_ID]}")

        # load the original signal
        orig_sig, orig_sr = librosa.load(orig_wav_path, sr=None)

        # resample it to 16000 Hz
        new_sig = librosa.resample(orig_sig, orig_sr, new_sr)

        # how long is the sample to extract
        if do_loud:
            sample_len = new_sr // 2
        else:
            sample_len = new_sr

        # split it in 1 second samples
        len_new_sig = new_sig.shape[0]
        num_samples = len_new_sig // sample_len

        sample_wav_template = f"ltts_{wav_ID}_{{:03d}}.wav"

        for i_sample in range(num_samples):

            # cut the sample
            sample_sig = new_sig[i_sample * sample_len : (i_sample + 1) * sample_len]

            # get the name of the wav file
            sample_name = sample_wav_template.format(i_sample)

            # save the sample
            sample_path = ltts_label_folder / sample_name
            write_wav(sample_path, new_sr, sample_sig)
            saved_samples += 1

            # the ID of the sample
            sample_id = f"{ltts_label}/{sample_name}"

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


def run_ltts_preprocess(args: argparse.Namespace) -> None:
    """TODO: What is ltts_preprocess doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_ltts_preprocess")
    logg.debug("Starting run_ltts_preprocess")

    do_loud = args.do_loud

    split_ltts(do_loud)


if __name__ == "__main__":
    args = setup_env()
    run_ltts_preprocess(args)
