from pathlib import Path
import librosa  # type: ignore
import argparse
import logging

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


def extract_loudes_section() -> None:
    """MAKEDOC: what is extract_loudes_section doing?"""
    logg = logging.getLogger(f"c.{__name__}.extract_loudes_section")
    # logg.setLevel("INFO")
    logg.debug("Start extract_loudes_section")


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

    for word in words:
        orig_word_folder = orig_base_folder / word
        logg.debug(f"orig_word_folder: {orig_word_folder}")

        for orig_wav_path in orig_word_folder:

            # load the original signal
            orig_sig, orig_sr = librosa.load(orig_wav_path, sr=None)


def run_gs_dataset(args: argparse.Namespace) -> None:
    """TODO: What is gs_dataset doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_gs_dataset")
    logg.debug("Starting run_gs_dataset")

    shorten_gs_utterances()


if __name__ == "__main__":
    args = setup_env()
    run_gs_dataset(args)
