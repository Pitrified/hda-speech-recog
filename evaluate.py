import argparse
import logging
from pathlib import Path
import json

# import numpy as np

from utils import setup_logger


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
    setup_logger()

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 evaluate.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def evaluate_results_recap():
    """TODO: what is evaluate_results_recap doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_recap")
    logg.debug("Start evaluate_results_recap")

    info_folder = Path("info")

    for model_folder in info_folder.iterdir():
        logg.debug(f"model_folder: {model_folder}")

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue

        results_recap = json.loads(res_recap_path.read_text())
        # logg.debug(f"results_recap: {results_recap}")
        logg.debug(f"results_recap['cm']: {results_recap['cm']}")


def run_evaluate(args):
    """TODO: What is evaluate doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate")
    logg.debug("Starting run_evaluate")

    evaluate_results_recap()


if __name__ == "__main__":
    args = setup_env()
    run_evaluate(args)
