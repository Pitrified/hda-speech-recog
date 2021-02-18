from pathlib import Path
from scipy.io import wavfile  # type: ignore
from time import sleep
import argparse
import logging
import typing as ty

from tqdm import tqdm  # type: ignore
import librosa  # type: ignore
import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore

from utils import setup_logger
from utils import find_free_file_index


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-rt",
        "--recording_type",
        type=str,
        default="record_samples",
        choices=["record_samples", "split_noise"],
        help="What to execute",
    )

    parser.add_argument(
        "-dl",
        "--do_loud",
        dest="do_loud",
        action="store_true",
        help="Extract half a second samples",
    )

    parser.add_argument(
        "-nr",
        "--num_rec",
        type=int,
        default=5,
        help="How many samples to record",
    )

    parser.add_argument(
        "-ls",
        "--len_sec",
        type=int,
        default=10,
        help="The duration of each recording",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 background_noise.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def record_samples(len_sec: int, num_rec: int) -> None:
    """MAKEDOC: what is record_samples doing?

    Record 10s samples, rescale them at 16KHz and save them
    """
    logg = logging.getLogger(f"c.{__name__}.record_samples")
    # logg.setLevel("INFO")
    logg.debug("Start record_samples")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    # where to save the recorded audios
    audio_folder = this_file_folder / "background_audios"
    if not audio_folder.exists():
        audio_folder.mkdir(parents=True, exist_ok=True)

    # audio path template name
    audio_path_fmt = "recorded_audio_{:03d}.wav"

    rec_samplerate = 16000
    # len_sec = 10
    rec_length = int(len_sec * rec_samplerate)
    # num_rec = 2

    for i in range(num_rec):
        logg.debug(f"Start recording {i+1:03d}/{num_rec:03d}")

        # record
        myrecording = sd.rec(rec_length, samplerate=rec_samplerate, channels=1)
        for _ in range(len_sec):
            print(".", end="", flush=True)
            sleep(1)
        print()
        sd.wait()  # Wait until recording is finished

        # resample
        # new_sig = librosa.resample(orig_sig, orig_sr, new_sr)

        # find a free file and save it there
        rec_index = find_free_file_index(audio_folder, audio_path_fmt)
        audio_path = audio_folder / audio_path_fmt.format(rec_index)
        logg.debug(f"Saving in {audio_path}")
        wavfile.write(audio_path, rec_samplerate, myrecording)


def split_noise(do_loud) -> None:
    """MAKEDOC: what is split_noise doing?"""
    logg = logging.getLogger(f"c.{__name__}.split_noise")
    # logg.setLevel("INFO")
    logg.debug("Start split_noise")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    # the raw audio folder for the dataset
    raw_word_fol = Path("data_raw")

    # the location of the original dataset(s)
    base_folders = []

    # my audios
    audio_folder = this_file_folder / "background_audios"
    base_folders.append(audio_folder)

    # Google Speech Commands audios
    gcsd_background = raw_word_fol / "_background_noise_"
    logg.debug(f"gcsd_background: {gcsd_background}")
    base_folders.append(gcsd_background)

    # the desired samplerate
    silence_samplerate = 16000

    # find all wav files and load the wavs
    all_signals_orig: ty.List[np.ndarray] = []
    for base_folder in base_folders:
        # logg.debug(f"base_folder: {base_folder}")

        for a_file in base_folder.iterdir():
            if not a_file.suffix == ".wav":
                continue

            # logg.debug(f"Processing {a_file}")
            orig_sig, orig_sr = librosa.load(a_file, sr=None)

            if orig_sr != silence_samplerate:
                logg.warn(f"Resampling: {a_file}")
                orig_sig = librosa.resample(orig_sig, orig_sr, silence_samplerate)

            # logg.debug(f"orig_sig.shape: {orig_sig.shape}")
            all_signals_orig.append(orig_sig)

    all_silence = np.concatenate(all_signals_orig)
    recap = f"all_silence.shape: {all_silence.shape}"
    recap += f" or {all_silence.shape[0]//silence_samplerate} seconds"
    logg.debug(recap)

    # how long is the silence available as numpy
    silence_sig_len = all_silence.shape[0]

    # how long is the desired sample saved
    if do_loud:
        silence_sample_len = 8000
        loud_tag = "_loud"
    else:
        silence_sample_len = 16000
        loud_tag = ""

    # where to save the split samples
    background_label = f"_background{loud_tag}"
    background_folder = Path("data_raw") / background_label
    logg.debug(f"background_folder: {background_folder}")
    if not background_folder.exists():
        background_folder.mkdir(parents=True, exist_ok=True)

    # the file to list the file names for the testing fold
    val_list_path = raw_word_fol / f"validation_list_background{loud_tag}.txt"
    word_val_list: ty.List[str] = []
    test_list_path = raw_word_fol / f"testing_list_background{loud_tag}.txt"
    word_test_list: ty.List[str] = []

    # the sizes of the test and validation folds
    val_fold_size = 0.1
    test_fold_size = 0.1

    # a random number generator to use
    rng = np.random.default_rng(12345)

    # numpy template name
    sample_name_template = "background_{:06d}.wav"

    # how many splits to extract
    num_silence_samples = 5000

    # how much to leave from the end
    end_sample = silence_sig_len - silence_sample_len - 10

    for i in tqdm(range(num_silence_samples)):
        sample_name = sample_name_template.format(i)

        start = rng.integers(0, end_sample)
        end = start + silence_sample_len

        silence_sample = all_silence[start:end]
        # recap = f"silence_sample.shape: {silence_sample.shape}"
        # recap += f" {start} : {end}"
        # logg.debug(recap)

        # the ID of the sample
        sample_id = f"{background_label}/{sample_name}"

        # split in trainin/validation/testing
        x = rng.random()
        if x < test_fold_size:
            word_test_list.append(sample_id)
        elif x < test_fold_size + val_fold_size:
            word_val_list.append(sample_id)

        sample_path = background_folder / sample_name
        wavfile.write(sample_path, silence_samplerate, silence_sample)

    # save the IDs
    word_val_str = "\n".join(word_val_list)
    val_list_path.write_text(word_val_str)
    word_test_str = "\n".join(word_test_list)
    test_list_path.write_text(word_test_str)


def run_background_noise(args: argparse.Namespace) -> None:
    """TODO: What is background_noise doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_background_noise")
    logg.debug("Starting run_background_noise")

    recording_type = args.recording_type
    len_sec = args.len_sec
    num_rec = args.num_rec
    do_loud = args.do_loud

    if recording_type == "record_samples":
        record_samples(len_sec, num_rec)
    elif recording_type == "split_noise":
        split_noise(do_loud)


if __name__ == "__main__":
    args = setup_env()
    run_background_noise(args)
