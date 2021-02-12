from pathlib import Path
from tensorflow_addons.image import sparse_image_warp  # type: ignore
from tqdm import tqdm  # type: ignore
import argparse
import librosa  # type: ignore
import logging
import math
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import typing as ty

from utils import setup_gpus
from utils import get_val_test_list
from utils import setup_logger
from utils import words_types


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    aug_dict = get_aug_dict()
    aug_keys = list(aug_dict.keys())
    aug_keys.extend(["2345", "6789", "10123", "14567"])
    aug_keys.extend(["auA1234", "auA5678"])
    parser.add_argument(
        "-at",
        "--augmentation_type",
        type=str,
        default="aug01",
        choices=aug_keys,
        help="Which augmentation to perform",
    )

    parser.add_argument(
        "-wt",
        "--words_type",
        type=str,
        default="f2",
        choices=words_types.keys(),
        help="Words to augment",
    )

    parser.add_argument(
        "-fp",
        "--force_augment",
        dest="force_augment",
        action="store_true",
        help="Force the augment and overwrite the previous results",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env():
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 augment_data.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def pad_signal(sig: np.ndarray, req_len: int) -> np.ndarray:
    """MAKEDOC: what is pad_signal doing?"""
    sig_len = len(sig)

    if sig_len >= req_len:
        new_sig = np.copy(sig[:req_len])

    else:
        diff = req_len - sig_len
        pad_needed = math.floor(diff / 2), math.ceil(diff / 2)
        new_sig = np.pad(sig, pad_needed, "constant")

    return new_sig


def stretch_signal(sig, rate) -> np.ndarray:
    """MAKEDOC: what is stretch_signal doing?"""
    old_len = len(sig)
    stretched = librosa.effects.time_stretch(sig, rate)
    stretched = pad_signal(stretched, old_len)
    return stretched


def sig2mel(
    signal, mel_kwargs, p2d_kwargs, requested_length, sample_rate=16000
) -> np.ndarray:
    """MAKEDOC: what is sig2mel doing?"""
    # logg = logging.getLogger(f"c.{__name__}.sig2mel")
    # logg.setLevel("INFO")
    # logg.debug("Start sig2mel")

    mel = librosa.feature.melspectrogram(signal, sr=sample_rate, **mel_kwargs)
    log_mel = librosa.power_to_db(mel, **p2d_kwargs)

    # the shape is not consistent, pad it

    pad_needed = requested_length - log_mel.shape[1]
    pad_needed = max(0, pad_needed)

    # recap = f"signal.shape: {signal.shape}"
    # recap += f" log_mel.shape: {log_mel.shape}"
    # recap += f" pad_needed: {pad_needed}"
    # logg.debug(recap)

    # number of values padded to the edges of each axis.
    pad_width = ((0, 0), (0, pad_needed))
    padded_log_mel = np.pad(log_mel, pad_width=pad_width)
    return padded_log_mel


def get_aug_dict() -> ty.Dict[str, ty.Any]:
    """MAKEDOC: what is get_aug_dict doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_aug_dict")
    logg.setLevel("INFO")
    logg.debug("Start get_aug_dict")

    aug_dict: ty.Dict[str, ty.Any] = {}

    mel_01 = {
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 256,
        "fmin": 40,
        "fmax": 8000,
    }  # (64, 64)
    mel_02 = {
        "n_mels": 128,
        "n_fft": 2048,
        "hop_length": 512,
        "fmin": 40,
        "fmax": 8000,
    }  # (128, 32)
    mel_03 = {
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 256,
    }  # (64, 64)
    mel_05 = {
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 128,
    }  # (128, 128)
    mel_01_loud = {
        "n_mels": 64,
        "n_fft": 512,
        "hop_length": 128,
        "fmin": 40,
        "fmax": 8000,
    }  # (64, 64)
    mel_05_loud = {
        "n_mels": 128,
        "n_fft": 512,
        "hop_length": 64,
    }  # (128, 128)
    mel_a1_loud = {
        "n_mels": 80,
        "n_fft": 1024,
        "hop_length": 128,
        "fmin": 40,
    }  # (80, 64)

    aug_dict["aug01"] = {
        "max_time_shifts": [1600, 3200],
        "stretch_rates": [0.8, 1.2],
        "mel_kwargs": mel_01,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 6},
    }

    # mel_02
    aug_dict["aug02"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_02,
        "aug_shape": (128, 32),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["aug03"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_02,
        "aug_shape": (128, 32),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["aug04"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_02,
        "aug_shape": (128, 32),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["aug05"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_02,
        "aug_shape": (128, 32),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_01
    aug_dict["aug06"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["aug07"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["aug08"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["aug09"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_01_loud
    aug_dict["auL06"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01_loud,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["auL07"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01_loud,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["auL08"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01_loud,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["auL09"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_01_loud,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_03
    aug_dict["aug10"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["aug11"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["aug12"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["aug13"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_03, smaller max_warp_time / max_warp_freq, more landmarks
    aug_dict["aug14"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 4, "max_warp_time": 2, "max_warp_freq": 2},
    }
    aug_dict["aug15"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 4, "max_warp_time": 2, "max_warp_freq": 0},
    }
    aug_dict["aug16"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 4, "max_warp_time": 0, "max_warp_freq": 2},
    }
    aug_dict["aug17"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_03,
        "aug_shape": (64, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_05
    aug_dict["aug18"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["aug19"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["aug20"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["aug21"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_05_loud
    aug_dict["auL18"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05_loud,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["auL19"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05_loud,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["auL20"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05_loud,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["auL21"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_05_loud,
        "aug_shape": (128, 128),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_a1_loud
    aug_dict["auA01"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 5},
    }
    aug_dict["auA02"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 5, "max_warp_freq": 0},
    }
    aug_dict["auA03"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 3, "max_warp_time": 0, "max_warp_freq": 5},
    }
    aug_dict["auA04"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }

    # mel_a1_loud, smaller max_warp_time / max_warp_freq, more landmarks
    aug_dict["auA05"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 4, "max_warp_time": 2, "max_warp_freq": 2},
    }
    aug_dict["auA06"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 4, "max_warp_time": 2, "max_warp_freq": 0},
    }
    aug_dict["auA07"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 4, "max_warp_time": 0, "max_warp_freq": 2},
    }
    aug_dict["auA08"] = {
        "max_time_shifts": [],
        "stretch_rates": [],
        "mel_kwargs": mel_a1_loud,
        "aug_shape": (80, 64),
        "keep_originals": True,
        "warp_params": {"num_landmarks": 0, "max_warp_time": 0, "max_warp_freq": 0},
    }
    return aug_dict


def load_wav(
    all_wavs_path: ty.List[Path],
    word: str,
    which_fold: str,
    validation_names: ty.Iterable[str],
    testing_names: ty.Iterable[str],
) -> ty.List[np.ndarray]:
    """MAKEDOC: what is load_wav doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_wav")
    logg.setLevel("INFO")
    logg.debug("Start load_wav")

    # the loaded audios
    sig_original = []

    for wav_path in tqdm(all_wavs_path[:]):

        # the name to lookup in the val/test list
        wav_name = f"{word}/{wav_path.name}"

        if wav_name in validation_names:
            word_fold = "validation"
        elif wav_name in testing_names:
            word_fold = "testing"
        else:
            word_fold = "training"

        if word_fold != which_fold:
            continue

        sig, sample_rate = librosa.load(wav_path, sr=None)
        # sig = pad_signal(sig, 16000)
        sig_original.append(sig)

    logg.debug(f"Loaded {len(sig_original)} of {word} for {which_fold}")
    return sig_original


def roll_signals(
    sig_original: ty.List[np.ndarray],
    max_time_shifts: ty.List[int],
    rng: np.random.Generator,
) -> ty.List[np.ndarray]:
    """MAKEDOC: what is roll_signals doing?"""
    sig_rolled: ty.List[np.ndarray] = []
    for s in tqdm(sig_original):
        for max_shift in max_time_shifts:
            time_shift = rng.integers(-max_shift, max_shift + 1)
            rolled = np.roll(s, time_shift)
            sig_rolled.append(rolled)
    return sig_rolled


def stretch_signals(
    sig_original: ty.List[np.ndarray],
    stretch_rates: ty.List[float],
    rng: np.random.Generator,
) -> ty.List[np.ndarray]:
    """MAKEDOC: what is stretch_signals doing?"""
    sig_stretched: ty.List[np.ndarray] = []
    for s in tqdm(sig_original):
        for stretch_rate in stretch_rates:
            stretched = stretch_signal(s, stretch_rate)
            sig_stretched.append(stretched)
    return sig_stretched


def compute_spectrograms(
    signals: ty.List[np.ndarray], mel_kwargs, p2d_kwargs, requested_length
) -> np.ndarray:
    """MAKEDOC: what is compute_spectrograms doing?"""
    logg = logging.getLogger(f"c.{__name__}.compute_spectrograms")
    # logg.setLevel("INFO")
    # logg.debug("Start compute_spectrograms")

    specs = []
    for s in tqdm(signals):
        log_mel = sig2mel(s, mel_kwargs, p2d_kwargs, requested_length)
        img_mel = log_mel.reshape((*log_mel.shape, 1))
        specs.append(img_mel)
    data_specs = np.stack(specs)
    logg.debug(f"data_specs.shape: {data_specs.shape} req_len {requested_length}")

    return data_specs


def warp_spectrograms(
    specs: np.ndarray,
    num_landmarks: int,
    max_warp_time: int,
    max_warp_freq: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """MAKEDOC: what is warp_spectrograms doing?"""
    logg = logging.getLogger(f"c.{__name__}.warp_spectrograms")
    logg.setLevel("INFO")
    logg.debug("Start warp_spectrograms")

    # extract info on data and spectrogram shapes
    num_samples = specs.shape[0]
    spec_dim = specs.shape[1:3]
    logg.debug(f"num_samples {num_samples} spec_dim {spec_dim}")

    # the shape of the landmark for one dimension
    land_shape = num_samples, num_landmarks

    # the source point has to be at least max_warp_* from the border
    bounds_time = (max_warp_time, spec_dim[0] - max_warp_time)
    bounds_freq = (max_warp_freq, spec_dim[1] - max_warp_freq)

    # generate (num_sample, num_landmarks) time/freq positions
    source_land_t = rng.uniform(*bounds_time, size=land_shape)
    source_land_f = rng.uniform(*bounds_freq, size=land_shape)
    source_landmarks = np.dstack((source_land_t, source_land_f))
    logg.debug(f"land_t.shape: {source_land_t.shape}")
    logg.debug(f"source_landmarks.shape: {source_landmarks.shape}")

    # generate the deltas, how much to shift each point
    delta_t = rng.uniform(-max_warp_time, max_warp_time, size=land_shape)
    delta_f = rng.uniform(-max_warp_freq, max_warp_freq, size=land_shape)
    dest_land_t = source_land_t + delta_t
    dest_land_f = source_land_f + delta_f
    dest_landmarks = np.dstack((dest_land_t, dest_land_f))
    logg.debug(f"dest_landmarks.shape: {dest_landmarks.shape}")

    # data_specs = data_specs.astype("float32")
    # source_landmarks = source_landmarks.astype("float32")
    # dest_landmarks = dest_landmarks.astype("float32")
    # data_warped, _ = sparse_image_warp(
    #     data_specs, source_landmarks, dest_landmarks, num_boundary_points=2
    # )
    # logg.debug(f"data_warped.shape: {data_warped.shape}")

    data_specs = tf.convert_to_tensor(specs, dtype=tf.float32)
    source_landmarks = tf.convert_to_tensor(source_landmarks, dtype=tf.float32)
    dest_landmarks = tf.convert_to_tensor(dest_landmarks, dtype=tf.float32)

    # siw = tf.function(sparse_image_warp, experimental_relax_shapes=True)
    # https://www.tensorflow.org/guide/function#controlling_retracing
    siw = tf.function(
        sparse_image_warp,
        experimental_relax_shapes=True,
        input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),),
    )
    data_warped, _ = siw(
        data_specs, source_landmarks, dest_landmarks, num_boundary_points=2
    )
    logg.debug(f"data_warped.shape: {data_warped.shape}")

    return data_warped


def augment_signals(
    sig_original: ty.List[np.ndarray],
    augmentation_type: str,
    rng: np.random.Generator,
    which_fold: str = "training",
    log_level: str = "INFO",
) -> np.ndarray:
    """MAKEDOC: what is augment_signals doing?"""
    logg = logging.getLogger(f"c.{__name__}.augment_signals")
    logg.setLevel(log_level)
    logg.debug("Start augment_signals")

    # get the params for the augmentation
    aug_dict = get_aug_dict()

    # the shape that we want for the spectrogram
    aug_shape = aug_dict[augmentation_type]["aug_shape"]
    requested_length = aug_shape[1]

    # the augmentation params
    aug_param = aug_dict[augmentation_type]
    max_time_shifts = aug_param["max_time_shifts"]
    stretch_rates = aug_param["stretch_rates"]
    keep_originals = aug_param["keep_originals"]
    mel_kwargs = aug_param["mel_kwargs"]
    num_landmarks = aug_param["warp_params"]["num_landmarks"]
    max_warp_time = aug_param["warp_params"]["max_warp_time"]
    max_warp_freq = aug_param["warp_params"]["max_warp_freq"]

    # args for the power_to_db function
    p2d_kwargs = {"ref": np.max}

    # the signals you are generating
    all_signals = []
    if keep_originals:
        all_signals.extend(sig_original)

    # if len(max_time_shifts) > 0 and which_fold != "testing":
    if len(max_time_shifts) > 0 and which_fold == "training":
        logg.info("Rolling")
        sig_rolled = roll_signals(sig_original, max_time_shifts, rng)
        all_signals.extend(sig_rolled)

    # if len(stretch_rates) > 0 and which_fold != "testing":
    if len(stretch_rates) > 0 and which_fold == "training":
        logg.info("Stretching")
        sig_stretched = stretch_signals(sig_original, stretch_rates, rng)
        all_signals.extend(sig_stretched)

    logg.debug(f"len(all_signals): {len(all_signals)}")

    # compute the spectrograms
    logg.info("Computing melspectrograms")
    # FIXME extract the mel shape and pass it
    data_specs: np.ndarray = compute_spectrograms(
        all_signals, mel_kwargs, p2d_kwargs, requested_length
    )

    # warp the spectrograms
    # if num_landmarks > 0 and which_fold != "testing":
    if num_landmarks > 0 and which_fold == "training":
        logg.info("Warping...")
        data_warped: np.ndarray = warp_spectrograms(
            data_specs, num_landmarks, max_warp_time, max_warp_freq, rng
        )
        all_data = np.concatenate((data_specs, data_warped), axis=0)
    else:
        all_data = data_specs
    logg.debug(f"all_data.shape: {all_data.shape}")

    # remove the last dimension, will be added again when loading (legacy load)
    squoze_data = tf.squeeze(all_data, axis=[-1])
    logg.debug(f"squoze_data.shape: {squoze_data.shape}")

    return squoze_data


def do_augmentation(
    augmentation_type: str, words_type: str, force_augment: bool = False,
) -> None:
    """MAKEDOC: what is do_augmentation doing?

    * Load wav
    * Time shift (data_roll = np.roll(data, 1600))
    * Time warp (librosa.effects.time_stretch)
    * Compute the right spectrogram
    * Warp the spectrogram (tfa.image.sparse_image_warp)

    """
    logg = logging.getLogger(f"c.{__name__}.do_augmentation")
    logg.setLevel("INFO")
    logg.debug("Start do_augmentation")

    # root of the Google dataset
    raw_data_fol = Path("data_raw")

    # the validation and testing lists
    validation_names, testing_names = get_val_test_list(raw_data_fol)

    # output folder
    aug_fol = Path("data_proc") / f"{augmentation_type}"
    if not aug_fol.exists():
        aug_fol.mkdir(parents=True, exist_ok=True)

    # get the list of words
    words = words_types[words_type]

    # a random number generator to use
    rng = np.random.default_rng(12345)

    # do everything on the three folds
    # for which_fold in ["training"]:
    for which_fold in ["training", "validation", "testing"]:

        for word in words:

            word_aug_path = aug_fol / f"{word}_{which_fold}.npy"
            if word_aug_path.exists():
                logg.debug(f"word_aug_path: {word_aug_path} already augmented")
                if force_augment:
                    logg.warn("OVERWRITING the previous results")
                else:
                    continue

            raw_word_fol = raw_data_fol / word
            logg.info(f"\nProcessing folder: {raw_word_fol} {which_fold}")

            # load the waveforms
            all_wavs_path = list(raw_word_fol.iterdir())
            sig_original = load_wav(
                all_wavs_path, word, which_fold, validation_names, testing_names
            )

            if len(sig_original) == 0:
                continue

            squoze_data = augment_signals(
                sig_original, augmentation_type, rng, which_fold
            )

            # save the thingy
            np.save(word_aug_path, squoze_data)


def run_augment_data(args: argparse.Namespace) -> None:
    """MAKEDOC: What is augment_data doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_augment_data")
    logg.debug("Starting run_augment_data")

    # magic to fix the GPUs
    setup_gpus()

    augmentation_type = args.augmentation_type
    words_type = args.words_type
    force_augment = args.force_augment

    if augmentation_type == "2345":
        aug_type_list = ["aug02", "aug03", "aug04", "aug05"]
    elif augmentation_type == "6789":
        aug_type_list = ["aug06", "aug07", "aug08", "aug09"]
    elif augmentation_type == "10123":
        aug_type_list = ["aug10", "aug11", "aug12", "aug13"]
    elif augmentation_type == "14567":
        aug_type_list = ["aug14", "aug15", "aug16", "aug17"]
    elif augmentation_type == "auA1234":
        aug_type_list = ["auA01", "auA02", "auA03", "auA04"]
    elif augmentation_type == "auA5678":
        aug_type_list = ["auA05", "auA06", "auA07", "auA08"]
    else:
        aug_type_list = [augmentation_type]

    for at in aug_type_list:
        do_augmentation(at, words_type, force_augment)


if __name__ == "__main__":
    args = setup_env()
    run_augment_data(args)
