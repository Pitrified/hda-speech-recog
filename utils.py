from scipy.io import wavfile  # type: ignore
from pathlib import Path
from sklearn.metrics import confusion_matrix  # type: ignore
from time import sleep
from copy import copy
import logging

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

import typing as ty


def define_words_types():

    words_all = [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero",
    ]

    words_num = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    words_direction = ["up", "down", "forward", "backward", "left", "right"]

    words_kaggle_1 = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]

    words_task_20 = [
        "down",
        "eight",
        "five",
        "four",
        "go",
        "left",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "six",
        "stop",
        "three",
        "two",
        "up",
        "yes",
        "zero",
    ]

    # words from the free spoken digit dataset
    words_num_fsdd = [f"fsdd_{w}" for w in words_num]

    # words shortened to the loudest section
    words_all_loud = [f"loudest_{w}" for w in words_all]
    words_num_loud = [f"loudest_{w}" for w in words_num]

    additional_ljs = ["_other_ljs"]
    words_ljs_all = copy(words_all)
    words_ljs_all.extend(additional_ljs)
    words_ljs_num = copy(words_num)
    words_ljs_num.extend(additional_ljs)

    additional_ljs_loud = ["_other_ljs_loud"]
    words_ljs_all_loud = copy(words_all_loud)
    words_ljs_all_loud.extend(additional_ljs_loud)
    words_ljs_num_loud = copy(words_num_loud)
    words_ljs_num_loud.extend(additional_ljs_loud)

    additional_ltts = ["_other_ltts"]
    words_ltts_all = copy(words_all)
    words_ltts_all.extend(additional_ltts)
    words_ltts_num = copy(words_num)
    words_ltts_num.extend(additional_ltts)

    additional_ltts_loud = ["_other_ltts_loud"]
    words_ltts_all_loud = copy(words_all_loud)
    words_ltts_all_loud.extend(additional_ltts_loud)
    words_ltts_num_loud = copy(words_num_loud)
    words_ltts_num_loud.extend(additional_ltts_loud)

    words_types = {}

    words_gs_standard = {
        "all": words_all,
        "dir": words_direction,
        "num": words_num,
        "k1": words_kaggle_1,
        "w2": words_task_20,
        "f1": ["happy", "learn", "wow", "visual"],
        "f2": ["backward", "eight", "go", "yes"],
    }
    words_types.update(words_gs_standard)

    words_loud = {
        "allLS": words_all_loud,
        "numLS": words_num_loud,
        "LJallLS": words_ljs_all_loud,
        "LJnumLS": words_ljs_num_loud,
        "LTallLS": words_ltts_all_loud,
        "LTnumLS": words_ltts_num_loud,
    }
    words_types.update(words_loud)

    words_fsdd = {
        "fsdd": words_num_fsdd,
    }
    words_types.update(words_fsdd)

    words_ljs = {
        "LJall": words_ljs_all,
        "LJnum": words_ljs_num,
    }
    words_types.update(words_ljs)

    words_ltts = {
        "LTall": words_ltts_all,
        "LTnum": words_ltts_num,
    }
    words_types.update(words_ltts)

    words_universe = set(
        [
            *words_all,
            *words_all_loud,
            *words_ljs_all,
            *words_ljs_all_loud,
            *words_ltts_all,
            *words_ltts_all_loud,
        ]
    )
    words_single = {f"_{w}": [w] for w in words_universe}
    words_types.update(words_single)
    # words_single = {f"_{w}": [w] for w in words_all}
    # words_types.update(words_single)
    # words_single = {f"_{w}": [w] for w in words_all_loud}
    # words_types.update(words_single)

    return words_types


words_types = define_words_types()


def analyze_confusion(confusion, true_labels):
    """Compute the F-score from the confusion matrix, and print the intermediate results

    Precision: TP / ( TP + FP)
    Recall: TP / ( TP + FN)
    F-score: 2 (PxR) / (P+R)
    """
    logg = logging.getLogger(f"c.{__name__}.analyze_confusion")
    logg.setLevel("INFO")
    logg.debug("Start analyze_confusion")

    logg.debug("Confusion matrix:")
    logg.debug(row_fmt("Pre\\Tru", true_labels))

    for line, label in zip(confusion, true_labels):
        logg.debug(row_fmt(f"{label}", line))

    TP = confusion.diagonal()
    FN = np.sum(confusion, axis=0) - TP
    FP = np.sum(confusion, axis=1) - TP

    logg.debug("")
    logg.debug(row_fmt("TP", TP))
    logg.debug(row_fmt("FP", FP))
    logg.debug(row_fmt("FN", FN))

    # https://stackoverflow.com/a/37977222
    #  P = TP / ( TP + FP)
    #  R = TP / ( TP + FN)
    dP = TP + FP
    P = np.divide(TP, dP, out=np.zeros_like(TP, dtype=float), where=dP != 0)
    dR = TP + FN
    R = np.divide(TP, dR, out=np.zeros_like(TP, dtype=float), where=dR != 0)

    logg.debug("\nPrecision = TP / ( TP + FP)\tRecall = TP / ( TP + FN)")
    logg.debug(row_fmt("Prec", P, ":.4f"))
    logg.debug(row_fmt("Recall", R, ":.4f"))

    avgP = np.sum(P) / len(true_labels)
    avgR = np.sum(R) / len(true_labels)
    logg.debug(f"Average P: {avgP:.4f}\tR: {avgR:.4f}")

    logg.debug("F-score = 2 (PxR) / (P+R)")
    #  F = 2 (PxR) / (P+R)
    PdR = 2 * P * R
    PpR = P + R
    F = np.divide(PdR, PpR, out=np.zeros_like(TP, dtype=float), where=PpR != 0)
    logg.debug(row_fmt("F-score", F, ":.4f"))

    avgF = np.sum(F) / len(true_labels)
    logg.debug(f"Average F-score {avgF}")

    return avgF


def row_fmt(header, iterable, formatter=""):
    row = header
    for item in iterable:
        #  row += f'\t{item{formatter}}'
        row += "\t{{i{f}}}".format(f=formatter).format(i=item)
    return row


def setup_logger(logLevel="DEBUG"):
    """Setup logger that outputs to console for the module"""
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    # log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # log_format_module = "%(name)s - %(levelname)s: %(message)s"
    # log_format_module = '%(levelname)s: %(message)s'
    log_format_module = '%(name)s: %(message)s'
    # log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logroot.log(5, 'Exceedingly verbose debug')


def setup_gpus():
    """TODO: what is setup_gpus doing?

    https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    https://github.com/tensorflow/tensorflow/issues/25138
    """
    logg = logging.getLogger(f"c.{__name__}.setup_gpus")
    # logg.debug("Start setup_gpus")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        logg.debug("Broski non ho trovato le GPU")


def compute_permutation(words):
    """"""
    sorted_words = sorted(words)
    perm_index = [sorted_words.index(w) for w in words]
    return perm_index


def pred_hot_2_cm(y_hot, y_pred, labels):
    """Computes the confusion matrix from prediction and 1hot encoded labels

    y_hot: one-hot encoded true labels (n, num_labels)
    y_pred: probabilities of the predicted labels (n, num_labels)
    labels: list of the labels in the dataset used
    """
    # logg = logging.getLogger(f"c.{__name__}.pred_hot_2_cm")
    # logg.debug("Start pred_hot_2_cm")

    # logg.debug(f"y_pred.shape: {y_pred.shape}")
    # logg.debug(f"y_hot.shape: {y_hot.shape}")

    y_pred_am = np.argmax(y_pred, axis=1)
    # logg.debug(f"y_pred_am.shape: {y_pred_am.shape}")
    y_pred_labels = np.array([labels[y] for y in y_pred_am])

    y_hot_am = np.argmax(y_hot, axis=1)
    # logg.debug(f"y_hot_am.shape: {y_hot_am.shape}")
    y_hot_labels = np.array([labels[y] for y in y_hot_am])

    cm = confusion_matrix(y_hot_labels, y_pred_labels)

    return cm


def record_audios(
    words: ty.List[str],
    audio_folder: Path,
    audio_path_fmt: str = "{}.wav",
    len_sec: float = 1,
    fs: int = 16000,
    timeout: int = 0,
) -> ty.List[np.ndarray]:
    """TODO: what is record_audios doing?"""
    logg = logging.getLogger(f"c.{__name__}.record_audios")
    logg.setLevel("INFO")
    logg.debug("Start record_audios")

    # importing sounddevice takes time, only do it if needed
    import sounddevice as sd  # type: ignore

    audios = []

    # if there is no timeout give it at least in the beginning
    if timeout == 0:
        logg.info(f"Get ready to start recording {words[0]}")
        sleep(1)

    for word in words:
        for i in range(timeout, 0, -1):
            logg.info(f"Start recording {word} in {i}s")
            sleep(1)
        logg.info(f"Start recording {word} NOW!")

        # record
        myrecording = sd.rec(int(len_sec * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished

        logg.info("Stop recording")

        # save the audio
        audio_path = audio_folder / audio_path_fmt.format(word)
        wavfile.write(audio_path, fs, myrecording)  # Save as WAV file

        audios.append(myrecording)

    return audios


def find_rowcol(n: int) -> ty.Tuple[int, int]:
    """TODO: what is find_rowcol doing?"""
    sn = np.sqrt(n)
    sn_low = int(np.floor(sn))
    sn_high = int(np.ceil(sn))

    if sn_low * sn_high >= n:
        return sn_low, sn_high
    else:
        return sn_high, sn_high


def get_val_test_list(dataset_path: Path) -> ty.Tuple[ty.List[str], ty.List[str]]:
    """TODO: what is get_val_test_list doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_val_test_list")
    logg.setLevel("INFO")
    logg.debug("Start get_val_test_list")

    # find the txt files that contain the lists
    validation_paths = []
    testing_paths = []
    for item in dataset_path.iterdir():
        file_name = item.name

        if file_name.startswith("validation_list"):
            validation_paths.append(item)

        elif file_name.startswith("testing_list"):
            testing_paths.append(item)

    ###### list of file names for validation
    validation_names = []
    for validation_path in validation_paths:
        with validation_path.open() as fvp:
            for line in fvp:
                validation_names.append(line.strip())
    # logg.debug(f"validation_names: {validation_names[:10]}")

    ###### list of file names for testing
    testing_names = []
    for testing_path in testing_paths:
        with testing_path.open() as fvp:
            for line in fvp:
                testing_names.append(line.strip())
    # logg.debug(f"testing_names: {testing_names[:10]}")

    return validation_names, testing_names
