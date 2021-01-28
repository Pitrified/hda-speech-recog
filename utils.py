import logging
import numpy as np  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
import tensorflow as tf  # type: ignore


def define_words_types():

    WORDS_ALL = [
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

    WORDS_NUMBERS = [
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

    WORDS_DIRECTION = ["up", "down", "forward", "backward", "left", "right"]

    WORDS_KAGGLE_1 = [
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

    WORDS_TASK_20 = [
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

    words_types = {
        "all": WORDS_ALL,
        "dir": WORDS_DIRECTION,
        "num": WORDS_NUMBERS,
        "k1": WORDS_KAGGLE_1,
        "w2": WORDS_TASK_20,
        "f1": ["happy", "learn", "wow", "visual"],
        "f2": ["backward", "eight", "go", "yes"],
        "_backward": ["backward"],
        "_bed": ["bed"],
        "_bird": ["bird"],
        "_cat": ["cat"],
        "_dog": ["dog"],
        "_down": ["down"],
        "_eight": ["eight"],
        "_five": ["five"],
        "_follow": ["follow"],
        "_forward": ["forward"],
        "_four": ["four"],
        "_go": ["go"],
        "_happy": ["happy"],
        "_house": ["house"],
        "_learn": ["learn"],
        "_left": ["left"],
        "_marvin": ["marvin"],
        "_nine": ["nine"],
        "_no": ["no"],
        "_off": ["off"],
        "_on": ["on"],
        "_one": ["one"],
        "_right": ["right"],
        "_seven": ["seven"],
        "_sheila": ["sheila"],
        "_six": ["six"],
        "_stop": ["stop"],
        "_three": ["three"],
        "_tree": ["tree"],
        "_two": ["two"],
        "_up": ["up"],
        "_visual": ["visual"],
        "_wow": ["wow"],
        "_yes": ["yes"],
        "_zero": ["zero"],
    }

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

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = '%(name)s: %(message)s'
    log_format_module = "%(message)s"

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
    logg.debug("Start setup_gpus")

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
