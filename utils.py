import logging
import numpy as np  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
import tensorflow as tf  # type: ignore

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
