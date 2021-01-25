import numpy as np  # type: ignore
from math import floor
from librosa import display as ld  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# import logging


def plot_waveform(sample_sig, ax, title="Sample waveform", sample_rate=16000):
    """TODO: what is plot_waveform doing?"""
    ax.plot(sample_sig)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    # one tick every tenth of a second
    num_ticks = floor(sample_sig.shape[0] / (sample_rate / 10))
    ax.set_xticks(np.linspace(0, sample_sig.shape[0], num_ticks + 1))
    # length of sample in seconds
    sig_len = sample_sig.shape[0] / sample_rate
    ax.set_xticklabels(np.linspace(0, sig_len, num_ticks + 1).round(2))
    return ax


def plot_spec(log_spec, ax, title="Spectrogram", sample_rate=16000):
    """TODO: what is plot_spec doing?"""
    # logg = logging.getLogger(f"c.{__name__}.plot_spec")
    # logg.debug("Start plot_spec")
    ld.specshow(
        log_spec, sr=sample_rate, x_axis="time", y_axis="mel", cmap="viridis", ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    # the ticks are wrong if the sample is not 1 sec
    # ax.set_xticks(np.linspace(0, 1, 11))
    return ax


def plot_loss(train_loss, val_loss, ax, model_name):
    """TODO: what is plot_loss doing?"""
    # logg = logging.getLogger(f"c.{__name__}.plot_loss")
    # logg.debug("Start plot_loss")

    ax.plot(train_loss)
    ax.plot(val_loss)
    ax.set_title(f"{model_name} loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Val"], loc="upper right")


def plot_pred(pred, words, ax):
    """"""
    x = np.arange(len(words))
    ax.bar(x, pred, tick_label=words)


def plot_cat_acc(train_cat_acc, val_cat_acc, ax, model_name):
    """TODO: what is plot_cat_acc doing?"""
    # logg = logging.getLogger(f"c.{__name__}.plot_cat_acc")
    # logg.debug("Start plot_cat_acc")

    ax.plot(train_cat_acc)
    ax.plot(val_cat_acc)
    ax.set_title(f"{model_name} categorical accuracy")
    ax.set_ylabel("Categorical accuracy")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Val"], loc="lower right")


def plot_confusion_matrix(conf_mat, ax, model_name, words):
    """TODO: what is plot_confusion_matrix doing?

    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    # logg = logging.getLogger(f"c.{__name__}.plot_confusion_matrix")
    # logg.debug("Start plot_confusion_matrix")

    conf_mat_percent = np.divide(conf_mat.T, conf_mat.sum(axis=1)).T
    ax.imshow(conf_mat_percent, cmap=plt.cm.Blues)

    tick_marks = np.arange(len(words))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(words, rotation=45, fontsize=14)
    ax.set_yticklabels(words, fontsize=14)

    ax.set_title(f"{model_name} confusion matrix", fontsize=22)
    ax.set_ylabel("Observed", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=16)

    mid = conf_mat.max() / 2
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            c = "white" if conf_mat[i, j] > mid else "black"
            ax.text(j, i, conf_mat[i, j], ha="center", va="center", color=c, size=13)
