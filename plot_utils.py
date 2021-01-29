import numpy as np  # type: ignore
from math import floor
from librosa import display as ld  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore

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


def plot_pred(pred, words, ax, title="Predictions", correct_index=None):
    """"""
    num_words = len(words)

    # https://matplotlib.org/users/dflt_style_changes.html#colors-in-default-property-cycle
    if correct_index is not None:
        colors = ["C3"] * num_words
        colors[correct_index] = "C0"
    else:
        colors = ["C0"] * num_words

    # x contains the bar positions
    x = np.arange(num_words)
    ax.bar(x, pred, color=colors)

    # https://stackoverflow.com/a/63755285
    if num_words <= 6:
        ax.xaxis.set_major_locator(mticker.FixedLocator(x))
        ax.set_xticklabels(words)
    elif num_words <= 10:
        ax.xaxis.set_major_locator(mticker.FixedLocator(x))
        ax.set_xticklabels(words, rotation=30)
    else:
        # find where the predictions are high
        high_pred = pred > 0.1
        # add the correct word
        high_pred[correct_index] = True
        # filter the x locations
        high_x = x[high_pred]
        # set the locations
        ax.xaxis.set_major_locator(mticker.FixedLocator(high_x))
        # convert words in np array and filter them
        np_words = np.array(words)
        high_words = np_words[high_pred]
        # set the labels
        ax.set_xticklabels(high_words, rotation=90)

    ax.set_title(title)


def plot_att_weights(weights, ax, title="Attention weights"):
    """"""
    num_weights = len(weights)
    x = np.arange(num_weights)
    ax.bar(x, weights)
    ax.set_title(title)


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


def plot_confusion_matrix(conf_mat, ax, model_name, words, fscore=None):
    """TODO: what is plot_confusion_matrix doing?

    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    # logg = logging.getLogger(f"c.{__name__}.plot_confusion_matrix")
    # logg.debug("Start plot_confusion_matrix")
    num_words = len(words)

    conf_mat_percent = np.divide(conf_mat.T, conf_mat.sum(axis=1)).T
    ax.imshow(conf_mat_percent, cmap=plt.cm.Blues)

    if num_words <= 10:
        xtlr = 45
        lfs = 13
    else:
        xtlr = 90
        lfs = 10

    tick_marks = np.arange(num_words)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(words, rotation=xtlr, fontsize=14)
    ax.set_yticklabels(words, fontsize=14)
    ax.set_ylabel("Observed", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=16)

    title = f"{model_name}\nConfusion matrix"
    if fscore is not None:
        title += f" (F-score: {fscore:.3f})"
    ax.set_title(title, fontsize=22)

    thresh = conf_mat.max() * 0.3
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            c = "white" if conf_mat[i, j] > thresh else "black"
            ax.text(j, i, conf_mat[i, j], ha="center", va="center", color=c, size=lfs)
