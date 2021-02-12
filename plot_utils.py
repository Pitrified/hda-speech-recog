from cycler import cycler  # type: ignore
from librosa import display as ld  # type: ignore
from math import floor
from pathlib import Path
from tqdm import tqdm  # type: ignore
import logging

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import typing as ty

from utils import find_rowcol


def plot_waveform(sample_sig, ax, title="Sample waveform", sample_rate=16000):
    """MAKEDOC: what is plot_waveform doing?"""
    ax.plot(sample_sig)
    ax.set_title(title, fontsize=20)
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
    """MAKEDOC: what is plot_spec doing?"""
    # logg = logging.getLogger(f"c.{__name__}.plot_spec")
    # logg.debug("Start plot_spec")
    ld.specshow(
        log_spec, sr=sample_rate, x_axis="time", y_axis="mel", cmap="viridis", ax=ax
    )
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Time (s)")
    # the ticks are wrong if the sample is not 1 sec
    # ax.set_xticks(np.linspace(0, 1, 11))
    return ax


def plot_loss(train_loss, val_loss, ax, model_name):
    """MAKEDOC: what is plot_loss doing?"""
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
    """MAKEDOC: what is plot_cat_acc doing?"""
    # logg = logging.getLogger(f"c.{__name__}.plot_cat_acc")
    # logg.debug("Start plot_cat_acc")

    ax.plot(train_cat_acc)
    ax.plot(val_cat_acc)
    ax.set_title(f"{model_name} categorical accuracy")
    ax.set_ylabel("Categorical accuracy")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Val"], loc="lower right")


def plot_confusion_matrix(conf_mat, ax, model_name, words, fscore=None):
    """MAKEDOC: what is plot_confusion_matrix doing?

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


def plot_double_data(
    lab_values, lab_names, f_mean, f_min, f_max, f_std, outer_label, ax
):
    """Plot groups of columns

    f_mean.shape (x, y)

    y groups of columns
    x columns per group
    """

    title = f"{lab_names[2]} {outer_label}\n"
    title += f"{lab_names[0]} grouped by {lab_names[1]}"
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(f"{lab_names[1]}", fontsize=14)
    ax.set_ylabel("F-score (min/mean/max and std-dev)", fontsize=14)

    f_dim = f_mean.shape
    width_group = 0.8
    width_col = width_group / f_dim[0]

    x_pos = np.arange(f_dim[1])
    x_ticks = x_pos + (width_group / 2)

    for ix in range(f_dim[0]):
        shift_col = width_col * ix
        x_col = x_pos + shift_col

        y_f = f_mean[ix, :]

        y_min = y_f - f_min[ix, :]
        y_max = f_max[ix, :] - y_f
        y_err = np.vstack((y_min, y_max))

        ax.bar(
            x=x_col,
            height=y_f,
            width=width_col,
            yerr=y_err,
            label=lab_values[0][ix],
            align="edge",
            capsize=5,
        )

        y_std = f_std[ix, :]
        ax.errorbar(
            x_col + width_col / 2,
            y_f,
            yerr=y_std,
            linestyle="None",
            capsize=3,
            capthick=4,
            ecolor="b",
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(lab_values[1])
    ax.legend()


def plot_triple_data(
    ax: plt.Axes,
    lab_values: ty.List[ty.List[str]],
    lab_names: ty.Tuple[str, str, str, str],
    f_mean: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    f_std: np.ndarray,
    outer_label: str = None,
    outer_value: str = None,
    mean_all: float = None,
    max_all: float = None,
    min_all: float = None,
    min_lower_limit: float = 0,
):
    """Plot groups of groups of columns

    f_mean.shape (x, y, z)

    z groups of (groups of columns)
    y groups of columns
    x columns per group

        example of (3, 4, 2) shape
    ^
    |                   x x
    |x                  xxx xx   x
    |xxx xx             xxx xx  xx  x
    |xxx xxx x   x x    xxx xxx xxx x x
    |xxx xxx xxx xxx    xxx xxx xxx xxx
    |xxx xxx xxx xxx    xxx xxx xxx xxx
    .----------------------------------->
    """
    # logg = logging.getLogger(f"c.{__name__}.plot_triple_data")
    # logg.setLevel("INFO")
    # logg.debug("Start plot_triple_data")
    # logg.debug(f"f_mean.shape: {f_mean.shape}")

    title = ""
    if outer_label is not None and outer_value is not None:
        title += f"{outer_label}"
        # title += f": \\textbf{{{outer_value}}}"
        title += f": $\\bf{{{outer_value}}}$"
        title += "\n"
    title += f"{lab_names[0]}"
    title += f": {lab_values[0]}"
    title += "\n"
    title += f" grouped by {lab_names[1]}"
    title += f": {lab_values[1]}"
    title += "\n"
    title += f" grouped by {lab_names[2]}"
    title += f": {lab_values[2]}"
    ax.set_title(title, fontsize=14)

    lab_fontsize = 14
    ax.set_ylabel("F-score (min/mean/max and std-dev)", fontsize=lab_fontsize)
    ax.set_xlabel(f"{lab_names[1]} ({lab_names[2]})", fontsize=lab_fontsize)

    f_dim = f_mean.shape

    # the width of each super group
    width_outer_group = 0.9

    # scale the available space by 0.8 to leave space between subgroups
    width_inner_group = width_outer_group * 0.8 / f_dim[1]

    # the columns are side by side
    width_inner_col = width_inner_group / f_dim[0]

    # where the z super groups of columns start
    x_outer_pos = np.arange(f_dim[2])

    # where the y sub groups start
    x_inner_pos = np.arange(f_dim[1]) * width_outer_group / f_dim[1]

    # where to put the ticks for each subgroup
    x_inner_ticks = x_inner_pos + (width_inner_group / 2)

    # all the ticks and the relative labels
    all_x_ticks = np.array([])
    all_xticklabels = []

    # if there are too many groups of columns draw less info
    too_many_groups = f_dim[1] * f_dim[2] > 12

    if not too_many_groups:
        err_capsize = 5
        std_capsize = 3
        std_capthick = 4
        xticklabels_rot = 0
        xticklabels_ha = "center"
    else:
        err_capsize = 3
        std_capsize = 2
        std_capthick = 3
        xticklabels_rot = 30
        xticklabels_ha = "right"

    # for each super group
    for iz in range(f_dim[2]):

        # where this group starts
        shift_group = x_outer_pos[iz]

        # where to put the ticks
        this_ticks = x_inner_ticks + shift_group
        all_x_ticks = np.hstack((all_x_ticks, this_ticks))
        this_labels = [f"{vy} ({lab_values[2][iz]})" for vy in lab_values[1]]
        all_xticklabels.extend(this_labels)

        # reset the cycler
        cc = cycler(
            color=[
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
        )

        ax.set_prop_cycle(cc)

        # for each column
        for ix in range(f_dim[0]):

            # how much this batch of y columns must be shifted
            shift_col = width_inner_col * ix
            x_col = x_inner_pos + shift_col + shift_group

            # extract the values of the y columns in this batch
            y_f = f_mean[ix, :, iz]

            # only put the label for the first z slice
            the_label = lab_values[0][ix] if iz == 0 else None

            # plot the bars
            ax.bar(
                x=x_col,
                height=y_f,
                width=width_inner_col,
                label=the_label,
                align="edge",
                capsize=err_capsize,
            )

            # compute the relative min/max
            y_min = y_f - f_min[ix, :, iz]
            y_max = f_max[ix, :, iz] - y_f
            y_err = np.vstack((y_min, y_max))
            ax.errorbar(
                x=x_col + width_inner_col / 2,
                y=y_f,
                yerr=y_err,
                linestyle="None",
                capsize=err_capsize,
                ecolor="k",
            )

            # get the standard deviation
            y_std = f_std[ix, :, iz]
            ax.errorbar(
                x_col + width_inner_col / 2,
                y_f,
                yerr=y_std,
                linestyle="None",
                capsize=std_capsize,
                capthick=std_capthick,
                ecolor="b",
            )

    if min_all is not None and min_all < min_lower_limit:
        # logg.debug(f"Resettin min_all: {min_all} to min_lower_limit")
        min_all = min_lower_limit

    if max_all is not None and min_all is not None:
        ax.set_ylim(top=max_all * 1.01, bottom=min_all * 0.99)
    elif max_all is not None:
        ax.set_ylim(top=max_all * 1.01)
    elif min_all is not None:
        ax.set_ylim(bottom=min_all * 0.99)

    if mean_all is not None:
        ax.axhline(mean_all)

        bottom, top = ax.get_ylim()
        mean_rescaled = ((mean_all - bottom) / (top - bottom)) * 1.01
        ax.annotate(
            text=f"{mean_all:.03f}",
            xy=(0.005, mean_rescaled),
            xycoords="axes fraction",
            fontsize=13,
        )

    ax.set_xticks(all_x_ticks)
    ax.set_xticklabels(
        labels=all_xticklabels,
        rotation=xticklabels_rot,
        horizontalalignment=xticklabels_ha,
    )
    ax.legend(title=f"{lab_names[0]}", title_fontsize=lab_fontsize)


def quad_plotter(
    all_hp_to_plot: ty.List[ty.Tuple[str, str, str, str]],
    hypa_grid: ty.Dict[str, ty.List[str]],
    results_df: pd.DataFrame,
    pdf_split_fol: Path,
    png_split_fol: Path,
    pdf_grid_fol: Path,
    png_grid_fol: Path,
    do_single_images: bool = True,
    min_at_zero: bool = False,
    min_lower_limit: float = 0,
) -> None:
    """MAKEDOC: what is quad_plotter doing?"""
    logg = logging.getLogger(f"c.{__name__}.quad_plotter")
    # logg.setLevel("INFO")
    # logg.debug("Start quad_plotter")

    for hp_to_plot in tqdm(all_hp_to_plot):
        outer_hp: str = hp_to_plot[-1]
        inner_hp: ty.List[str] = hp_to_plot[:-1]
        # logg.debug(f"outer_hp: {outer_hp} inner_hp: {inner_hp}")

        # split outer value (changes across subplots)
        outer_values: ty.List[str] = hypa_grid[outer_hp]
        outer_dim: int = len(outer_values)

        # and inner values (change within a subplot)
        inner_values: ty.List[ty.List[str]] = [hypa_grid[hptp] for hptp in inner_hp]
        labels_dim: ty.List[int] = [len(lab) for lab in inner_values]

        # build the grid of subplots
        nrows, ncols = find_rowcol(outer_dim)
        base_figsize = 10
        figsize = (ncols * base_figsize * 1.5, nrows * base_figsize)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey="all")
        # make flat_ax always a list of axes
        flat_ax = list(axes.flat) if outer_dim > 1 else [axes]

        f_mean = np.zeros((*labels_dim, outer_dim))
        f_min = np.zeros((*labels_dim, outer_dim))
        f_max = np.zeros((*labels_dim, outer_dim))
        f_std = np.zeros((*labels_dim, outer_dim))

        # first we compute all the f values
        for iv, outer_value in enumerate(outer_values):
            for iz, vz in enumerate(inner_values[2]):
                for iy, vy in enumerate(inner_values[1]):
                    for ix, vx in enumerate(inner_values[0]):
                        q_str = f"({hp_to_plot[0]} == '{vx}')"
                        q_str += f" and ({hp_to_plot[1]} == '{vy}')"
                        q_str += f" and ({hp_to_plot[2]} == '{vz}')"
                        q_str += f" and ({outer_hp} == '{outer_value}')"
                        q_df = results_df.query(q_str)

                        if len(q_df) == 0:
                            continue

                        f_mean[ix, iy, iz, iv] = q_df.fscore.mean()
                        f_min[ix, iy, iz, iv] = q_df.fscore.min()
                        f_max[ix, iy, iz, iv] = q_df.fscore.max()
                        f_std[ix, iy, iz, iv] = q_df.fscore.std()

                        if f_std[ix, iy, iz, iv] is None:
                            logg.debug("Found None std")

        f_mean_nonzero = f_mean[f_mean > 0]
        f_mean_all = f_mean_nonzero.mean()
        f_min_nonzero = f_min[f_min > 0]
        f_min_all = f_min_nonzero.min()

        # then we plot them
        for iv, outer_value in enumerate(outer_values):
            # plot on the outer grid
            plot_triple_data(
                flat_ax[iv],
                inner_values,
                hp_to_plot,
                f_mean[:, :, :, iv],
                f_min[:, :, :, iv],
                f_max[:, :, :, iv],
                f_std[:, :, :, iv],
                outer_hp,
                outer_value,
                f_mean_all,
                f_max.max(),
                f_min_all,
                min_lower_limit,
            )

            if do_single_images:
                # plot on a single image
                base_figsize = 10
                figsize_in = (1.5 * base_figsize, base_figsize)
                fig_in, ax_in = plt.subplots(figsize=figsize_in)

                plot_triple_data(
                    ax_in,
                    inner_values,
                    hp_to_plot,
                    f_mean[:, :, :, iv],
                    f_min[:, :, :, iv],
                    f_max[:, :, :, iv],
                    f_std[:, :, :, iv],
                    outer_hp,
                    outer_value,
                    f_mean_all,
                    f_max.max(),
                    f_min_all,
                    min_lower_limit,
                )

                # save and close the single image
                fig_name = "Fscore"
                fig_name += f"_{hp_to_plot[2]}_{vz}"
                fig_name += f"_{hp_to_plot[1]}_{vy}"
                fig_name += f"_{hp_to_plot[0]}_{vx}"
                fig_name += f"_{outer_hp}_{outer_value}.{{}}"
                fig_in.tight_layout()
                fig_in.savefig(pdf_split_fol / fig_name.format("pdf"))
                fig_in.savefig(png_split_fol / fig_name.format("png"))
                plt.close(fig_in)

        # save and close the composite image
        fig_name = "Fscore"
        fig_name += f"__{hp_to_plot[2]}"
        fig_name += f"__{hp_to_plot[1]}"
        fig_name += f"__{hp_to_plot[0]}"
        fig_name += f"__{outer_hp}.{{}}"
        fig.tight_layout()
        fig.savefig(pdf_grid_fol / fig_name.format("pdf"))
        fig.savefig(png_grid_fol / fig_name.format("png"))
        plt.close(fig)
