from itertools import combinations
from pathlib import Path
import argparse
import json
import logging

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
import typing as ty

from plot_utils import plot_confusion_matrix
from plot_utils import plot_pred
from plot_utils import plot_spec
from plot_utils import plot_waveform
from plot_utils import quad_plotter
from preprocess_data import get_spec_dict
from preprocess_data import load_processed
from preprocess_data import wav2mel
from train import build_cnn_name
from utils import analyze_confusion
from utils import compute_permutation
from utils import pred_hot_2_cm
from utils import record_audios
from utils import setup_gpus
from utils import setup_logger
from utils import words_types


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="results",
        choices=[
            "results",
            "model",
            "audio",
            "delete_bad_models",
            "make_plots",
        ],
        help="Which evaluation to perform",
    )

    tra_types = [w for w in words_types.keys() if not w.startswith("_")]
    parser.add_argument(
        "-tw",
        "--train_words_type",
        type=str,
        default="f2",
        choices=tra_types,
        help="Words the dataset was trained on",
    )

    rec_types = [w for w in words_types.keys() if not w.startswith("_")]
    rec_types.append("train")
    parser.add_argument(
        "-rw",
        "--rec_words_type",
        type=str,
        default="train",
        choices=rec_types,
        help="Words to record and test",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 evaluate.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def build_cnn_results_df() -> pd.DataFrame:
    """TODO: what is build_cnn_results_df doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_cnn_results_df")
    logg.setLevel("INFO")
    logg.debug("Start build_cnn_results_df")

    info_folder = Path("info")

    pandito: ty.Dict[str, ty.List[str]] = {
        "dense_width": [],
        "filters": [],
        "batch_size": [],
        "dataset": [],
        "dropout": [],
        "epoch_num": [],
        "kernel_size": [],
        "pool_size": [],
        "lr": [],
        "opt": [],
        "words": [],
        "fscore": [],
        "loss": [],
        "cat_acc": [],
        "model_name": [],
    }

    for model_folder in info_folder.iterdir():
        logg.debug(f"model_folder: {model_folder}")

        model_name = model_folder.name
        if not model_name.startswith("CNN"):
            continue

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue

        results_recap = json.loads(res_recap_path.read_text())
        logg.debug(f"results_recap['cm']: {results_recap['cm']}")

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())
        logg.debug(f"recap['words']: {recap['words']}")

        cm = np.array(results_recap["cm"])
        fscore = analyze_confusion(cm, recap["words"])

        categorical_accuracy = results_recap["categorical_accuracy"]
        logg.debug(f"categorical_accuracy: {categorical_accuracy}")

        pandito["dense_width"].append(recap["hypa"]["base_dense_width"])
        pandito["filters"].append(recap["hypa"]["base_filters"])
        pandito["batch_size"].append(recap["hypa"]["batch_size"])
        pandito["dataset"].append(recap["hypa"]["dataset"])
        pandito["dropout"].append(recap["hypa"]["dropout_type"])
        pandito["epoch_num"].append(recap["hypa"]["epoch_num"])
        pandito["kernel_size"].append(recap["hypa"]["kernel_size_type"])
        pandito["pool_size"].append(recap["hypa"]["pool_size_type"])
        pandito["words"].append(recap["hypa"]["words"])
        pandito["cat_acc"].append(results_recap["categorical_accuracy"])
        pandito["loss"].append(results_recap["loss"])
        pandito["model_name"].append(results_recap["model_name"])
        pandito["fscore"].append(fscore)

        if "version" in recap:
            if recap["version"] == "001":
                pandito["lr"].append(recap["hypa"]["learning_rate_type"])
                pandito["opt"].append(recap["hypa"]["optimizer_type"])
        else:
            pandito["lr"].append("default")
            pandito["opt"].append("adam")

    df = pd.DataFrame(pandito)
    return df


def evaluate_results_cnn(args):
    """TODO: what is evaluate_results_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_cnn")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_cnn")

    results_df = build_cnn_results_df()
    fscore_df = results_df.sort_values("fscore", ascending=False).head(30)
    logg.info(f"{fscore_df}")


def make_plots_cnn() -> None:
    """TODO: what is make_plots_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.make_plots_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start make_plots_cnn")

    results_df = build_cnn_results_df()

    # the output folders
    plot_fol = Path("plot_results") / "cnn"
    pdf_split_fol = plot_fol / "pdf_split"
    pdf_grid_fol = plot_fol / "pdf_grid"
    png_split_fol = plot_fol / "png_split"
    png_grid_fol = plot_fol / "png_grid"
    for f in pdf_split_fol, pdf_grid_fol, png_split_fol, png_grid_fol:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)

    # the real hypa values are not str but we only need them for the query
    hypa_grid: ty.Dict[str, ty.List[str]] = {}
    hypa_grid["filters"] = ["10", "20", "30", "32", "64", "128"]
    hypa_grid["kernel_size"] = ["01", "02", "03"]
    hypa_grid["pool_size"] = ["01", "02"]
    hypa_grid["dense_width"] = ["16", "32", "64", "128"]
    hypa_grid["dropout"] = ["01", "02"]
    hypa_grid["batch_size"] = ["16", "32", "64"]
    hypa_grid["epoch_num"] = ["15", "30", "60"]
    hypa_grid["lr"] = ["01", "02", "03"]
    hypa_grid["opt"] = ["a1", "r1"]
    ds = ["mel01", "mel02", "mel03", "mel04"]
    ds.extend(["mfcc01", "mfcc02", "mfcc03", "mfcc04"])
    hypa_grid["dataset"] = ds
    hypa_grid["words"] = ["f2", "f1", "num", "dir", "k1", "w2", "all"]

    all_hp_to_plot = list(combinations(hypa_grid.keys(), 4))
    logg.debug(f"len(all_hp_to_plot): {len(all_hp_to_plot)}")

    all_hp_to_plot = []
    all_hp_to_plot.append(("epoch_num", "dataset", "lr", "filters"))
    all_hp_to_plot.append(("batch_size", "filters", "kernel_size", "dropout"))
    all_hp_to_plot.append(("words", "dense_width", "batch_size", "dataset"))
    all_hp_to_plot.append(("dataset", "words", "dense_width", "batch_size"))
    all_hp_to_plot.append(("dataset", "batch_size", "dense_width", "words"))
    all_hp_to_plot.append(("batch_size", "dense_width", "words", "dataset"))
    all_hp_to_plot.append(("dense_width", "batch_size", "dataset", "words"))

    quad_plotter(
        all_hp_to_plot,
        hypa_grid,
        results_df,
        pdf_split_fol,
        png_split_fol,
        pdf_grid_fol,
        png_grid_fol,
        do_single_images=True,
    )

    # for hp_to_plot in tqdm(all_hp_to_plot[:]):
    #     outer_hp = hp_to_plot[-1]
    #     inner_hp = hp_to_plot[:-1]
    #     # logg.debug(f"outer_hp: {outer_hp} inner_hp: {inner_hp}")

    #     # split outer value (changes across subplots)
    #     outer_values = hypa_grid[outer_hp]
    #     outer_dim = len(outer_values)

    #     # and inner values (change within a subplot)
    #     inner_values = [hypa_grid[hptp] for hptp in inner_hp]
    #     labels_dim = [len(lab) for lab in inner_values]

    #     # build the grid of subplots
    #     nrows, ncols = find_rowcol(outer_dim)
    #     base_figsize = 10
    #     figsize = (ncols * base_figsize * 1.5, nrows * base_figsize)
    #     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey="all")
    #     # make flat_ax always a list of axes
    #     flat_ax = list(axes.flat) if outer_dim > 1 else [axes]

    #     f_mean = np.zeros((*labels_dim, outer_dim))
    #     f_min = np.zeros((*labels_dim, outer_dim))
    #     f_max = np.zeros((*labels_dim, outer_dim))
    #     f_std = np.zeros((*labels_dim, outer_dim))

    #     # first we compute all the f values
    #     for iv, outer_value in enumerate(outer_values):
    #         for iz, vz in enumerate(inner_values[2]):
    #             for iy, vy in enumerate(inner_values[1]):
    #                 for ix, vx in enumerate(inner_values[0]):
    #                     q_str = f"({hp_to_plot[0]} == '{vx}')"
    #                     q_str += f" and ({hp_to_plot[1]} == '{vy}')"
    #                     q_str += f" and ({hp_to_plot[2]} == '{vz}')"
    #                     q_str += f" and ({outer_hp} == '{outer_value}')"
    #                     q_df = results_df.query(q_str)

    #                     if len(q_df) == 0:
    #                         continue

    #                     f_mean[ix, iy, iz, iv] = q_df.fscore.mean()
    #                     f_min[ix, iy, iz, iv] = q_df.fscore.min()
    #                     f_max[ix, iy, iz, iv] = q_df.fscore.max()
    #                     f_std[ix, iy, iz, iv] = q_df.fscore.std()

    #                     if f_std[ix, iy, iz, iv] is None:
    #                         logg.debug("Found None std")

    #     f_mean_nonzero = f_mean[f_mean > 0]
    #     f_mean_all = f_mean_nonzero.mean()
    #     # logg.debug(f"f_mean_all: {f_mean_all}")

    #     # then we plot them
    #     for iv, outer_value in enumerate(outer_values):
    #         # plot on the outer grid
    #         plot_triple_data(
    #             flat_ax[iv],
    #             inner_values,
    #             hp_to_plot,
    #             f_mean[:, :, :, iv],
    #             f_min[:, :, :, iv],
    #             f_max[:, :, :, iv],
    #             f_std[:, :, :, iv],
    #             outer_hp,
    #             outer_value,
    #             f_mean_all,
    #             f_max.max(),
    #         )

    #         # do_single_images = True
    #         do_single_images = False
    #         if do_single_images:
    #             # plot on a single image
    #             base_figsize = 10
    #             figsize_in = (1.5 * base_figsize, base_figsize)
    #             fig_in, ax_in = plt.subplots(figsize=figsize_in)

    #             plot_triple_data(
    #                 ax_in,
    #                 inner_values,
    #                 hp_to_plot,
    #                 f_mean[:, :, :, iv],
    #                 f_min[:, :, :, iv],
    #                 f_max[:, :, :, iv],
    #                 f_std[:, :, :, iv],
    #                 outer_hp,
    #                 outer_value,
    #                 f_mean_all,
    #                 f_max.max(),
    #             )

    #             # save and close the single image
    #             fig_name = "Fscore"
    #             fig_name += f"_{hp_to_plot[2]}_{vz}"
    #             fig_name += f"_{hp_to_plot[1]}_{vy}"
    #             fig_name += f"_{hp_to_plot[0]}_{vx}"
    #             fig_name += f"_{outer_hp}_{outer_value}.{{}}"
    #             fig_in.tight_layout()
    #             fig_in.savefig(pdf_split_fol / fig_name.format("pdf"))
    #             fig_in.savefig(png_split_fol / fig_name.format("png"))
    #             plt.close(fig_in)

    #     # save and close the composite image
    #     fig_name = "Fscore"
    #     fig_name += f"__{hp_to_plot[2]}"
    #     fig_name += f"__{hp_to_plot[1]}"
    #     fig_name += f"__{hp_to_plot[0]}"
    #     fig_name += f"__{outer_hp}.{{}}"
    #     fig.tight_layout()
    #     fig.savefig(pdf_grid_fol / fig_name.format("pdf"))
    #     fig.savefig(png_grid_fol / fig_name.format("png"))
    #     plt.close(fig)


def evaluate_model_cnn(args):
    """TODO: what is evaluate_model_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_model_cnn")

    train_words_type = args.train_words_type
    dataset = "mel01"

    # magic to fix the GPUs
    setup_gpus()

    # setup the parameters
    hypa: ty.Dict[str, ty.Union[str, int]] = {}
    hypa["base_dense_width"] = 32
    hypa["base_filters"] = 20
    hypa["batch_size"] = 32
    hypa["dropout_type"] = "01"
    hypa["epoch_num"] = 16
    hypa["kernel_size_type"] = "02"
    hypa["pool_size_type"] = "02"
    hypa["learning_rate_type"] = "02"
    hypa["optimizer_type"] = "a1"
    hypa["dataset"] = dataset
    hypa["words"] = train_words_type

    # get the words
    train_words = words_types[train_words_type]

    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        raise FileNotFoundError

    model = tf.keras.models.load_model(model_path)
    model.summary()

    # input data
    processed_path = Path("data_proc") / f"{dataset}"
    data, labels = load_processed(processed_path, train_words)
    logg.debug(f"data['testing'].shape: {data['testing'].shape}")

    # evaluate on the words you trained on
    logg.debug("Evaluate on test data:")
    model.evaluate(data["testing"], labels["testing"])
    # model.evaluate(data["validation"], labels["validation"])

    # predict labels/cm/fscore
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, train_words)
    # y_pred = model.predict(data["validation"])
    # cm = pred_hot_2_cm(labels["validation"], y_pred, train_words)
    fscore = analyze_confusion(cm, train_words)
    logg.debug(f"fscore: {fscore}")

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, train_words, fscore)

    plt.show()


def evaluate_audio_cnn(args):
    """TODO: what is evaluate_audio_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_audio_cnn")
    logg.debug("Start evaluate_audio_cnn")

    # magic to fix the GPUs
    setup_gpus()

    # need to know on which dataset the model was trained to compute specs
    dataset_name = "mel01"

    # words that the dataset was trained on
    train_words_type = args.train_words_type
    train_words = words_types[train_words_type]

    # permutation from sorted to your wor(l)d order
    perm_pred = compute_permutation(train_words)

    rec_words_type = args.rec_words_type
    if rec_words_type == "train":
        rec_words = train_words
    else:
        rec_words = words_types[rec_words_type]
    num_rec_words = len(rec_words)

    # where to save the audios
    audio_folder = Path("recorded_audio")
    if not audio_folder.exists():
        audio_folder.mkdir(parents=True, exist_ok=True)

    # record the audios and save them in audio_folder
    audio_path_fmt = "{}_02.wav"
    audios = record_audios(rec_words, audio_folder, audio_path_fmt, timeout=0)

    # compute the spectrograms and build the dataset of correct shape
    img_specs = []
    spec_dict = get_spec_dict()
    spec_kwargs = spec_dict[dataset_name]
    p2d_kwargs = {"ref": np.max}
    for word in rec_words:
        # get the name
        audio_path = audio_folder / audio_path_fmt.format(word)

        # convert it to mel
        log_spec = wav2mel(audio_path, spec_kwargs, p2d_kwargs)
        img_spec = log_spec.reshape((*log_spec.shape, 1))
        logg.debug(f"img_spec.shape: {img_spec.shape}")  # img_spec.shape: (128, 32, 1)

        img_specs.append(img_spec)

    # the data needs to look like this data['testing'].shape: (735, 128, 32, 1)
    # data = log_spec.reshape((1, *log_spec.shape, 1))
    data = np.stack(img_specs)
    logg.debug(f"data.shape: {data.shape}")

    hypa: ty.Dict[str, ty.Union[str, int]] = {}
    hypa["base_dense_width"] = 32
    hypa["base_filters"] = 20
    hypa["batch_size"] = 32
    hypa["dropout_type"] = "01"
    hypa["epoch_num"] = 16
    hypa["kernel_size_type"] = "02"
    hypa["pool_size_type"] = "02"
    hypa["learning_rate_type"] = "02"
    hypa["optimizer_type"] = "a1"
    hypa["dataset"] = dataset_name
    hypa["words"] = train_words_type

    # get the words
    train_words = words_types[train_words_type]

    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    # model_folder = Path("trained_models")
    model_folder = Path("saved_models")
    model_path = model_folder / f"{model_name}.h5"
    if not model_path.exists():
        logg.error(f"Model not found at: {model_path}")
        raise FileNotFoundError

    model = tf.keras.models.load_model(model_path)
    model.summary()

    pred = model.predict(data)
    # logg.debug(f"pred: {pred}")

    # plot the thing
    plot_size = 5
    fw = plot_size * 3
    fh = plot_size * num_rec_words
    fig, axes = plt.subplots(nrows=num_rec_words, ncols=3, figsize=(fw, fh))
    fig.suptitle("Recorded audios", fontsize=18)

    for i, word in enumerate(rec_words):
        plot_waveform(audios[i], axes[i][0])
        spec = img_specs[i][:, :, 0]
        plot_spec(spec, axes[i][1])
        plot_pred(
            pred[i][perm_pred],
            train_words,
            axes[i][2],
            f"Prediction for {rec_words[i]}",
            train_words.index(word),
        )

    # https://stackoverflow.com/q/8248467
    # https://stackoverflow.com/q/2418125
    fig.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.97])

    fig_name = f"{model_name}_{train_words_type}_{rec_words_type}.png"
    results_path = audio_folder / fig_name
    fig.savefig(results_path)

    if num_rec_words <= 6:
        plt.show()


def delete_bad_models_cnn(args) -> None:
    """TODO: what is delete_bad_models_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start delete_bad_models_cnn")

    info_folder = Path("info")
    trained_folder = Path("trained_models")
    f_tresh = 0.87
    ca_tresh = 0.95
    deleted = 0
    recreated = 0

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue

        results_recap = json.loads(res_recap_path.read_text())
        # logg.debug(f"results_recap['cm']: {results_recap['cm']}")

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())
        # logg.debug(f"recap['words']: {recap['words']}")

        cm = np.array(results_recap["cm"])
        fscore = analyze_confusion(cm, recap["words"])
        # logg.debug(f"fscore: {fscore}")

        categorical_accuracy = results_recap["categorical_accuracy"]
        # logg.debug(f"categorical_accuracy: {categorical_accuracy}")

        if fscore < f_tresh and categorical_accuracy < ca_tresh:
            model_name = model_folder.name
            model_path = trained_folder / f"{model_name}.h5"

            if model_path.exists():
                # model_path.unlink()
                deleted += 1
                logg.debug(f"Deleting model_path: {model_path}")
                logg.debug(f"fscore: {fscore}")
                logg.debug(f"categorical_accuracy: {categorical_accuracy}")
                # model_path.write_text("Deleted")

            # you gone goofed and deleted a model, put the placeholder
            else:
                # model_path.write_text("Deleted")
                logg.debug(f"Recreating model_path: {model_path}")
                recreated += 1

    logg.debug(f"deleted: {deleted}")
    logg.debug(f"recreated: {recreated}")


def run_evaluate_cnn(args: argparse.Namespace) -> None:
    """TODO: What is evaluate_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_cnn")
    logg.debug("Starting run_evaluate_cnn")

    evaluation_type = args.evaluation_type
    # train_words_type = args.train_words_type
    # rec_words_type = args.rec_words_type

    pd.set_option("max_colwidth", 100)

    if evaluation_type == "results":
        evaluate_results_cnn(args)
    elif evaluation_type == "make_plots":
        make_plots_cnn()
    elif evaluation_type == "model":
        evaluate_model_cnn(args)
    elif evaluation_type == "audio":
        evaluate_audio_cnn(args)
    elif evaluation_type == "delete_bad_models":
        delete_bad_models_cnn(args)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_cnn(args)