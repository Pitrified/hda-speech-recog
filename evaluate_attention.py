from itertools import combinations
from pathlib import Path
import argparse
import json
import logging

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore

from plot_utils import plot_att_weights
from plot_utils import plot_pred
from plot_utils import plot_spec
from plot_utils import quad_plotter
from preprocess_data import load_processed
from train import build_attention_name
from utils import compute_permutation
from utils import setup_gpus
from utils import setup_logger
from utils import words_types

# import typing as ty
from typing import Dict
from typing import List
from typing import Any


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="results",
        choices=[
            "results",
            "attention_weights",
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


def build_att_results_df() -> pd.DataFrame:
    """TODO: what is build_att_results_df doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_att_results_df")
    # logg.setLevel("INFO")
    logg.debug("Start build_att_results_df")

    pandito: Dict[str, List[str]] = {
        "words": [],
        "dataset": [],
        "conv": [],
        "dropout": [],
        "kernel": [],
        "lstm": [],
        "att": [],
        "query": [],
        "dense": [],
        "lr": [],
        "optimizer": [],
        "batch": [],
        "epoch": [],
        "use_val": [],
        "loss": [],
        "cat_acc": [],
        "precision": [],
        "recall": [],
        "fscore": [],
        "model_name": [],
    }

    info_folder = Path("info")

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")
        model_name = model_folder.name
        if not model_name.startswith("ATT"):
            continue
        # logg.debug(f"model_name: {model_name}")

        res_path = model_folder / "results_recap.json"
        if not res_path.exists():
            logg.info(f"Skipping res_path: {res_path}, not found")
            continue
        res = json.loads(res_path.read_text())

        recap_path = model_folder / "recap.json"
        if not recap_path.exists():
            logg.info(f"Skipping recap_path: {recap_path}, not found")
            continue
        recap = json.loads(recap_path.read_text())

        if res["results_recap_version"] == "001":
            logg.debug("\nWHAT ARE YOU DOING using this version\n")

        hypa: Dict[str, Any] = recap["hypa"]

        pandito["words"].append(hypa["words_type"])
        pandito["dataset"].append(hypa["dataset_name"])
        pandito["conv"].append(hypa["conv_size_type"])
        pandito["dropout"].append(hypa["dropout_type"])
        pandito["kernel"].append(hypa["kernel_size_type"])
        pandito["lstm"].append(hypa["lstm_units_type"])
        pandito["att"].append(hypa["att_sample_type"])
        pandito["query"].append(hypa["query_style_type"])
        pandito["dense"].append(hypa["dense_width_type"])

        pandito["lr"].append(hypa["learning_rate_type"])
        pandito["optimizer"].append(hypa["optimizer_type"])
        pandito["batch"].append(hypa["batch_size_type"])
        pandito["epoch"].append(hypa["epoch_num_type"])

        pandito["use_val"].append(recap["use_validation"])
        pandito["loss"].append(res["loss"])
        pandito["cat_acc"].append(res["categorical_accuracy"])
        pandito["precision"].append(res["precision"])
        pandito["recall"].append(res["recall"])
        pandito["fscore"].append(res["fscore"])

        pandito["model_name"].append(res["model_name"])

    pd.set_option("max_colwidth", 100)
    df = pd.DataFrame(pandito)

    return df


def evaluate_results_attention() -> None:
    """TODO: what is evaluate_results_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_attention")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_results_attention")

    results_df = build_att_results_df()

    df_f = results_df
    # filter the dataframe to find the best hypas
    # df_f = df_f.query("use_val == True")
    # df_f = df_f.query("words == 'k1'")
    # df_f = df_f.query("words == 'w2'")
    df_f = df_f.sort_values("fscore", ascending=False)
    logg.info(f"{df_f.head(30)}")
    logg.info(f"{df_f.tail()}")

    # produce a dict for recreating the training
    for index, row in df_f.iterrows():
        hp = "    {"
        hp += f"        'words_type': '{row['words']}',"
        hp += f"        'dataset_name': '{row['dataset']}',"
        hp += f"        'conv_size_type': '{row['conv']}',"
        hp += f"        'dropout_type': '{row['dropout']}',"
        hp += f"        'kernel_size_type': '{row['kernel']}',"
        hp += f"        'lstm_units_type': '{row['lstm']}',"
        hp += f"        'att_sample_type': '{row['att']}',"
        hp += f"        'query_style_type': '{row['query']}',"
        hp += f"        'dense_width_type': '{row['dense']}',"
        hp += f"        'learning_rate_type': '{row['lr']}',"
        hp += f"        'optimizer_type': '{row['optimizer']}',"
        hp += f"        'batch_size_type': '{row['batch']}',"
        hp += f"        'epoch_num_type': '{row['epoch']}',"
        hp += "    },"
        # logg.debug(f"hp: {hp}")

    # compare the rankings with and without validation
    best_val = results_df.query("use_val==True").sort_values("fscore", ascending=False)
    best_noval = results_df.query("use_val==False").sort_values(
        "fscore", ascending=False
    )

    rank: Dict[str, List[int]] = {}
    num_head = 30
    for i, (_, row) in enumerate(best_val.head(num_head).iterrows()):
        model_name = row["model_name"]
        hypa_str = model_name[4:]  # chop off ATT_ at the beginning
        # logg.debug(f"hypa_str: {hypa_str}")
        rank[hypa_str] = [i, 0]

    for i, (_, row) in enumerate(best_noval.head(num_head).iterrows()):
        model_name = row["model_name"]
        hypa_str = model_name[4:]  # chop off ATT_ at the beginning
        hypa_str = hypa_str[:-6]  # chop off _noval at the end
        # logg.debug(f"hypa_str: {hypa_str}")

        # if you are a fool and trained with noval and not with val the key is not there
        if hypa_str in rank:
            rank[hypa_str][1] = i
        else:
            rank[hypa_str] = [0, i]

    rank_pd: Dict[str, Any] = {
        "hypa_str": list(rank.keys()),
        "rank_val": [rank[hs][0] for hs in rank],
        "rank_noval": [rank[hs][1] for hs in rank],
    }
    rank_df = pd.DataFrame(rank_pd)
    ranked = rank_df.sort_values("rank_val", ascending=True).head(30)
    logg.debug(f"ranked:\n{ranked}")


def evaluate_attention_weights(train_words_type: str) -> None:
    """TODO: what is evaluate_attention_weights doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_attention_weights")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_attention_weights")

    # magic to fix the GPUs
    setup_gpus()

    # ATT_ct02_dr02_ks02_lu01_as01_qt01_dw01_opa1_lr01_bs01_en01_dsmel04_wk1
    dataset_name = "mel04"

    hypa: Dict[str, str] = {}

    hypa["conv_size_type"] = "02"
    hypa["dropout_type"] = "02"
    hypa["kernel_size_type"] = "02"
    hypa["lstm_units_type"] = "01"
    hypa["att_sample_type"] = "01"
    hypa["query_style_type"] = "01"
    hypa["dense_width_type"] = "01"
    hypa["optimizer_type"] = "a1"
    hypa["learning_rate_type"] = "01"
    hypa["batch_size_type"] = "01"
    hypa["epoch_num_type"] = "01"

    hypa["dataset_name"] = dataset_name
    hypa["words_type"] = train_words_type

    use_validation = True

    model_name = build_attention_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # load the model
    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"

    # model = tf.keras.models.load_model(model_path)
    # https://github.com/keras-team/keras/issues/5088#issuecomment-401498334
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "backend": tf.keras.backend,
        },
    )

    att_weight_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer("output").output,
            model.get_layer("att_softmax").output,
        ],
    )
    att_weight_model.summary()
    logg.debug(f"att_weight_model.outputs: {att_weight_model.outputs}")

    # get the words
    train_words = words_types[train_words_type]
    logg.debug(f"train_words: {train_words}")
    perm_pred = compute_permutation(train_words)

    # select a word
    correct_index = 0
    word = train_words[correct_index]

    # input data
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{dataset_name}"
    data, labels = load_processed(processed_path, [word])

    # get prediction and attention weights
    pred, att_weights = att_weight_model.predict(data["testing"])
    logg.debug(f"att_weights.shape: {att_weights.shape}")
    logg.debug(f"att_weights[0].shape: {att_weights[0].shape}")

    # plot the spectrogram and the weights
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    fig.suptitle(f"Attention weights and prediction for {word}")

    # which word in the dataset to plot
    word_id = 0

    # extract the spectrogram, data shape (?, x, y, 1)
    word_data = data["testing"][word_id][:, :, -1]
    logg.debug(f"word_data.shape: {word_data.shape}")
    title = f"Spectrogram for {word}"
    plot_spec(word_data, axes[0], title=title)

    # plot the weights
    word_att_weights = att_weights[word_id]
    title = f"Attention weights for {word}"
    plot_att_weights(word_att_weights, axes[1], title)

    # plot the predictions
    word_pred = pred[word_id]
    # permute the prediction from sorted to the order you have
    word_pred = word_pred[perm_pred]
    pred_index = np.argmax(word_pred)
    title = f"Predictions for {word}"
    plot_pred(word_pred, train_words, axes[2], title, pred_index)

    fig.tight_layout()

    plt.show()


def make_plots_attention() -> None:
    """TODO: what is make_plots_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.make_plots_attention")
    # logg.setLevel("INFO")
    logg.debug("Start make_plots_attention")

    results_df = build_att_results_df()

    # the output folders
    plot_fol = Path("plot_results") / "att"
    pdf_split_fol = plot_fol / "pdf_split"
    pdf_grid_fol = plot_fol / "pdf_grid"
    png_split_fol = plot_fol / "png_split"
    png_grid_fol = plot_fol / "png_grid"
    for f in pdf_split_fol, pdf_grid_fol, png_split_fol, png_grid_fol:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)

    for col in results_df:
        if col in ["model_name", "loss", "fscore", "recall", "precision", "cat_acc"]:
            continue
        logg.debug(f"hypa_grid['{col}'] = {results_df[col].unique()}")

    hypa_grid: Dict[str, List[str]] = {}
    hypa_grid["words"] = ["k1", "w2"]
    hypa_grid["dataset"] = ["mela1", "mel04", "mel05", "mel01"]
    hypa_grid["conv"] = ["01", "02"]
    hypa_grid["dropout"] = ["01", "02"]
    hypa_grid["kernel"] = ["01", "02"]
    hypa_grid["lstm"] = ["01"]
    hypa_grid["att"] = ["01", "02"]
    hypa_grid["query"] = ["01", "03", "02", "04"]
    hypa_grid["dense"] = ["01", "02"]
    hypa_grid["lr"] = ["01"]
    hypa_grid["optimizer"] = ["a1"]
    hypa_grid["batch"] = ["01", "02"]
    hypa_grid["epoch"] = ["01", "02"]

    hp_to_plot_names = [
        "words",
        "dataset",
        "conv",
        # "dropout",
        # "kernel",
        # "lstm",
        "att",
        "query",
        # "dense",
        # "lr",
        # "optimizer",
        "batch",
        "epoch",
    ]

    # all_hp_to_plot = list(combinations(hypa_grid.keys(), 4))
    all_hp_to_plot = list(combinations(hp_to_plot_names, 4))
    logg.debug(f"len(all_hp_to_plot): {len(all_hp_to_plot)}")

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


def run_evaluate_attention(args: argparse.Namespace) -> None:
    """TODO: What is evaluate_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_attention")
    logg.debug("Starting run_evaluate_attention")

    evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type
    # rec_words_type = args.rec_words_type

    pd.set_option("max_colwidth", 100)

    if evaluation_type == "results":
        evaluate_results_attention()
    elif evaluation_type == "attention_weights":
        evaluate_attention_weights(train_words_type)
    elif evaluation_type == "make_plots":
        make_plots_attention()


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_attention(args)
