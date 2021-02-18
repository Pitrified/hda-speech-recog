from pathlib import Path
from tensorflow.keras import models as tf_models  # type: ignore
from utils import setup_gpus
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import typing as ty

from preprocess_data import load_processed
from train_area import build_area_name
from utils import compute_permutation
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
            # "evaluate_batch_epoch",
            "attention_weights",
            # "make_plots_hypa",
            # "make_plots_clr",
            # "audio",
            "delete_bad_models",
        ],
        help="Which evaluation to perform",
    )

    tra_types = [w for w in words_types.keys() if not w.startswith("_")]
    parser.add_argument(
        "-tw",
        "--train_words_type",
        type=str,
        default="f1",
        choices=tra_types,
        help="Words the dataset was trained on",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 evaluate_area.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def build_area_results_df() -> pd.DataFrame:
    """MAKEDOC: what is build_area_results_df doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_area_results_df")
    # logg.setLevel("INFO")
    logg.debug("Start build_area_results_df")

    pandito: ty.Dict[str, ty.List[str]] = {
        "words": [],
        "dataset": [],
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

    info_folder = Path("info") / "area"

    for model_folder in info_folder.iterdir():
        # model_name = model_folder.name
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

        hypa: ty.Dict[str, ty.Any] = recap["hypa"]

        pandito["words"].append(hypa["words_type"])
        pandito["dataset"].append(hypa["dataset_name"])

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


def evaluate_results_area() -> None:
    """MAKEDOC: what is evaluate_results_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_area")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_results_area")

    results_df = build_area_results_df()

    # all the unique values of the hypa ever used
    for col in results_df:
        if col in ["model_name", "loss", "fscore", "recall", "precision", "cat_acc"]:
            continue
        logg.info(f"hypa_grid['{col}'] = {results_df[col].unique()}")
        logg.info(f"frequencies for '{col}'\n{results_df[col].value_counts()}")

    # all the results so far
    df_f = results_df
    df_f = df_f.sort_values("fscore", ascending=False)
    logg.info("All results:")
    logg.info(f"{df_f.head(30)}")
    logg.info(f"{df_f.tail(10)}")

    # by word
    for words_type in ["dir", "k1", "w2", "f2", "f1", "all", "num"]:
        word_list = [w for w in results_df.words.unique() if words_type in w]

        df_f = results_df
        # df_f = df_f.query("use_val == True")
        # df_f = df_f.query(f"words == '{words_type}'")
        df_f = df_f[df_f["words"].isin(word_list)]
        if len(df_f) == 0:
            continue

        df_f = df_f.sort_values("fscore", ascending=False)
        logg.info(f"\nOnly on {words_type}")
        logg.info(f"{df_f.head(30)}\n{df_f.tail()}")

    df_f = results_df
    df_f = df_f.query("words == 'LTnumLS'")
    df_f = df_f.sort_values("fscore", ascending=False)
    logg.info(f"{df_f.head(10)}")


def load_trained_model_area(
    hypa: ty.Dict[str, ty.List[str]], use_validation: bool
) -> None:
    """MAKEDOC: what is load_trained_model_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model_area")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model_area")


def evaluate_attention_weights(train_words_type: str) -> None:
    """MAKEDOC: what is evaluate_attention_weights doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_attention_weights")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_attention_weights")

    # magic to fix the GPUs
    setup_gpus()

    # dataset_name = "mela1"
    # dataset_name = "mel04"
    dataset_name = "aug14"

    # hypa = {
    #     "net_type": "AAN",
    #     "batch_size_type": "32",
    #     # "dataset_name": "mela1",
    #     "dataset_name": dataset_name,
    #     "epoch_num_type": "15",
    #     "learning_rate_type": "03",
    #     "optimizer_type": "a1",
    #     # "words_type": "LTnum",
    #     "words_type": train_words_type,
    # }

    # hypa = {
    #     "batch_size_type": "32",
    #     "dataset_name": "auA05",
    #     "epoch_num_type": "15",
    #     "learning_rate_type": "03",
    #     "net_type": "AAN",
    #     "optimizer_type": "a1",
    #     "words_type": "LTnumLS",
    # }

    # AAN_opa1_lr03_bs32_en15_dsaug14_wLTnum
    # hypa = {
    #     "batch_size_type": "32",
    #     "dataset_name": "aug14",
    #     "epoch_num_type": "15",
    #     "learning_rate_type": "03",
    #     "net_type": "AAN",
    #     "optimizer_type": "a1",
    #     "words_type": "LTnum",
    # }

    # VAN_opa1_lr04_bs32_en15_dsaug14_wLTall
    # hypa = {
    #     "batch_size_type": "32",
    #     "dataset_name": "aug14",
    #     "epoch_num_type": "15",
    #     "learning_rate_type": "04",
    #     "net_type": "VAN",
    #     "optimizer_type": "a1",
    #     "words_type": "LTall",
    # }

    # VAN_opa1_lr05_bs32_en15_dsaug07_wLTall
    hypa = {
        "batch_size_type": "32",
        "dataset_name": "aug07",
        "epoch_num_type": "15",
        "learning_rate_type": "05",
        "net_type": "VAN",
        "optimizer_type": "a1",
        "words_type": "LTall",
    }

    dataset_name = hypa["dataset_name"]

    use_validation = True

    model_name = build_area_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # load the model
    model_folder = Path("trained_models") / "area"
    model_path = model_folder / f"{model_name}.h5"

    model = tf_models.load_model(model_path)

    # for layer in model.layers:
    #     logg.debug(layer.name)
    name_output_layer = model.layers[-1].name
    logg.debug(f"name_output_layer: {name_output_layer}")

    att_weight_model = tf_models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(name_output_layer).output,
            # model.outputs,
            model.get_layer("area_values").output,
        ],
    )
    att_weight_model.summary()
    # logg.debug(f"att_weight_model.outputs: {att_weight_model.outputs}")

    # get the training words
    train_words = words_types[train_words_type]
    perm_pred = compute_permutation(train_words)
    logg.debug(f"perm_pred: {perm_pred}")

    # load data if you do not want to record new audios
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{dataset_name}"

    # evaluate on all data because im confused
    data, labels = load_processed(processed_path, train_words)
    logg.debug(f"data['testing'].shape: {data['testing'].shape}")
    logg.debug(f"labels['testing'].shape: {labels['testing'].shape}")
    eval_testing = model.evaluate(data["testing"], labels["testing"])
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")

    # which word in the dataset to plot
    # word_id = 5
    # word_id = 7
    word_id = 12

    # the loaded spectrograms
    rec_data_l: ty.List[np.ndarray] = []

    # for now we do not record new words
    # rec_words = train_words[-3:]
    # rec_words = train_words[:3]
    # rec_words = train_words[:2]
    # rec_words = train_words[10:21]
    # rec_words = train_words[20:31]
    rec_words = train_words[30:32]
    num_rec_words = len(rec_words)

    logg.debug(f"processed_path: {processed_path}")
    for i, word in enumerate(rec_words):
        logg.debug(f"word: {word}")

        data, labels = load_processed(processed_path, [word])
        logg.debug(f"data['testing'].shape: {data['testing'].shape}")
        logg.debug(f"labels['testing'].shape: {labels['testing'].shape}")

        # eval_testing = model.evaluate(data["testing"], labels["testing"])
        # for metrics_name, value in zip(model.metrics_names, eval_testing):
        #     logg.debug(f"{metrics_name}: {value}")

        # get one of the spectrograms
        word_data = data["testing"][word_id]
        rec_data_l.append(word_data)

    # turn the list into np array
    rec_data = np.stack(rec_data_l)
    logg.debug(f"rec_data.shape: {rec_data.shape}")

    # get prediction and attention weights
    pred, att_weights = att_weight_model.predict(rec_data)
    logg.debug(f"att_weights.shape: {att_weights.shape}")
    logg.debug(f"att_weights[0].shape: {att_weights[0].shape}")

    plot_size = 5
    fw = plot_size * num_rec_words
    nrows = 2
    fh = plot_size * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=num_rec_words, figsize=(fw, fh))
    fig.suptitle("Attention weights computed with VerticalAreaNet", fontsize=20)

    sorted_train_words = sorted(train_words)
    logg.debug(f"sorted(train_words): {sorted(train_words)}")

    for i, word in enumerate(rec_words):
        logg.debug(f"recword: {word}")

        # show the spectrogram
        word_spec = rec_data[i][:, :, 0]
        # logg.debug(f"word_spec: {word_spec}")
        axes[0][i].set_title(f"Spectrogram for {word}", fontsize=20)
        axes[0][i].imshow(word_spec, origin="lower")

        axes[1][i].set_title(f"Attention weights for {word}", fontsize=20)
        att_w = att_weights[i][:, :, 0]
        axes[1][i].imshow(att_w, origin="lower")
        logg.debug(f"att_w.max(): {att_w.max()}")

        # axes[0][i].imshow(
        #     att_w, origin="lower", extent=img.get_extent(), cmap="gray", alpha=0.4
        # )

        # weighted = word_spec * att_w
        # axes[2][i].imshow(weighted, origin="lower")

        word_pred = pred[i]
        pred_index = np.argmax(word_pred)
        pred_word = sorted_train_words[pred_index]
        logg.debug(f"sorted pred_word: {pred_word} pred_index {pred_index}")

        # # plot the predictions
        word_pred = pred[i]
        logg.debug(f"word_pred: {word_pred}")
        # # permute the prediction from sorted to the order you have
        word_pred = word_pred[perm_pred]
        logg.debug(f"word_pred permuted: {word_pred}")
        pred_index = np.argmax(word_pred)
        pred_word = train_words[pred_index]
        logg.debug(f"pred_word: {pred_word} pred_index {pred_index}")
        # title = f"Predictions for {word}"
        # plot_pred(word_pred, train_words, axes[2][i], title, pred_index)

    fig.tight_layout()

    fig_name = f"{model_name}"
    fig_name += "_0002.{}"
    plot_folder = Path("plot_results")
    results_path = plot_folder / fig_name.format("pdf")
    fig.savefig(results_path)

    plt.show()


def delete_bad_models_area() -> None:
    """MAKEDOC: what is delete_bad_models_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models_area")
    # logg.setLevel("INFO")
    logg.debug("Start delete_bad_models_area")

    train_type_tag = "area"
    info_folder = Path("info") / train_type_tag
    trained_folder = Path("trained_models") / train_type_tag
    deleted = 0
    recreated = 0
    bad_models = 0
    good_models = 0

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")

        model_name = model_folder.name
        model_path = trained_folder / f"{model_name}.h5"
        placeholder_path = trained_folder / f"{model_name}.txt"

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            logg.warn(f"Skipping {res_recap_path}, not found")
            continue
        results_recap = json.loads(res_recap_path.read_text())

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())

        # load info
        words_type = recap["hypa"]["words_type"]
        fscore = results_recap["fscore"]

        if "all" in words_type:
            f_tresh = 0.96
        elif "f1" in words_type:
            f_tresh = 0.985
        elif "yn" in words_type:
            f_tresh = 0.998
        elif "num" in words_type:
            f_tresh = 0.985

        if fscore < f_tresh:
            bad_models += 1

            if model_path.exists():
                # manually uncomment this when ready
                model_path.unlink()
                deleted += 1
                logg.debug(f"Deleting model_path: {model_path}")
                logg.debug(f"\tfscore: {fscore}")

            # check that a placeholder is there, you have info for this model
            else:
                if not placeholder_path.exists():
                    placeholder_path.write_text("Deleted")
                    logg.debug(f"Recreating placeholder_path: {placeholder_path}")
                    recreated += 1

        else:
            # logg.debug(f"Good model_path {model_path} {words_type}")
            # logg.debug(f"\tfscore: {fscore}")
            good_models += 1

    logg.info(f"bad_models: {bad_models}")
    logg.info(f"good_models: {good_models}")
    logg.info(f"deleted: {deleted}")
    logg.info(f"recreated: {recreated}")


def run_evaluate_area(args: argparse.Namespace) -> None:
    """MAKEDOC: What is evaluate_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_area")
    logg.debug("Starting run_evaluate_area")

    evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type

    pd.set_option("max_colwidth", 100)
    pd.set_option("display.max_rows", None)

    if evaluation_type == "results":
        evaluate_results_area()
    elif evaluation_type == "attention_weights":
        evaluate_attention_weights(train_words_type)
    elif evaluation_type == "delete_bad_models":
        delete_bad_models_area()


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_area(args)
