from pathlib import Path
from tensorflow.keras import models as tf_models  # type: ignore
from utils import setup_gpus
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import re
import typing as ty

from plot_utils import plot_confusion_matrix
from preprocess_data import load_processed
from train_area import build_area_name
from utils import analyze_confusion
from utils import compute_permutation
from utils import pred_hot_2_cm
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
            "evaluate_model_area",
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

    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default="VAN_opa1_lr03_bs32_en15_dsaug07_wnum_noval",
        help="Which model to use",
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
    # info_folder = Path("info") / "image"

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

    word_res = [
        "all",
        "LTall",
        "LTBall",
        "allLS",
        "LTallLS",
        "LTBallLS",
        "BallLS",
        "Ball",
        "num",
        "LTnum",
        "LTBnum",
        "numLS",
        "LTnumLS",
        "LTBnumLS",
        "BnumLS",
        "Bnum",
    ]
    for word in word_res:
        df_f = results_df
        df_f = df_f.query(f"words == '{word}'")
        if len(df_f) == 0:
            continue
        df_f = df_f.sort_values("fscore", ascending=False)
        logg.info(f"\nOnly on {word}")
        logg.info(f"{df_f.head(10)}")

    arch_type = ["VAN", "AAN"]
    word_res = ["LTBall", "LTBallLS", "LTBnum", "LTBnumLS"]
    for word in word_res:
        for arch in arch_type:
            df_f = results_df
            df_f = df_f.query(f"words == '{word}'")
            df_f = df_f.query(f"model_name.str.startswith('{arch}')")
            if len(df_f) == 0:
                continue
            df_f = df_f.sort_values("fscore", ascending=False)
            logg.info(f"\nOnly on {word} {arch}")
            logg.info(f"{df_f.head(1)}")

    # df_f = results_df
    # df_f = df_f.query("words == 'LTnumLS'")
    # df_f = df_f.sort_values("fscore", ascending=False)
    # logg.info(f"{df_f.head(10)}")


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
    use_validation = True
    dataset_name = hypa["dataset_name"]
    batch_size = int(hypa["batch_size_type"])

    # get the model name
    model_name = build_area_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # load the model
    model_folder = Path("trained_models") / "area"
    model_path = model_folder / f"{model_name}.h5"
    model = tf_models.load_model(model_path)

    # get the output layer because you forgot to name it
    name_output_layer = model.layers[-1].name
    logg.debug(f"name_output_layer: {name_output_layer}")

    # build a model on top of that to get the weights
    att_weight_model = tf_models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(name_output_layer).output,
            model.get_layer("area_values").output,
        ],
    )
    att_weight_model.summary()

    # get the training words
    train_words = words_types[train_words_type]
    perm_pred = compute_permutation(train_words)
    logg.debug(f"perm_pred: {perm_pred}")
    sorted_train_words = sorted(train_words)
    logg.debug(f"sorted(train_words): {sorted(train_words)}")

    # load data if you do not want to record new audios
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{dataset_name}"
    logg.debug(f"processed_path: {processed_path}")

    # # evaluate on all data because im confused
    # data, labels = load_processed(processed_path, train_words)
    # logg.debug(f"data['testing'].shape: {data['testing'].shape}")
    # logg.debug(f"labels['testing'].shape: {labels['testing'].shape}")
    # eval_testing = model.evaluate(data["testing"], labels["testing"])
    # for metrics_name, value in zip(model.metrics_names, eval_testing):
    #     logg.debug(f"{metrics_name}: {value}")

    # which word in the dataset to plot
    # word_id = 5
    # word_id = 7
    word_id = 12

    # the loaded spectrograms
    rec_data_l: ty.List[np.ndarray] = []

    # for now we do not record new words
    rec_words = train_words[30:32]
    num_rec_words = len(rec_words)

    logg.debug(f"processed_path: {processed_path}")
    for i, word in enumerate(rec_words):
        logg.debug(f"\nword: {word}")

        data, labels = load_processed(processed_path, [word])
        logg.debug(f"data['testing'].shape: {data['testing'].shape}")
        logg.debug(f"labels['testing'].shape: {labels['testing'].shape}")

        # eval_testing = model.evaluate(data["testing"], labels["testing"])
        # for metrics_name, value in zip(model.metrics_names, eval_testing):
        #     logg.debug(f"{metrics_name}: {value}")

        # get one of the spectrograms
        word_data = data["testing"][word_id]
        rec_data_l.append(word_data)

        pred, att_weights = att_weight_model.predict(data["testing"])
        logg.debug(f"pred.shape: {pred.shape}")
        logg.debug(f"pred[0].shape: {pred[0].shape}")

        pred_am_all = np.argmax(pred, axis=1)
        logg.debug(f"pred_am_all: {pred_am_all}")

        pred_index = np.argmax(pred[0])
        pred_word = sorted_train_words[pred_index]
        logg.debug(f"sorted pred_word: {pred_word} pred_index {pred_index}")

        # test EVERY SINGLE spectrogram
        spec_num = data["testing"].shape[0]
        for wid in range(spec_num):

            # get the word
            word_data = data["testing"][wid]
            logg.debug(f"word_data.shape: {word_data.shape}")
            batch_word_data = np.expand_dims(word_data, axis=0)
            logg.debug(f"batch_word_data.shape: {batch_word_data.shape}")

            shape_batch = (batch_size, *word_data.shape)
            logg.debug(f"shape_batch: {shape_batch}")

            batch_word_data_big = np.zeros(shape_batch, dtype=np.float32)
            for i in range(batch_size):
                batch_word_data_big[i, :, :, :] = batch_word_data
            # batch_word_data_big[0, :, :, :] = batch_word_data

            # predict it
            # pred, att_weights = att_weight_model.predict(batch_word_data)
            pred, att_weights = att_weight_model.predict(batch_word_data_big)

            # show all prediction
            # pred_am = np.argmax(pred, axis=1)
            # logg.debug(f"pred_am: {pred_am}")

            # focus on first prediction
            word_pred = pred[0]
            pred_index = np.argmax(word_pred)
            pred_word = sorted_train_words[pred_index]

            recap = ""
            if pred_word == word:
                recap += "correct "
            else:
                recap += "  wrong "
                pred_am = np.argmax(pred, axis=1)
                logg.debug(f"pred_am: {pred_am}")

            recap += f"sorted pred_word: {pred_word} pred_index {pred_index}"
            recap += f" word_pred.shape {word_pred.shape}"
            recap += f" pred_am_all[wid] {pred_am_all[wid]}"

            # pred_f = ", ".join([f"{p:.3f}" for p in pred[0]])
            # recap += f" pred_f: {pred_f}"

            logg.debug(recap)

            # break

    # turn the list into np array
    rec_data = np.stack(rec_data_l)
    logg.debug(f"\nrec_data.shape: {rec_data.shape}")

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
        # logg.debug(f"word_pred: {word_pred}")
        # # permute the prediction from sorted to the order you have
        word_pred = word_pred[perm_pred]
        # logg.debug(f"word_pred permuted: {word_pred}")
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


def evaluate_model_area(model_name: str, test_words_type: str) -> None:
    r"""MAKEDOC: what is evaluate_model_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model_area")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_model_area")

    # magic to fix the GPUs
    setup_gpus()

    # # VAN_opa1_lr05_bs32_en15_dsaug07_wLTall
    # hypa = {
    #     "batch_size_type": "32",
    #     "dataset_name": "aug07",
    #     "epoch_num_type": "15",
    #     "learning_rate_type": "03",
    #     "net_type": "VAN",
    #     "optimizer_type": "a1",
    #     # "words_type": "LTall",
    #     "words_type": train_words_type,
    # }
    # # use_validation = True
    # use_validation = False
    # dataset_name = hypa["dataset_name"]

    # get the model name
    # model_name = build_area_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    dataset_re = re.compile("_ds(.*?)_")
    match = dataset_re.search(model_name)
    if match is not None:
        logg.debug(f"match[1]: {match[1]}")
        dataset_name = match[1]

    train_words_type_re = re.compile("_w(.*?)[_.]")
    match = train_words_type_re.search(model_name)
    if match is not None:
        logg.debug(f"match[1]: {match[1]}")
        train_words_type = match[1]

    # load the model
    model_folder = Path("trained_models") / "area"
    model_path = model_folder / f"{model_name}.h5"
    model = tf_models.load_model(model_path)
    # model.summary()

    train_words = words_types[train_words_type]
    logg.debug(f"train_words: {train_words}")
    test_words = words_types[test_words_type]
    logg.debug(f"test_words: {test_words}")

    # input data
    processed_path = Path("data_proc") / f"{dataset_name}"
    data, labels = load_processed(processed_path, test_words)
    logg.debug(f"list(data.keys()): {list(data.keys())}")
    logg.debug(f"data['testing'].shape: {data['testing'].shape}")

    # evaluate on the words you trained on
    logg.debug("Evaluate on test data:")
    model.evaluate(data["testing"], labels["testing"])
    # model.evaluate(data["validation"], labels["validation"])

    # predict labels/cm/fscore
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, test_words)
    # y_pred = model.predict(data["validation"])
    # cm = pred_hot_2_cm(labels["validation"], y_pred, test_words)
    fscore = analyze_confusion(cm, test_words)
    logg.debug(f"fscore: {fscore}")

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, test_words, fscore, train_words)

    fig_name = f"{model_name}_test{test_words_type}_cm.{{}}"
    cm_folder = Path("plot_results") / "cm"
    if not cm_folder.exists():
        cm_folder.mkdir(parents=True, exist_ok=True)

    plot_cm_path = cm_folder / fig_name.format("png")
    fig.savefig(plot_cm_path)
    plot_cm_path = cm_folder / fig_name.format("pdf")
    fig.savefig(plot_cm_path)

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
    model_name = args.model_name

    rec_words_type = args.rec_words_type
    if rec_words_type == "train":
        rec_words_type = train_words_type

    pd.set_option("max_colwidth", 100)
    pd.set_option("display.max_rows", None)

    if evaluation_type == "results":
        evaluate_results_area()
    elif evaluation_type == "attention_weights":
        evaluate_attention_weights(train_words_type)
    elif evaluation_type == "delete_bad_models":
        delete_bad_models_area()
    elif evaluation_type == "evaluate_model_area":
        evaluate_model_area(model_name, rec_words_type)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_area(args)
