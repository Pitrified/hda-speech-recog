from pathlib import Path
from tensorflow.keras import models  # type: ignore
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore
import typing as ty

from preprocess_data import preprocess_spec
from augment_data import do_augmentation
from plot_utils import plot_confusion_matrix
from preprocess_data import load_processed
from utils import analyze_confusion
from utils import pred_hot_2_cm
from utils import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-rt",
        "--rename_type",
        type=str,
        default="rename_att_v1_to_v2",
        # choices=aug_keys,
        help="Which rename/fix to perform",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env():
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 train.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def build_attention_name_v1(hypa: ty.Dict[str, str], use_validation: bool) -> str:
    """TODO: what is build_attention_name doing?"""
    model_name = "ATT"

    model_name += f"_ct{hypa['conv_size_type']}"
    model_name += f"_dr{hypa['dropout_type']}"
    model_name += f"_ks{hypa['kernel_size_type']}"
    model_name += f"_lu{hypa['lstm_units_type']}"
    model_name += f"_as{hypa['att_sample_type']}"
    model_name += f"_qt{hypa['query_style_type']}"
    model_name += f"_dw{hypa['dense_width_type']}"
    model_name += f"_op{hypa['optimizer_type']}"
    model_name += f"_lr{hypa['learning_rate_type']}"
    model_name += f"_bs{hypa['batch_size_type']}"
    model_name += f"_en{hypa['epoch_num_type']}"

    model_name += f"_ds{hypa['dataset_name']}"
    model_name += f"_w{hypa['words_type']}"

    if not use_validation:
        model_name += "_noval"
    return model_name


def build_attention_name_v2(hypa: ty.Dict[str, str], use_validation: bool) -> str:
    """TODO: what is build_attention_name doing?"""
    model_name = "ATT"

    model_name += f"_ct{hypa['conv_size_type']}"
    model_name += f"_dr{hypa['dropout_type']}"
    model_name += f"_ks{hypa['kernel_size_type']}"
    model_name += f"_lu{hypa['lstm_units_type']}"
    model_name += f"_qt{hypa['query_style_type']}"
    model_name += f"_dw{hypa['dense_width_type']}"
    model_name += f"_op{hypa['optimizer_type']}"
    model_name += f"_lr{hypa['learning_rate_type']}"
    model_name += f"_bs{hypa['batch_size_type']}"
    model_name += f"_en{hypa['epoch_num_type']}"

    model_name += f"_ds{hypa['dataset_name']}"
    model_name += f"_w{hypa['words_type']}"

    if not use_validation:
        model_name += "_noval"
    return model_name


def extract_att_hypa_v1(model_name: str) -> ty.Tuple[ty.Dict[str, str], bool, bool]:
    """Extracts the hypas from a name made by build_attention_name_v1"""
    # logg = logging.getLogger(f"c.{__name__}.extract_att_hypa_v1")
    # logg.setLevel("INFO")
    # logg.debug("Start extract_att_hypa_v1")

    hypa: ty.Dict[str, str] = {}

    hypa["conv_size_type"] = model_name[6:8]
    hypa["dropout_type"] = model_name[11:13]
    hypa["kernel_size_type"] = model_name[16:18]
    hypa["lstm_units_type"] = model_name[21:23]
    hypa["att_sample_type"] = model_name[26:28]
    hypa["query_style_type"] = model_name[31:33]
    hypa["dense_width_type"] = model_name[36:38]
    hypa["optimizer_type"] = model_name[41:43]
    hypa["learning_rate_type"] = model_name[46:48]
    hypa["batch_size_type"] = model_name[51:53]
    hypa["epoch_num_type"] = model_name[56:58]
    hypa["dataset_name"] = model_name[61:66]
    hypa["words_type"] = model_name[68:70]

    use_validation = not model_name.endswith("_noval")

    # check that the hypas are correct
    re_name = build_attention_name_v1(hypa, use_validation)

    if re_name == model_name:
        is_correct = True
    else:
        is_correct = False

    return hypa, use_validation, is_correct


def rename_att_v1_to_v2() -> None:
    """Changes the hypas in the name

    * folder_name
    * recap.json
    * results_recap.json
    * model_name.h5
    """
    logg = logging.getLogger(f"c.{__name__}.rename_att_v1_to_v2")
    # logg.setLevel("INFO")
    logg.debug("Start rename_att_v1_to_v2")

    trained_folder = Path("trained_models")
    info_folder = Path("info")
    skipped = []

    # done = 0 done += 1 if done > 2: return
    for model_folder_old in info_folder.iterdir():

        # get the model name
        old_name = model_folder_old.name
        if not old_name.startswith("ATT"):
            continue
        # logg.debug(f"\nmodel_name: {old_name}")
        logg.debug("\n")

        recap_path = model_folder_old / "recap.json"
        res_recap_path = model_folder_old / "results_recap.json"
        if not res_recap_path.exists() or not recap_path.exists():
            logg.info(f"Skipping {old_name}, not found a recap")
            skipped.append(old_name)
            continue

        hypa_old, use_validation, is_correct = extract_att_hypa_v1(old_name)
        # logg.debug(f"hypa_old {hypa_old} use_val {use_validation} is_cor {is_correct}")

        if not is_correct:
            logg.debug(f"Failed to parse the name {old_name}")
            continue

        hypa_new: ty.Dict[str, str] = {}
        hypa_new["conv_size_type"] = hypa_old["conv_size_type"]
        hypa_new["dropout_type"] = hypa_old["dropout_type"]
        hypa_new["kernel_size_type"] = hypa_old["kernel_size_type"]
        hypa_new["lstm_units_type"] = hypa_old["lstm_units_type"]

        # no more att_sample_type
        ast_old = hypa_old["att_sample_type"]

        # got to fiddle with query_style_type
        qst_old = hypa_old["query_style_type"]
        if qst_old == "01":
            if ast_old == "01":
                qst_new = "01"
            elif ast_old == "02":
                logg.debug(f"ast_old: {ast_old}")
                qst_new = "05"
        else:
            qst_new = qst_old
        hypa_new["query_style_type"] = qst_new

        hypa_new["dense_width_type"] = hypa_old["dense_width_type"]
        hypa_new["optimizer_type"] = hypa_old["optimizer_type"]
        hypa_new["learning_rate_type"] = hypa_old["learning_rate_type"]
        hypa_new["batch_size_type"] = hypa_old["batch_size_type"]
        hypa_new["epoch_num_type"] = hypa_old["epoch_num_type"]
        hypa_new["dataset_name"] = hypa_old["dataset_name"]
        hypa_new["words_type"] = hypa_old["words_type"]

        new_name = build_attention_name_v2(hypa_new, use_validation)
        logg.debug(f"old_name: {old_name}")
        logg.debug(f"new_name: {new_name}")

        query_style_types = {
            "01": "dense01",
            "02": "conv01",
            "03": "conv02",
            "04": "conv03",
            "05": "dense02",
        }
        query_style = query_style_types[hypa_new["query_style_type"]]

        # fix the recaps
        recap = json.loads(recap_path.read_text())
        recap["model_name"] = new_name
        recap["hypa"] = hypa_new
        recap["model_param"]["query_style"] = query_style
        del recap["model_param"]["att_sample"]
        recap_path.write_text(json.dumps(recap, indent=4))

        res_recap = json.loads(res_recap_path.read_text())
        res_recap["model_name"] = new_name
        res_recap_path.write_text(json.dumps(res_recap, indent=4))

        # rename the trained model
        trained_model_old = trained_folder / f"{old_name}.h5"
        trained_model_new = trained_folder / f"{new_name}.h5"
        trained_model_old.rename(trained_model_new)

        # rename the info folder
        model_folder_new = info_folder / f"{new_name}"
        model_folder_old.rename(model_folder_new)

    for sk in skipped:
        logg.debug(f"{sk}")


def recompute_fscore_cnn() -> None:
    """TODO: what is recompute_fscore_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.recompute_fscore_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start recompute_fscore_cnn")

    info_folder = Path("info")
    trained_folder = Path("trained_models")

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")

        # check that it is a CNN
        model_name = model_folder.name
        if not model_name.startswith("CNN"):
            continue

        # check that the model is trained and not a placeholder
        model_path = trained_folder / f"{model_name}.h5"
        found_model = False
        if model_path.exists():
            if model_path.stat().st_size > 100:
                found_model = True
        if not found_model:
            continue

        # load it
        model = models.load_model(model_path)

        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue
        results_recap = json.loads(res_recap_path.read_text())
        # logg.debug(f"results_recap['cm']: {results_recap['cm']}")

        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())
        # logg.debug(f"recap['words']: {recap['words']}")

        words = recap["words"]
        hypa = recap["hypa"]

        # check that the data is available
        dn = hypa["dataset"]
        wt = hypa["words"]
        if dn.startswith("mel") or dn.startswith("mfcc"):
            preprocess_spec(dn, wt)
        elif dn.startswith("aug"):
            do_augmentation(dn, wt)

        processed_path = Path("data_proc") / f"{hypa['dataset']}"
        data, labels = load_processed(processed_path, words)

        y_pred = model.predict(data["testing"])
        cm = pred_hot_2_cm(labels["testing"], y_pred, words)
        fscore = analyze_confusion(cm, words)
        # logg.debug(f"fscore: {fscore}")

        # overwrite the cm
        results_recap["cm"] = cm.tolist()
        # add the fscore
        results_recap["fscore"] = fscore
        # increase the version
        results_recap["results_recap_version"] = "002"
        # write the new results
        res_recap_path.write_text(json.dumps(results_recap, indent=4))

        # increase the recap version (shows that it is after this debacle)
        recap["version"] = "002"
        recap_path.write_text(json.dumps(recap, indent=4))

        # save the new plots
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_confusion_matrix(cm, ax, model_name, words, fscore)
        plot_cm_path = info_folder / "test_confusion_matrix.png"
        fig.savefig(plot_cm_path)
        plt.close(fig)


def create_placeholders() -> None:
    """TODO: what is create_placeholders doing?"""
    logg = logging.getLogger(f"c.{__name__}.create_placeholders")
    # logg.setLevel("INFO")
    logg.debug("Start create_placeholders")

    train_type_tag = "cnn"
    trained_folder = Path("trained_models") / train_type_tag

    for model_path in trained_folder.iterdir():
        model_name = model_path.stem
        placeholder_path = trained_folder / f"{model_name}.txt"
        if not placeholder_path.exists():
            logg.debug(f"Creating placeholder_path: {placeholder_path}")
            placeholder_path.write_text("Trained.")


def run_renamer(args: argparse.Namespace) -> None:
    """TODO: What is renamer doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_renamer")
    logg.debug("Starting run_renamer")

    rename_type = args.rename_type

    if rename_type == "rename_att_v1_to_v2":
        rename_att_v1_to_v2()
    elif rename_type == "recompute_fscore_cnn":
        recompute_fscore_cnn()
    elif rename_type == "create_placeholders":
        create_placeholders()


if __name__ == "__main__":
    args = setup_env()
    run_renamer(args)
