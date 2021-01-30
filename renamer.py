from pathlib import Path
import argparse
import logging
import json

import typing as ty

from utils import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

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


def run_renamer(args: argparse.Namespace) -> None:
    """TODO: What is renamer doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_renamer")
    logg.debug("Starting run_renamer")

    rename_att_v1_to_v2()


if __name__ == "__main__":
    args = setup_env()
    run_renamer(args)
