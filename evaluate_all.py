from pathlib import Path
from tensorflow.keras import models as tf_models  # type: ignore
import argparse
import json
import logging
import math
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import re
import typing as ty

from augment_data import do_augmentation
from evaluate_area import build_area_results_df
from evaluate_attention import build_att_results_df
from evaluate_cnn import build_cnn_results_df
from evaluate_transfer import build_tra_results_df
from plot_utils import plot_confusion_matrix
from preprocess_data import load_processed
from preprocess_data import preprocess_spec
from utils import analyze_confusion
from utils import pred_hot_2_cm
from utils import setup_gpus
from utils import which_arch
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
            "evaluate_augmentation",
            "build_megacomparison_v01",
            "build_megacomparison_v02",
            "build_megacomparison_v03",
            "evaluate_loud_section",
            "evaluate_model_cm",
            "evaluate_model_cm_all",
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
    recap = "python3 evaluate_all.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def evaluate_augmentation() -> None:
    """MAKEDOC: what is evaluate_augmentation doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_augmentation")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_augmentation")

    res_att_df = build_att_results_df()
    logg.debug(f"res_att_df['dataset'].unique(): {res_att_df['dataset'].unique()}")
    res_cnn_df = build_cnn_results_df()
    logg.debug(f"res_cnn_df['dataset'].unique(): {res_cnn_df['dataset'].unique()}")

    results_df = res_cnn_df

    # word_list = ["k1", "LTnum", "num", "LTnumLS"]
    # word_list = ["all", "LTall"]
    word_list = ["f1"]
    # word_list = ["dir"]
    results_df = results_df[results_df["words"].isin(word_list)]

    # results_df = results_df.query("fscore > 0.5")
    results_df = results_df.query("cat_acc > 0.5")

    mels = ["mel01"], ["mel04"], ["mel05"], ["mela1"]
    # mels = ["mel01", "mel04", "mel05", "mela1"]
    aug2345 = ["aug02", "aug03", "aug04", "aug05"]
    aug6789 = ["aug06", "aug07", "aug08", "aug09"]
    aug0123 = ["aug10", "aug11", "aug12", "aug13"]
    aug4567 = ["aug14", "aug15", "aug16", "aug17"]
    # meLs = ["meL04", "meLa1", "meLa2", "meLa3", "meLa4"]

    all_aug: ty.List[str] = []
    all_aug.extend((*aug2345, *aug6789, *aug0123, *aug4567))
    logg.debug(f"all_aug: {all_aug}")
    # results_df = results_df[results_df["dataset"].isin(all_aug)]

    aug_res: ty.Dict[str, ty.Dict[str, float]] = {}

    for aug_type in (*mels, aug2345[:-1], aug6789[:-1], aug0123[:-1], aug4567[:-1]):
        logg.debug(f"\naug_type: {aug_type}")

        df_f = results_df
        logg.debug(f"res len(df_f): {len(df_f)}")

        df_f = df_f[df_f["dataset"].isin(aug_type)]
        logg.debug(f"len(df_f): {len(df_f)}")

        # for dn in aug_type:
        #     df_q = df_f.query(f"dataset == '{dn}'")
        #     logg.debug(f"len(df_q): {len(df_q)}")
        # fscore_mean = df_f.fscore.mean()
        fscore_mean = df_f.cat_acc.mean()
        logg.debug(f"fscore_mean: {fscore_mean}")
        # fscore_stddev = df_f.fscore.std()
        fscore_stddev = df_f.cat_acc.std()
        logg.debug(f"fscore_stddev: {fscore_stddev}")

        aug_key = "_".join(aug_type)
        aug_res[aug_key] = {}
        aug_res[aug_key]["mean"] = fscore_mean
        aug_res[aug_key]["stddev"] = fscore_stddev

    logg.debug(f"aug_res: {json.dumps(aug_res, indent=4)}")
    for k in aug_res:
        recap = f"{k} & mel &"
        recap += f" ${aug_res[k]['mean']:.3f} \\pm"
        recap += f" {aug_res[k]['stddev']:.3f}$ \\\\"
        logg.debug(recap)

    dir_res: ty.Dict[str, ty.Dict[str, float]] = {}
    for dir_type in zip(aug2345, aug6789, aug0123, aug4567):
        logg.debug(f"\ndir_type: {dir_type}")

        df_f = results_df
        logg.debug(f"res len(df_f): {len(df_f)}")

        df_f = df_f[df_f["dataset"].isin(dir_type)]
        logg.debug(f"len(df_f): {len(df_f)}")

        logg.debug(f"df_f.fscore.mean(): {df_f.fscore.mean()}")

        # fscore_mean = df_f.fscore.mean()
        fscore_mean = df_f.cat_acc.mean()
        logg.debug(f"fscore_mean: {fscore_mean}")
        # fscore_stddev = df_f.fscore.std()
        fscore_stddev = df_f.cat_acc.std()
        logg.debug(f"fscore_stddev: {fscore_stddev}")

        dir_key = "_".join(dir_type)
        dir_res[dir_key] = {}
        dir_res[dir_key]["mean"] = fscore_mean
        dir_res[dir_key]["stddev"] = fscore_stddev

        for dn in dir_type:
            df_q = df_f.query(f"dataset == '{dn}'")
            logg.debug(f"{dn} len(df_q): {len(df_q)}")
            fscore_mean = df_q.cat_acc.mean()
            logg.debug(f"fscore_mean: {fscore_mean}")

    logg.debug(f"dir_res: {json.dumps(dir_res, indent=4)}")
    for k in dir_res:
        recap = f"{k} & mel &"
        recap += f" ${dir_res[k]['mean']:.3f} \\pm"
        recap += f" {dir_res[k]['stddev']:.3f}$ \\\\"
        logg.debug(recap)


def evaluate_loud_section() -> None:
    """MAKEDOC: what is evaluate_loud_section doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_loud_section")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_loud_section")

    res_att_df = build_att_results_df()
    logg.debug(f"res_att_df['words'].unique(): {res_att_df['words'].unique()}")
    res_cnn_df = build_cnn_results_df()
    logg.debug(f"res_cnn_df['words'].unique(): {res_cnn_df['words'].unique()}")
    res_area_df = build_area_results_df()
    logg.debug(f"res_area_df['words'].unique(): {res_area_df['words'].unique()}")

    res_df = {}
    res_df["att"] = res_att_df
    res_df["cnn"] = res_cnn_df
    res_df["area"] = res_area_df

    word_list = ["LTnum", "LTnumLS"]
    # word_list += ["LTall", "LTallLS"]

    # for results_df in res_att_df, res_cnn_df, res_area_df:
    logg.debug(f" & {' & '.join(word_list)} \\\\")
    for arch_type in res_df:
        results_df = res_df[arch_type]
        results_df = results_df[results_df["words"].isin(word_list)]
        results_df = results_df.query("fscore > 0.5")
        # logg.debug(f"res len(results_df): {len(results_df)}")

        recap = f"{arch_type} &"
        for w in word_list:
            # logg.debug(f"w: {w}")
            df_f = results_df
            df_f = df_f.query(f"words == '{w}'")
            fscore_mean = df_f.fscore.mean()
            fscore_stddev = df_f.fscore.std()
            recap += f" ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"
        recap += " \\\\"
        logg.debug(f"{recap}")

    # res_att_df = res_att_df[res_att_df["words"].isin(comb_word_list)]
    # res_att_df = res_att_df.query("fscore > 0.5")
    # logg.debug(f"res len(res_att_df): {len(res_att_df)}")

    # res_area_df = res_area_df[res_area_df["words"].isin(comb_word_list)]
    # res_area_df = res_area_df.query("fscore > 0.5")
    # logg.debug(f"res len(res_area_df): {len(res_area_df)}")

    # for w in comb_word_list:
    #     df_f = res_att_df
    #     # logg.debug(f"res_att_df len(df_f): {len(df_f)}")
    #     df_f = df_f.query(f"words == '{w}'")
    #     fscore_mean = df_f.fscore.mean()
    #     fscore_stddev = df_f.fscore.std()
    #     # logg.debug(f"{w} att fscore_mean: {fscore_mean:.3f}")
    #     recap = f"{w} &"
    #     recap += f" ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"

    #     df_f = res_area_df
    #     # logg.debug(f"res_area_df len(df_f): {len(df_f)}")
    #     df_f = df_f.query(f"words == '{w}'")
    #     fscore_mean = df_f.fscore.mean()
    #     fscore_stddev = df_f.fscore.std()
    #     # logg.debug(f"{w} area fscore_mean: {fscore_mean:.3f}")
    #     recap += f" & ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$ \\\\"

    #     logg.debug(recap)


def build_table(
    table_str,
    caption="Mega comparison",
    autogen_cmd="python magic",
    num_data_col=1,
    label_str="mega_comparison",
) -> str:
    """MAKEDOC: what is build_table doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_table")
    # logg.setLevel("INFO")
    logg.debug("Start build_table")

    template_table = ""
    template_table += "% Autogenerated by"
    # template_table += " python evaluate_all.py -et build_megacomparison_v01"
    template_table += " {autogen_cmd}"
    template_table += "\n"
    template_table += r"\begin{{table*}}[t!]"
    template_table += "\n"
    template_table += r"    \centering"
    template_table += "\n"
    # template_table += r"    \caption{{Mega comparison}}"
    template_table += r"    \caption{{{caption}}}"
    template_table += "\n"
    # template_table += r"    \label{{tab:mega_comparison}}"
    template_table += r"    \label{{tab:{label_str}}}"
    template_table += "\n"
    # template_table += r"    \begin{{tabular}}{{|c|ccc|}}"
    template_table += r"    \begin{{tabular}}{{|c|{cc_str}|}}"
    template_table += "\n"
    template_table += "{table_str}"
    template_table += r"    \end{{tabular}}"
    template_table += "\n"
    template_table += r"\end{{table*}}"

    cc_str = "c" * num_data_col
    the_table = template_table.format(
        autogen_cmd=autogen_cmd,
        caption=caption,
        label_str=label_str,
        table_str=table_str,
        cc_str=cc_str,
    )

    return the_table


def build_megacomparison_v01() -> None:
    """MAKEDOC: what is build_megacomparison_v01 doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_megacomparison_v01")
    # logg.setLevel("INFO")
    logg.debug("Start build_megacomparison_v01")

    all_res_df = {}
    all_res_df["area"] = build_area_results_df()
    all_res_df["att"] = build_att_results_df()
    all_res_df["cnn"] = build_cnn_results_df()
    all_res_df["tra"] = build_tra_results_df()

    # remove failed trainings
    for arch in all_res_df:
        all_res_df[arch] = all_res_df[arch].query("fscore > 0.5")
        logg.debug(f"len(all_res_df[{arch}]): {len(all_res_df[arch])}")

    word_lists = [["f1"], ["k1"], ["yn", "LTyn"], ["num", "LTnum"], ["all", "LTall"]]

    type_per_arch: ty.Dict[str, ty.List[str]] = {}
    type_per_arch["cnn"] = ["CNN"]
    type_per_arch["tra"] = ["TRA", "TD1", "TB0", "TB4", "TB7"]
    type_per_arch["att"] = ["ATT"]
    type_per_arch["area"] = ["AAN", "ARN", "SIM", "SI2", "VAN"]

    # mean
    # stddev
    # max
    # min
    # how many trained
    # best 5, everything again

    latex = ""
    indent = "    "

    header = indent * 2
    header += "Arch & Mean $\\pm$ StdDev & Min & Max"
    header += " \\\\\n"

    hline = indent * 2
    hline += "\\hline\n"

    for wl in word_lists:
        wl_str = ", ".join(wl)
        # task_str = f"Task: {wl_str}"
        task_str = "Task"
        task_str += "s" if len(wl) > 1 else ""
        task_str += f": {wl_str}"

        latex += hline

        latex += indent * 2
        latex += f"\\multicolumn{{4}}{{|c|}}{{{task_str}}}"
        latex += " \\\\\n"

        latex += hline
        latex += header

        latex += hline

        best_arch_name = "notfoundyet"
        best_fscore_max = 0

        all_fscores: ty.Dict[str, ty.Dict[str, float]] = {}

        for arch in type_per_arch:
            for arch_name in type_per_arch[arch]:
                df_f = all_res_df[arch]
                df_f = df_f[df_f["words"].isin(wl)]
                df_f = df_f.query(f"model_name.str.startswith('{arch_name}')")

                recap = f"{wl} {arch} {arch_name}"
                recap += f" len(df_f): {len(df_f)}"
                logg.debug(recap)

                fscore_mean = df_f.fscore.mean()
                fscore_stddev = df_f.fscore.std()
                fscore_min = df_f.fscore.min()
                fscore_max = df_f.fscore.max()

                if not math.isnan(fscore_mean) and not math.isnan(fscore_stddev):
                    all_fscores[arch_name] = {}
                    all_fscores[arch_name]["mean"] = fscore_mean
                    all_fscores[arch_name]["stddev"] = fscore_stddev
                    all_fscores[arch_name]["min"] = fscore_min
                    all_fscores[arch_name]["max"] = fscore_max

                    if fscore_max > best_fscore_max:
                        best_arch_name = arch_name
                        best_fscore_max = fscore_max
                    # latex += indent * 2
                    # latex += f"{arch_name}"
                    # latex += f" & ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"
                    # latex += f" & ${fscore_min:.3f}$"
                    # latex += f" & ${fscore_max:.3f}$"
                    # latex += " \\\\\n"

        for arch_name in all_fscores:
            fscore_mean = all_fscores[arch_name]["mean"]
            fscore_stddev = all_fscores[arch_name]["stddev"]
            fscore_min = all_fscores[arch_name]["min"]
            fscore_max = all_fscores[arch_name]["max"]

            latex += indent * 2
            latex += f"{arch_name}"
            latex += f" & ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"
            latex += f" & ${fscore_min:.3f}$"

            latex += " & $"
            latex += "\\bf{" if arch_name == best_arch_name else ""
            latex += f"{fscore_max:.3f}"
            latex += "}" if arch_name == best_arch_name else ""
            latex += "$"

            latex += " \\\\\n"

        latex += hline

    logg.debug(f"{latex}")

    this_file_folder = Path(__file__).parent.absolute()
    report_folder = this_file_folder / "report"
    output_file = report_folder / "megacomparison_auto.tex"
    template_table = ""
    template_table += "% Autogenerated by"
    template_table += " python evaluate_all.py -et build_megacomparison_v01"
    template_table += "\n"
    template_table += r"\begin{{table}}[t!]"
    template_table += "\n"
    template_table += r"    \centering"
    template_table += "\n"
    template_table += r"    \caption{{Mega comparison}}"
    template_table += "\n"
    template_table += r"    \label{{tab:mega_comparison}}"
    template_table += "\n"
    template_table += r"    \begin{{tabular}}{{|c|ccc|}}"
    template_table += "\n"
    template_table += "{}"
    template_table += r"    \end{{tabular}}"
    template_table += "\n"
    template_table += r"\end{{table}}"
    output_file.write_text(template_table.format(latex))


def build_megacomparison_v02() -> None:
    """MAKEDOC: what is build_megacomparison_v02 doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_megacomparison_v02")
    # logg.setLevel("INFO")
    logg.debug("Start build_megacomparison_v02")

    all_res_df = {}
    all_res_df["area"] = build_area_results_df()
    all_res_df["att"] = build_att_results_df()
    all_res_df["cnn"] = build_cnn_results_df()
    all_res_df["tra"] = build_tra_results_df()

    # remove failed trainings
    for arch in all_res_df:
        all_res_df[arch] = all_res_df[arch].query("fscore > 0.5")
        logg.debug(f"len(all_res_df[{arch}]): {len(all_res_df[arch])}")

    word_lists = [["f1"], ["k1"], ["yn", "LTyn"], ["num", "LTnum"], ["all", "LTall"]]

    type_per_arch: ty.Dict[str, ty.List[str]] = {}
    type_per_arch["cnn"] = ["CNN"]
    type_per_arch["tra"] = ["TRA", "TD1", "TB0", "TB4", "TB7"]
    type_per_arch["att"] = ["ATT"]
    # type_per_arch["area"] = ["AAN", "ARN", "SIM", "SI2", "VAN"]
    type_per_arch["area"] = ["AAN", "SIM", "SI2", "VAN"]

    # all_fscores[wl_str][arch_name]['mean'] = fmean
    all_fscores: ty.Dict[str, ty.Dict[str, ty.Dict[str, float]]] = {}

    # track the best
    best_arch_names: ty.Dict[str, ty.List[str]] = {}
    best_fscore_max: ty.Dict[str, float] = {}

    # build the fscore dict
    for wl in word_lists:

        # identifier for this task
        wl_str = ", ".join(wl)

        # fscores for this task
        wl_fscores: ty.Dict[str, ty.Dict[str, float]] = {}

        for arch in type_per_arch:
            for arch_name in type_per_arch[arch]:
                df_f = all_res_df[arch]
                df_f = df_f[df_f["words"].isin(wl)]
                df_f = df_f.query(f"model_name.str.startswith('{arch_name}')")

                fscore_mean = df_f.fscore.mean()
                fscore_stddev = df_f.fscore.std()
                fscore_min = df_f.fscore.min()
                fscore_max = df_f.fscore.max()

                if not math.isnan(fscore_mean) and not math.isnan(fscore_stddev):
                    wl_fscores[arch_name] = {}
                    wl_fscores[arch_name]["mean"] = fscore_mean
                    wl_fscores[arch_name]["stddev"] = fscore_stddev
                    wl_fscores[arch_name]["min"] = fscore_min
                    wl_fscores[arch_name]["max"] = fscore_max

                    # get the current max or 0
                    # curr_best_fscore_max = best_fscore_max.setdefault(wl_str, 0)
                    if wl_str in best_fscore_max:
                        curr_best_fscore_max = best_fscore_max[wl_str]
                    else:
                        curr_best_fscore_max = 0

                    # add the value if it is not increasing
                    if math.isclose(fscore_max, curr_best_fscore_max, abs_tol=1e-5):
                        # if wl_str in best_arch_names:
                        best_arch_names[wl_str].append(arch_name)
                        # logg.debug(f"isclose {arch_name} for {wl_str}")

                    # the value is a new max, reset the list
                    # we only arrive here if it is NOT close, avoid float errors
                    elif fscore_max > curr_best_fscore_max:
                        best_arch_names[wl_str] = [arch_name]
                        best_fscore_max[wl_str] = fscore_max

        all_fscores[wl_str] = wl_fscores

    # logg.debug(f"all_fscores: {json.dumps(all_fscores, indent=4)}")
    logg.debug(f"best_arch_names: {json.dumps(best_arch_names, indent=4)}")

    #################################################################
    #   compose the latex string
    #################################################################

    num_data_col = len(word_lists)

    # misc util str
    table_str = ""
    indent = "    "
    newline = " \\\\\n"
    hline = indent * 2 + "\\hline\n"
    # \cline{2-6}

    cline = indent * 2
    cline += f"\\cline{{2-{num_data_col+1}}}\n"

    # build the header line
    header = indent * 2
    header += "Arch"
    for wl_str in all_fscores:
        header += f" & {wl_str}"
    header += newline

    # mount the header
    table_str += hline
    table_str += header
    table_str += hline
    table_str += hline

    for arch in type_per_arch:
        for ani, arch_name in enumerate(type_per_arch[arch]):

            if ani != 0:
                table_str += cline

            table_str += indent * 2
            # table_str += f"{arch_name} "
            table_str += f"\\multirow{{2}}{{*}}{{{arch_name}}}"
            table_str += "\n"

            # write the mean and stddev
            table_str += indent * 2
            table_str += "    "

            for wl_str in all_fscores:
                if arch_name in all_fscores[wl_str]:
                    fscore_mean = all_fscores[wl_str][arch_name]["mean"]
                    fscore_stddev = all_fscores[wl_str][arch_name]["stddev"]
                    table_str += f" & ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"
                else:
                    table_str += " & -                "

            table_str += newline

            # write the max
            table_str += indent * 2
            table_str += "    "

            for wl_str in all_fscores:
                if arch_name in all_fscores[wl_str]:
                    fscore_max = all_fscores[wl_str][arch_name]["max"]

                    best_arch_name = best_arch_names[wl_str]
                    table_str += " & $"
                    table_str += "\\bf{" if arch_name in best_arch_name else "    "
                    table_str += f"{fscore_max:.3f}"
                    table_str += "}" if arch_name in best_arch_name else " "
                    table_str += "$     "

                    if arch_name in best_arch_name:
                        logg.debug(f"is best {arch_name} for {wl_str}")

                else:
                    table_str += " & -                "

            table_str += newline

        table_str += hline

    # logg.debug(f"{table_str}")

    caption = "A nice mega comparison"
    autogen_cmd = "python evaluate_all.py -et build_megacomparison_v02"
    label_str = "mega_comparison"

    the_table = build_table(
        autogen_cmd=autogen_cmd,
        caption=caption,
        label_str=label_str,
        table_str=table_str,
        num_data_col=num_data_col,
    )
    # logg.debug(f"the_table:\n{the_table}")

    this_file_folder = Path(__file__).parent.absolute()
    report_folder = this_file_folder / "report"
    output_file = report_folder / "megacomparison_auto.tex"
    output_file.write_text(the_table)


def build_megacomparison_v03() -> None:
    """MAKEDOC: what is build_megacomparison_v03 doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_megacomparison_v03")
    # logg.setLevel("INFO")
    logg.debug("Start build_megacomparison_v03")

    all_res_df = {}
    all_res_df["area"] = build_area_results_df()
    all_res_df["att"] = build_att_results_df()
    all_res_df["cnn"] = build_cnn_results_df()
    all_res_df["tra"] = build_tra_results_df()

    # remove failed trainings
    for arch in all_res_df:
        all_res_df[arch] = all_res_df[arch].query("fscore > 0.5")
        logg.debug(f"len(all_res_df[{arch}]): {len(all_res_df[arch])}")

    # word_lists = [["f1"], ["k1"], ["yn", "LTyn"], ["num", "LTnum"], ["all", "LTall"]]
    word_lists = [
        ["f1"],
        ["k1"],
        ["yn"],
        ["num", "LTnum", "LTBnum"],
        ["all", "LTall", "LTBall"],
    ]

    type_per_arch: ty.Dict[str, ty.List[str]] = {}
    type_per_arch["cnn"] = ["CNN"]
    type_per_arch["tra"] = ["TRA", "TD1", "TB0", "TB4", "TB7"]
    type_per_arch["att"] = ["ATT"]
    # type_per_arch["area"] = ["AAN", "ARN", "SIM", "SI2", "VAN"]
    type_per_arch["area"] = ["AAN", "SIM", "SI2", "VAN"]

    # all_fscores[wl_str][arch_name]['mean'] = fmean
    all_fscores: ty.Dict[str, ty.Dict[str, ty.Dict[str, float]]] = {}

    # track the best
    best_arch_names: ty.Dict[str, ty.List[str]] = {}
    best_fscore_max: ty.Dict[str, float] = {}

    # build the fscore dict
    for wl in word_lists:

        # identifier for this task
        wl_str = ", ".join(wl)

        # fscores for this task
        wl_fscores: ty.Dict[str, ty.Dict[str, float]] = {}

        for arch in type_per_arch:
            for arch_name in type_per_arch[arch]:
                df_f = all_res_df[arch]
                df_f = df_f[df_f["words"].isin(wl)]
                df_f = df_f.query(f"model_name.str.startswith('{arch_name}')")

                fscore_mean = df_f.fscore.mean()
                fscore_stddev = df_f.fscore.std()
                fscore_min = df_f.fscore.min()
                fscore_max = df_f.fscore.max()

                if not math.isnan(fscore_mean) and not math.isnan(fscore_stddev):
                    wl_fscores[arch_name] = {}
                    wl_fscores[arch_name]["mean"] = fscore_mean
                    wl_fscores[arch_name]["stddev"] = fscore_stddev
                    wl_fscores[arch_name]["min"] = fscore_min
                    wl_fscores[arch_name]["max"] = fscore_max

                    # get the current max or 0
                    # curr_best_fscore_max = best_fscore_max.setdefault(wl_str, 0)
                    if wl_str in best_fscore_max:
                        curr_best_fscore_max = best_fscore_max[wl_str]
                    else:
                        curr_best_fscore_max = 0

                    # add the value if it is not increasing
                    if math.isclose(fscore_max, curr_best_fscore_max, abs_tol=1e-5):
                        # if wl_str in best_arch_names:
                        best_arch_names[wl_str].append(arch_name)
                        # logg.debug(f"isclose {arch_name} for {wl_str}")

                    # the value is a new max, reset the list
                    # we only arrive here if it is NOT close, avoid float errors
                    elif fscore_max > curr_best_fscore_max:
                        best_arch_names[wl_str] = [arch_name]
                        best_fscore_max[wl_str] = fscore_max

        all_fscores[wl_str] = wl_fscores

    # logg.debug(f"all_fscores: {json.dumps(all_fscores, indent=4)}")
    logg.debug(f"best_arch_names: {json.dumps(best_arch_names, indent=4)}")

    #################################################################
    #   compose the latex string
    #################################################################

    num_data_col = len(word_lists)

    # misc util str
    table_str = ""
    indent = "    "
    newline = " \\\\\n"
    hline = indent * 2 + "\\hline\n"
    # \cline{2-6}

    cline = indent * 2
    cline += f"\\cline{{2-{num_data_col+1}}}\n"

    # build the header line
    header = indent * 2
    header += "Arch"
    for wl_str in all_fscores:
        header += f" & {wl_str}"
    header += newline

    # mount the header
    table_str += hline
    table_str += header
    table_str += hline
    table_str += hline

    for arch in type_per_arch:
        for ani, arch_name in enumerate(type_per_arch[arch]):

            if ani != 0:
                table_str += cline

            table_str += indent * 2
            # table_str += f"{arch_name} "
            table_str += f"\\multirow{{2}}{{*}}{{{arch_name}}}"
            table_str += "\n"

            # write the mean and stddev
            table_str += indent * 2
            table_str += "    "

            for wl_str in all_fscores:
                if arch_name in all_fscores[wl_str]:
                    fscore_mean = all_fscores[wl_str][arch_name]["mean"]
                    fscore_stddev = all_fscores[wl_str][arch_name]["stddev"]
                    table_str += f" & ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"
                else:
                    table_str += " & -                "

            table_str += newline

            # write the max
            table_str += indent * 2
            table_str += "    "

            for wl_str in all_fscores:
                if arch_name in all_fscores[wl_str]:
                    fscore_max = all_fscores[wl_str][arch_name]["max"]

                    best_arch_name = best_arch_names[wl_str]
                    table_str += " & $"
                    table_str += "\\bf{" if arch_name in best_arch_name else "    "
                    table_str += f"{fscore_max:.3f}"
                    table_str += "}" if arch_name in best_arch_name else " "
                    table_str += "$     "

                    if arch_name in best_arch_name:
                        logg.debug(f"is best {arch_name} for {wl_str}")

                else:
                    table_str += " & -                "

            table_str += newline

        table_str += hline

    # logg.debug(f"{table_str}")

    caption = "Architecture comparison for different tasks."
    caption += " Each cell shows the mean, standard deviation and max F-Score value."
    autogen_cmd = "python evaluate_all.py -et build_megacomparison_v03"
    label_str = "mega_comparison"

    the_table = build_table(
        autogen_cmd=autogen_cmd,
        caption=caption,
        label_str=label_str,
        table_str=table_str,
        num_data_col=num_data_col,
    )
    # logg.debug(f"the_table:\n{the_table}")

    this_file_folder = Path(__file__).parent.absolute()
    report_folder = this_file_folder / "report"
    output_file = report_folder / "megacomparison_auto.tex"
    output_file.write_text(the_table)


def evaluate_model_cm_all(test_words_type: str) -> None:
    r"""MAKEDOC: what is evaluate_model_cm_all doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model_cm_all")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_model_cm_all")

    this_file_folder = Path(__file__).parent.absolute()

    trained_folder = this_file_folder / "trained_models"

    train_type_tags = []
    train_type_tags.append("area")
    train_type_tags.append("attention")
    # train_type_tags.append("cnn")
    # train_type_tags.append("transfer")

    train_words_type_re = re.compile("_w(.*?)[_.]")

    all_models = []

    # find all the trained models for num
    for train_type_tag in train_type_tags:
        model_folder = trained_folder / train_type_tag
        logg.debug(f"model_folder: {model_folder}")

        for model_path in model_folder.iterdir():
            model_name = model_path.name

            if not model_path.suffix == ".h5":
                continue

            match = train_words_type_re.search(model_name)
            if match is not None:
                # logg.debug(f"match[1]: {match[1]}")
                train_words_type = match[1]
            else:
                continue

            if train_words_type == "num":
                # logg.debug(f"model_path: {model_path}")
                all_models.append(model_path.stem)

    logg.debug(f"len(all_models): {len(all_models)}")

    cm_folder: Path = Path("plot_results") / "cm_all01"
    res_path: Path = cm_folder / "fscore_num_results.json"

    # load the existing results
    if res_path.exists():
        fsdd_fscores = json.loads(res_path.read_text())
    else:
        fsdd_fscores = {}

    # compute the new fscores on fsdd
    for model_name in all_models:
        if model_name in fsdd_fscores:
            logg.debug(f"{model_name} fscore {fsdd_fscores[model_name]}")
            continue
        fscore = evaluate_model_cm(model_name, test_words_type)
        fsdd_fscores[model_name] = fscore

    res_path.write_text(json.dumps(fsdd_fscores, indent=4))

    train_fscore = {}

    # get the fscores on the num data
    for model_name in all_models:
        train_type_tag = which_arch(model_name)

        info_folder = Path("info") / train_type_tag
        model_info_folder = info_folder / model_name

        res_path = model_info_folder / "results_recap.json"
        if not res_path.exists():
            logg.info(f"Skipping res_path: {res_path}, not found")
            continue
        res = json.loads(res_path.read_text())

        fscore = res["fscore"]
        train_fscore[model_name] = fscore

    # build the DataFrame

    pandito: ty.Dict[str, ty.List[ty.Any]] = {
        "model_name": [],
        "arch_type": [],
        "train_fscore": [],
        "fsdd_fscore": [],
    }

    for model_name in train_fscore:
        pandito["model_name"].append(model_name)
        pandito["train_fscore"].append(train_fscore[model_name])
        pandito["fsdd_fscore"].append(fsdd_fscores[model_name])
        arch_type = model_name[:3]
        pandito["arch_type"].append(arch_type)

    res_df = pd.DataFrame(pandito)

    ###### build the table

    table_str = ""
    indent = "    "
    newline = " \\\\\n"
    hline = indent * 2 + "\\hline\n"

    header = indent * 2
    header += "Arch & Train & FSDD"
    header += newline

    table_str += hline
    table_str += header
    table_str += hline

    arch_types = res_df["arch_type"].unique()

    for arch_type in arch_types:
        logg.debug(f"arch_type: {arch_type}")

        df_f = res_df
        df_f = df_f.query(f"arch_type == '{arch_type}'")

        fsdd_fscore_mean = df_f.fsdd_fscore.mean()
        fsdd_fscore_stddev = df_f.fsdd_fscore.std()
        # fsdd_fscore_min = df_f.fsdd_fscore.min()
        fsdd_fscore_max = df_f.fsdd_fscore.max()

        train_fscore_mean = df_f.train_fscore.mean()
        train_fscore_stddev = df_f.train_fscore.std()
        # train_fscore_min = df_f.train_fscore.min()
        train_fscore_max = df_f.train_fscore.max()

        # write the arch name
        table_str += indent * 2
        table_str += f"\\multirow{{2}}{{*}}{{{arch_type}}}"
        table_str += "\n"

        # write the train and fsdd mean/std
        table_str += indent * 2
        table_str += f"    & ${train_fscore_mean:.3f} \\pm {train_fscore_stddev:.3f}$"
        table_str += f"    & ${fsdd_fscore_mean:.3f} \\pm {fsdd_fscore_stddev:.3f}$"
        table_str += newline

        # write the train and fsdd max
        table_str += indent * 2
        table_str += f"    & ${train_fscore_max:.3f}$"
        table_str += f"              & ${fsdd_fscore_max:.3f}$"
        table_str += newline

    table_str += hline
    logg.debug(f"{table_str}")


def evaluate_model_cm(model_name: str, test_words_type: str) -> float:
    r"""MAKEDOC: what is evaluate_model_cm doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_model_cm")
    # logg.setLevel("INFO")
    # logg.debug("\nStart evaluate_model_cm")

    # magic to fix the GPUs
    setup_gpus()

    logg.debug(f"\nmodel_name: {model_name}")

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

    arch_type = model_name[:3]

    if arch_type == "ATT":
        train_type_tag = "attention"
    else:
        train_type_tag = "area"

    # load the model
    model_folder = Path("trained_models") / train_type_tag
    model_path = model_folder / f"{model_name}.h5"
    model = tf_models.load_model(model_path)
    # model.summary()

    train_words = words_types[train_words_type]
    logg.debug(f"train_words: {train_words}")
    test_words = words_types[test_words_type]
    logg.debug(f"test_words: {test_words}")

    # input data must exist
    if dataset_name.startswith("mel"):
        preprocess_spec(dataset_name, test_words_type)
    elif dataset_name.startswith("aug"):
        do_augmentation(dataset_name, test_words_type)

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
    cm_folder = Path("plot_results") / "cm_all01"
    if not cm_folder.exists():
        cm_folder.mkdir(parents=True, exist_ok=True)

    plot_cm_path = cm_folder / fig_name.format("png")
    fig.savefig(plot_cm_path)
    plot_cm_path = cm_folder / fig_name.format("pdf")
    fig.savefig(plot_cm_path)

    # plt.show()
    return fscore


def evaluate_results_all() -> None:
    """MAKEDOC: what is evaluate_results_all doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_all")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_results_all")


def run_evaluate_all(args: argparse.Namespace) -> None:
    """TODO: What is evaluate_all doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_all")
    logg.debug("Starting run_evaluate_all")

    evaluation_type = args.evaluation_type

    train_words_type = args.train_words_type
    model_name = args.model_name

    rec_words_type = args.rec_words_type
    if rec_words_type == "train":
        rec_words_type = train_words_type

    if evaluation_type == "results":
        evaluate_results_all()
    elif evaluation_type == "evaluate_augmentation":
        evaluate_augmentation()
    elif evaluation_type == "evaluate_loud_section":
        evaluate_loud_section()
    elif evaluation_type == "build_megacomparison_v01":
        build_megacomparison_v01()
    elif evaluation_type == "build_megacomparison_v02":
        build_megacomparison_v02()
    elif evaluation_type == "build_megacomparison_v03":
        build_megacomparison_v03()
    elif evaluation_type == "evaluate_model_cm":
        evaluate_model_cm(model_name, rec_words_type)
    elif evaluation_type == "evaluate_model_cm_all":
        evaluate_model_cm_all(rec_words_type)


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_all(args)
