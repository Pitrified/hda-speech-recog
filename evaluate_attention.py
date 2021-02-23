from copy import deepcopy
from itertools import combinations
from itertools import product
from itertools import permutations
from pathlib import Path
import argparse
import json
import logging

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
import typing as ty

from keras_diagram import ascii_model
from plot_utils import plot_pred
from plot_utils import plot_spec
from plot_utils import plot_waveform
from plot_utils import quad_plotter
from preprocess_data import get_spec_dict
from preprocess_data import load_processed
from preprocess_data import wav2mel
from train_attention import build_attention_name
from utils import compute_permutation
from utils import pr2fscore
from utils import record_audios
from utils import setup_gpus
from utils import setup_logger
from utils import words_types


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
            "evaluate_batch_epoch",
            "attention_weights",
            "make_plots_hypa",
            "make_plots_clr",
            "audio",
            "delete_bad_models",
            "compare_augmentation",
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
    """MAKEDOC: what is build_att_results_df doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_att_results_df")
    # logg.setLevel("INFO")
    logg.debug("Start build_att_results_df")

    pandito: ty.Dict[str, ty.List[str]] = {
        "words": [],
        "dataset": [],
        "conv": [],
        "dropout": [],
        "kernel": [],
        "lstm": [],
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

    info_folder = Path("info") / "attention"

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
            logg.error("\nWHAT ARE YOU DOING using this version\n")

        hypa: ty.Dict[str, ty.Any] = recap["hypa"]

        pandito["words"].append(hypa["words_type"])
        pandito["dataset"].append(hypa["dataset_name"])
        pandito["conv"].append(hypa["conv_size_type"])
        pandito["dropout"].append(hypa["dropout_type"])
        pandito["kernel"].append(hypa["kernel_size_type"])
        pandito["lstm"].append(hypa["lstm_units_type"])
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
    """MAKEDOC: what is evaluate_results_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_results_attention")
    logg.setLevel("INFO")
    logg.debug("Start evaluate_results_attention")

    results_df = build_att_results_df()

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
    logg.info(f"{df_f.tail(100)}")

    # by word
    for words_type in ["dir", "k1", "w2", "f2", "f1", "all", "num"]:
        word_list = [w for w in results_df.words.unique() if words_type in w]

        df_f = results_df
        # df_f = df_f.query("use_val == True")
        # df_f = df_f.query(f"words == '{words_type}'")
        df_f = df_f[df_f["words"].isin(word_list)]
        df_f = df_f.sort_values("fscore", ascending=False)
        logg.info(f"\nOnly on {words_type}")
        logg.info(f"{df_f.head(30)}\n{df_f.tail()}")

    # produce a dict for recreating the training
    for index, row in df_f.iterrows():
        hp = "    {"
        hp += f"        'words_type': '{row['words']}',"
        hp += f"        'dataset_name': '{row['dataset']}',"
        hp += f"        'conv_size_type': '{row['conv']}',"
        hp += f"        'dropout_type': '{row['dropout']}',"
        hp += f"        'kernel_size_type': '{row['kernel']}',"
        hp += f"        'lstm_units_type': '{row['lstm']}',"
        hp += f"        'query_style_type': '{row['query']}',"
        hp += f"        'dense_width_type': '{row['dense']}',"
        hp += f"        'learning_rate_type': '{row['lr']}',"
        hp += f"        'optimizer_type': '{row['optimizer']}',"
        hp += f"        'batch_size_type': '{row['batch']}',"
        hp += f"        'epoch_num_type': '{row['epoch']}',"
        hp += "    },"
        # logg.debug(f"hp: {hp}")

    aug_list = [dn for dn in df_f.dataset.unique() if dn.startswith("aug")]
    logg.info(f"Only on aug_list: {aug_list}")
    df_f = results_df
    df_f = df_f.query("use_val == True")
    df_f = df_f.query("words == 'k1'")
    df_f = df_f[df_f["dataset"].isin(aug_list)]
    df_f = df_f.sort_values("fscore", ascending=False)
    logg.info(f"{df_f.head(30)}\n{df_f.tail()}")

    # df['a'].value_counts()
    logg.info(f"results_df['conv'].value_count(): {results_df['conv'].value_counts()}")

    df_f = results_df
    df_f = df_f.query("words == 'LTBnum'")
    # df_f = df_f.query("words == 'LTBnumLS'")
    df_f = df_f.sort_values("fscore", ascending=False)
    logg.info(f"{df_f.head(10)}")

    df_f = results_df
    df_f = df_f.query("words == 'k1'")
    df_f = df_f.query("epoch == '01'")
    df_f = df_f.sort_values("fscore", ascending=False)
    logg.info(f"{df_f.head(10)}")


def evaluate_batch_epoch() -> None:
    """MAKEDOC: what is evaluate_batch_epoch doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_batch_epoch")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_batch_epoch")

    results_df = build_att_results_df()

    # all the results so far
    df_f = results_df
    df_f = df_f.query("use_val == True")
    df_f = df_f.query("words == 'k1'")

    epoch_nums = df_f["epoch"].unique()
    batch_sizes = df_f["batch"].unique()
    for en, bs in product(epoch_nums, batch_sizes):
        logg.debug(f"\nen {en} bs {bs}")
        eb_f = df_f
        eb_f = eb_f.query(f"epoch == '{en}'")
        eb_f = eb_f.query(f"batch == '{bs}'")
        eb_f = eb_f.sort_values("fscore", ascending=False)
        logg.info(f"{eb_f.head(10)}")
        logg.info(f"{eb_f.tail(4)}")
        fscore_mean = eb_f.fscore.mean()
        logg.debug(f"fscore_mean: {fscore_mean}")


def evaluate_attention_weights(
    train_words_type: str, rec_words_type: str, do_new_record: bool = False
) -> None:
    """MAKEDOC: what is evaluate_attention_weights doing?"""
    logg = logging.getLogger(f"c.{__name__}.evaluate_attention_weights")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_attention_weights")

    # magic to fix the GPUs
    setup_gpus()

    # ATT_ct02_dr02_ks02_lu01_as01_qt01_dw01_opa1_lr01_bs01_en01_dsmel04_wk1
    # hypa: ty.Dict[str, str] = {}
    # hypa["conv_size_type"] = "02"
    # hypa["dropout_type"] = "02"
    # hypa["kernel_size_type"] = "02"
    # hypa["lstm_units_type"] = "01"
    # hypa["query_style_type"] = "01"
    # hypa["dense_width_type"] = "01"
    # hypa["optimizer_type"] = "a1"
    # hypa["learning_rate_type"] = "01"
    # hypa["batch_size_type"] = "01"
    # hypa["epoch_num_type"] = "01"
    # dataset_name = "mel04"
    # hypa["dataset_name"] = dataset_name
    # hypa["words_type"] = train_words_type
    # use_validation = True

    # ATT_ct02_dr01_ks01_lu01_qt05_dw01_opa1_lr03_bs02_en02_dsaug07_wLTnum
    hypa = {
        "batch_size_type": "02",
        "conv_size_type": "02",
        "dataset_name": "aug07",
        "dense_width_type": "01",
        "dropout_type": "01",
        "epoch_num_type": "02",
        "kernel_size_type": "01",
        "learning_rate_type": "03",
        "lstm_units_type": "01",
        "optimizer_type": "a1",
        "query_style_type": "05",
        "words_type": "LTnum",
    }
    use_validation = True

    dataset_name = hypa["dataset_name"]

    model_name = build_attention_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # load the model
    model_folder = Path("trained_models") / "attention"
    model_path = model_folder / f"{model_name}.h5"

    # model = tf.keras.models.load_model(model_path)
    # https://github.com/keras-team/keras/issues/5088#issuecomment-401498334
    model = tf.keras.models.load_model(
        model_path, custom_objects={"backend": tf.keras.backend}
    )
    model.summary()
    logg.debug(f"ascii_model(model): {ascii_model(model)}")

    att_weight_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer("output").output,
            model.get_layer("att_softmax").output,
            model.get_layer("bidirectional_1").output,
        ],
    )
    att_weight_model.summary()
    # logg.debug(f"att_weight_model.outputs: {att_weight_model.outputs}")

    # get the training words
    train_words = words_types[train_words_type]
    # logg.debug(f"train_words: {train_words}")
    perm_pred = compute_permutation(train_words)

    rec_words_type = args.rec_words_type
    if rec_words_type == "train":
        rec_words = train_words[-3:]
        # rec_words = train_words[:]
        logg.debug(f"Using rec_words: {rec_words}")
    else:
        rec_words = words_types[rec_words_type]
    num_rec_words = len(rec_words)

    # record new audios
    if do_new_record:

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
            # logg.debug(f"img_spec.shape: {img_spec.shape}")
            # img_spec.shape: (128, 32, 1)

            img_specs.append(img_spec)

        # the data needs to look like this data['testing'].shape: (735, 128, 32, 1)
        rec_data = np.stack(img_specs)
        # logg.debug(f"rec_data.shape: {rec_data.shape}")

    # load data if you do not want to record new audios
    else:

        # input data
        processed_folder = Path("data_proc")
        processed_path = processed_folder / f"{dataset_name}"

        # which word in the dataset to plot
        word_id = 2

        # the loaded spectrograms
        rec_data_l: ty.List[np.ndarray] = []

        for i, word in enumerate(rec_words):
            data, labels = load_processed(processed_path, [word])

            # get one of the spectrograms
            word_data = data["testing"][word_id]
            rec_data_l.append(word_data)

        # turn the list into np array
        rec_data = np.stack(rec_data_l)

    # get prediction and attention weights
    pred, att_weights, LSTM_out = att_weight_model.predict(rec_data)
    # logg.debug(f"att_weights.shape: {att_weights.shape}")
    # logg.debug(f"att_weights[0].shape: {att_weights[0].shape}")

    # if we recorded fresh audios we also have the waveform to plot
    ax_add = 1 if do_new_record else 0

    # plot the wave, spectrogram, weights and predictions in each column
    plot_size = 5
    fw = plot_size * num_rec_words
    nrows = 3 + ax_add
    # nrows = 4 + ax_add
    fh = plot_size * nrows * 0.7
    fig, axes = plt.subplots(
        nrows=nrows, ncols=num_rec_words, figsize=(fw, fh), sharey="row"
    )
    fig.suptitle(f"Attention weights and predictions for {rec_words}", fontsize=20)

    for i, word in enumerate(rec_words):
        word_spec = rec_data[i][:, :, 0]
        # logg.debug(f"word_spec.shape: {word_spec.shape}")

        # plot the waveform
        if do_new_record:
            plot_waveform(audios[i], axes[0][i])

        # plot the spectrogram
        title = f"Spectrogram for {word}"
        plot_spec(word_spec, axes[0 + ax_add][i], title=title)

        # plot the weights
        word_att_weights = att_weights[i]
        # plot_att_weights(word_att_weights, axes[1 + ax_add][i], title)

        word_att_weights_img = np.expand_dims(word_att_weights, axis=-1).T
        axes[1 + ax_add][i].imshow(word_att_weights_img, origin="lower", aspect="auto")
        title = f"Attention weights for {word}"
        axes[1 + ax_add][i].set_title(title)

        # plot the predictions
        word_pred = pred[i]
        # permute the prediction from sorted to the order you have
        word_pred = word_pred[perm_pred]
        pred_index = np.argmax(word_pred)
        title = f"Predictions for {word}"
        plot_pred(word_pred, train_words, axes[2 + ax_add][i], title, pred_index)

        # axes[3 + ax_add][i].imshow(LSTM_out[i], origin="lower")

    # fig.tight_layout()
    fig.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.97])

    fig_name = f"{model_name}"
    fig_name += f"_{train_words_type}"
    fig_name += f"_{rec_words_type}_img"
    if do_new_record:
        fig_name += "_new.{}"
    else:
        fig_name += "_data.{}"

    plot_folder = Path("plot_results")
    results_path = plot_folder / fig_name.format("png")
    fig.savefig(results_path)
    results_path = plot_folder / fig_name.format("pdf")
    fig.savefig(results_path)

    if num_rec_words <= 6:
        plt.show()


def compare_augmentation() -> None:
    """MAKEDOC: what is compare_augmentation doing?"""
    logg = logging.getLogger(f"c.{__name__}.compare_augmentation")
    # logg.setLevel("INFO")
    logg.debug("Start compare_augmentation")

    res_att_df = build_att_results_df()
    res_att_df = res_att_df.query("fscore > 0.5")
    res_att_df = res_att_df.query("words == 'k1'")
    logg.debug(f"res_att_df['dataset'].unique(): {res_att_df['dataset'].unique()}")

    # aug2345 = ["aug02", "aug03", "aug04", "aug05"]
    # aug6789 = ["aug06", "aug07", "aug08", "aug09"]
    # aug0123 = ["aug10", "aug11", "aug12", "aug13"]
    # aug4567 = ["aug14", "aug15", "aug16", "aug17"]
    aug4567 = ["aug14", "aug15", "aug16"]
    # all_aug = (*aug2345, *aug6789, *aug0123, *aug4567)
    all_aug = aug4567
    logg.debug(f"all_aug: {all_aug}")

    mels = ["mel01", "mel04", "mel05", "mela1"]
    # meLs = ["meL04", "meLa1", "meLa2", "meLa3", "meLa4"]
    # all_mels = (*mels, *meLs)
    all_mels = mels
    logg.debug(f"all_mels: {all_mels}")

    data_names = ["mel", "aug"]
    for ep in ["01", "03", "04"]:
        recap = f"{ep}"
        for i, data in enumerate((mels, all_aug)):
            logg.debug(f"{data_names[i]} ep: {ep}")
            df_f = res_att_df
            df_f = df_f[df_f["dataset"].isin(data)]
            df_f = df_f.query(f"epoch == '{ep}'")
            df_f = df_f.sort_values("fscore", ascending=False)
            # logg.info(f"{df_f.head(5)}")
            fscore_mean = df_f.fscore.mean()
            logg.debug(f"fscore_mean: {fscore_mean:.3f}")
            fscore_stddev = df_f.fscore.std()
            # logg.debug(f"fscore_stddev: {fscore_stddev:.3f}")

            recap += f" & ${fscore_mean:.3f} \\pm {fscore_stddev:.3f}$"
        recap += " \\\\"
        logg.debug(f"{recap}")

    # df_f = res_att_df
    # # df_f = df_f[df_f["dataset"].isin(all_aug)]
    # df_f = df_f[df_f["dataset"].isin(all_mels)]
    # df_f = df_f.query("epoch == '04'")
    # df_f = df_f.sort_values("fscore", ascending=False)
    # # logg.info(f"{df_f.head(5)}")
    # fscore_mean = df_f.fscore.mean()
    # logg.debug(f"fscore_mean: {fscore_mean}")

    df_f = res_att_df
    # df_f = df_f[df_f["dataset"].isin(all_mels)]
    df_f = df_f[df_f["dataset"].isin(all_aug)]
    df_f = df_f.query("words == 'k1'")
    df_f = df_f.query("epoch == '01' or epoch == '02'")
    df_f = df_f.sort_values("fscore", ascending=False)
    # logg.info(f"{df_f.head(5)}")
    fscore_mean = df_f.fscore.mean()
    logg.debug(f"fscore_mean: {fscore_mean}")

    ##########################################################################
    #   Plot learning
    ##########################################################################

    info_folder = Path("info") / "attention"

    # ATT_ct02_dr01_ks02_lu01_qt05_dw01_opa1_lr07_bs02_en02_dsmel04_wk1
    # ATT_ct02_dr01_ks02_lu01_qt05_dw01_opa1_lr07_bs02_en01_dsmel04_wk1
    # NOT augmented, lr 07, en 02
    # model_name_mel = "ATT_ct02_dr01_ks02_lu01_qt05_dw01_opa1_lr07_bs02_en02_dsmel04_wk1"
    # model_name_mel = "ATT_ct02_dr01_ks02_lu01_qt05_dw01_opa1_lr07_bs02_en01_dsmel04_wk1"
    model_name_mel = "ATT_ct02_dr01_ks01_lu01_qt04_dw01_opa1_lr03_bs01_en01_dsmel04_wk1"
    res_path_mel = info_folder / model_name_mel / "results_recap.json"
    res_mel = json.loads(res_path_mel.read_text())
    logg.debug(f"res_mel.keys(): {res_mel.keys()}")

    # compute F-Score
    mel_prec_val = np.array(res_mel["history_val"]["val_precision"])
    mel_recall_val = np.array(res_mel["history_val"]["val_recall"])
    mel_fscore_val = pr2fscore(mel_prec_val, mel_recall_val)
    logg.debug(f"mel_fscore_val: {mel_fscore_val}")
    mel_prec_train = np.array(res_mel["history_train"]["precision"])
    mel_recall_train = np.array(res_mel["history_train"]["recall"])
    mel_fscore_train = pr2fscore(mel_prec_train, mel_recall_train)
    logg.debug(f"mel_fscore_train: {mel_fscore_train}")

    mel_loss_val = np.array(res_mel["history_val"]["val_loss"])
    mel_loss_train = np.array(res_mel["history_train"]["loss"])

    # ATT_ct02_dr01_ks02_lu01_qt05_dw01_opa1_lr06_bs02_en04_dsaug14_wk1
    # augmented, lr 06, en 04
    # model_name_aug = "ATT_ct02_dr01_ks02_lu01_qt05_dw01_opa1_lr06_bs02_en04_dsaug14_wk1"
    model_name_aug = "ATT_ct02_dr01_ks01_lu01_qt01_dw01_opa1_lr04_bs02_en01_dsaug15_wk1"

    res_path_aug = info_folder / model_name_aug / "results_recap.json"
    res_aug = json.loads(res_path_aug.read_text())
    logg.debug(f"res_aug.keys(): {res_aug.keys()}")

    # compute F-Score
    aug_prec_val = np.array(res_aug["history_val"]["val_precision"])
    aug_recall_val = np.array(res_aug["history_val"]["val_recall"])
    aug_fscore_val = pr2fscore(aug_prec_val, aug_recall_val)
    logg.debug(f"aug_fscore_val: {aug_fscore_val}")
    aug_prec_train = np.array(res_aug["history_train"]["precision"])
    aug_recall_train = np.array(res_aug["history_train"]["recall"])
    aug_fscore_train = pr2fscore(aug_prec_train, aug_recall_train)
    logg.debug(f"aug_fscore_train: {aug_fscore_train}")

    aug_loss_val = np.array(res_aug["history_val"]["val_loss"])
    aug_loss_train = np.array(res_aug["history_train"]["loss"])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle("Comparison of learning speed for regular/augmented data", fontsize=18)

    x_aug = np.arange(len(aug_fscore_val)) + 1
    x_mel = np.arange(len(mel_fscore_val)) + 1
    ax[0].plot(x_mel, mel_fscore_train, label="Standard train F-Score")
    ax[0].plot(x_mel, mel_fscore_val, label="Standard val F-Score")
    ax[0].plot(x_aug, aug_fscore_train, label="Augmented train F-Score")
    ax[0].plot(x_aug, aug_fscore_val, label="Augmented val F-Score")
    ax[0].set_xlabel("Epoch number", fontsize=15)
    ax[0].set_ylabel("F-Score", fontsize=15)

    ax[1].plot(x_mel, mel_loss_train, label="Standard train loss")
    ax[1].plot(x_mel, mel_loss_val, label="Standard val loss")
    ax[1].plot(x_aug, aug_loss_train, label="Augmented train loss")
    ax[1].plot(x_aug, aug_loss_val, label="Augmented val loss")
    ax[1].set_xlabel("Epoch number", fontsize=15)
    ax[1].set_ylabel("Loss", fontsize=15)

    ax[0].legend(loc="lower right")
    ax[1].legend()
    fig.tight_layout()

    fig_name = "comparison_augmentation_en0101.{}"
    plot_folder = Path("plot_results")
    results_path = plot_folder / fig_name.format("png")
    fig.savefig(results_path)
    results_path = plot_folder / fig_name.format("pdf")
    fig.savefig(results_path)

    plt.show()


def make_plots_hypa() -> None:
    """MAKEDOC: what is make_plots_hypa doing?"""
    logg = logging.getLogger(f"c.{__name__}.make_plots_hypa")
    # logg.setLevel("INFO")
    logg.debug("Start make_plots_hypa")

    results_df = build_att_results_df()

    # print all unique values to get an idea of what is inside the dataframe
    for col in results_df:
        if col in ["model_name", "loss", "fscore", "recall", "precision", "cat_acc"]:
            continue
        logg.debug(f"hypa_grid['{col}'] = {results_df[col].unique()}")

    # full hypa_grid
    hypa_grid_all: ty.Dict[str, ty.List[str]] = {}
    hypa_grid_all["words"] = ["k1", "LTnumLS", "w2", "FJall", "f2", "LTnum", "LTall"]
    ds = []
    ds.extend(["mel01", "mel04", "mel05", "mela1"])
    ds.extend(["aug01"])
    ds.extend(["aug02", "aug03", "aug04", "aug05"])
    ds.extend(["aug06", "aug07", "aug08", "aug09"])
    ds.extend(["aug10", "aug11", "aug12", "aug13"])
    ds.extend(["aug14", "aug15", "aug16", "aug17"])
    ds.extend(["meL04", "meLa1", "meLa2", "meLa3"])
    hypa_grid_all["dataset"] = ds
    hypa_grid_all["conv"] = ["01", "02"]
    hypa_grid_all["dropout"] = ["01", "02"]
    hypa_grid_all["kernel"] = ["01", "02"]
    hypa_grid_all["lstm"] = ["01"]
    # hypa_grid_all["query"] = ["01", "02", "03", "04", "05"]
    hypa_grid_all["query"] = ["01", "05", "02", "03", "04"]
    hypa_grid_all["dense"] = ["01", "02"]
    hypa_grid_all["lr"] = ["01", "03", "04", "10", "05", "06", "07", "08", "09"]
    hypa_grid_all["optimizer"] = ["a1"]
    hypa_grid_all["batch"] = ["02", "01"]
    hypa_grid_all["epoch"] = ["03", "04", "01", "02"]

    hypa_labels: ty.Dict[str, ty.Dict[str, str]] = {}
    hypa_labels["epoch"] = {"01": "15", "02": "30", "03": "2", "04": "4"}
    hypa_labels["batch"] = {"01": "32", "02": "16"}

    hypa_labels["lr"] = {}
    hypa_labels["lr"]["01"] = "fixed01"
    hypa_labels["lr"]["02"] = "fixed02"
    hypa_labels["lr"]["03"] = "exp_step_01"
    hypa_labels["lr"]["04"] = "exp_smooth_01"
    hypa_labels["lr"]["05"] = "clr_tri2_01"
    hypa_labels["lr"]["06"] = "clr_tri2_02"
    hypa_labels["lr"]["07"] = "clr_tri2_03"
    hypa_labels["lr"]["08"] = "clr_tri2_04"
    hypa_labels["lr"]["09"] = "clr_tri2_05"
    hypa_labels["lr"]["10"] = "exp_smooth_02"

    hypa_labels["query"] = {}
    hypa_labels["query"]["01"] = "dense_last"
    hypa_labels["query"]["05"] = "dense_mid"
    hypa_labels["query"]["02"] = "conv_LSTM1"
    hypa_labels["query"]["03"] = "conv_LSTM2"
    hypa_labels["query"]["04"] = "conv_spec"

    hp_to_plot_names_all = [
        "words",
        "dataset",
        "conv",
        "dropout",
        "kernel",
        "lstm",
        "query",
        "dense",
        "lr",
        "optimizer",
        "batch",
        "epoch",
    ]
    logg.debug(f"hp_to_plot_names_all: {hp_to_plot_names_all}")
    # hp_to_plot_names: ty.List[str] = []

    # a unique name for this filtering
    filter_tag = "104"

    # clone the results
    df_f = results_df

    # remove failed trainings
    df_f = df_f[df_f["fscore"] > 0.5]

    # in each tag overwrite the hypas to reduce
    # and set which hp to plot
    hypa_grid: ty.Dict[str, ty.List[str]] = deepcopy(hypa_grid_all)

    if filter_tag == "101":
        hypa_grid = deepcopy(hypa_grid_all)

        hypa_grid["words"] = ["k1"]

        ds = []
        # ds.extend(["mel01", "mel04", "mel05", "mela1"])
        ds.extend(["mel04"])
        # ds.extend(["aug02", "aug03", "aug04", "aug05"])
        # ds.extend(["aug06", "aug07", "aug08", "aug09"])
        # ds.extend(["aug10", "aug11", "aug12", "aug13"])
        ds.extend(["aug14", "aug15", "aug16", "aug17"])
        hypa_grid["dataset"] = ds

        # hypa_grid["epoch"] = ["01", "02"]
        hypa_grid["lr"] = ["01", "03", "04", "07"]

        hp_to_plot_names = [
            "dataset",
            "lr",
            "epoch",
            "query",
        ]
        min_lower_limit = 0.92

        sub_tag = "__".join(hp_to_plot_names)

        # filter by epochs
        # epoch_list = ["01", "02"]
        # df_f = df_f[df_f["epoch"].isin(epoch_list)]

    elif filter_tag == "103":
        logg.debug(f"filter_tag: {filter_tag}")
        # TODO: dropout 12 kernel 2 query 12345 dataset pick 2 (or dense pick 2)
        hypa_grid = deepcopy(hypa_grid_all)
        hp_to_plot_names = [
            "query",
            "conv",
            "dropout",
            # "kernel",
            # "lstm",  # only one, kinda useless
            "dense",
        ]

        word_list = []

        # word_list.append("FJall")
        # word_list.append("LTBall")
        # word_list.append("LTBallLS")
        # word_list.append("LTall")

        # word_list.append("LTBnum")
        # word_list.append("LTBnumLS")

        # word_list.append("LTnum")
        # word_list.append("LTnum")
        # word_list.append("LTnumLS")

        # word_list.append("f1")
        # word_list.append("f2")

        # word_list.append("k1")
        # word_list.append("num")

        word_list.append("w2")

        # word_list.append("yn")

        # word_list = ["num", "LTnum", "k1"]
        # word_list = ["num", "k1"]
        # word_list = ["k1", "LTnum", "LTnumLS", "w2", "FJall", "LTall"]
        # word_list = ["w2", "FJall", "LTall"]
        # word_list = ["k1"]

        df_f = df_f[df_f["words"].isin(word_list)]

        word_list = ["nowordfilter"]
        sub_tag = "_".join(word_list)

        min_lower_limit = 0.94

    elif filter_tag == "104":
        logg.debug(f"filter_tag: {filter_tag}")

        # TODO all lr, all batch, all epochs on k1

        hypa_grid = deepcopy(hypa_grid_all)
        # hypa_grid["words"] = ["k1", "w2", "f2"]
        hypa_grid["words"] = ["k1"]
        word_list = ["k1"]
        df_f = df_f[df_f["words"].isin(word_list)]

        ds = []
        ds.extend(["mel01", "mel04", "mel05", "mela1"])
        hypa_grid["dataset"] = ds
        df_f = df_f[df_f["dataset"].isin(ds)]

        hp_to_plot_names = [
            "batch",
            "lr",
            "epoch",
            "words",
        ]

        sub_tag = "_".join(word_list)
        min_lower_limit = 0.92

    elif filter_tag == "105":
        logg.debug(f"filter_tag: {filter_tag}")
        hypa_grid = deepcopy(hypa_grid_all)

        ds = []
        # ds.extend(["mel01", "mel04", "mel05", "mela1"])
        # ds.extend(["aug01"])
        # ds.extend(["aug02", "aug03", "aug04", "aug05"])
        # ds.extend(["aug06", "aug07", "aug08", "aug09"])
        # ds.extend(["aug10", "aug11", "aug12", "aug13"])
        # ds.extend(["aug14", "aug15", "aug16", "aug17"])
        # ds.extend(["meL04", "meLa1", "meLa2", "meLa3", "meLa4"])

        # hypa_grid["dataset"] = ds
        # df_f = df_f[df_f["dataset"].isin(ds)]

        word_list = ["k1", "LTnum", "LTnumLS"]
        hypa_grid["words"] = word_list
        df_f = df_f[df_f["words"].isin(word_list)]

        hp_to_plot_names = [
            "optimizer",
            "dataset",
            "lstm",
            "words",
        ]

        sub_tag = "_".join(word_list)
        # sub_tag = "nofilter"
        min_lower_limit = 0.92

    all_hp_to_plot = list(combinations(hp_to_plot_names, 4))
    logg.debug(f"len(all_hp_to_plot): {len(all_hp_to_plot)}")

    all_hp_to_plot = []
    perm = list(permutations(hp_to_plot_names[:3]))
    logg.debug(f"perm: {perm}")
    for p in perm:
        hp_to_plot = p[0], p[1], p[2], hp_to_plot_names[3]
        logg.debug(f"hp_to_plot: {hp_to_plot}")
        all_hp_to_plot.append(hp_to_plot)
    logg.debug(f"all_hp_to_plot: {all_hp_to_plot}")

    # add additional combinations if needed
    # all_hp_to_plot.append(("epoch", "dataset", "batch", "query"))

    # the output folders
    plot_fol = Path("plot_results") / "att"
    filter_fol = plot_fol / filter_tag / sub_tag
    pdf_split_fol = filter_fol / "pdf_split"
    pdf_grid_fol = filter_fol / "pdf_grid"
    png_split_fol = filter_fol / "png_split"
    png_grid_fol = filter_fol / "png_grid"
    for f in pdf_split_fol, pdf_grid_fol, png_split_fol, png_grid_fol:
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True)

    quad_plotter(
        all_hp_to_plot[:],
        hypa_grid,
        df_f,
        pdf_split_fol,
        png_split_fol,
        pdf_grid_fol,
        png_grid_fol,
        # do_single_images=True,
        do_single_images=False,
        min_at_zero=False,
        min_lower_limit=min_lower_limit,
        hypa_labels=hypa_labels,
    )


def make_plots_clr() -> None:
    """MAKEDOC: what is make_plots_clr doing?"""
    logg = logging.getLogger(f"c.{__name__}.make_plots_clr")
    # logg.setLevel("INFO")
    logg.debug("Start make_plots_clr")

    # the output folders
    plot_fol = Path("plot_results") / "att" / "clr"
    if not plot_fol.exists():
        plot_fol.mkdir(parents=True, exist_ok=True)

    info_folder = Path("info") / "attention"

    # mf
    # for mf in info_folder.iterdir()
    # if (("_lr07" in mf.name or "_lr08" in mf.name) and "_en01" in mf.name)
    info_folder_dirs = [
        mf for mf in info_folder.iterdir() if ("_lr07" in mf.name or "_lr08" in mf.name)
    ]
    # info_folder_dirs = list(info_folder.iterdir())
    logg.debug(f"len(info_folder_dirs): {len(info_folder_dirs)}")

    # si = 14
    # si = 0
    # for model_folder in info_folder_dirs[si : si + 1]:
    for model_folder in info_folder_dirs[:]:
        model_name = model_folder.name
        # if "_lr05" not in model_name and "_lr06" not in model_name:
        # if "_lr08" not in model_name:
        #     continue
        logg.debug(f"model_name: {model_name}")

        clr_recap_path = model_folder / "clr_recap.json"
        if not clr_recap_path.exists():
            logg.info(f"Skipping clr_recap: {clr_recap_path}, not found")
            continue
        clr_recap = json.loads(clr_recap_path.read_text())

        metrics_names = list(clr_recap.keys())
        metrics_names.remove("iterations")
        metrics_names_train = [mn for mn in metrics_names if "val" not in mn]
        metrics_names_val = [mn for mn in metrics_names if "val" in mn]
        logg.debug(f"metrics_names: {metrics_names}")
        logg.debug(f"train {metrics_names_train} val {metrics_names_val}")

        # fscore data for training
        prec_train = np.array(clr_recap["precision"])
        recall_train = np.array(clr_recap["recall"])
        pr_prod = 2 * prec_train * recall_train
        pr_sum = prec_train + recall_train
        fscores_train = np.zeros_like(pr_prod)
        np.divide(pr_prod, pr_sum, out=fscores_train, where=pr_sum > 0)

        # loss for train
        loss_train = np.array(clr_recap["loss"])

        # fscore data for validation
        prec_val = np.array(clr_recap["val_precision"])
        recall_val = np.array(clr_recap["val_recall"])
        pr_prod = 2 * prec_val * recall_val
        pr_sum = prec_val + recall_val
        fscores_val = np.zeros_like(pr_prod)
        np.divide(pr_prod, pr_sum, out=fscores_val, where=pr_sum > 0)
        logg.debug(f"len(fscores_val): {len(fscores_val)}")

        # loss for validation
        loss_val = np.array(clr_recap["val_loss"])
        logg.debug(f"len(loss_val): {len(loss_val)}")
        # x_val = np.arange(len(loss_val))

        # lr
        lr = clr_recap["lr"]

        # x_axis
        iterations = clr_recap["iterations"]

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

        # https://matplotlib.org/gallery/subplots_axes_and_figures/two_scales.html

        (line_fscore,) = ax[0].plot(iterations, fscores_train, color="C0")
        # ax[0].plot(iterations, loss_train, label="loss_train")
        ax[0].set_title("Loss and F-score")
        ax[0].set_xlabel("Iterations")
        ax[0].set_ylabel("F-score")
        # ax[0].legend()

        ax_loss = ax[0].twinx()
        ax_loss.set_ylabel("Loss")
        (line_loss,) = ax_loss.plot(iterations, loss_train, color="C1")
        # ax_loss.legend()

        ax[0].legend((line_fscore, line_loss), ("F-score", "loss"), loc="center right")

        ax[1].plot(iterations, lr)
        ax[1].set_title("Learning rate")
        ax[1].set_xlabel("Iterations")
        ax[1].set_ylabel("Learning rate")

        fig.tight_layout()

        fig_name = f"{model_name}"
        fig_name += "_fscore_loss_clr.{}"

        results_path = plot_fol / fig_name.format("png")
        fig.savefig(results_path)
        results_path = plot_fol / fig_name.format("pdf")
        fig.savefig(results_path)

    plt.show()


def delete_bad_models_att() -> None:
    """MAKEDOC: what is delete_bad_models_att doing?"""
    logg = logging.getLogger(f"c.{__name__}.delete_bad_models_att")
    # logg.setLevel("INFO")
    logg.debug("Start delete_bad_models_att")

    info_folder = Path("info") / "attention"
    trained_folder = Path("trained_models") / "attention"
    deleted = 0
    recreated = 0
    bad_models = 0
    good_models = 0

    for model_folder in info_folder.iterdir():
        # logg.debug(f"model_folder: {model_folder}")

        model_name = model_folder.name
        if not model_name.startswith("ATT"):
            continue

        model_name = model_folder.name
        model_path = trained_folder / f"{model_name}.h5"
        placeholder_path = trained_folder / f"{model_name}.txt"

        # load results dict
        res_recap_path = model_folder / "results_recap.json"
        if not res_recap_path.exists():
            continue
        results_recap = json.loads(res_recap_path.read_text())

        # load recap dict
        recap_path = model_folder / "recap.json"
        recap = json.loads(recap_path.read_text())

        # extract info
        words_type = recap["hypa"]["words_type"]
        fscore = results_recap["fscore"]

        if "all" in words_type:
            # TODO
            f_tresh = 0.94
        elif "w2" in words_type:
            f_tresh = 0.965
        elif "k1" in words_type:
            f_tresh = 0.972
        elif "num" in words_type:
            f_tresh = 0.975
        elif "f2" in words_type:
            f_tresh = 0.98
        else:
            logg.warn(f"Not specified f_tresh for words_type: {words_type}")
            f_tresh = 0.8

        if fscore < f_tresh:
            bad_models += 1

            if model_path.exists():
                # manually uncomment this when ready
                # model_path.unlink()
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
            logg.debug(f"Good model_path {model_path} {words_type}")
            logg.debug(f"\tfscore: {fscore}")
            good_models += 1

    logg.info(f"bad_models: {bad_models}")
    logg.info(f"good_models: {good_models}")
    logg.info(f"deleted: {deleted}")
    logg.info(f"recreated: {recreated}")


def run_evaluate_attention(args: argparse.Namespace) -> None:
    """MAKEDOC: What is evaluate_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_attention")
    logg.debug("Starting run_evaluate_attention")

    evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type
    rec_words_type = args.rec_words_type

    pd.set_option("max_colwidth", 100)
    pd.set_option("display.max_rows", None)

    if evaluation_type == "results":
        evaluate_results_attention()
    elif evaluation_type == "attention_weights":
        evaluate_attention_weights(train_words_type, rec_words_type, False)
    elif evaluation_type == "audio":
        evaluate_attention_weights(train_words_type, rec_words_type, True)
    elif evaluation_type == "make_plots_hypa":
        make_plots_hypa()
    elif evaluation_type == "make_plots_clr":
        make_plots_clr()
    elif evaluation_type == "evaluate_batch_epoch":
        evaluate_batch_epoch()
    elif evaluation_type == "delete_bad_models":
        delete_bad_models_att()
    elif evaluation_type == "compare_augmentation":
        compare_augmentation()


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_attention(args)
