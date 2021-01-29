from multiprocessing import Pool
from pathlib import Path
import argparse
import json
import logging

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

# from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.data import Dataset  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # type: ignore

from sklearn.model_selection import ParameterGrid  # type: ignore

from models import AttentionModel
from models import CNNmodel
from models import TRAmodel

from plot_utils import plot_confusion_matrix
from preprocess_data import compose_spec
from preprocess_data import load_processed
from preprocess_data import load_triple
from preprocess_data import preprocess_spec
from utils import analyze_confusion
from utils import pred_hot_2_cm
from utils import setup_gpus
from utils import setup_logger
from utils import words_types

from typing import Any
from typing import Dict
from typing import List
from typing import Union


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-tt",
        "--training_type",
        type=str,
        default="transfer",
        choices=["smallCNN", "transfer", "attention"],
        help="Which training to execute",
    )
    parser.add_argument(
        "-wt",
        "--words_type",
        type=str,
        default="f2",
        choices=words_types.keys(),
        help="Words to preprocess",
    )

    parser.add_argument(
        "-ft",
        "--force_retrain",
        dest="force_retrain",
        action="store_true",
        help="Force the training and overwrite the previous results",
    )

    parser.add_argument(
        "-nv",
        "--no_use_validation",
        dest="use_validation",
        action="store_false",
        help="Do not use validation data while training",
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


def build_cnn_name(hypa: Dict[str, Union[str, int]]) -> str:
    """Build the name to load a CNN model

    There is an older version of the name without lr and opt, kept for loading purposes
    """
    model_name = "CNN"
    model_name += f"_nf{hypa['base_filters']}"
    model_name += f"_ks{hypa['kernel_size_type']}"
    model_name += f"_ps{hypa['pool_size_type']}"
    model_name += f"_dw{hypa['base_dense_width']}"
    model_name += f"_dr{hypa['dropout_type']}"
    if hypa["learning_rate_type"] != "default":
        model_name += f"_lr{hypa['learning_rate_type']}"
    if hypa["optimizer_type"] != "adam":
        model_name += f"_op{hypa['optimizer_type']}"
    model_name += f"_ds{hypa['dataset']}"
    model_name += f"_bs{hypa['batch_size']}"
    model_name += f"_en{hypa['epoch_num']}"
    model_name += f"_w{hypa['words']}"
    return model_name


def hyper_train(args):
    """TODO: what is hyper_train doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train")
    logg.debug("Start hyper_train")

    words_type = args.words_type

    hypa_grid = {}
    hypa_grid["base_filters"] = [20, 32]
    hypa_grid["kernel_size_type"] = ["01", "02"]
    hypa_grid["pool_size_type"] = ["01", "02"]
    hypa_grid["base_dense_width"] = [32]
    hypa_grid["dropout_type"] = ["01", "02"]
    hypa_grid["batch_size"] = [32]
    hypa_grid["epoch_num"] = [15, 30, 60]
    hypa_grid["learning_rate_type"] = ["01", "02", "03"]
    hypa_grid["optimizer_type"] = ["a1"]
    hypa_grid["dataset"] = ["mel04"]
    hypa_grid["words"] = [words_type]
    the_grid = list(ParameterGrid(hypa_grid))

    # best params generally
    hypa_grid = {}
    hypa_grid["base_dense_width"] = [32]
    hypa_grid["base_filters"] = [20]
    hypa_grid["batch_size"] = [32]
    hypa_grid["dataset"] = ["mel01"]
    hypa_grid["dropout_type"] = ["01"]
    hypa_grid["epoch_num"] = [16]
    hypa_grid["kernel_size_type"] = ["02"]
    hypa_grid["pool_size_type"] = ["02"]
    hypa_grid["learning_rate_type"] = ["02"]
    hypa_grid["optimizer_type"] = ["a1"]
    hypa_grid["words"] = ["f2", "f1", "num", "dir", "k1", "w2", "all"]
    the_grid = list(ParameterGrid(hypa_grid))

    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    # check that the data is available
    for dn in hypa_grid["dataset"]:
        for wt in hypa_grid["words"]:
            preprocess_spec(dn, wt)

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(train_model, (hypa,))


def train_model(hypa):
    """TODO: What is train_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.train_model")
    # logg.debug("Starting train_model")

    # get the words
    words = words_types[hypa["words"]]

    # name the model
    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    # save the trained model here
    model_folder = Path("trained_models")
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / f"{model_name}.h5"
    # logg.debug(f"model_path: {model_path}")

    # check if this model has already been trained
    if model_path.exists():
        return "Already trained"

    # magic to fix the GPUs
    setup_gpus()

    # input data
    processed_path = Path("data_proc") / f"{hypa['dataset']}"
    data, labels = load_processed(processed_path, words)

    # from hypa extract model param
    model_param = {}
    model_param["num_labels"] = len(words)
    model_param["input_shape"] = data["training"][0].shape
    model_param["base_filters"] = hypa["base_filters"]
    model_param["base_dense_width"] = hypa["base_dense_width"]

    # translate types to actual values
    kernel_size_types = {"01": [(2, 2), (2, 2), (2, 2)], "02": [(5, 1), (3, 3), (3, 3)]}
    model_param["kernel_sizes"] = kernel_size_types[hypa["kernel_size_type"]]

    pool_size_types = {"01": [(2, 2), (2, 2), (2, 2)], "02": [(2, 1), (2, 2), (2, 2)]}
    model_param["pool_sizes"] = pool_size_types[hypa["pool_size_type"]]

    dropout_types = {"01": [0.03, 0.01], "02": [0.3, 0.1]}
    model_param["dropouts"] = dropout_types[hypa["dropout_type"]]

    # setup learning rates
    e1 = ExponentialDecay(0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
    learning_rate_types = {"01": 0.01, "02": 0.001, "03": 0.0001, "e1": e1}
    lr = learning_rate_types[hypa["learning_rate_type"]]

    optimizer_types = {
        "a1": Adam(learning_rate=lr),
        "r1": RMSprop(learning_rate=lr),
    }
    opt = optimizer_types[hypa["optimizer_type"]]

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

    # a dict to recreate this training
    recap = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["model_name"] = model_name
    recap["version"] = "001"
    # logg.debug(f"recap: {recap}")
    recap_path = info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    # create the model
    model = CNNmodel(**model_param)

    # model = AttRNNmodel(len(words), data["training"][0].shape)
    # model = AttentionModel(len(words), data["training"][0].shape)

    model.compile(
        optimizer=opt,
        loss=["categorical_crossentropy"],
        metrics=["categorical_accuracy"],
    )
    # model.summary()

    # get training parameters
    BATCH_SIZE = hypa["batch_size"]
    SHUFFLE_BUFFER_SIZE = BATCH_SIZE
    EPOCH_NUM = hypa["epoch_num"]

    # load the datasets
    datasets = {}
    for which in ["training", "validation", "testing"]:
        # logg.debug(f"data[{which}].shape: {data[which].shape}")
        datasets[which] = Dataset.from_tensor_slices((data[which], labels[which]))
        datasets[which] = datasets[which].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    # setup early stopping
    early_stop = EarlyStopping(
        # monitor="val_categorical_accuracy",
        monitor="val_loss",
        patience=4,
        verbose=1,
        restore_best_weights=True,
    )

    # train the model
    results = model.fit(
        data["training"],
        labels["training"],
        validation_data=datasets["validation"],
        batch_size=BATCH_SIZE,
        epochs=EPOCH_NUM,
        verbose=1,
        callbacks=[early_stop],
    )

    # save the trained model
    model.save(model_path)

    results_recap = {}
    results_recap["model_name"] = model_name

    # version of the results saved
    results_recap["results_recap_version"] = "001"

    # quickly evaluate the results
    # logg.debug(f"\nmodel.metrics_names: {model.metrics_names}")
    # for which in ["training", "validation", "testing"]:
    #     model_eval = model.evaluate(datasets[which])
    #     logg.debug(f"{which}: model_eval: {model_eval}")

    # save the evaluation results
    logg.debug("Evaluate on test data:")
    eval_testing = model.evaluate(datasets["testing"])
    results_recap[model.metrics_names[0]] = eval_testing[0]
    results_recap[model.metrics_names[1]] = eval_testing[1]

    # compute the confusion matrix
    y_pred = model.predict(datasets["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, words)
    # logg.debug(f"cm: {cm}")
    results_recap["cm"] = cm.tolist()

    # compute the fscore
    fscore = analyze_confusion(cm, words)
    logg.debug(f"fscore: {fscore}")

    # plot the cm
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words, fscore)
    plot_cm_path = info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)
    plt.close(fig)

    # save the histories
    results_recap["history"] = {
        "loss": results.history["loss"],
        "val_loss": results.history["val_loss"],
        "categorical_accuracy": results.history["categorical_accuracy"],
        "val_categorical_accuracy": results.history["val_categorical_accuracy"],
    }

    # save the results
    res_recap_path = info_folder / "results_recap.json"
    res_recap_path.write_text(json.dumps(results_recap, indent=4))

    # plt.show()
    return results_recap


def build_transfer_name(hypa: Dict[str, str], use_validation: bool) -> str:
    """TODO: what is build_transfer_name doing?"""

    model_name = "TRA"
    model_name += f"_dw{hypa['dense_width_type']}"
    model_name += f"_dr{hypa['dropout_type']}"
    model_name += f"_bs{hypa['batch_size_type']}"
    model_name += f"_en{hypa['epoch_num_type']}"
    model_name += f"_lr{hypa['learning_rate_type']}"
    model_name += f"_op{hypa['optimizer_type']}"
    model_name += f"_ds{hypa['datasets_type']}"
    model_name += f"_w{hypa['words_type']}"
    if not use_validation:
        model_name += "_noval"

    return model_name


def get_datasets_types():
    """"""
    datasets_types = {
        "01": ["mel05", "mel09", "mel10"],
        "02": ["mel05", "mel10", "mfcc07"],
        "03": ["mfcc06", "mfcc07", "mfcc08"],
        "04": ["mel05", "mfcc06", "melc1"],
        "05": ["melc1", "melc2", "melc4"],
    }
    return datasets_types


def hyper_train_transfer(args: argparse.Namespace) -> None:
    """TODO: what is hyper_train_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_transfer")

    words_type = args.words_type
    force_retrain = args.force_retrain
    use_validation = args.use_validation

    hypa_grid: Dict[str, List[str]] = {}

    # TODO test again dense_width_type 1234 on f1

    # hypa_grid["dense_width_type"] = ["01", "02", "03", "04"]
    # hypa_grid["dense_width_type"] = ["01", "02"]
    hypa_grid["dense_width_type"] = ["03"]
    # hypa_grid["dense_width_type"] = ["02"]

    # hypa_grid["dropout_type"] = ["01", "03"]
    hypa_grid["dropout_type"] = ["01"]

    # hypa_grid["batch_size_type"] = ["01", "02"]
    # hypa_grid["batch_size_type"] = ["01"]
    hypa_grid["batch_size_type"] = ["02"]

    hypa_grid["epoch_num_type"] = ["01"]
    hypa_grid["learning_rate_type"] = ["01"]

    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["r1"]

    # hypa_grid["datasets_type"] = ["01", "02", "03", "04", "05"]
    hypa_grid["datasets_type"] = ["01"]

    # hypa_grid["words_type"] = [words_type]
    hypa_grid["words_type"] = [words_type]
    # hypa_grid["words_type"] = ["f2", "f1", "dir", "num", "k1", "w2", "all"]
    # hypa_grid["words_type"] = ["f2", "f1", "dir", "num"]
    the_grid = list(ParameterGrid(hypa_grid))

    logg.debug(f"hypa_grid: {hypa_grid}")

    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    # for each type of dataset that will be used
    datasets_types = get_datasets_types()
    for dt in hypa_grid["datasets_type"]:
        # get the dataset name list
        dataset_names = datasets_types[dt]
        for dn in dataset_names:
            # and check that the data is available for each word type
            for wt in hypa_grid["words_type"]:
                logg.debug(f"\nwt: {wt} dn: {dn}\n")
                if dn.startswith("melc"):
                    compose_spec(dn, wt)
                else:
                    preprocess_spec(dn, wt)

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(train_transfer, (hypa, force_retrain, use_validation))


def train_transfer(
    hypa: Dict[str, str], force_retrain: bool, use_validation: bool
) -> None:
    """TODO: what is train_transfer doing?

    https://www.tensorflow.org/guide/keras/transfer_learning/#build_a_model
    """
    logg = logging.getLogger(f"c.{__name__}.train_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start train_transfer")

    # name the model
    model_name = build_transfer_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # save the trained model here
    model_folder = Path("trained_models")
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / f"{model_name}.h5"
    # logg.debug(f"model_path: {model_path}")

    # check if this model has already been trained
    if model_path.exists():
        if force_retrain:
            logg.warn("\nRETRAINING MODEL!!\n")
        else:
            logg.debug("Already trained")
            return

    # magic to fix the GPUs
    setup_gpus()

    # get the word list
    words = words_types[hypa["words_type"]]
    num_labels = len(words)

    # get the dataset name list
    datasets_types = get_datasets_types()
    dataset_names = datasets_types[hypa["datasets_type"]]

    # load datasets
    processed_folder = Path("data_proc")
    data_paths = [processed_folder / f"{dn}" for dn in dataset_names]
    data, labels = load_triple(data_paths, words)

    val_data = None
    if use_validation:
        x = data["training"]
        y = labels["training"]
        val_data = (data["validation"], labels["validation"])
        logg.debug("Using validation data")
        # logg.debug(f"val_data[0].shape: {val_data[0].shape}")
    else:
        logg.debug("NOT using validation data")
        # logg.debug(f"data['training'].shape: {data['training'].shape}")
        # logg.debug(f"data['validation'].shape: {data['validation'].shape}")
        # logg.debug(f"labels['training'].shape: {labels['training'].shape}")
        # logg.debug(f"labels['validation'].shape: {labels['validation'].shape}")
        x = np.concatenate((data["training"], data["validation"]))
        y = np.concatenate((labels["training"], labels["validation"]))
        # logg.debug(f"x.shape: {x.shape}")
        # logg.debug(f"y.shape: {y.shape}")

    model_param: Dict[str, Union[List[int], int, float]] = {}
    model_param["num_labels"] = num_labels
    model_param["input_shape"] = data["training"][0].shape

    dense_width_types = {"01": [4, 0], "02": [16, 16], "03": [0, 0], "04": [64, 64]}
    model_param["dense_widths"] = dense_width_types[hypa["dense_width_type"]]

    dropout_types = {"01": 0.2, "02": 0.1, "03": 0}
    model_param["dropout"] = dropout_types[hypa["dropout_type"]]

    batch_size_types = {"01": [32, 32], "02": [16, 16]}
    batch_sizes = batch_size_types[hypa["batch_size_type"]]

    epoch_num_types = {"01": [20, 10]}
    epoch_nums = epoch_num_types[hypa["epoch_num_type"]]

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

    # a dict to recreate this training
    recap: Dict[str, Any] = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["use_validation"] = use_validation
    recap["model_name"] = model_name
    recap["version"] = "002"
    # logg.debug(f"recap: {recap}")
    recap_path = info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    # get the model
    model, base_model = TRAmodel(data=data, **model_param)
    model.summary()

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]

    learning_rate_types = {"01": [1e-3, 1e-5]}
    lr = learning_rate_types[hypa["learning_rate_type"]]

    optimizer_types = {
        "a1": [Adam(learning_rate=lr[0]), Adam(learning_rate=lr[1])],
        "r1": [RMSprop(learning_rate=lr[0]), RMSprop(learning_rate=lr[1])],
    }
    opt = optimizer_types[hypa["optimizer_type"]]

    model.compile(
        optimizer=opt[0],
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
    )

    logg.debug("Start fit")
    results_freeze = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=epoch_nums[0],
        batch_size=batch_sizes[0],
        verbose=0,
    )

    results_freeze_recap: Dict[str, Any] = {}
    results_freeze_recap["model_name"] = model_name
    results_freeze_recap["results_recap_version"] = "001"
    results_freeze_recap["history_train"] = {
        mn: results_freeze.history[mn] for mn in model.metrics_names
    }
    if use_validation:
        results_freeze_recap["history_val"] = {
            f"val_{mn}": results_freeze.history[f"val_{mn}"]
            for mn in model.metrics_names
        }

    # save the results
    res_recap_path = info_folder / "results_freeze_recap.json"
    res_recap_path.write_text(json.dumps(results_freeze_recap, indent=4))

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=opt[1],  # Low learning rate
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
    )

    results_full = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=epoch_nums[1],
        batch_size=batch_sizes[1],
    )

    results_full_recap: Dict[str, Any] = {}
    results_full_recap["model_name"] = model_name
    results_full_recap["results_recap_version"] = "001"

    eval_testing = model.evaluate(data["testing"], labels["testing"])
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")
        results_full_recap[metrics_name] = value

    # compute the confusion matrix
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, words)
    # logg.debug(f"cm: {cm}")
    results_full_recap["cm"] = cm.tolist()

    # compute the fscore
    fscore = analyze_confusion(cm, words)
    logg.debug(f"fscore: {fscore}")
    results_full_recap["fscore"] = fscore

    # plot the cm
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words, fscore)
    plot_cm_path = info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)
    plt.close(fig)

    results_full_recap["history_train"] = {
        mn: results_full.history[mn] for mn in model.metrics_names
    }
    if use_validation:
        results_full_recap["history_val"] = {
            f"val_{mn}": results_full.history[f"val_{mn}"] for mn in model.metrics_names
        }

    # save the results
    res_recap_path = info_folder / "results_full_recap.json"
    res_recap_path.write_text(json.dumps(results_full_recap, indent=4))

    # save the trained model
    model.save(model_path)


def build_attention_name(hypa: Dict[str, str], use_validation: bool) -> str:
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


def hyper_train_attention(
    words_type: str, force_retrain: bool, use_validation: bool
) -> None:
    """TODO: what is hyper_train_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_attention")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_attention")

    hypa_grid: Dict[str, List[str]] = {}

    # the words to train on
    hypa_grid["words_type"] = [words_type]

    # the dataset to train on
    # hypa_grid["dataset_name"] = ["mel01", "mel04", "mel05", "mela1"]
    # hypa_grid["dataset_name"] = ["mela1"]
    # hypa_grid["dataset_name"] = ["mel04"]
    hypa_grid["dataset_name"] = ["mela1", "mel04"]

    # how big are the first conv layers
    hypa_grid["conv_size_type"] = ["01", "02"]
    # hypa_grid["conv_size_type"] = ["02"]

    # dropout after conv, 0 to skip it
    # hypa_grid["dropout_type"] = ["01", "02"]
    hypa_grid["dropout_type"] = ["01"]

    # the shape of the kernels in the conv layers
    # hypa_grid["kernel_size_type"] = ["01", "02"]
    hypa_grid["kernel_size_type"] = ["01"]

    # the dimension of the LSTM
    # hypa_grid["lstm_units_type"] = ["01", "02"]
    hypa_grid["lstm_units_type"] = ["01"]

    # the vector picked for attention
    # hypa_grid["att_sample_type"] = ["01", "02"]
    hypa_grid["att_sample_type"] = ["02"]

    # the query style type
    hypa_grid["query_style_type"] = ["01", "02", "03", "04"]
    # hypa_grid["query_style_type"] = ["04"]

    # the width of the dense layers
    # hypa_grid["dense_width_type"] = ["01", "02"]
    hypa_grid["dense_width_type"] = ["01"]

    # the learning rates for the optimizer
    hypa_grid["learning_rate_type"] = ["01"]

    # which optimizer to use
    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["a1"]

    # the batch size to use
    hypa_grid["batch_size_type"] = ["01"]

    # the number of epochs
    # hypa_grid["epoch_num_type"] = ["01", "02"]
    hypa_grid["epoch_num_type"] = ["01"]

    logg.debug(f"hypa_grid: {hypa_grid}")

    # create the grid
    the_grid = list(ParameterGrid(hypa_grid))

    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    # check that the data is available
    for dn in hypa_grid["dataset_name"]:
        preprocess_spec(dn, words_type)

    # train each combination
    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(train_attention, (hypa, force_retrain, use_validation))


def train_attention(
    hypa: Dict[str, str], force_retrain: bool, use_validation: bool
) -> None:
    """TODO: what is train_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.train_attention")
    # logg.setLevel("INFO")
    logg.debug("Start train_attention")

    # build the model name
    model_name = build_attention_name(hypa, use_validation)

    # save the trained model here
    model_folder = Path("trained_models")
    if not model_folder.exists():
        model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / f"{model_name}.h5"

    # check if this model has already been trained
    if model_path.exists():
        if force_retrain:
            logg.warn("\nRETRAINING MODEL!!\n")
        else:
            logg.debug("Already trained")
            return

    # get the word list
    words = words_types[hypa["words_type"]]
    num_labels = len(words)

    # load data
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{hypa['dataset_name']}"
    data, labels = load_processed(processed_path, words)

    # concatenate train and val for final train
    val_data = None
    if use_validation:
        x = data["training"]
        y = labels["training"]
        val_data = (data["validation"], labels["validation"])
        logg.debug("Using validation data")
    else:
        x = np.concatenate((data["training"], data["validation"]))
        y = np.concatenate((labels["training"], labels["validation"]))
        logg.debug("NOT using validation data")

    # from hypa extract model param
    model_param: Dict[str, Any] = {}
    model_param["num_labels"] = num_labels
    model_param["input_shape"] = data["training"][0].shape

    # translate types to actual values
    conv_size_types = {"01": [10, 0, 1], "02": [10, 10, 1]}
    model_param["conv_sizes"] = conv_size_types[hypa["conv_size_type"]]

    dropout_types = {"01": 0.2, "02": 0}
    model_param["dropout"] = dropout_types[hypa["dropout_type"]]

    kernel_size_types = {"01": [(5, 1), (5, 1), (5, 1)], "02": [(3, 3), (3, 3), (3, 3)]}
    model_param["kernel_sizes"] = kernel_size_types[hypa["kernel_size_type"]]

    lstm_units_types = {"01": [64, 64], "02": [64, 0]}
    model_param["lstm_units"] = lstm_units_types[hypa["lstm_units_type"]]

    att_sample_types = {"01": "last", "02": "mid"}
    model_param["att_sample"] = att_sample_types[hypa["att_sample_type"]]

    query_style_types = {
        "01": "dense01",
        "02": "conv01",
        "03": "conv02",
        "04": "conv03",
    }
    model_param["query_style"] = query_style_types[hypa["query_style_type"]]

    dense_width_types = {"01": 32, "02": 64}
    model_param["dense_width"] = dense_width_types[hypa["dense_width_type"]]

    batch_size_types = {"01": 32, "02": 16}
    batch_sizes = batch_size_types[hypa["batch_size_type"]]

    epoch_num_types = {"01": 15, "02": 30}
    epoch_nums = epoch_num_types[hypa["epoch_num_type"]]

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

    # a dict to recreate this training
    recap: Dict[str, Any] = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["use_validation"] = use_validation
    recap["model_name"] = model_name
    recap["version"] = "001"
    # logg.debug(f"recap: {recap}")
    recap_path = info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    # magic to fix the GPUs
    setup_gpus()

    model = AttentionModel(**model_param)
    model.summary()

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]

    learning_rate_types = {"01": 1e-3, "02": 1e-4}
    lr = learning_rate_types[hypa["learning_rate_type"]]

    optimizer_types = {"a1": Adam(learning_rate=lr), "r1": RMSprop(learning_rate=lr)}
    opt = optimizer_types[hypa["optimizer_type"]]

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
    )

    # setup early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    # model_checkpoint = ModelCheckpoint(
    #     model_name,
    #     monitor="val_loss",
    #     save_best_only=True,
    # )

    # callbacks = [early_stop, model_checkpoint]
    callbacks = [early_stop]

    results = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=epoch_nums,
        batch_size=batch_sizes,
        callbacks=callbacks,
    )

    results_recap: Dict[str, Any] = {}
    results_recap["model_name"] = model_name
    results_recap["results_recap_version"] = "002"

    # eval performance on the various metrics
    eval_testing = model.evaluate(data["testing"], labels["testing"])
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")
        results_recap[metrics_name] = value

    # compute the confusion matrix
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, words)
    # logg.debug(f"cm: {cm}")
    results_recap["cm"] = cm.tolist()

    # compute the fscore
    fscore = analyze_confusion(cm, words)
    logg.debug(f"fscore: {fscore}")
    results_recap["fscore"] = fscore

    # save the histories
    results_recap["history_train"] = {
        mn: results.history[mn] for mn in model.metrics_names
    }
    if use_validation:
        results_recap["history_val"] = {
            f"val_{mn}": results.history[f"val_{mn}"] for mn in model.metrics_names
        }

    # plot the cm
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words, fscore)
    plot_cm_path = info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)
    plt.close(fig)

    # save the results
    res_recap_path = info_folder / "results_recap.json"
    res_recap_path.write_text(json.dumps(results_recap, indent=4))

    # save the trained model
    model.save(model_path)


def run_train(args):
    """TODO: what is run_train doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train")
    logg.debug("Start run_train")

    training_type = args.training_type
    words_type = args.words_type
    force_retrain = args.force_retrain
    use_validation = args.use_validation

    if training_type == "smallCNN":
        hyper_train(args)
    elif training_type == "transfer":
        hyper_train_transfer(args)
    elif training_type == "attention":
        hyper_train_attention(words_type, force_retrain, use_validation)


if __name__ == "__main__":
    args = setup_env()
    run_train(args)
