from multiprocessing import Pool
from pathlib import Path
from sklearn.model_selection import ParameterGrid  # type: ignore
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import typing as ty

# from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

from models import TRAmodel
from plot_utils import plot_confusion_matrix
from preprocess_data import compose_spec
from preprocess_data import load_triple
from preprocess_data import preprocess_spec
from utils import analyze_confusion
from utils import pred_hot_2_cm
from utils import setup_gpus
from utils import setup_logger
from utils import words_types


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

    tra_types = [w for w in words_types.keys() if not w.startswith("_")]
    parser.add_argument(
        "-wt",
        "--words_type",
        type=str,
        default="f2",
        choices=tra_types,
        help="Words to preprocess",
    )

    parser.add_argument(
        "-lrf",
        "--do_find_best_lr",
        dest="do_find_best_lr",
        action="store_true",
        help="Find the best values for the learning rate",
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

    parser.add_argument(
        "-dr",
        "--dry_run",
        action="store_true",
        help="Do a dry run for the hypa grid",
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


def build_transfer_name(hypa: ty.Dict[str, str], use_validation: bool) -> str:
    """TODO: what is build_transfer_name doing?"""

    model_name = "TB4"
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


def hyper_train_transfer(
    words_type: str, force_retrain: bool, use_validation: bool, dry_run: bool
) -> None:
    """TODO: what is hyper_train_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_transfer")

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

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

    # hypa_grid["epoch_num_type"] = ["01", "02"]
    # hypa_grid["epoch_num_type"] = ["01"]
    hypa_grid["epoch_num_type"] = ["02"]

    hypa_grid["learning_rate_type"] = ["01"]

    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["a1"]

    # hypa_grid["datasets_type"] = ["01", "02", "03", "04", "05"]
    hypa_grid["datasets_type"] = ["01"]

    # hypa_grid["words_type"] = [words_type]
    # hypa_grid["words_type"] = [words_type]
    # hypa_grid["words_type"] = ["f2", "f1", "dir", "num", "k1", "w2", "all"]
    # hypa_grid["words_type"] = ["f2", "f1", "dir", "num"]
    hypa_grid["words_type"] = ["f1"]
    the_grid = list(ParameterGrid(hypa_grid))

    logg.debug(f"hypa_grid: {hypa_grid}")

    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    if dry_run:
        tra_info = {"already_trained": 0, "to_train": 0}
        for hypa in the_grid:
            train_status = train_model_tra_dry(hypa, use_validation)
            tra_info[train_status] += 1
        logg.debug(f"tra_info: {tra_info}")
        return

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


def train_model_tra_dry(hypa, use_validation: bool) -> str:
    """TODO: what is train_model_tra_dry doing?"""
    model_name = build_transfer_name(hypa, use_validation)
    model_folder = Path("trained_models")
    model_path = model_folder / f"{model_name}.h5"

    if model_path.exists():
        return "already_trained"

    return "to_train"


def train_transfer(
    hypa: ty.Dict[str, str], force_retrain: bool, use_validation: bool
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

    model_param: ty.Dict[str, ty.Union[ty.List[int], int, float]] = {}
    model_param["num_labels"] = num_labels
    model_param["input_shape"] = data["training"][0].shape

    dense_width_types = {"01": [4, 0], "02": [16, 16], "03": [0, 0], "04": [64, 64]}
    model_param["dense_widths"] = dense_width_types[hypa["dense_width_type"]]

    dropout_types = {"01": 0.2, "02": 0.1, "03": 0}
    model_param["dropout"] = dropout_types[hypa["dropout_type"]]

    batch_size_types = {"01": [32, 32], "02": [16, 16]}
    batch_sizes = batch_size_types[hypa["batch_size_type"]]

    epoch_num_types = {"01": [20, 10], "02": [40, 20]}
    epoch_nums = epoch_num_types[hypa["epoch_num_type"]]

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

    # a dict to recreate this training
    recap: ty.Dict[str, ty.Any] = {}
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

    # setup callbacks
    callbacks = []

    metric_to_monitor = "val_loss" if use_validation else "loss"
    early_stop = EarlyStopping(
        monitor=metric_to_monitor,
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stop)

    results_freeze = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=epoch_nums[0],
        batch_size=batch_sizes[0],
        callbacks=[early_stop],
    )

    results_freeze_recap: ty.Dict[str, ty.Any] = {}
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
        callbacks=[early_stop],
    )

    results_full_recap: ty.Dict[str, ty.Any] = {}
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


def run_train_transfer(args: argparse.Namespace) -> None:
    """TODO: What is train_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train_transfer")
    logg.debug("Starting run_train_transfer")

    training_type = args.training_type
    words_type = args.words_type
    force_retrain = args.force_retrain
    use_validation = args.use_validation
    dry_run = args.dry_run
    # do_find_best_lr = args.do_find_best_lr

    if training_type == "transfer":
        hyper_train_transfer(words_type, force_retrain, use_validation, dry_run)


if __name__ == "__main__":
    args = setup_env()
    run_train_transfer(args)
