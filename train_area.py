from functools import partial
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

from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

from area_model import AreaNet
from augment_data import do_augmentation
from plot_utils import plot_confusion_matrix
from preprocess_data import load_processed
from preprocess_data import preprocess_spec
from schedules import exp_decay_smooth
from schedules import exp_decay_step
from utils import analyze_confusion
from utils import pred_hot_2_cm
from utils import setup_gpus
from utils import setup_logger
from utils import words_types


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-tt",
        "--training_type",
        type=str,
        default="hypa_tune",
        choices=["hypa_tune"],
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
        "-dr", "--dry_run", action="store_true", help="Do a dry run for the hypa grid",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 train_area.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def train_model_area_dry(
    hypa: ty.Dict[str, str], use_validation: bool, trained_folder: Path
) -> str:
    """TODO: what is train_model_area_dry doing?"""
    logg = logging.getLogger(f"c.{__name__}.train_model_area_dry")
    # logg.setLevel("INFO")
    logg.debug("Start train_model_area_dry")

    model_name = build_area_name(hypa, use_validation)
    model_path = trained_folder / f"{model_name}.h5"

    if model_path.exists():
        return "already_trained"

    return "to_train"


def hyper_train_area(
    words_type: str, force_retrain: bool, use_validation: bool, dry_run: bool,
) -> None:
    """TODO: what is hyper_train_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_area")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_area")

    ##########################################################
    #   Hyper-parameters grid
    ##########################################################

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

    ###### the words to train on
    hypa_grid["words_type"] = [words_type]

    ###### the dataset to train on
    ds = []
    ds.extend(["mel04"])
    hypa_grid["dataset_name"] = ds

    ###### the learning rates for the optimizer
    lr = []
    # lr.extend(["01", "02"])  # fixed
    lr.extend(["03"])  # exp_decay_step_01
    lr.extend(["04"])  # exp_decay_smooth_01
    hypa_grid["learning_rate_type"] = lr

    ###### which optimizer to use
    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["a1"]

    ###### the batch size (the key is converted to int)
    bs = []
    bs.extend(["32"])
    hypa_grid["batch_size_type"] = bs

    ###### the number of epochs
    en = []
    en.extend(["15"])
    hypa_grid["epoch_num_type"] = en

    ###### build the combinations
    logg.debug(f"hypa_grid = {hypa_grid}")
    the_grid = list(ParameterGrid(hypa_grid))
    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    ##########################################################
    #   Setup pre train
    ##########################################################

    model_tag = "area"

    # where to save the trained models
    trained_folder = Path("trained_models") / model_tag
    if not trained_folder.exists():
        trained_folder.mkdir(parents=True, exist_ok=True)

    # where to put the info folders
    root_info_folder = Path("info") / model_tag
    if not root_info_folder.exists():
        root_info_folder.mkdir(parents=True, exist_ok=True)

    # count how many models are left to train
    if dry_run:
        tra_info = {"already_trained": 0, "to_train": 0}
        for hypa in the_grid:
            train_status = train_model_area_dry(hypa, use_validation, trained_folder)
            tra_info[train_status] += 1
        logg.debug(f"tra_info: {tra_info}")
        return

    # check that the data is available
    for dn in hypa_grid["dataset_name"]:
        for wt in hypa_grid["words_type"]:
            logg.debug(f"\nwt: {wt} dn: {dn}\n")
            if dn.startswith("mel"):
                preprocess_spec(dn, wt)
            elif dn.startswith("aug"):
                do_augmentation(dn, wt)

    ##########################################################
    #   Train all hypas
    ##########################################################

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(
                train_area,
                (hypa, force_retrain, use_validation, trained_folder, root_info_folder),
            )


def build_area_name(hypa: ty.Dict[str, str], use_validation: bool) -> str:
    """TODO: what is build_area_name doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_area_name")
    logg.setLevel("INFO")
    logg.debug("Start build_area_name")

    model_name = "ARN"
    model_name += f"_op{hypa['optimizer_type']}"
    model_name += f"_lr{hypa['learning_rate_type']}"
    model_name += f"_bs{hypa['batch_size_type']}"
    model_name += f"_en{hypa['epoch_num_type']}"

    model_name += f"_ds{hypa['dataset_name']}"
    model_name += f"_w{hypa['words_type']}"

    if not use_validation:
        model_name += "_noval"

    return model_name


def get_model_param_area(
    hypa: ty.Dict[str, str], num_labels: int, input_shape: ty.Tuple[int, int, int]
) -> ty.Dict[str, ty.Any]:
    """TODO: what is get_model_param_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_model_param_area")
    # logg.setLevel("INFO")
    logg.debug("Start get_model_param_area")

    model_param: ty.Dict[str, ty.Any] = {}

    model_param["num_classes"] = num_labels
    model_param["input_shape"] = input_shape

    return model_param


def get_training_param_area(
    hypa: ty.Dict[str, str], use_validation: bool
) -> ty.Dict[str, ty.Any]:
    """TODO: what is get_training_param_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_training_param_area")
    # logg.setLevel("INFO")
    logg.debug("Start get_training_param_area")

    # TODO extract epoch/batch/opt/lr
    training_param: ty.Dict[str, ty.Any] = {}

    training_param["callbacks"] = []

    # it seems silly but the key must be 2 char long for a consistent model name
    # batch_size_types = {"32": 32, "16": 16}
    # batch_size = batch_size_types[hypa["batch_size_type"]]
    # training_param["batch_size"] = batch_size
    training_param["batch_size"] = int(hypa["batch_size_type"])

    # epochs_types = {"15": 15, "30": 30, "02": 2, "04": 4}
    # epochs = epochs_types[hypa["epochs"]]
    # training_param["epochs"] = epochs
    training_param["epochs"] = int(hypa["epoch_num_type"])

    # translate from short key to long name
    learning_rate_types = {
        "01": "fixed01",
        "02": "fixed02",
        "03": "exp_decay_step_01",
        "04": "exp_decay_smooth_01",
    }
    learning_rate_type = hypa["learning_rate_type"]
    lr_name = learning_rate_types[learning_rate_type]
    training_param["lr_name"] = lr_name

    if lr_name.startswith("fixed"):
        if lr_name == "fixed01":
            lr = 1e-3
        elif lr_name == "fixed02":
            lr = 1e-4
    else:
        lr = 1e-3

    optimizer_types = {"a1": Adam(learning_rate=lr), "r1": RMSprop(learning_rate=lr)}
    training_param["opt"] = optimizer_types[hypa["optimizer_type"]]

    callbacks = []

    if lr_name.startswith("exp_decay"):
        if lr_name == "exp_decay_step_01":
            exp_decay_part = partial(exp_decay_step, epochs_drop=5)
        elif lr_name == "exp_decay_smooth_01":
            exp_decay_part = partial(exp_decay_smooth, epochs_drop=5)
        lrate = LearningRateScheduler(exp_decay_part)
        callbacks.append(lrate)

    if lr_name.startswith("fixed") or lr_name.startswith("exp_decay"):
        metric_to_monitor = "val_loss" if use_validation else "loss"
        early_stop = EarlyStopping(
            monitor=metric_to_monitor, patience=4, restore_best_weights=True, verbose=1,
        )
        callbacks.append(early_stop)

    training_param["callbacks"] = callbacks

    return training_param


def train_area(
    hypa, force_retrain, use_validation, trained_folder, root_info_folder
) -> None:
    """TODO: what is train_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.train_area")
    # logg.setLevel("INFO")
    logg.debug("Start train_area")

    ##########################################################
    #   Setup folders
    ##########################################################

    # name the model
    model_name = build_area_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # save the trained model here
    model_path = trained_folder / f"{model_name}.h5"
    placeholder_path = trained_folder / f"{model_name}.txt"

    # check if this model has already been trained
    if placeholder_path.exists():
        if force_retrain:
            logg.warn("\nRETRAINING MODEL!!\n")
        else:
            logg.debug("Already trained")
            return

    # save info regarding the model training in this folder
    model_info_folder = root_info_folder / model_name
    if not model_info_folder.exists():
        model_info_folder.mkdir(parents=True, exist_ok=True)

    # magic to fix the GPUs
    setup_gpus()

    ##########################################################
    #   Load data
    ##########################################################

    # get the words
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

    ##########################################################
    #   Setup model
    ##########################################################

    # the shape of each sample
    input_shape = data["training"][0].shape

    # from hypa extract model param
    model_param = get_model_param_area(hypa, num_labels, input_shape)

    # get the model with the chosen params
    model = AreaNet.build(**model_param)

    # from hypa extract training param (epochs, batch, opt, ...)
    training_param = get_training_param_area(hypa, use_validation)

    # a few metrics to track
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]

    # compile the model
    model.compile(
        optimizer=training_param["opt"],
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
    )

    # recap
    recap: ty.Dict[str, ty.Any] = {}
    recap["model_name"] = model_name
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["use_validation"] = use_validation
    recap["batch_size"] = training_param["batch_size"]
    recap["epochs"] = training_param["epochs"]
    recap["lr_name"] = training_param["lr_name"]
    recap["version"] = "001"

    # logg.debug(f"recap: {recap}")
    recap_path = model_info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    ##########################################################
    #   Fit model
    ##########################################################

    results = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=training_param["epochs"],
        batch_size=training_param["batch_size"],
        callbacks=training_param["callbacks"],
    )

    ##########################################################
    #   Save results, history, performance
    ##########################################################

    # results_recap
    results_recap: ty.Dict[str, ty.Any] = {}
    results_recap["model_name"] = model_name
    results_recap["results_recap_version"] = "001"

    # evaluate performance
    eval_testing = model.evaluate(data["testing"], labels["testing"])
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")
        results_recap[metrics_name] = value

    # confusion matrix
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, words)
    results_recap["cm"] = cm.tolist()

    # fscore
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

    # save the results
    res_recap_path = model_info_folder / "results_recap.json"
    res_recap_path.write_text(json.dumps(results_recap, indent=4))

    # plot the cm
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words, fscore)
    plot_cm_path = model_info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)
    plt.close(fig)

    # save the trained model
    model.save(model_path)

    # save the placeholder
    placeholder_path.write_text(f"Trained. F-score: {fscore}")


def run_train_area(args: argparse.Namespace) -> None:
    """TODO: What is train_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train_area")
    logg.debug("Starting run_train_area")

    training_type = args.training_type
    words_type = args.words_type
    use_validation = args.use_validation
    dry_run = args.dry_run
    force_retrain = args.force_retrain

    if training_type == "hypa_tune":
        hyper_train_area(words_type, force_retrain, use_validation, dry_run)


if __name__ == "__main__":
    args = setup_env()
    run_train_area(args)