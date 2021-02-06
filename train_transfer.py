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

# from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

from models import TRAmodel
from plot_utils import plot_confusion_matrix
from preprocess_data import compose_spec
from preprocess_data import load_triple
from preprocess_data import preprocess_spec
from schedules import exp_decay_smooth
from schedules import exp_decay_step
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
        choices=["transfer"],
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
        "-dr", "--dry_run", action="store_true", help="Do a dry run for the hypa grid",
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

    # model_name = "TB4"
    # model_name = "TB7"
    model_name = hypa["net_type"]
    model_name += f"_dw{hypa['dense_width_type']}"
    model_name += f"_dr{hypa['dropout_type']}"
    model_name += f"_bs{hypa['batch_size_type']}"
    model_name += f"_en{hypa['epoch_nums_type']}"
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

    ##########################################################
    #   Hyper-parameters grid
    ##########################################################

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

    # TODO test again dense_width_type 1234 on f1

    ###### the architecture to train on
    arch = []
    # arch.append("TRA")  # Xception
    arch.append("TD1")  # DenseNet121
    # arch.append("TB4")  # EfficientNetB4
    # arch.append("TB7")  # EfficientNetB7
    hypa_grid["net_type"] = arch

    ###### the dataset to train on
    # hypa_grid["datasets_type"] = ["01", "02", "03", "04", "05"]
    hypa_grid["datasets_type"] = ["01"]

    ###### the words to train on
    hypa_grid["words_type"] = [words_type]
    # hypa_grid["words_type"] = ["k1", "w2", "all"]
    # hypa_grid["words_type"] = ["f2", "f1", "dir", "num"]

    ###### the dense width of the classifier
    dw = []
    # dw.extend(["01"])  # [4, 0]
    # dw.extend(["02"])  # [16, 16]
    dw.extend(["03"])  # [0, 0]
    # dw.extend(["04"])  # [64, 64]
    hypa_grid["dense_width_type"] = dw

    ###### the dropout to use
    dr = []
    dr.extend(["01"])  # 0.2
    # dr.extend(["02"])  # 0.1
    # dr.extend(["03"])  # 0
    hypa_grid["dropout_type"] = dr

    ###### the batch sizes to use
    bs = []
    # bs.extend(["01"])  # [32, 32]
    bs.extend(["02"])  # [16, 16]
    hypa_grid["batch_size_type"] = bs

    ###### the number of epochs
    en = []
    # en.extend(["01"])  # [20, 10]
    en.extend(["02"])  # [40, 20]
    hypa_grid["epoch_nums_type"] = en

    ###### the learning rates for the optimizer
    hypa_grid["learning_rate_type"] = ["01"]

    ###### which optimizer to use
    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["a1"]

    ###### build the combinations
    logg.debug(f"hypa_grid: {hypa_grid}")
    the_grid = list(ParameterGrid(hypa_grid))
    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    ##########################################################
    #   Setup pre train
    ##########################################################

    train_type_tag = "transfer"

    # where to save the trained models
    trained_folder = Path("trained_models") / train_type_tag
    if not trained_folder.exists():
        trained_folder.mkdir(parents=True, exist_ok=True)

    # where to put the info folders
    root_info_folder = Path("info") / train_type_tag
    if not root_info_folder.exists():
        root_info_folder.mkdir(parents=True, exist_ok=True)

    # count how many models are left to train
    if dry_run:
        tra_info = {"already_trained": 0, "to_train": 0}
        for hypa in the_grid:
            train_status = train_model_tra_dry(hypa, use_validation, trained_folder)
            tra_info[train_status] += 1
        logg.debug(f"tra_info: {tra_info}")
        return

    # check that the data is available
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

    ##########################################################
    #   Train all hypas
    ##########################################################

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(
                train_transfer,
                (hypa, force_retrain, use_validation, trained_folder, root_info_folder),
            )


def train_model_tra_dry(hypa, use_validation: bool, trained_folder: Path) -> str:
    """TODO: what is train_model_tra_dry doing?"""
    model_name = build_transfer_name(hypa, use_validation)
    # model_folder = Path("trained_models") / "transfer"
    model_path = trained_folder / f"{model_name}.h5"

    if model_path.exists():
        return "already_trained"

    return "to_train"


def get_model_param_transfer(
    hypa: ty.Dict[str, str], num_labels: int, input_shape: ty.Tuple[int, int, int]
) -> ty.Dict[str, ty.Any]:
    """TODO: what is get_model_param_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_model_param_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start get_model_param_transfer")

    model_param: ty.Dict[str, ty.Any] = {}
    model_param["num_labels"] = num_labels
    model_param["input_shape"] = input_shape

    model_param["net_type"] = hypa["net_type"]

    dense_width_types = {"01": [4, 0], "02": [16, 16], "03": [0, 0], "04": [64, 64]}
    model_param["dense_widths"] = dense_width_types[hypa["dense_width_type"]]

    dropout_types = {"01": 0.2, "02": 0.1, "03": 0}
    model_param["dropout"] = dropout_types[hypa["dropout_type"]]

    return model_param


def get_training_param_transfer(
    hypa: ty.Dict[str, str], use_validation: bool
) -> ty.Dict[str, ty.Any]:
    """TODO: what is get_training_param_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_training_param_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start get_training_param_transfer")

    training_param: ty.Dict[str, ty.Any] = {}

    batch_size_types = {"01": [32, 32], "02": [16, 16]}
    batch_sizes = batch_size_types[hypa["batch_size_type"]]
    training_param["batch_sizes"] = batch_sizes

    epoch_nums_types = {"01": [20, 10], "02": [40, 20]}
    epoch_nums = epoch_nums_types[hypa["epoch_nums_type"]]
    training_param["epoch_nums"] = epoch_nums

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
            lr = [1e-3, 1e-5]
        elif lr_name == "fixed02":
            lr = [5e-4, 5e-6]
    else:
        lr = [1e-3, 1e-5]

    optimizer_types = {
        "a1": [Adam(learning_rate=lr[0]), Adam(learning_rate=lr[1])],
        "r1": [RMSprop(learning_rate=lr[0]), RMSprop(learning_rate=lr[1])],
    }
    training_param["opt"] = optimizer_types[hypa["optimizer_type"]]

    ###### setup callbacks
    callbacks: ty.List[ty.List[tf.keras.callbacks]] = [[], []]

    if lr_name.startswith("exp_decay"):
        if lr_name == "exp_decay_step_01":
            exp_decay_part_frozen = partial(exp_decay_step, epochs_drop=5)
            exp_decay_part_fine = partial(
                exp_decay_step, epochs_drop=5, initial_lrate=1e-5, min_lrate=1e-6
            )
        elif lr_name == "exp_decay_smooth_01":
            exp_decay_part_frozen = partial(exp_decay_smooth, epochs_drop=5)
            exp_decay_part_fine = partial(
                exp_decay_smooth, epochs_drop=5, initial_lrate=1e-5, min_lrate=1e-6
            )
        lrate = LearningRateScheduler(exp_decay_part_frozen)
        callbacks.append(lrate)
        lrate = LearningRateScheduler(exp_decay_part_fine)
        callbacks.append(lrate)

    # add early stop to learning_rate_types where it makes sense
    if lr_name.startswith("fixed") or lr_name.startswith("exp_decay"):
        metric_to_monitor = "val_loss" if use_validation else "loss"
        early_stop = EarlyStopping(
            monitor=metric_to_monitor, patience=6, restore_best_weights=True, verbose=1
        )
        callbacks[0].append(early_stop)
        callbacks[1].append(early_stop)

    training_param["callbacks"] = callbacks

    ###### a few metrics to track
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]
    training_param["metrics"] = [metrics, metrics]

    return training_param


def train_transfer(
    hypa: ty.Dict[str, str],
    force_retrain: bool,
    use_validation: bool,
    trained_folder: Path,
    root_info_folder: Path,
) -> None:
    """TODO: what is train_transfer doing?

    https://www.tensorflow.org/guide/keras/transfer_learning/#build_a_model
    """
    logg = logging.getLogger(f"c.{__name__}.train_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start train_transfer")

    ##########################################################
    #   Setup folders
    ##########################################################

    # name the model
    model_name = build_transfer_name(hypa, use_validation)
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
    model_param = get_model_param_transfer(hypa, num_labels, input_shape)

    # get the model
    model, base_model = TRAmodel(data=data, **model_param)
    model.summary()

    # from hypa extract training param (epochs, batch, opt, ...)
    training_param = get_training_param_transfer(hypa, use_validation)

    # a dict to recreate this training
    recap: ty.Dict[str, ty.Any] = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["use_validation"] = use_validation
    recap["model_name"] = model_name
    recap["batch_sizes"] = training_param["batch_sizes"]
    recap["epoch_nums"] = training_param["epoch_nums"]
    recap["version"] = "003"

    # logg.debug(f"recap: {recap}")
    recap_path = model_info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    ##########################################################
    #   Compile and fit model the first time
    ##########################################################

    model.compile(
        optimizer=training_param["opt"][0],
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=training_param["metrics"][0],
    )

    results_freeze = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=training_param["epoch_nums"][0],
        batch_size=training_param["batch_sizes"][0],
        callbacks=training_param["callbacks"][0],
    )

    ##########################################################
    #   Save results, history, performance
    ##########################################################

    # results_freeze_recap
    results_freeze_recap: ty.Dict[str, ty.Any] = {}
    results_freeze_recap["model_name"] = model_name
    results_freeze_recap["results_recap_version"] = "001"

    # save the histories
    results_freeze_recap["history_train"] = {
        mn: results_freeze.history[mn] for mn in model.metrics_names
    }
    if use_validation:
        results_freeze_recap["history_val"] = {
            f"val_{mn}": results_freeze.history[f"val_{mn}"]
            for mn in model.metrics_names
        }

    # save the results
    res_recap_path = model_info_folder / "results_freeze_recap.json"
    res_recap_path.write_text(json.dumps(results_freeze_recap, indent=4))

    ##########################################################
    #   Compile and fit model the second time
    ##########################################################

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=training_param["opt"][1],  # Low learning rate
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=training_param["metrics"][1],
    )

    results_full = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=training_param["epoch_nums"][1],
        batch_size=training_param["batch_sizes"][1],
        callbacks=training_param["callbacks"][1],
    )

    ##########################################################
    #   Save results, history, performance
    ##########################################################

    results_full_recap: ty.Dict[str, ty.Any] = {}
    results_full_recap["model_name"] = model_name
    results_full_recap["results_recap_version"] = "001"

    # evaluate performance
    eval_testing = model.evaluate(data["testing"], labels["testing"])
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")
        results_full_recap[metrics_name] = value

    # compute the confusion matrix
    y_pred = model.predict(data["testing"])
    cm = pred_hot_2_cm(labels["testing"], y_pred, words)
    results_full_recap["cm"] = cm.tolist()

    # compute the fscore
    fscore = analyze_confusion(cm, words)
    logg.debug(f"fscore: {fscore}")
    results_full_recap["fscore"] = fscore

    # plot the cm
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words, fscore)
    plot_cm_path = model_info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)
    plt.close(fig)

    # save the histories
    results_full_recap["history_train"] = {
        mn: results_full.history[mn] for mn in model.metrics_names
    }
    if use_validation:
        results_full_recap["history_val"] = {
            f"val_{mn}": results_full.history[f"val_{mn}"] for mn in model.metrics_names
        }

    # save the results
    res_recap_path = model_info_folder / "results_full_recap.json"
    res_recap_path.write_text(json.dumps(results_full_recap, indent=4))

    # save the trained model
    model.save(model_path)

    # save the placeholder
    placeholder_path.write_text(f"Trained. F-score: {fscore}")


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
