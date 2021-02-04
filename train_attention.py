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

from augment_data import do_augmentation
from clr_callback import CyclicLR
from models import AttentionModel
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
from lr_finder import LearningRateFinder


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


def build_attention_name(hypa: ty.Dict[str, str], use_validation: bool) -> str:
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


def hyper_train_attention(
    words_type: str,
    force_retrain: bool,
    use_validation: bool,
    dry_run: bool,
    do_find_best_lr: bool,
) -> None:
    """TODO: what is hyper_train_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_attention")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_attention")

    # TODO do them with more epochs (at least on those with early_stop)
    # TODO augment on 422 @ 4,15 epochs

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

    ###### the words to train on
    hypa_grid["words_type"] = [words_type]

    ###### the dataset to train on
    ds = []
    # ds.extend(["mfcc01", "mfcc03", "mfcc04", "mfcc08"])
    # ds.extend(["mfcc04"])
    # ds.extend(["mel01", "mel04", "mel05", "mela1"])
    # ds.extend(["aug02", "aug03", "aug04", "aug05"])
    # ds.extend(["aug06", "aug07", "aug08", "aug09"])
    # ds.extend(["aug10", "aug11", "aug12", "aug13"])
    ds.extend(["aug14", "aug15", "aug16", "aug17"])
    # ds.extend(["aug14"])
    # hypa_grid["dataset_name"] = ["mela1"]
    # hypa_grid["dataset_name"] = ["mel04"]
    # hypa_grid["dataset_name"] = ["aug01"]
    # hypa_grid["dataset_name"] = ["aug01", "mela1", "mel04"]
    # hypa_grid["dataset_name"] = ["mela1", "mel04"]
    hypa_grid["dataset_name"] = ds

    ###### how big are the first conv layers
    # hypa_grid["conv_size_type"] = ["01", "02"]
    hypa_grid["conv_size_type"] = ["02"]

    ###### dropout after conv, 0 to skip it
    # hypa_grid["dropout_type"] = ["01", "02"]
    hypa_grid["dropout_type"] = ["01"]

    ###### the shape of the kernels in the conv layers
    # hypa_grid["kernel_size_type"] = ["01", "02"]
    hypa_grid["kernel_size_type"] = ["01"]

    ###### the dimension of the LSTM
    # hypa_grid["lstm_units_type"] = ["01", "02"]
    hypa_grid["lstm_units_type"] = ["01"]

    ###### the query style type
    qs = []
    # hypa_grid["query_style_type"] = ["01", "02", "03", "04", "05"]
    qs.extend(["01"])  # dense01
    qs.extend(["03"])  # conv02 (LSTM)
    qs.extend(["04"])  # conv03 (inputs)
    qs.extend(["05"])  # dense02
    hypa_grid["query_style_type"] = qs

    ###### the width of the dense layers
    # hypa_grid["dense_width_type"] = ["01", "02"]
    hypa_grid["dense_width_type"] = ["01"]

    ###### the learning rates for the optimizer
    lr = []
    # lr.extend(["01", "02"])  # fixed
    lr.extend(["03"])  # exp_decay_step_01
    lr.extend(["04"])  # exp_decay_smooth_01
    # lr.extend(["07"])  # clr_triangular2_03
    # lr.extend(["09"])  # clr_triangular2_05
    lr.extend(["10"])  # exp_decay_smooth_02
    hypa_grid["learning_rate_type"] = lr

    ###### which optimizer to use
    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["a1"]

    ###### the batch size to use
    # hypa_grid["batch_size_type"] = ["01", "02"]
    hypa_grid["batch_size_type"] = ["02"]

    ###### the number of epochs
    en = []
    en.extend(["01"])  # 15
    # en.extend(["02"]) # 30
    # en.extend(["03", "04"])
    hypa_grid["epoch_num_type"] = en

    # the grid you are generating from (useful to recreate the training)
    logg.debug(f"hypa_grid: {hypa_grid}")

    # create the list of combinations
    the_grid = list(ParameterGrid(hypa_grid))
    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    # check which models need to be trained
    if dry_run:
        tra_info = {"already_trained": 0, "to_train": 0}
        for hypa in the_grid:
            train_status = train_model_att_dry(hypa, use_validation)
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

    # useful to pick good values of lr
    if do_find_best_lr:
        hypa = the_grid[0]
        logg.debug(f"\nSTARTING find_best_lr with hypa: {hypa}")
        find_best_lr(hypa)
        return

    # train each combination
    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(train_attention, (hypa, force_retrain, use_validation))


def get_model_param_attention(
    hypa: ty.Dict[str, str], num_labels: int, input_shape: np.ndarray
) -> ty.Dict[str, ty.Any]:
    """TODO: what is get_model_param_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_model_param_attention")
    # logg.setLevel("INFO")
    logg.debug("Start get_model_param_attention")

    model_param: ty.Dict[str, ty.Any] = {}
    model_param["num_labels"] = num_labels
    # model_param["input_shape"] = data["training"][0].shape
    model_param["input_shape"] = input_shape

    # translate types to actual values
    conv_size_types = {"01": [10, 0, 1], "02": [10, 10, 1]}
    model_param["conv_sizes"] = conv_size_types[hypa["conv_size_type"]]

    dropout_types = {"01": 0.2, "02": 0}
    model_param["dropout"] = dropout_types[hypa["dropout_type"]]

    kernel_size_types = {"01": [(5, 1), (5, 1), (5, 1)], "02": [(3, 3), (3, 3), (3, 3)]}
    model_param["kernel_sizes"] = kernel_size_types[hypa["kernel_size_type"]]

    lstm_units_types = {"01": [64, 64], "02": [64, 0]}
    model_param["lstm_units"] = lstm_units_types[hypa["lstm_units_type"]]

    query_style_types = {
        "01": "dense01",
        "02": "conv01",
        "03": "conv02",
        "04": "conv03",
        "05": "dense02",
    }
    model_param["query_style"] = query_style_types[hypa["query_style_type"]]

    dense_width_types = {"01": 32, "02": 64}
    model_param["dense_width"] = dense_width_types[hypa["dense_width_type"]]

    return model_param


def train_model_att_dry(hypa, use_validation: bool) -> str:
    """TODO: what is train_model_att_dry doing?"""
    model_folder = Path("trained_models")

    model_name = build_attention_name(hypa, use_validation)
    model_path = model_folder / f"{model_name}.h5"

    if model_path.exists():
        return "already_trained"

    return "to_train"


def train_attention(
    hypa: ty.Dict[str, str], force_retrain: bool, use_validation: bool
) -> None:
    """TODO: what is train_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.train_attention")
    # logg.setLevel("INFO")
    logg.debug("Start train_attention")

    # build the model name
    model_name = build_attention_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

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

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

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

    # the shape of each sample
    input_shape = data["training"][0].shape

    # from hypa extract model param
    model_param = get_model_param_attention(hypa, num_labels, input_shape)

    batch_size_types = {"01": 32, "02": 16}
    batch_size = batch_size_types[hypa["batch_size_type"]]

    epoch_num_types = {"01": 15, "02": 30, "03": 2, "04": 4}
    epoch_num = epoch_num_types[hypa["epoch_num_type"]]

    # magic to fix the GPUs
    setup_gpus()

    model = AttentionModel(**model_param)
    # model.summary()

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]

    learning_rate_types = {
        "01": "fixed01",
        "02": "fixed02",
        "03": "exp_decay_step_01",
        "04": "exp_decay_smooth_01",
        "05": "clr_triangular2_01",
        "06": "clr_triangular2_02",
        "07": "clr_triangular2_03",
        "08": "clr_triangular2_04",
        "09": "clr_triangular2_05",
        "10": "exp_decay_smooth_02",
    }
    learning_rate_type = hypa["learning_rate_type"]
    lr_value = learning_rate_types[learning_rate_type]

    # setup opt fixed lr values
    if lr_value.startswith("fixed"):
        if lr_value == "fixed01":
            lr = 1e-3
        elif lr_value == "fixed02":
            lr = 1e-4
    else:
        lr = 1e-3

    optimizer_types = {"a1": Adam(learning_rate=lr), "r1": RMSprop(learning_rate=lr)}
    opt = optimizer_types[hypa["optimizer_type"]]

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
    )

    # setup callbacks
    callbacks = []

    # setup exp decay step / smooth
    if lr_value.startswith("exp_decay"):
        if lr_value == "exp_decay_step_01":
            exp_decay_part = partial(exp_decay_step, epochs_drop=5)
        elif lr_value == "exp_decay_smooth_01":
            exp_decay_part = partial(exp_decay_smooth, epochs_drop=5)
        elif lr_value == "exp_decay_smooth_02":
            exp_decay_part = partial(
                exp_decay_smooth, epochs_drop=5, initial_lrate=1e-2
            )
        lrate = LearningRateScheduler(exp_decay_part)
        callbacks.append(lrate)

    # setup cyclic learning rate
    if lr_value.startswith("clr_triangular2"):
        base_lr = 1e-5
        max_lr = 1e-3

        # training iteration per epoch = num samples // batch size
        # step size suggested = 2~8 * iterations
        if lr_value == "clr_triangular2_01":
            step_factor = 8
            step_size = step_factor * x.shape[0] // batch_size

        elif lr_value == "clr_triangular2_02":
            step_factor = 2
            step_size = step_factor * x.shape[0] // batch_size

        # target_cycles = the number of cycles we want in those epochs
        # it_per_epoch = num_samples // batch_size
        # total_iterations = it_per_epoch * epoch_num
        # step_size = total_iterations // target_cycles
        elif lr_value == "clr_triangular2_03":
            # the number of cycles we want in those epochs
            target_cycles = 4
            it_per_epoch = x.shape[0] // batch_size
            total_iterations = it_per_epoch * epoch_num
            step_size = total_iterations // (target_cycles * 2)

        elif lr_value == "clr_triangular2_04":
            # the number of cycles we want in those epochs
            target_cycles = 2
            it_per_epoch = x.shape[0] // batch_size
            total_iterations = it_per_epoch * epoch_num
            step_size = total_iterations // (target_cycles * 2)

        elif lr_value == "clr_triangular2_05":
            # the number of cycles we want in those epochs
            target_cycles = 2
            it_per_epoch = x.shape[0] // batch_size
            total_iterations = it_per_epoch * epoch_num
            step_size = total_iterations // (target_cycles * 2)
            # set bigger starting value
            max_lr = 1e-2

        logg.debug(f"x.shape[0]: {x.shape[0]}")
        logg.debug(f"CLR is using step_size: {step_size}")

        mode = "triangular2"
        cyclic_lr = CyclicLR(base_lr, max_lr, step_size, mode)
        callbacks.append(cyclic_lr)

    # setup early stopping
    if learning_rate_type in ["01", "02", "03", "04"]:
        metric_to_monitor = "val_loss" if use_validation else "loss"
        early_stop = EarlyStopping(
            monitor=metric_to_monitor,
            patience=4,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)

    # model_checkpoint = ModelCheckpoint(
    #     model_name,
    #     monitor="val_loss",
    #     save_best_only=True,
    # )

    # a dict to recreate this training
    # FIXME this should be right before fit and have epoch_num/batch_size/lr info
    recap: ty.Dict[str, ty.Any] = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["use_validation"] = use_validation
    recap["model_name"] = model_name
    recap["version"] = "001"
    # logg.debug(f"recap: {recap}")
    recap_path = info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    results = model.fit(
        x,
        y,
        validation_data=val_data,
        epochs=epoch_num,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    results_recap: ty.Dict[str, ty.Any] = {}
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

    # if cyclic_lr was used save the history
    if lr_value.startswith("clr_triangular2"):
        logg.debug(f"cyclic_lr.history.keys(): {cyclic_lr.history.keys()}")
        clr_recap = {}
        for metric_name, values in cyclic_lr.history.items():
            clr_recap[metric_name] = list(float(v) for v in values)
        clr_recap_path = info_folder / "clr_recap.json"
        clr_recap_path.write_text(json.dumps(clr_recap, indent=4))

    # save the trained model
    model.save(model_path)

    logg.debug(f"results.history.keys(): {results.history.keys()}")


def find_best_lr(hypa: ty.Dict[str, str]) -> None:
    """TODO: what is find_best_lr doing?"""
    logg = logging.getLogger(f"c.{__name__}.find_best_lr")
    # logg.setLevel("INFO")
    logg.debug("Start find_best_lr")

    # get the word list
    words = words_types[hypa["words_type"]]
    num_labels = len(words)

    # load data
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{hypa['dataset_name']}"
    data, labels = load_processed(processed_path, words)

    # no need for validation
    x = np.concatenate((data["training"], data["validation"]))
    y = np.concatenate((labels["training"], labels["validation"]))

    # the shape of each sample
    input_shape = data["training"][0].shape

    # from hypa extract model param
    model_param = get_model_param_attention(hypa, num_labels, input_shape)

    # magic to fix the GPUs
    setup_gpus()

    model = AttentionModel(**model_param)
    # model.summary()

    start_lr = 1e-9
    end_lr = 1e1

    batch_size_types = {"01": 32, "02": 16}
    batch_size = batch_size_types[hypa["batch_size_type"]]

    epoch_num_types = {"01": 15, "02": 30, "03": 2}
    epoch_num = epoch_num_types[hypa["epoch_num_type"]]

    optimizer_types = {"a1": Adam(), "r1": RMSprop()}
    opt = optimizer_types[hypa["optimizer_type"]]

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
    )

    # find the best values
    lrf = LearningRateFinder(model)
    lrf.find((x, y), start_lr, end_lr, epochs=epoch_num, batchSize=batch_size)

    model_name = build_attention_name(hypa, False)

    fig_title = "LR_sweep"
    fig_title += f"_bs{batch_size}"
    fig_title += f"_en{epoch_num}"
    fig_title += f"__{model_name}"
    fig, ax = plt.subplots(figsize=(8, 8))

    # get the plot
    lrf.plot_loss(ax=ax, title=fig_title)

    # save the plot
    plot_fol = Path("plot_results") / "att" / "find_best_lr"
    if not plot_fol.exists():
        plot_fol.mkdir(parents=True, exist_ok=True)
    fig_name = fig_title + ".{}"
    fig.savefig(plot_fol / fig_name.format("png"))
    fig.savefig(plot_fol / fig_name.format("pdf"))

    plt.show()


def run_train_attention(args: argparse.Namespace) -> None:
    """TODO: What is train_attention doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train_attention")
    logg.debug("Starting run_train_attention")

    training_type = args.training_type
    words_type = args.words_type
    force_retrain = args.force_retrain
    use_validation = args.use_validation
    dry_run = args.dry_run
    do_find_best_lr = args.do_find_best_lr

    if training_type == "attention":
        hyper_train_attention(
            words_type, force_retrain, use_validation, dry_run, do_find_best_lr
        )


if __name__ == "__main__":
    args = setup_env()
    run_train_attention(args)
