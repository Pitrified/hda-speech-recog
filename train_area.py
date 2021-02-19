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
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

from area_model import ActualAreaNet
from area_model import AreaNet
from area_model import SimpleNet
from area_model import VerticalAreaNet
from augment_data import do_augmentation
from clr_callback import CyclicLR
from lr_finder import LearningRateFinder
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
        "-dr",
        "--dry_run",
        action="store_true",
        help="Do a dry run for the hypa grid",
    )

    parser.add_argument(
        "-lrf",
        "--do_find_best_lr",
        dest="do_find_best_lr",
        action="store_true",
        help="Find the best values for the learning rate",
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
    """MAKEDOC: what is train_model_area_dry doing?"""
    # logg = logging.getLogger(f"c.{__name__}.train_model_area_dry")
    # logg.setLevel("INFO")
    # logg.debug("Start train_model_area_dry")

    model_name = build_area_name(hypa, use_validation)
    placeholder_path = trained_folder / f"{model_name}.txt"

    if placeholder_path.exists():
        return "already_trained"

    return "to_train"


def hyper_train_area(
    words_type: str,
    force_retrain: bool,
    use_validation: bool,
    dry_run: bool,
    do_find_best_lr: bool,
) -> None:
    """MAKEDOC: what is hyper_train_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_area")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_area")

    ##########################################################
    #   Hyper-parameters grid
    ##########################################################

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

    ###### the net type
    nt = []
    # nt.append("SIM")
    nt.append("SI2")
    nt.append("AAN")
    nt.append("VAN")
    hypa_grid["net_type"] = nt

    ###### the words to train on
    # hypa_grid["words_type"] = [words_type]
    # hypa_grid["words_type"] = ["LTnum", "LTall", "yn"]
    # hypa_grid["words_type"] = ["LTnum", "yn"]
    # hypa_grid["words_type"] = ["LTnum", "yn", "f1", "k1"]
    # hypa_grid["words_type"] = ["LTBnum", "LTBall"]
    # hypa_grid["words_type"] = ["LTBnumLS", "LTBallLS"]
    hypa_grid["words_type"] = ["lr"]

    ###### the dataset to train on
    ds = []

    # TODO VAN on LTall
    # TODO AAN/SIM/VAN on LTnum
    # TODO AAN on LTnum for all datasets, only one lr
    ds.extend(["mel04"])
    ds.extend(["mela1"])
    ds.extend(["aug07"])
    ds.extend(["aug14"])
    ds.extend(["aug15"])

    # TODO auL6789 auL18901 on all net_type
    # TODO auA5678 on VAN (on LTnumLS)
    # TODO auA5678 with lr04 (on LTnumLS to complete lr03 is done)
    # MAYBE start removing ARN, too much time
    # ds.extend(["auA01", "auA02", "auA03", "auA04"])
    # ds.extend(["auA05", "auA06", "auA07", "auA08"])
    # ds.extend(["auA04"])
    # ds.extend(["mel04L"])

    # TODO just the 3 best per architecture on noval

    hypa_grid["dataset_name"] = ds

    ###### the learning rates for the optimizer
    lr = []
    # lr.extend(["01", "02"])  # fixed
    lr.extend(["03"])  # exp_decay_step_01
    lr.extend(["04"])  # exp_decay_smooth_01
    # lr.extend(["05"])  # clr_triangular2_01
    # lr.extend(["06"])  # clr_triangular2_02
    hypa_grid["learning_rate_type"] = lr

    ###### which optimizer to use
    # hypa_grid["optimizer_type"] = ["a1", "r1"]
    hypa_grid["optimizer_type"] = ["a1"]

    ###### the batch size (the key is converted to int)
    bs = []
    bs.extend(["32"])
    # bs.extend(["16"])
    hypa_grid["batch_size_type"] = bs

    ###### the number of epochs (the key is converted to int)
    en = []
    en.extend(["15"])
    # en.extend(["13"])
    # en.extend(["10"])
    hypa_grid["epoch_num_type"] = en

    ###### build the combinations
    logg.debug(f"hypa_grid = {hypa_grid}")
    the_grid = list(ParameterGrid(hypa_grid))

    # hijack the grid
    # hypa = {
    #     "batch_size_type": "32",
    #     "dataset_name": "auA07",
    #     "epoch_num_type": "15",
    #     "learning_rate_type": "03",
    #     "net_type": "VAN",
    #     "optimizer_type": "a1",
    #     "words_type": "LTnumLS",
    # }
    # the_grid = [hypa]

    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    ##########################################################
    #   Setup pre train
    ##########################################################

    train_type_tag = "area"

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

    # useful to pick good values of lr
    if do_find_best_lr:
        hypa = the_grid[0]
        logg.debug(f"\nSTARTING find_best_lr with hypa: {hypa}")
        find_best_lr(hypa)
        return

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
    """MAKEDOC: what is build_area_name doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_area_name")
    logg.setLevel("INFO")
    logg.debug("Start build_area_name")

    # model_name = "ARN"
    # model_name = "SIM"
    # model_name = "AAN"
    model_name = hypa["net_type"]
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
    """MAKEDOC: what is get_model_param_area doing?"""
    # logg = logging.getLogger(f"c.{__name__}.get_model_param_area")
    # logg.setLevel("INFO")
    # logg.debug("Start get_model_param_area")

    model_param: ty.Dict[str, ty.Any] = {}

    model_param["num_classes"] = num_labels
    model_param["input_shape"] = input_shape

    return model_param


def get_training_param_area(
    hypa: ty.Dict[str, str],
    use_validation: bool,
    model_path: ty.Optional[Path],
    num_samples: int,
) -> ty.Dict[str, ty.Any]:
    """MAKEDOC: what is get_training_param_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_training_param_area")
    # logg.setLevel("INFO")
    # logg.debug("Start get_training_param_area")

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
        "05": "clr_triangular2_01",
        "06": "clr_triangular2_02",
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

    # setup cyclic learning rate
    elif lr_name.startswith("clr_triangular2"):

        # target_cycles = the number of cycles we want in those epochs
        # it_per_epoch = num_samples // batch_size
        # total_iterations = it_per_epoch * epoch_num
        # step_size = total_iterations // target_cycles

        if lr_name == "clr_triangular2_01":
            target_cycles = 2
            it_per_epoch = num_samples // training_param["batch_size"]
            total_iterations = it_per_epoch * training_param["epochs"]
            step_size = total_iterations // (target_cycles * 2)
            base_lr = 1e-5
            max_lr = 1e-3
        elif lr_name == "clr_triangular2_02":
            target_cycles = 8
            it_per_epoch = num_samples // training_param["batch_size"]
            total_iterations = it_per_epoch * training_param["epochs"]
            step_size = total_iterations // (target_cycles * 2)
            base_lr = 1e-6
            max_lr = 1e-3

        logg.debug(f"target_cycles: {target_cycles}")
        logg.debug(f"it_per_epoch: {it_per_epoch}")
        logg.debug(f"total_iterations: {total_iterations}")
        logg.debug(f"num_samples: {num_samples}")
        logg.debug(f"CLR is using step_size: {step_size}")

        mode = "triangular2"
        cyclic_lr = CyclicLR(base_lr, max_lr, step_size, mode)
        callbacks.append(cyclic_lr)

    # which metric to monitor for early_stop and model_checkpoint
    metric_to_monitor = "val_loss" if use_validation else "loss"

    if lr_name.startswith("fixed") or lr_name.startswith("exp_decay"):
        early_stop = EarlyStopping(
            monitor=metric_to_monitor,
            patience=4,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)

    # to inhibit checkpointing by passing None
    if model_path is not None:
        model_checkpoint = ModelCheckpoint(
            str(model_path), monitor=metric_to_monitor, verbose=1, save_best_only=True
        )
        callbacks.append(model_checkpoint)

    training_param["callbacks"] = callbacks

    return training_param


def train_area(
    hypa: ty.Dict[str, str],
    force_retrain: bool,
    use_validation: bool,
    trained_folder: Path,
    root_info_folder: Path,
) -> None:
    """MAKEDOC: what is train_area doing?"""
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
    net_type = hypa["net_type"]
    if net_type == "ARN":
        model = AreaNet.build(**model_param)
    elif net_type == "AAN":
        model = ActualAreaNet.build(**model_param)
    elif net_type == "VAN":
        model = VerticalAreaNet.build(**model_param)
    elif net_type.startswith("SI"):
        if net_type == "SIM":
            sim_type = "1"
        elif net_type == "SI2":
            sim_type = "2"
        model = SimpleNet.build(sim_type=sim_type, **model_param)

    num_samples = x.shape[0]
    logg.debug(f"num_samples: {num_samples}")

    # from hypa extract training param (epochs, batch, opt, ...)
    training_param = get_training_param_area(
        hypa, use_validation, model_path, num_samples
    )

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
    recap["version"] = "002"

    # logg.debug(f"recap: {recap}")
    recap_path = model_info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    # https://stackoverflow.com/a/45546663/2237151
    model_summary_path = model_info_folder / "model_summary.txt"
    with model_summary_path.open("w") as msf:
        model.summary(line_length=150, print_fn=lambda x: msf.write(x + "\n"))

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


def find_best_lr(hypa: ty.Dict[str, str]) -> None:
    """MAKEDOC: what is find_best_lr doing?"""
    logg = logging.getLogger(f"c.{__name__}.find_best_lr")
    # logg.setLevel("INFO")
    logg.debug("Start find_best_lr")

    # get the word list
    words = words_types[hypa["words_type"]]
    num_labels = len(words)

    # no validation just find the LR
    use_validation = False

    # name the model
    model_name = build_area_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    # load data
    processed_folder = Path("data_proc")
    processed_path = processed_folder / f"{hypa['dataset_name']}"
    data, labels = load_processed(processed_path, words)

    # the shape of each sample
    input_shape = data["training"][0].shape

    # from hypa extract model param
    model_param = get_model_param_area(hypa, num_labels, input_shape)

    # no need for validation
    x = np.concatenate((data["training"], data["validation"]))
    y = np.concatenate((labels["training"], labels["validation"]))

    # magic to fix the GPUs
    setup_gpus()

    # get the model with the chosen params
    net_type = hypa["net_type"]
    if net_type == "ARN":
        model = AreaNet.build(**model_param)
    elif net_type == "AAN":
        model = ActualAreaNet.build(**model_param)
    elif net_type == "VAN":
        model = VerticalAreaNet.build(**model_param)
    elif net_type.startswith("SI"):
        if net_type == "SIM":
            sim_type = "1"
        elif net_type == "SI2":
            sim_type = "2"
        model = SimpleNet.build(sim_type=sim_type, **model_param)

    num_samples = x.shape[0]
    logg.debug(f"num_samples: {num_samples}")

    # from hypa extract training param (epochs, batch, opt, ...)
    training_param = get_training_param_area(
        hypa, use_validation, model_path=None, num_samples=num_samples
    )

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

    # boundary values
    start_lr = 1e-9
    end_lr = 1e1

    # find the best values
    lrf = LearningRateFinder(model)
    lrf.find(
        (x, y),
        start_lr,
        end_lr,
        epochs=training_param["epochs"],
        batchSize=training_param["batch_size"],
    )

    fig_title = "LR_sweep"
    fig_title += f"__{model_name}"

    fig, ax = plt.subplots(figsize=(8, 8))

    # get the plot
    lrf.plot_loss(ax=ax, title=fig_title)

    # save the plot
    plot_fol = Path("plot_results") / "area" / "find_best_lr"
    if not plot_fol.exists():
        plot_fol.mkdir(parents=True, exist_ok=True)
    fig_name = fig_title + ".{}"
    fig.savefig(plot_fol / fig_name.format("png"))
    fig.savefig(plot_fol / fig_name.format("pdf"))

    recap_loss = {}
    recap_loss["lrs"] = [float(lr) for lr in lrf.lrs[:]]
    recap_loss["losses"] = [float(loss) for loss in lrf.losses[:]]
    loss_path = plot_fol / f"loss_{fig_title}.json"
    loss_path.write_text(json.dumps(recap_loss, indent=4))

    plt.show()


def run_train_area(args: argparse.Namespace) -> None:
    """MAKEDOC: What is train_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train_area")
    logg.debug("Starting run_train_area")

    training_type = args.training_type
    words_type = args.words_type
    use_validation = args.use_validation
    dry_run = args.dry_run
    force_retrain = args.force_retrain
    do_find_best_lr = args.do_find_best_lr

    if training_type == "hypa_tune":
        hyper_train_area(
            words_type, force_retrain, use_validation, dry_run, do_find_best_lr
        )


if __name__ == "__main__":
    args = setup_env()
    run_train_area(args)
