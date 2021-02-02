from functools import partial
from multiprocessing import Pool
from pathlib import Path
from sklearn.model_selection import ParameterGrid  # type: ignore
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore
import tensorflow as tf  # type: ignore
import typing as ty

# from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.data import Dataset  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # type: ignore

from augment_data import do_augmentation
from models import CNNmodel
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


def build_cnn_name(hypa: ty.Dict[str, ty.Union[str, int]]) -> str:
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


def hyper_train(words_type, force_retrain, use_validation, dry_run):
    """TODO: what is hyper_train doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train")
    logg.debug("Start hyper_train")

    # TODO add dense_width 64, dropout 02

    # big grid
    hypa_grid_big = {}
    # hypa_grid_big["base_dense_width"] = [16, 32, 64, 128]
    hypa_grid_big["base_dense_width"] = [32]
    # hypa_grid_big["base_filters"] = [10, 20, 30, 32, 64, 128]
    hypa_grid_big["base_filters"] = [20, 32]
    # hypa_grid_big["batch_size"] = [16, 32, 64]
    hypa_grid_big["batch_size"] = [32]
    ds = []
    ds.extend(["mel01", "mel04", "mela1"])
    # ds.extend(["mel01", "mel02", "mel03", "mel04"])
    # ds.extend(["mfcc01", "mfcc02", "mfcc03", "mfcc04"])
    ds.extend(["aug02", "aug03", "aug04", "aug05"])
    ds.extend(["aug06", "aug07", "aug08", "aug09"])
    hypa_grid_big["dataset"] = ds
    # hypa_grid_big["dropout_type"] = ["01", "02"]
    hypa_grid_big["dropout_type"] = ["01"]
    # hypa_grid_big["epoch_num"] = [15, 30, 60]
    hypa_grid_big["epoch_num"] = [15]
    hypa_grid_big["kernel_size_type"] = ["01", "02"]
    # hypa_grid_big["kernel_size_type"] = ["02"]
    # hypa_grid_big["pool_size_type"] = ["01", "02"]
    hypa_grid_big["pool_size_type"] = ["01"]
    lr = []
    lr.extend(["01", "02", "03"])  # fixed
    # lr.extend(["e1"])  # exp_decay_keras_01
    lr.extend(["04"])  # exp_decay_step_01
    lr.extend(["05"])  # exp_decay_smooth_01
    lr.extend(["06"])  # exp_decay_smooth_02
    hypa_grid_big["learning_rate_type"] = lr
    hypa_grid_big["optimizer_type"] = ["a1"]
    hypa_grid_big["words"] = [words_type]

    # tiny grid
    hypa_grid_tiny = {}
    hypa_grid_tiny["base_filters"] = [20]
    hypa_grid_tiny["kernel_size_type"] = ["01"]
    hypa_grid_tiny["pool_size_type"] = ["01"]
    hypa_grid_tiny["base_dense_width"] = [32]
    hypa_grid_tiny["dropout_type"] = ["01"]
    hypa_grid_tiny["batch_size"] = [32]
    hypa_grid_tiny["epoch_num"] = [15]
    hypa_grid_tiny["learning_rate_type"] = ["01"]
    hypa_grid_tiny["optimizer_type"] = ["a1"]
    hypa_grid_tiny["dataset"] = ["mel04"]
    hypa_grid_tiny["words"] = [words_type]

    # best params generally
    hypa_grid_best = {}
    hypa_grid_best["base_dense_width"] = [32]
    hypa_grid_best["base_filters"] = [20]
    hypa_grid_best["batch_size"] = [32]
    hypa_grid_best["dataset"] = ["mel01"]
    hypa_grid_best["dropout_type"] = ["01"]
    hypa_grid_best["epoch_num"] = [16]
    hypa_grid_best["kernel_size_type"] = ["02"]
    hypa_grid_best["pool_size_type"] = ["02"]
    hypa_grid_best["learning_rate_type"] = ["02"]
    hypa_grid_best["optimizer_type"] = ["a1"]
    # hypa_grid_best["words"] = ["f2", "f1", "num", "dir", "k1", "w2", "all"]
    hypa_grid_best["words"] = [words_type]

    # hypa_grid = hypa_grid_tiny
    # hypa_grid = hypa_grid_best
    hypa_grid = hypa_grid_big
    logg.debug(f"hypa_grid: {hypa_grid}")

    the_grid = list(ParameterGrid(hypa_grid))

    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    if dry_run:
        tra_info = {"already_trained": 0, "to_train": 0}
        for hypa in the_grid:
            train_status = train_model_cnn_dry(hypa)
            tra_info[train_status] += 1
        logg.debug(f"tra_info: {tra_info}")
        return

    # check that the data is available
    for dn in hypa_grid["dataset"]:
        for wt in hypa_grid["words"]:
            logg.debug(f"\nwt: {wt} dn: {dn}\n")
            if dn.startswith("mel"):
                preprocess_spec(dn, wt)
            elif dn.startswith("aug"):
                do_augmentation(dn, wt)

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(train_model, (hypa, force_retrain))


def train_model_cnn_dry(hypa) -> str:
    """TODO: what is train_model_cnn_dry doing?"""
    model_folder = Path("trained_models")

    model_name = build_cnn_name(hypa)
    model_path = model_folder / f"{model_name}.h5"
    if model_path.exists():
        return "already_trained"

    # the model might have been trained before introducing lr and opt hypas
    if hypa["learning_rate_type"] == "01" and hypa["optimizer_type"] == "a1":
        hypa["learning_rate_type"] = "default"
        hypa["optimizer_type"] = "adam"

    model_name = build_cnn_name(hypa)
    model_path = model_folder / f"{model_name}.h5"
    if model_path.exists():
        return "already_trained"

    return "to_train"


def train_model(hypa, force_retrain):
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
        if force_retrain:
            logg.warn("\nRETRAINING MODEL!!\n")
        else:
            logg.debug("Already trained")
            return

    # save info regarding the model training in this folder
    info_folder = Path("info") / model_name
    if not info_folder.exists():
        info_folder.mkdir(parents=True, exist_ok=True)

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

    learning_rate_types = {
        "01": "fixed01",
        "02": "fixed02",
        "03": "fixed03",
        "e1": "exp_decay_keras_01",
        "04": "exp_decay_step_01",
        "05": "exp_decay_smooth_01",
        "06": "exp_decay_smooth_02",
    }
    learning_rate_type = hypa["learning_rate_type"]
    lr_value = learning_rate_types[learning_rate_type]

    # setup opt fixed lr values
    if lr_value.startswith("fixed"):
        if lr_value == "fixed01":
            lr = 1e-2
        elif lr_value == "fixed02":
            lr = 1e-3
        elif lr_value == "fixed03":
            lr = 1e-4
    else:
        lr = 1e-3

    if lr_value == "exp_decay_keras_01":
        lr = ExponentialDecay(0.1, decay_steps=100000, decay_rate=0.96, staircase=True)

    optimizer_types = {
        "a1": Adam(learning_rate=lr),
        "r1": RMSprop(learning_rate=lr),
    }
    opt = optimizer_types[hypa["optimizer_type"]]

    # create the model
    model = CNNmodel(**model_param)
    # model.summary()

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

    # setup early stopping
    early_stop = EarlyStopping(
        # monitor="val_categorical_accuracy",
        monitor="val_loss",
        patience=4,
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stop)

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

    # train the model
    results = model.fit(
        data["training"],
        labels["training"],
        validation_data=datasets["validation"],
        batch_size=BATCH_SIZE,
        epochs=EPOCH_NUM,
        verbose=1,
        callbacks=callbacks,
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
    return "done_training"


def run_train_cnn(args: argparse.Namespace) -> None:
    """TODO: What is train_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train_cnn")
    logg.debug("Starting run_train_cnn")

    training_type = args.training_type
    words_type = args.words_type
    force_retrain = args.force_retrain
    use_validation = args.use_validation
    dry_run = args.dry_run
    # do_find_best_lr = args.do_find_best_lr

    if training_type == "smallCNN":
        hyper_train(words_type, force_retrain, use_validation, dry_run)


if __name__ == "__main__":
    args = setup_env()
    run_train_cnn(args)
