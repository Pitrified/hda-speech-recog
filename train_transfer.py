from functools import partial
from multiprocessing import Pool
from pathlib import Path
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn.model_selection import ParameterGrid  # type: ignore
import argparse
import json
import logging
import matplotlib.pyplot as plt  # type: ignore
import tensorflow.keras.callbacks as tf_callbacks  # type: ignore
import tensorflow.keras.losses as tf_losses  # type: ignore
import tensorflow.keras.metrics as tf_metrics  # type: ignore
import tensorflow.keras.optimizers as tf_optimizers  # type: ignore
import typing as ty

from audio_generator import AudioGenerator
from audio_generator import get_generator_mean_var_cached
from models import TRAmodel
from plot_utils import plot_confusion_matrix
from preprocess_data import prepare_partitions
from preprocess_data import preprocess_split
from schedules import exp_decay_smooth
from schedules import exp_decay_step
from utils import analyze_confusion
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
    """MAKEDOC: what is build_transfer_name doing?"""

    # model_name = "TB4"
    # model_name = "TB7"
    model_name = hypa["net_type"]
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


def get_datasets_types() -> ty.Tuple[
    ty.Dict[str, ty.List[str]], ty.Dict[str, ty.Tuple[int, int]]
]:
    """"""
    datasets_types = {
        "01": ["mel05", "mel09", "mel10"],
        "02": ["mel05", "mel10", "mfcc07"],
        "03": ["mfcc06", "mfcc07", "mfcc08"],
        "04": ["mel05", "mfcc06", "melc1"],
        "05": ["melc1", "melc2", "melc4"],
    }
    # if you actually use them add the dims lol
    datasets_shapes = {
        "01": (128, 128),
        "02": (-1, -1),
        "03": (-1, -1),
        "04": (-1, -1),
        "05": (-1, -1),
    }
    return datasets_types, datasets_shapes


def hyper_train_transfer(
    words_type: str, force_retrain: bool, use_validation: bool, dry_run: bool
) -> None:
    """MAKEDOC: what is hyper_train_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_transfer")

    ##########################################################
    #   Hyper-parameters grid
    ##########################################################

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

    ###### the architecture to train on
    arch = []
    arch.append("TRA")  # Xception
    # arch.append("TD1")  # DenseNet121
    # arch.append("TB0")  # EfficientNetB0
    # arch.append("TB4")  # EfficientNetB4
    # arch.append("TB7")  # EfficientNetB7
    hypa_grid["net_type"] = arch

    ###### the dataset to train on
    # hypa_grid["datasets_type"] = ["01", "02", "03", "04", "05"]
    hypa_grid["datasets_type"] = ["01"]

    ###### the words to train on
    # wtl = []
    wtl: ty.List[str] = []
    # wtl.extend(["f2", "f1", "dir", "num"])
    # wtl.extend(["k1", "w2", "all"])
    wtl.extend([words_type])
    hypa_grid["words_type"] = wtl

    ###### the dense width of the classifier
    dw = []
    dw.extend(["01"])  # [4, 0]
    dw.extend(["02"])  # [16, 16]
    dw.extend(["03"])  # [0, 0]
    dw.extend(["04"])  # [64, 64]
    hypa_grid["dense_width_type"] = dw

    ###### the dropout to use
    dr = []
    dr.extend(["01"])  # 0.2
    # dr.extend(["02"])  # 0.1
    # dr.extend(["03"])  # 0
    hypa_grid["dropout_type"] = dr

    ###### the batch sizes to use
    bs = []
    bs.extend(["01"])  # [32, 32]
    bs.extend(["02"])  # [16, 16]
    hypa_grid["batch_size_type"] = bs

    ###### the number of epochs
    en = []
    en.extend(["01"])  # [20, 10]
    en.extend(["02"])  # [40, 20]
    # en.extend(["03"])  # [1, 1]
    hypa_grid["epoch_num_type"] = en

    ###### the learning rates for the optimizer
    lr = []
    # lr.extend(["01", "02"])  # fixed
    # lr.extend(["01"])  # fixed
    lr.extend(["03"])  # exp_decay_step_01
    # lr.extend(["04"])  # exp_decay_smooth_01
    hypa_grid["learning_rate_type"] = lr

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

    # where to put the tensorboard logs
    tensorboard_logs_folder = Path("tensorboard_logs") / train_type_tag
    if not tensorboard_logs_folder.exists():
        tensorboard_logs_folder.mkdir(parents=True, exist_ok=True)

    # count how many models are left to train
    if dry_run:
        tra_info = {"already_trained": 0, "to_train": 0}
        for hypa in the_grid:
            train_status = train_model_tra_dry(hypa, use_validation, trained_folder)
            tra_info[train_status] += 1
        logg.debug(f"tra_info: {tra_info}")
        return

    # check that the data is available
    datasets_types, _ = get_datasets_types()
    # for each type of dataset that will be used
    for dt in hypa_grid["datasets_type"]:
        # get the dataset name list
        dataset_names = datasets_types[dt]
        for dn in dataset_names:
            # and check that the data is available for each word type
            for wt in hypa_grid["words_type"]:
                logg.debug(f"\nwt: {wt} dn: {dn}")
                preprocess_split(dn, wt)

    ##########################################################
    #   Train all hypas
    ##########################################################

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(
                train_transfer,
                (
                    hypa,
                    force_retrain,
                    use_validation,
                    trained_folder,
                    root_info_folder,
                    tensorboard_logs_folder,
                ),
            )


def train_model_tra_dry(hypa, use_validation: bool, trained_folder: Path) -> str:
    """MAKEDOC: what is train_model_tra_dry doing?"""
    model_name = build_transfer_name(hypa, use_validation)
    placeholder_path = trained_folder / f"{model_name}.txt"

    if placeholder_path.exists():
        return "already_trained"

    return "to_train"


def get_model_param_transfer(
    hypa: ty.Dict[str, str], num_labels: int, input_shape: ty.Tuple[int, int, int]
) -> ty.Dict[str, ty.Any]:
    """MAKEDOC: what is get_model_param_transfer doing?"""
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
    hypa: ty.Dict[str, str],
    use_validation: bool,
    tensorboard_logs_folder: Path,
    model_path: Path,
) -> ty.Dict[str, ty.Any]:
    """MAKEDOC: what is get_training_param_transfer doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_training_param_transfer")
    # logg.setLevel("INFO")
    logg.debug("Start get_training_param_transfer")

    training_param: ty.Dict[str, ty.Any] = {}

    batch_size_types = {"01": [32, 32], "02": [16, 16]}
    batch_sizes = batch_size_types[hypa["batch_size_type"]]
    training_param["batch_sizes"] = batch_sizes

    epoch_num_types = {"01": [20, 10], "02": [40, 20], "03": [1, 1]}
    epoch_num = epoch_num_types[hypa["epoch_num_type"]]
    training_param["epoch_num"] = epoch_num

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
        "a1": [
            tf_optimizers.Adam(learning_rate=lr[0]),
            tf_optimizers.Adam(learning_rate=lr[1]),
        ],
        "r1": [
            tf_optimizers.RMSprop(learning_rate=lr[0]),
            tf_optimizers.RMSprop(learning_rate=lr[1]),
        ],
    }
    training_param["opt"] = optimizer_types[hypa["optimizer_type"]]

    ###### setup callbacks
    callbacks: ty.List[ty.List[tf_callbacks]] = [[], []]

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
        lrate = tf_callbacks.LearningRateScheduler(exp_decay_part_frozen)
        callbacks.append(lrate)
        lrate = tf_callbacks.LearningRateScheduler(exp_decay_part_fine)
        callbacks.append(lrate)

    # monitor this metric in early_stop/model_checkpoint
    metric_to_monitor = "val_loss" if use_validation else "loss"

    # add early stop to learning_rate_types where it makes sense
    if lr_name.startswith("fixed") or lr_name.startswith("exp_decay"):
        early_stop = tf_callbacks.EarlyStopping(
            monitor=metric_to_monitor, patience=6, restore_best_weights=True, verbose=1
        )
        callbacks[0].append(early_stop)
        callbacks[1].append(early_stop)

    # TODO: check if you need to reload the best one at the end of fit
    # yes you should
    # add model_checkpoint to keep the best weights
    model_checkpoint = tf_callbacks.ModelCheckpoint(
        str(model_path), monitor=metric_to_monitor, verbose=1, save_best_only=True
    )
    callbacks[0].append(model_checkpoint)
    callbacks[1].append(model_checkpoint)

    # add the TensorBoard callback for fancy logs
    # log_dir = (
    #     tensorboard_logs_folder
    #     / "fit"
    #     / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # )
    # tensorboard_callback = tf_callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # callbacks[0].append(tensorboard_callback)
    # callbacks[1].append(tensorboard_callback)

    training_param["callbacks"] = callbacks

    ###### a few metrics to track
    metrics = [
        tf_metrics.CategoricalAccuracy(),
        tf_metrics.Precision(),
        tf_metrics.Recall(),
    ]
    training_param["metrics"] = [metrics, metrics]

    return training_param


def train_transfer(
    hypa: ty.Dict[str, str],
    force_retrain: bool,
    use_validation: bool,
    trained_folder: Path,
    root_info_folder: Path,
    tensorboard_logs_folder: Path,
) -> None:
    """MAKEDOC: what is train_transfer doing?

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

    # grab a few hypas
    words_type = hypa["words_type"]
    datasets_type = hypa["datasets_type"]

    # get the partition of the data
    partition, ids2labels = prepare_partitions(words_type)

    # get the word list
    words = words_types[words_type]
    num_labels = len(words)

    # get the dataset name list
    datasets_types, datasets_shapes = get_datasets_types()
    dataset_names = datasets_types[datasets_type]
    dataset_shape = datasets_shapes[datasets_type]

    # the shape of each sample
    input_shape = (*dataset_shape, 3)

    # from hypa extract training param (epochs, batch, opt, ...)
    training_param = get_training_param_transfer(
        hypa, use_validation, tensorboard_logs_folder, model_path
    )

    # load datasets
    processed_folder = Path("data_split")
    data_split_paths = [processed_folder / f"{dn}" for dn in dataset_names]
    # data, labels = load_triple(data_paths, words)

    # assemble the gen_param for the generators
    gen_param = {
        "dim": dataset_shape,
        "batch_size": training_param["batch_sizes"][0],
        "shuffle": True,
        "label_names": words,
        "data_split_paths": data_split_paths,
    }

    # maybe concatenate the valdation and training lists
    val_generator: ty.Optional[AudioGenerator] = None
    if use_validation:
        val_generator = AudioGenerator(partition["validation"], ids2labels, **gen_param)
        logg.debug("Using validation data")
    else:
        partition["training"].extend(partition["validation"])
        logg.debug("NOT using validation data")

    # create the training generator with the modified (maybe) list of IDs
    training_generator = AudioGenerator(partition["training"], ids2labels, **gen_param)
    logg.debug(f"len(training_generator): {len(training_generator)}")

    ###### always create the test generator
    # do not shuffle the test data
    gen_param["shuffle"] = False
    # do not batch it, no loss of stray data at the end
    gen_param["batch_size"] = 1
    testing_generator = AudioGenerator(partition["testing"], ids2labels, **gen_param)

    ##########################################################
    #   Setup model
    ##########################################################

    # from hypa extract model param
    model_param = get_model_param_transfer(hypa, num_labels, input_shape)

    # get mean and var to normalize the data
    data_mean, data_variance = get_generator_mean_var_cached(
        training_generator, words_type, datasets_type, processed_folder
    )

    # get the model
    model, base_model = TRAmodel(
        data_mean=data_mean, data_variance=data_variance, **model_param
    )
    model.summary()

    # a dict to recreate this training
    recap: ty.Dict[str, ty.Any] = {}
    recap["words"] = words
    recap["hypa"] = hypa
    recap["model_param"] = model_param
    recap["use_validation"] = use_validation
    recap["model_name"] = model_name
    recap["batch_sizes"] = training_param["batch_sizes"]
    recap["epoch_num"] = training_param["epoch_num"]
    recap["version"] = "003"

    # logg.debug(f"recap: {recap}")
    recap_path = model_info_folder / "recap.json"
    recap_path.write_text(json.dumps(recap, indent=4))

    ##########################################################
    #   Compile and fit model the first time
    ##########################################################

    model.compile(
        optimizer=training_param["opt"][0],
        loss=tf_losses.CategoricalCrossentropy(),
        metrics=training_param["metrics"][0],
    )

    results_freeze = model.fit(
        training_generator,
        validation_data=val_generator,
        epochs=training_param["epoch_num"][0],
        callbacks=training_param["callbacks"][0],
    )

    # reload the best weights saved by the ModelCheckpoint
    model.load_weights(str(model_path))

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
        loss=tf_losses.CategoricalCrossentropy(),
        metrics=training_param["metrics"][1],
    )

    results_full = model.fit(
        training_generator,
        validation_data=val_generator,
        epochs=training_param["epoch_num"][1],
        callbacks=training_param["callbacks"][1],
    )

    # reload the best weights saved by the ModelCheckpoint
    model.load_weights(str(model_path))

    ##########################################################
    #   Save results, history, performance
    ##########################################################

    results_full_recap: ty.Dict[str, ty.Any] = {}
    results_full_recap["model_name"] = model_name
    results_full_recap["results_recap_version"] = "001"

    # evaluate performance
    eval_testing = model.evaluate(testing_generator)
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")
        results_full_recap[metrics_name] = value

    # compute the confusion matrix
    y_pred = model.predict(testing_generator)
    y_pred_labels = testing_generator.pred2labelnames(y_pred)
    y_true = testing_generator.get_true_labels()
    # cm = pred_hot_2_cm(y_true, y_pred, words)
    cm = confusion_matrix(y_true, y_pred_labels)
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
    """MAKEDOC: What is train_transfer doing?"""
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
