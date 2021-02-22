from functools import partial
from multiprocessing import Pool
from pathlib import Path
import argparse
import json
import logging
import typing as ty

from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn.model_selection import ParameterGrid  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import tensorflow as tf  # type: ignore

from area_model import ActualAreaNet
from area_model import AreaNet
from area_model import SimpleNet
from area_model import VerticalAreaNet
from clr_callback import CyclicLR
from imagenet_generator import ImageNetGenerator
from imagenet_generator import prepare_partitions
from plot_utils import plot_confusion_matrix
from schedules import exp_decay_smooth
from schedules import exp_decay_step
from utils import analyze_confusion
from utils import setup_gpus


def parse_arguments() -> argparse.Namespace:
    r"""Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-lld",
        "--log_level_debug",
        type=str,
        default="DEBUG",
        help="Level for the debugging logger",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
    )

    parser.add_argument(
        "-llt",
        "--log_level_type",
        type=str,
        default="m",
        help="Message format for the debugging logger",
        choices=["anlm", "nlm", "lm", "nm", "m"],
    )

    parser.add_argument(
        "-tt",
        "--training_type",
        type=str,
        default="hypa_tune",
        choices=["hypa_tune"],
        help="Which training to execute",
    )

    parser.add_argument(
        "-wt",
        "--words_type",
        type=str,
        default="im01",
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


def setup_logger(logLevel: str = "DEBUG", msg_type: str = "m") -> None:
    r"""Setup logger that outputs to console for the module"""
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    if msg_type == "anlm":
        log_format_module = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif msg_type == "nlm":
        log_format_module = "%(name)s - %(levelname)s: %(message)s"
    elif msg_type == "lm":
        log_format_module = "%(levelname)s: %(message)s"
    elif msg_type == "nm":
        log_format_module = "%(name)s: %(message)s"
    else:
        log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)


def setup_env() -> argparse.Namespace:
    r"""Setup the logger and parse the args"""
    args = parse_arguments()
    setup_logger(args.log_level_debug, args.log_level_type)

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 imagenet_preprocess.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def build_img_name(hypa: ty.Dict[str, str], use_validation: bool) -> str:
    """MAKEDOC: what is build_img_name doing?"""
    logg = logging.getLogger(f"c.{__name__}.build_img_name")
    logg.setLevel("INFO")
    logg.debug("Start build_img_name")

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


def train_model_img_dry(
    hypa: ty.Dict[str, str], use_validation: bool, trained_folder: Path
) -> str:
    """MAKEDOC: what is train_model_img_dry doing?"""
    # logg = logging.getLogger(f"c.{__name__}.train_model_img_dry")
    # logg.setLevel("INFO")
    # logg.debug("Start train_model_img_dry")

    model_name = build_img_name(hypa, use_validation)
    placeholder_path = trained_folder / f"{model_name}.txt"

    if placeholder_path.exists():
        return "already_trained"

    return "to_train"


def hyper_train_img(
    words_type: str,
    force_retrain: bool,
    use_validation: bool,
    dry_run: bool,
    do_find_best_lr: bool,
) -> None:
    r"""MAKEDOC: what is hyper_train_img doing?"""
    logg = logging.getLogger(f"c.{__name__}.hyper_train_img")
    # logg.setLevel("INFO")
    logg.debug("Start hyper_train_img")

    ##########################################################
    #   Hyper-parameters grid
    ##########################################################

    hypa_grid: ty.Dict[str, ty.List[str]] = {}

    ###### the net type
    nt = []
    nt.append("SIM")
    nt.append("SI2")
    nt.append("AAN")
    # nt.append("VAN")
    hypa_grid["net_type"] = nt

    ###### the words to train on
    wtl = []
    # wtl.append("im01")
    wtl.append("im02")
    hypa_grid["words_type"] = wtl

    ###### the dataset to train on
    ds = []
    ds.append("aug01")
    # ds.append("aug02")
    # ds.append("aug03")
    hypa_grid["dataset_name"] = ds

    ###### the learning rates for the optimizer
    lr = []
    lr.append("03")  # exp_decay_step_01
    lr.append("04")  # exp_decay_smooth_01
    lr.append("05")  # clr_triangular2_01
    lr.append("06")  # clr_triangular2_02
    hypa_grid["learning_rate_type"] = lr

    ###### which optimizer to use
    hypa_grid["optimizer_type"] = ["a1"]

    ###### the batch size (the key is converted to int)
    bs = []
    # bs.append("32")
    bs.append("16")
    hypa_grid["batch_size_type"] = bs

    ###### the number of epochs (the key is converted to int)
    en = []
    # en.append("15")
    en.append("30")
    hypa_grid["epoch_num_type"] = en

    ###### build the combinations
    logg.debug(f"hypa_grid = {hypa_grid}")
    the_grid = list(ParameterGrid(hypa_grid))
    num_hypa = len(the_grid)
    logg.debug(f"num_hypa: {num_hypa}")

    ##########################################################
    #   Setup pre train
    ##########################################################

    train_type_tag = "image"

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
            train_status = train_model_img_dry(hypa, use_validation, trained_folder)
            tra_info[train_status] += 1
        logg.debug(f"tra_info: {tra_info}")
        return

    ##########################################################
    #   Train all hypas
    ##########################################################

    for i, hypa in enumerate(the_grid):
        logg.debug(f"\nSTARTING {i+1}/{num_hypa} with hypa: {hypa}")
        with Pool(1) as p:
            p.apply(
                train_img,
                (hypa, force_retrain, use_validation, trained_folder, root_info_folder),
            )


def get_label_list(label_type: str) -> ty.List[str]:
    r"""MAKEDOC: what is get_label_list doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_label_list")
    # logg.setLevel("INFO")
    logg.debug("Start get_label_list")

    if label_type == "im01":
        label_list = [
            "bee",
            "cannon",
            "cicada",
            "convertible",
            "crocodile",
            "daisy",
            "heron",
            "ice cream",
            "night heron",
            "tarantula",
        ]

    elif label_type == "im02":
        label_list = [
            "agaric",
            "anemone fish",
            "architecture",
            "armlet",
            "banana",
            "banana bread",
            "barred owl",
            "basenji",
            "basketball player",
            "bee",
            "belted kingfisher",
            "beverage",
            "bird",
            "bomber",
            "book",
            "bridesmaid",
            "cake mix",
            "calliandra",
            "canna",
            "cannon",
            "cardigan",
            "car mirror",
            "cassette tape",
            "cathedral",
            "chancel",
            "cherry tomato",
            "chickpea",
            "child",
            "chocolate",
            "chronograph",
            "cicada",
            "cichlid",
            "clasp",
            "column",
            "convertible",
            "cosmos",
            "covered bridge",
            "crocodile",
            "cue",
            "cygnet",
            "daisy",
            "Dalai Lama",
            "dandelion green",
            "drummer",
            "drunkard",
            "entree",
            "fairground",
            "fly",
            "footbridge",
            "garden",
            "gazebo",
            "giant panda",
            "granddaughter",
            "grandfather",
            "grate",
            "green snake",
            "helix",
            "hen",
            "heron",
            "hip",
            "hookah",
            "ice cream",
            "invertebrate",
            "junco",
            "kilt",
            "kitten",
            "lavender",
            "lizard",
            "moray",
            "musician",
            "niece",
            "Nigerian",
            "night heron",
            "office building",
            "pan",
            "Pembroke",
            "porthole",
            "rat snake",
            "ready-to-wear",
            "rhododendron",
            "robber fly",
            "rock",
            "rodent",
            "Roman arch",
            "rose mallow",
            "sea turtle",
            "Segway",
            "side dish",
            "sky",
            "snowbank",
            "sparrow",
            "spider web",
            "strawberry",
            "sulphur-crested cockatoo",
            "suspension bridge",
            "tarantula",
            "tennis racket",
            "toast",
            "tom",
            "tramline",
            "trolleybus",
            "Tulipa gesneriana",
            "viper",
            "wild carrot",
            "wing",
            "woman",
            "woodpecker",
            "world",
        ]

    return label_list


def get_model_param_img(
    hypa: ty.Dict[str, str], num_labels: int, input_shape: ty.Tuple[int, int, int]
) -> ty.Dict[str, ty.Any]:
    """MAKEDOC: what is get_model_param_img doing?"""
    # logg = logging.getLogger(f"c.{__name__}.get_model_param_img")
    # logg.setLevel("INFO")
    # logg.debug("Start get_model_param_img")

    model_param: ty.Dict[str, ty.Any] = {}

    model_param["num_classes"] = num_labels
    model_param["input_shape"] = input_shape

    return model_param


def get_training_param_img(
    hypa: ty.Dict[str, str],
    use_validation: bool,
    model_path: ty.Optional[Path],
    num_samples: int,
) -> ty.Dict[str, ty.Any]:
    """MAKEDOC: what is get_training_param_img doing?"""
    logg = logging.getLogger(f"c.{__name__}.get_training_param_img")
    # logg.setLevel("INFO")
    # logg.debug("Start get_training_param_img")

    training_param: ty.Dict[str, ty.Any] = {}

    training_param["batch_size"] = int(hypa["batch_size_type"])
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


def train_img(
    hypa: ty.Dict[str, str],
    force_retrain: bool,
    use_validation: bool,
    trained_folder: Path,
    root_info_folder: Path,
) -> None:
    """MAKEDOC: what is train_img doing?"""
    logg = logging.getLogger(f"c.{__name__}.train_img")
    # logg.setLevel("INFO")
    logg.debug("Start train_img")

    ##########################################################
    #   Setup folders
    ##########################################################

    # name the model
    model_name = build_img_name(hypa, use_validation)
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

    label_type = hypa["words_type"]
    label_list = get_label_list(label_type)
    num_labels = len(label_list)

    dataset_raw_folder = Path.home() / "datasets" / "imagenet" / "imagenet_images"
    dataset_proc_base_folder = Path.home() / "datasets" / "imagenet"

    # get the partition of the data
    partition, ids2labels = prepare_partitions(label_list, dataset_raw_folder)

    num_samples = len(partition["training"])

    # from hypa extract training param (epochs, batch, opt, ...)
    training_param = get_training_param_img(
        hypa, use_validation, model_path, num_samples
    )

    preprocess_type = hypa["dataset_name"]
    dataset_proc_folder = dataset_proc_base_folder / preprocess_type

    val_generator: ty.Optional[ImageNetGenerator] = None
    if use_validation:
        val_generator = ImageNetGenerator(
            partition["validation"],
            ids2labels,
            label_list,
            dataset_proc_folder=dataset_proc_folder,
            dataset_raw_folder=dataset_raw_folder,
            preprocess_type=preprocess_type,
            save_processed=True,
            batch_size=training_param["batch_size"],
            shuffle=True,
        )
        logg.debug("Using validation data")
    else:
        partition["training"].extend(partition["validation"])
        logg.debug("NOT using validation data")

    training_generator = ImageNetGenerator(
        partition["training"],
        ids2labels,
        label_list,
        dataset_proc_folder=dataset_proc_folder,
        dataset_raw_folder=dataset_raw_folder,
        preprocess_type=preprocess_type,
        save_processed=True,
        batch_size=training_param["batch_size"],
        shuffle=True,
    )

    testing_generator = ImageNetGenerator(
        partition["testing"],
        ids2labels,
        label_list,
        dataset_proc_folder=dataset_proc_folder,
        dataset_raw_folder=dataset_raw_folder,
        preprocess_type=preprocess_type,
        save_processed=True,
        batch_size=1,
        shuffle=False,
    )

    ##########################################################
    #   Setup model
    ##########################################################

    input_shape = training_generator.get_img_shape()

    # from hypa extract model param
    model_param = get_model_param_img(hypa, num_labels, input_shape)

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
    recap["words"] = label_list
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
        training_generator,
        validation_data=val_generator,
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
    eval_testing = model.evaluate(testing_generator)
    for metrics_name, value in zip(model.metrics_names, eval_testing):
        logg.debug(f"{metrics_name}: {value}")
        results_recap[metrics_name] = value

    # confusion matrix
    y_pred = model.predict(testing_generator)
    y_pred_labels = testing_generator.pred2labelnames(y_pred)
    y_true = testing_generator.get_true_labels()
    cm = confusion_matrix(y_true, y_pred_labels)
    results_recap["cm"] = cm.tolist()

    # fscore
    fscore = analyze_confusion(cm, label_list)
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
    plot_confusion_matrix(cm, ax, model_name, label_list, fscore)
    plot_cm_path = model_info_folder / "test_confusion_matrix.png"
    fig.savefig(plot_cm_path)
    plt.close(fig)

    # save the trained model
    model.save(model_path)

    # save the placeholder
    placeholder_path.write_text(f"Trained. F-score: {fscore}")


def run_train_imagenet(args: argparse.Namespace) -> None:
    r"""TODO: What is train_imagenet doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_train_imagenet")
    logg.debug("Starting run_train_imagenet")

    training_type = args.training_type
    words_type = args.words_type
    use_validation = args.use_validation
    dry_run = args.dry_run
    force_retrain = args.force_retrain
    do_find_best_lr = args.do_find_best_lr

    if training_type == "hypa_tune":
        hyper_train_img(
            words_type, force_retrain, use_validation, dry_run, do_find_best_lr
        )


if __name__ == "__main__":
    args = setup_env()
    run_train_imagenet(args)
