from functools import partial
from pathlib import Path
import argparse
import logging
import json

from tensorflow_addons.image import sparse_image_warp  # type: ignore
import librosa  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from augment_data import warp_spectrograms
from augment_data import do_augmentation
from preprocess_data import preprocess_spec
from plot_utils import plot_spec
from plot_utils import plot_waveform
from plot_utils import plot_confusion_matrix
from schedules import exp_decay_step
from schedules import exp_decay_smooth
from utils import find_rowcol
from utils import setup_gpus
from utils import setup_logger
from utils import words_types


def parse_arguments():
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-vt",
        "--visualization_type",
        type=str,
        default="augment",
        choices=["augment", "cm", "spec", "datasets", "waveform", "lr_decay", "loss"],
        help="Which visualization to perform",
    )

    parser.add_argument(
        "-wi",
        "--word_index",
        type=int,
        default=0,
        help="Which word to show",
    )

    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default="VAN_opa1_lr03_bs32_en15_dsaug14_wLTBnum",
        help="Which model to use",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env():
    setup_logger("DEBUG")

    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 visualize.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def visualize_waveform() -> None:
    """MAKEDOC: what is visualize_waveform doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_waveform")
    # logg.setLevel("INFO")
    logg.debug("Start visualize_waveform")

    dataset_path = Path("data_raw")
    logg.debug(f"dataset_path: {dataset_path}")

    # words = words_types["num"]
    # words = words_types["all"]
    words = words_types["num_loud"]
    for word in words:
        word_folder = dataset_path / word

        sample_path = list(word_folder.iterdir())[0]
        sample_sig, sr = librosa.load(sample_path, sr=None)
        # logg.debug(f"sample_sig.shape: {sample_sig.shape}")

        fig, ax = plt.subplots(2, 1, figsize=(10, 12))
        plot_waveform(sample_sig, ax[0], title=f"Waveform for {word}")
        plot_waveform(sample_sig ** 2, ax[1], title=f"Waveform**2 for {word}")
        fig.tight_layout()


def visualize_spec():
    """MAKEDOC: what is visualize_spec doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_spec")
    logg.debug("Start visualize_spec")

    plot_folder = Path("plot_models")

    dataset_path = Path("data_raw")
    logg.debug(f"dataset_path: {dataset_path}")

    # word = "happy"
    # word = "wow"
    # word = "six"
    # word = "eight"  # 3
    word = "loudest_eight"
    word_folder = dataset_path / word
    # sample_path = word_folder / "0a2b400e_nohash_0.wav"
    sample_path = list(word_folder.iterdir())[0]

    # sample_path = "/home/pmn/free_spoken_digit_dataset/recordings/3_theo_10.wav"
    # sample_path = "/home/pmn/uni/human_data/progetto2020/src/data_fsdd_raw/five/5_yweweler_30.wav"
    # sample_path = "/home/pmn/uni/human_data/progetto2020/src/data_fsdd_raw/fsdd_five/5_yweweler_30.wav"
    # sample_path = "/home/pmn/uni/human_data/progetto2020/src/data_fsdd_raw/fsdd_five/5_yweweler_33.wav"
    # sample_path = "/home/pmn/uni/human_data/progetto2020/src/data_raw/_other/ljs_LJ001-0018_005.wav"
    logg.debug(f"sample_path: {sample_path}")

    # fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    fig, ax = plt.subplots(4, 1, figsize=(10, 15))

    sample_sig, sr = librosa.load(sample_path, sr=None)
    logg.debug(f"sample_sig.shape: {sample_sig.shape}")
    plot_waveform(sample_sig, ax[0], sample_rate=sr, title=f"Waveform for {word}")
    plot_waveform(
        sample_sig ** 2, ax[1], sample_rate=sr, title=f"Waveform**2 for {word}"
    )

    sample_melspec = librosa.feature.melspectrogram(sample_sig, sr=sr)
    logg.debug(f"sample_melspec.shape: {sample_melspec.shape}")
    sample_log_melspec = librosa.power_to_db(sample_melspec, ref=np.max)
    logg.debug(f"sample_log_melspec.shape: {sample_log_melspec.shape}")
    plot_spec(sample_log_melspec, ax[2], title=f"Mel spectrogram for {word}")

    sample_mfcc = librosa.feature.mfcc(sample_sig, sr=sr)
    logg.debug(f"sample_mfcc.shape: {sample_mfcc.shape}")
    sample_log_mfcc = librosa.power_to_db(sample_mfcc, ref=np.max)
    logg.debug(f"sample_log_mfcc.shape: {sample_log_mfcc.shape}")
    plot_spec(sample_log_mfcc, ax[3], title=f"MFCCs for {word}")

    fig.tight_layout()
    fig.savefig(plot_folder / f"{word}_specs.pdf")

    sr = 16000
    n_fft = 2048
    hop_length = 512
    mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)

    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig, ax = plt.subplots(1, 1)
    librosa.display.specshow(
        mel_10,
        sr=sr,
        hop_length=hop_length,
        x_axis="linear",
        y_axis="linear",
        cmap="viridis",
        ax=ax,
    )
    ax.set_ylabel("Mel filter")
    ax.set_xlabel("Hz")
    fig.tight_layout()
    fig.savefig(plot_folder / "mel10_bins.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, m in enumerate(mel_10):
        ax.plot(m, label=i)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_folder / "mel10_filterbank.pdf")


def visualize_datasets(word_index):
    """MAKEDOC: what is visualize_datasets doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_datasets")
    logg.debug("Start visualize_datasets")

    # magic to fix the GPUs
    setup_gpus()

    # show different datasets

    # datasets = [ "mfcc01", "mfcc02", "mfcc03", "mfcc04", "mfcc05", "mfcc06", "mfcc07", "mfcc08"]
    # datasets = [ "mel01", "mel02", "mel03", "mel04", "mel05", "mel06", "mel07", "mel08",
    #     "mel09", "mel10", "mel11", "mel12", "mel13", "mel14", "mel15", "melc1", "melc2",
    #     "melc3", "melc4", "mela1", "meL04", "meLa1", "auL18", "aug18", ]
    # datasets = [ "mel01", "mel04", "mel06", "melc1" ]
    # datasets = ["mel09", "mel10", "mel11", "melc1"]
    datasets = ["mel04", "mel04a", "mel04b", "melc1"]

    # words = words_types["f1"]
    # a_word = words[0]
    # a_word = "loudest_one"
    a_word = "happy"
    # a_word = "_other_ltts_loud"

    # datasets = []
    # datasets.extend(["meL04", "meLa1", "meLa2", "meLa3", "meLa4"])
    # datasets.extend(["auL06", "auL07", "auL08", "auL09"])
    # datasets.extend(["auL18", "auL19", "auL20", "auL21"])
    # a_word = "loudest_two"

    # datasets = []
    # datasets.extend(["mel04", "mela1"])
    # datasets.extend(["aug14", "aug15"])
    # a_word = "forward"
    # datasets.extend(["aug14", "aug07"])
    # a_word = "one"
    # a_word = "_other_ltts"

    # which word in the dataset to plot
    iw = word_index

    processed_folder = Path("data_proc")

    # fig, axes = plt.subplots(4, 5, figsize=(12, 15))
    nrows, ncols = find_rowcol(len(datasets))
    base_figsize = 5
    figsize = (ncols * base_figsize * 1.5, nrows * base_figsize)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols > 1:
        axes_flat = axes.flat
    else:
        axes_flat = [axes]

    fig.suptitle(f"Various spectrograms for {a_word}", fontsize=20)
    for i, ax in enumerate(axes_flat[: len(datasets)]):

        # the current dataset being plotted
        dataset_name = datasets[i]
        processed_path = processed_folder / f"{dataset_name}"
        word_path = processed_path / f"{a_word}_training.npy"
        logg.debug(f"word_path: {word_path}")

        # FIXME this is shaky as hell
        if not word_path.exists():
            if dataset_name.startswith("me"):
                preprocess_spec(dataset_name, f"_{a_word}")
            elif dataset_name.startswith("au"):
                do_augmentation(dataset_name, f"_{a_word}")

        word_data = np.load(word_path, allow_pickle=True)
        logg.debug(f"{dataset_name} {a_word} shape: {word_data[iw].shape}")
        title = f"{dataset_name}: shape {word_data[iw].shape}"
        plot_spec(word_data[iw], ax, title=title)
    fig.tight_layout()

    plot_folder = Path("plot_models")
    dt_names = "_".join(datasets)
    fig.savefig(plot_folder / f"{a_word}_{dt_names}_specs.pdf")

    # show different words
    # for dataset_name in datasets:
    #     processed_path = processed_folder / f"{dataset_name}"
    #     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    #     for i, ax in enumerate(axes.flat):
    #         word_path = processed_path / f"{words[i]}_validation.npy"
    #         word_data = np.load(word_path, allow_pickle=True)
    #         # logg.debug(f"{dataset_name} word shape: {word_data[iw].shape}")
    #         plot_spec(word_data[iw], ax, title=words[i])
    #     title = f"{dataset_name} shape {word_data[iw].shape}"
    #     fig.suptitle(title)
    #     fig.tight_layout()


def visualize_augment() -> None:
    """MAKEDOC: what is visualize_augment doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_augment")
    # logg.setLevel("INFO")
    logg.debug("Start visualize_augment")

    # build a sample image to warp
    grid_stride = 8
    grid_w = 8
    grid_h = 8
    # grid_h = 4
    grid = np.zeros((grid_w * grid_stride, grid_h * grid_stride), dtype=np.float32)
    for x in range(grid_w):
        for y in range(grid_h):
            val = (x + 1) * (y + 1)
            xs = x * grid_stride
            xe = (x + 1) * grid_stride
            ys = y * grid_stride
            ye = (y + 1) * grid_stride
            grid[xs:xe, ys:ye] = val

    # word = "down"
    # which_fold = "training"
    # dataset_name = "mel04"
    # data_fol = Path("data_proc") / f"{dataset_name}"
    # word_aug_path = data_fol / f"{word}_{which_fold}.npy"
    # data = np.load(word_aug_path)
    # grid = data[0].T

    ################################
    #   Deterministic shift
    ################################

    # dx = grid_stride // 3
    # dy = grid_stride // 3
    # source_lnd = np.array(
    #     [
    #         [1 * grid_stride + dx, 1 * grid_stride + dy],
    #         [1 * grid_stride + dx, (grid_h - 2) * grid_stride + dy],
    #         [(grid_w - 2) * grid_stride + dx, 1 * grid_stride + dy],
    #         [(grid_w - 2) * grid_stride + dx, (grid_h - 2) * grid_stride + dy],
    #     ],
    #     dtype=np.float32,
    # )
    # logg.debug(f"source_lnd.shape: {source_lnd.shape}")

    # dw = grid_stride // 2
    # delta_land = np.array([[dw, dw], [dw, dw], [dw, dw], [dw, dw]], dtype=np.float32,)

    # dest_lnd = source_lnd + delta_land
    # logg.debug(f"dest_lnd:\n{dest_lnd}")

    # # add the batch dimension
    # grid_b = np.expand_dims(grid, axis=0)
    # logg.debug(f"grid_b.shape: {grid_b.shape}")
    # source_lnd_b = np.expand_dims(source_lnd, axis=0)
    # logg.debug(f"source_lnd_b.shape: {source_lnd_b.shape}")
    # dest_lnd_b = np.expand_dims(dest_lnd, axis=0)
    # grid_b = np.expand_dims(grid_b, axis=-1)
    # logg.debug(f"grid_b.shape: {grid_b.shape}")

    # # warp the image
    # grid_warped_b, _ = sparse_image_warp(
    #     grid_b, source_lnd_b, dest_lnd_b, num_boundary_points=2
    # )

    # logg.debug(f"grid_warped_b.shape: {grid_warped_b.shape}")

    # # extract the single image
    # grid_warped = grid_warped_b[0][:, :, 0].numpy()
    # logg.debug(f"grid_warped.shape: {grid_warped.shape}")

    # # plot all the results
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    # im_args = {"origin": "lower", "cmap": "YlOrBr"}
    # axes[0].imshow(grid.T, **im_args)
    # axes[1].imshow(grid_warped.T, **im_args)

    # pl_args = {"linestyle": "none", "color": "c", "markersize": 8}
    # pl_args["marker"] = "d"
    # axes[0].plot(*source_lnd.T, label="Source landmarks", **pl_args)
    # # axes[0].legend()
    # axes[0].set_title("Original image")

    # # pl_args["marker"] = "^"
    # axes[1].plot(*dest_lnd.T, label="Dest landmarks", **pl_args)
    # # axes[1].legend()
    # axes[1].set_title("Warped image")

    # fig.tight_layout()
    # plt.show()

    ################################
    #   Random shift
    ################################

    rng = np.random.default_rng()
    num_landmarks = 4
    max_warp_time = 2
    max_warp_freq = 2
    # num_landmarks = 3
    # max_warp_time = 5
    # max_warp_freq = 5
    num_samples = 1
    spec_dim = grid.shape

    # expand the grid with batch and channel dimension
    grid_b = np.expand_dims(grid, axis=0)
    grid_b = np.expand_dims(grid_b, axis=-1)

    # the shape of the landmark for one dimension
    land_shape = num_samples, num_landmarks

    # the source point has to be at least max_warp_* from the border
    bounds_time = (max_warp_time, spec_dim[0] - max_warp_time)
    bounds_freq = (max_warp_freq, spec_dim[1] - max_warp_freq)

    # generate (num_sample, num_landmarks) time/freq positions
    source_land_t = rng.uniform(*bounds_time, size=land_shape).astype(np.float32)
    source_land_f = rng.uniform(*bounds_freq, size=land_shape).astype(np.float32)
    source_lnd = np.dstack((source_land_t, source_land_f))
    logg.debug(f"land_t.shape: {source_land_t.shape}")
    logg.debug(f"source_lnd.shape: {source_lnd.shape}")

    # generate the deltas, how much to shift each point
    delta_t = rng.uniform(-max_warp_time, max_warp_time, size=land_shape)
    delta_f = rng.uniform(-max_warp_freq, max_warp_freq, size=land_shape)
    dest_land_t = source_land_t + delta_t
    dest_land_f = source_land_f + delta_f
    dest_lnd = np.dstack((dest_land_t, dest_land_f)).astype(np.float32)
    logg.debug(f"dest_lnd.shape: {dest_lnd.shape}")

    # data_specs = data_specs.astype("float32")
    # source_lnd = source_lnd.astype("float32")
    # dest_lnd = dest_lnd.astype("float32")
    data_warped, _ = sparse_image_warp(
        grid_b, source_lnd, dest_lnd, num_boundary_points=2
    )
    logg.debug(f"data_warped.shape: {data_warped.shape}")

    # data_specs = tf.convert_to_tensor(specs, dtype=tf.float32)
    # source_lnd = tf.convert_to_tensor(source_lnd, dtype=tf.float32)
    # dest_lnd = tf.convert_to_tensor(dest_lnd, dtype=tf.float32)
    # siw = tf.function(sparse_image_warp, experimental_relax_shapes=True)
    # data_warped, _ = siw(
    #     data_specs, source_lnd, dest_lnd, num_boundary_points=2
    # # )
    # logg.debug(f"data_warped.shape: {data_warped.shape}")

    grid_warped_b = warp_spectrograms(
        grid_b, num_landmarks, max_warp_time, max_warp_freq, rng
    )

    logg.debug(f"grid_warped_b.shape: {grid_warped_b.shape}")

    # extract the single image
    grid_warped = grid_warped_b[0][:, :, 0].numpy()
    logg.debug(f"grid_warped.shape: {grid_warped.shape}")

    # plot all the results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    im_args = {"origin": "lower", "cmap": "YlOrBr"}
    axes[0].imshow(grid.T, **im_args)
    axes[1].imshow(grid_warped.T, **im_args)

    pl_args = {"linestyle": "none", "color": "c", "markersize": 8}
    pl_args["marker"] = "d"
    axes[0].plot(*source_lnd.T, label="Source landmarks", **pl_args)
    # axes[0].legend()
    axes[0].set_title("Original image")

    # pl_args["marker"] = "^"
    axes[1].plot(*dest_lnd.T, label="Dest landmarks", **pl_args)
    # axes[1].legend()
    axes[1].set_title("Warped image")

    fig.tight_layout()

    plot_folder = Path("plot_models")
    fig.savefig(plot_folder / "warp_grid.pdf")

    plt.show()


def visualize_lr_decay() -> None:
    """MAKEDOC: what is visualize_lr_decay doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_lr_decay")
    # logg.setLevel("INFO")
    logg.debug("Start visualize_lr_decay")

    num_epochs = 30
    epochs = np.arange(num_epochs)

    exp_decay_part = partial(exp_decay_step, epochs_drop=5)
    lr_gen = (exp_decay_part(x, 0) for x in epochs)
    lr_step = np.fromiter(lr_gen, dtype=np.float32, count=num_epochs)

    exp_decay_part = partial(exp_decay_smooth, epochs_drop=5)
    lr_gen = (exp_decay_part(x, 0) for x in epochs)
    lr_smooth = np.fromiter(lr_gen, dtype=np.float32, count=num_epochs)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    axes.plot(epochs, lr_step, label="LR step")
    axes.plot(epochs, lr_smooth, label="LR smooth")
    axes.legend()
    axes.set_title("Comparison of learning rate schedules")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Learning rate")

    fig.tight_layout()

    plot_folder = Path("plot_results")
    fig_name = "lr_step_smooth.pdf"
    fig.savefig(plot_folder / fig_name)

    plt.show()


def visualize_cm(model_name: str) -> None:
    r"""MAKEDOC: what is visualize_cm doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_cm")
    # logg.setLevel("INFO")
    logg.debug("Start visualize_cm")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    if "im0" in model_name:
        train_type_tag = "image"
    elif model_name.startswith("ATT"):
        train_type_tag = "attention"
    elif model_name.startswith("VAN"):
        train_type_tag = "area"
    elif model_name.startswith("S"):
        train_type_tag = "area"
    elif model_name.startswith("AAN"):
        train_type_tag = "area"

    info_folder = Path("info") / train_type_tag
    model_folder = info_folder / model_name
    res_path = model_folder / "results_recap.json"
    res = json.loads(res_path.read_text())
    recap_path = model_folder / "recap.json"
    recap = json.loads(recap_path.read_text())

    cm = np.array(res["cm"])
    fscore = res["fscore"]
    words = recap["words"]

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(cm, ax, model_name, words, fscore)
    fig.tight_layout()

    fig_name = f"{model_name}_cm.{{}}"
    cm_folder = Path("plot_results") / "cm"
    if not cm_folder.exists():
        cm_folder.mkdir(parents=True, exist_ok=True)

    plot_cm_path = cm_folder / fig_name.format("png")
    fig.savefig(plot_cm_path)
    plot_cm_path = cm_folder / fig_name.format("pdf")
    fig.savefig(plot_cm_path)

    plt.close(fig)


def visualize_loss() -> None:
    r"""MAKEDOC: what is visualize_loss doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_loss")
    # logg.setLevel("INFO")
    logg.debug("Start visualize_loss")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    model_name = "CNN_nf32_ks02_ps03_dw32_dr01_lr04_opa1_dsmel04_bs32_en18_wf2"
    train_type_tag = "cnn"
    info_folder = Path("info") / train_type_tag
    model_folder = info_folder / model_name
    res_path = model_folder / "results_recap.json"
    res = json.loads(res_path.read_text())
    # recap_path = model_folder / "recap.json"
    # recap = json.loads(recap_path.read_text())
    loss = res["history"]["loss"]
    val_loss = res["history"]["val_loss"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(loss, label="Loss")
    ax.plot(val_loss, label="Validation loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()

    fig_name = f"{model_name}_loss.{{}}"
    loss_folder = Path("plot_results") / "loss"
    if not loss_folder.exists():
        loss_folder.mkdir(parents=True, exist_ok=True)

    plot_loss_path = loss_folder / fig_name.format("pdf")
    fig.savefig(plot_loss_path)

    plt.show()


def run_visualize(args):
    """MAKEDOC: What is visualize doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_visualize")
    logg.debug("Starting run_visualize")

    visualization_type = args.visualization_type
    word_index = args.word_index
    model_name = args.model_name

    if visualization_type == "augment":
        visualize_augment()
    elif visualization_type == "spec":
        visualize_spec()
    elif visualization_type == "datasets":
        visualize_datasets(word_index)
    elif visualization_type == "waveform":
        visualize_waveform()
    elif visualization_type == "lr_decay":
        visualize_lr_decay()
    elif visualization_type == "cm":
        visualize_cm(model_name)
    elif visualization_type == "loss":
        visualize_loss()

    plt.show()


if __name__ == "__main__":
    args = setup_env()
    run_visualize(args)
