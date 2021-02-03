from pathlib import Path
import argparse
import logging

from tensorflow_addons.image import sparse_image_warp  # type: ignore
import librosa  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from plot_utils import plot_spec
from plot_utils import plot_waveform
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
        choices=["augment", "spec", "datasets"],
        help="Which visualization to perform",
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


def visualize_spec():
    """TODO: what is visualize_spec doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_spec")
    logg.debug("Start visualize_spec")

    plot_folder = Path("plot_models")

    dataset_path = Path("data_raw")
    logg.debug(f"dataset_path: {dataset_path}")

    sample_path = dataset_path / "happy" / "0a2b400e_nohash_0.wav"
    logg.debug(f"sample_path: {sample_path}")
    sample_sig, sr = librosa.load(sample_path, sr=None)
    logg.debug(f"sample_sig.shape: {sample_sig.shape}")

    # fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    plot_waveform(sample_sig, ax[0], sample_rate=sr, title="Waveform for happy")
    # plot_waveform(sample_sig, ax[0], sample_rate=sr)

    sample_melspec = librosa.feature.melspectrogram(sample_sig, sr=sr)
    logg.debug(f"sample_melspec.shape: {sample_melspec.shape}")
    sample_log_melspec = librosa.power_to_db(sample_melspec, ref=np.max)
    logg.debug(f"sample_log_melspec.shape: {sample_log_melspec.shape}")
    plot_spec(sample_log_melspec, ax[1], title="Mel spectrogram for happy")
    # plot_spec(sample_log_melspec, ax[1])

    sample_mfcc = librosa.feature.mfcc(sample_sig, sr=sr)
    logg.debug(f"sample_mfcc.shape: {sample_mfcc.shape}")
    sample_log_mfcc = librosa.power_to_db(sample_mfcc, ref=np.max)
    logg.debug(f"sample_log_mfcc.shape: {sample_log_mfcc.shape}")
    plot_spec(sample_log_mfcc, ax[2], title="MFCCs for happy")
    # plot_spec(sample_log_mfcc, ax[2])

    fig.tight_layout()
    fig.savefig(plot_folder / "happy_specs.pdf")

    sr = 16000
    n_fft = 2048
    hop_length = 512
    mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
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


def visualize_datasets():
    """TODO: what is visualize_datasets doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_datasets")
    logg.debug("Start visualize_datasets")

    # show different datasets

    datasets = [
        "mfcc01",
        "mfcc02",
        "mfcc03",
        "mfcc04",
        "mfcc05",
        "mfcc06",
        "mfcc07",
        "mfcc08",
    ]
    datasets = [
        "mel01",
        "mel02",
        "mel03",
        "mel04",
        "mel05",
        "mel06",
        "mel07",
        "mel08",
        "mel09",
        "mel10",
        "mel11",
        "mel12",
        "mel13",
        "mel14",
        "mel15",
        "melc1",
        "melc2",
        "melc3",
        "melc4",
        "mela1",
    ]

    words = words_types["f1"]
    a_word = words[0]
    # which word in the dataset to plot
    iw = 3

    processed_folder = Path("data_proc")

    fig, axes = plt.subplots(4, 5, figsize=(12, 15))
    fig.suptitle(f"{a_word}")
    for i, ax in enumerate(axes.flat[: len(datasets)]):
        dataset_name = datasets[i]
        processed_path = processed_folder / f"{dataset_name}"
        word_path = processed_path / f"{a_word}_validation.npy"
        word_data = np.load(word_path, allow_pickle=True)
        logg.debug(f"{dataset_name} word shape: {word_data[iw].shape}")
        title = f"{dataset_name} shape {word_data[iw].shape}"
        plot_spec(word_data[iw], ax, title=title)
    fig.tight_layout()

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
    """TODO: what is visualize_augment doing?"""
    logg = logging.getLogger(f"c.{__name__}.visualize_augment")
    # logg.setLevel("INFO")
    logg.debug("Start visualize_augment")

    # build a sample image to warp
    grid_stride = 8
    grid_w = 8
    # grid_h = 8
    grid_h = 4
    grid = np.zeros((grid_w * grid_stride, grid_h * grid_stride), dtype=np.float32)
    for x in range(grid_w):
        for y in range(grid_h):
            val = (x + 1) * (y + 1)
            xs = x * grid_stride
            xe = (x + 1) * grid_stride
            ys = y * grid_stride
            ye = (y + 1) * grid_stride
            grid[xs:xe, ys:ye] = val

    # rng = np.random.default_rng(12345)

    # word = "down"
    # which_fold = "training"
    # dataset_name = "mel04"
    # data_fol = Path("data_proc") / f"{dataset_name}"
    # word_aug_path = data_fol / f"{word}_{which_fold}.npy"
    # data = np.load(word_aug_path)
    # grid = data[0].T

    dx = grid_stride // 3
    dy = grid_stride // 3
    source_lnd = np.array(
        [
            [1 * grid_stride + dx, 1 * grid_stride + dy],
            [1 * grid_stride + dx, (grid_h - 2) * grid_stride + dy],
            [(grid_w - 2) * grid_stride + dx, 1 * grid_stride + dy],
            [(grid_w - 2) * grid_stride + dx, (grid_h - 2) * grid_stride + dy],
        ],
        dtype=np.float32,
    )
    logg.debug(f"source_lnd.shape: {source_lnd.shape}")

    dw = grid_stride // 2
    delta_land = np.array([[dw, dw], [dw, dw], [dw, dw], [dw, dw]], dtype=np.float32,)

    dest_lnd = source_lnd + delta_land
    logg.debug(f"dest_lnd:\n{dest_lnd}")

    # add the batch dimension
    grid_b = np.expand_dims(grid, axis=0)
    source_lnd_b = np.expand_dims(source_lnd, axis=0)
    dest_lnd_b = np.expand_dims(dest_lnd, axis=0)
    grid_b = np.expand_dims(grid, axis=-1)

    # warp the image
    grid_warped_b, _ = sparse_image_warp(
        grid_b, source_lnd_b, dest_lnd_b, num_boundary_points=2
    )

    logg.debug(f"grid_warped_b.shape: {grid_warped_b.shape}")

    # extract the single image
    grid_warped = grid_warped_b[:, :, 0].numpy()
    logg.debug(f"grid_warped.shape: {grid_warped.shape}")

    # plot all the results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

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
    plt.show()


def run_visualize(args):
    """TODO: What is visualize doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_visualize")
    logg.debug("Starting run_visualize")

    visualization_type = args.visualization_type

    if visualization_type == "augment":
        visualize_augment()
    elif visualization_type == "spec":
        visualize_spec()
    elif visualization_type == "datasets":
        visualize_datasets()

    plt.show()


if __name__ == "__main__":
    args = setup_env()
    run_visualize(args)
