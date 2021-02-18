from random import seed as rseed
from pathlib import Path
from timeit import default_timer as timer
import argparse
import logging
import queue
import typing as ty

from tensorflow.keras import models as tf_models  # type: ignore
import librosa  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore

from augment_data import get_aug_dict
from preprocess_data import get_spec_dict
from preprocess_data import get_spec_shape_dict
from train_area import build_area_name
from utils import setup_gpus


def load_trained_model_area(train_dataset: str) -> tf_models.Model:
    """MAKEDOC: what is load_trained_model_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model_area")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model_area")

    hypa = {
        "batch_size_type": "32",
        # "dataset_name": "aug07",
        "dataset_name": train_dataset,
        "epoch_num_type": "15",
        "learning_rate_type": "05",
        "net_type": "VAN",
        "optimizer_type": "a1",
        "words_type": "LTnum",
    }
    use_validation = True
    model_name = build_area_name(hypa, use_validation)

    model_folder = Path("trained_models") / "area"
    model_path = model_folder / f"{model_name}.h5"

    model = tf_models.load_model(model_path)

    # grab the last layer that has no name because you were a fool
    name_output_layer = model.layers[-1].name
    logg.debug(f"name_output_layer: {name_output_layer}")

    att_weight_model = tf_models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(name_output_layer).output,
            model.get_layer("area_values").output,
        ],
    )

    return att_weight_model


class Demo:
    """

    class implementation of
    https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#plot-microphone-signal-s-in-real-time
    """

    def __init__(
        self,
        device,
        window,
        interval,
        samplerate_input,
        channels,
    ) -> None:
        """MAKEDOC: what is __init__ doing?"""
        logg = logging.getLogger(f"c.{__name__}.__init__")
        # logg.setLevel("INFO")
        logg.debug("Start __init__")

        self.device = device
        self.window = window
        self.interval = interval
        self.samplerate_input = int(samplerate_input)
        self.channels = channels
        self.mapping = [c - 1 for c in self.channels]  # Channel numbers start with 1

        self.plot_downsample = 1
        self.samplerate_train = 16000

        # self.train_dataset = "aug07"
        # self.train_dataset = "mel04"

        # the trained model with additional outputs
        # self.att_weight_model = load_trained_model_area(self.train_dataset)
        self.train_dataset = "aug07"
        self.att_weight_model = load_trained_model_area(self.train_dataset)

        # info on the spectrogram
        self.get_spec_aug_info()
        self.spec = np.zeros(self.spec_shape)
        logg.debug(f"self.spec.shape: {self.spec.shape}")
        logg.debug(f"self.spec_shape: {self.spec_shape}")

        # info on the predictions
        self.max_old_pred = 50
        self.num_labels = 11
        self.all_pred = np.zeros((self.max_old_pred, self.num_labels))

        # info on the att_weights
        self.att_weight_shape = (16, 1)
        self.att_weights = np.zeros(self.att_weight_shape)

        # the input audio stream
        self.stream = sd.InputStream(
            device=self.device,
            channels=max(self.channels),
            samplerate=self.samplerate_input,
            callback=self.audio_callback,
        )

        # the audio signal to be plotted, not filtered
        self.length = int(self.window * self.samplerate_input / 1000)
        self.audio_signal = np.zeros((self.length, len(self.channels)))

        # the Queue is filled by sounddevice and emptied by matplotlib
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # the figs and axis
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        # the waveform ax
        ax_wave = self.axes[0][0]
        self.lines_wave = ax_wave.plot(self.audio_signal[:: self.plot_downsample, 0])
        logg.debug(f"self.lines_wave: {self.lines_wave}")
        ax_wave.axis((0, len(self.audio_signal // self.plot_downsample), -1, 1))
        ax_wave.set_yticks([0])
        ax_wave.yaxis.grid(True)

        # the spectrogram ax
        ax_spec = self.axes[1][0]
        self.im_spec = ax_spec.imshow(self.spec, origin="lower", vmin=-80, vmax=-10)
        logg.debug(f"self.im_spec: {self.im_spec}")

        # the attention weights ax
        ax_attw = self.axes[0][1]
        self.im_attw = ax_attw.imshow(self.all_pred, origin="lower", vmin=0, vmax=1)
        logg.debug(f"self.im_attw: {self.im_attw}")

        # the prediction ax
        ax_pred = self.axes[1][1]
        self.im_pred = ax_pred.imshow(self.all_pred, origin="lower", vmin=0, vmax=1)
        logg.debug(f"self.im_pred: {self.im_pred}")

        # fig setup
        self.fig.suptitle("The demo")
        self.fig.tight_layout()

        # setup the animations
        self.animation_waveform = FuncAnimation(
            self.fig, self.update_plot_waveform, interval=self.interval, blit=True
        )
        self.animation_spectrogram = FuncAnimation(
            self.fig, self.update_plot_spectrogram, interval=self.interval, blit=True
        )
        self.animation_predictions = FuncAnimation(
            self.fig, self.update_plot_predictions, interval=self.interval, blit=True
        )
        self.animation_weights = FuncAnimation(
            self.fig, self.update_plot_weights, interval=self.interval, blit=True
        )

    def audio_callback(self, indata, frames, time, status) -> None:
        """MAKEDOC: what is audio_callback doing?

        This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.
        """
        # logg = logging.getLogger(f"c.{__name__}.audio_callback")
        # # logg.setLevel("INFO")
        # # logg.debug("Start audio_callback")
        # time_001 = timer()
        # from_last_update = time_001 - self.last_update
        # recap = f"now: {time_001:6f}"
        # recap += f"   from_last_update: {from_last_update:6f}"
        # recap += f"   indata.shape: {indata.shape}"
        # logg.debug(recap)

        # self.audio_queue.put(indata[:: self.downsample, self.mapping])
        self.audio_queue.put(indata[::, self.mapping])

    def update_plot_waveform(self, frame) -> None:
        """MAKEDOC: what is update_plot_waveform doing?"""
        logg = logging.getLogger(f"c.{__name__}.update_plot_waveform")
        logg.setLevel("INFO")
        # logg.debug("Start update_plot_waveform")

        # aquire data from queue
        while True:
            try:
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            shift = len(data)
            self.audio_signal = np.roll(self.audio_signal, -shift, axis=0)
            self.audio_signal[-shift:, :] = data
            # logg.debug(f"data.shape: {data.shape}")
            # logg.debug(f"data: {data}")

        # plot but not everything
        for column, line in enumerate(self.lines_wave):
            # logg.debug(f"setting {line} {column}")
            # logg.debug(f"self.audio_signal.shape: {self.audio_signal.shape}")
            line.set_ydata(self.audio_signal[:: self.plot_downsample, column])

        # logg.debug(f"self.audio_signal: {self.audio_signal.T}")
        # logg.debug(f"self.audio_signal.shape: {self.audio_signal.shape}")
        # logg.debug(f"self.audio_signal.min(): {self.audio_signal.min()}")
        # logg.debug(f"self.audio_signal.max(): {self.audio_signal.max()}")

        return self.lines_wave

    def update_plot_spectrogram(self, frame) -> ty.Any:
        """MAKEDOC: what is update_plot_spectrogram doing?

        https://stackoverflow.com/a/17837600/2237151
        http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
        https://stackoverflow.com/a/57259405/2237151
        """
        logg = logging.getLogger(f"c.{__name__}.update_plot_spectrogram")
        # logg.setLevel("INFO")
        # logg.debug("Start update_plot_spectrogram")

        time_001 = timer()

        #################################################################
        #   compute the spectrogram
        #################################################################

        sig_16k = librosa.resample(
            self.audio_signal[:, 0], self.samplerate_input, self.samplerate_train
        )
        mel = librosa.feature.melspectrogram(
            sig_16k, sr=self.samplerate_train, **self.mel_kwargs
        )
        log_mel = librosa.power_to_db(mel, **self.p2d_kwargs)

        # pad it
        pad_needed = self.spec_shape[1] - log_mel.shape[1]
        pad_needed = max(0, pad_needed)
        pad_width = ((0, 0), (0, pad_needed))
        padded_log_mel = np.pad(log_mel, pad_width=pad_width)
        self.spec = padded_log_mel

        self.im_spec.set_data(self.spec)

        time_002 = timer()
        recap = f"self.spec.shape: {self.spec.shape}"
        recap += f" self.spec.min() {self.spec.min()}"
        recap += f" self.spec.max() {self.spec.min()}"
        recap += f"    spectrogram time: {time_002-time_001}"
        logg.debug(recap)

        # now use the spectrogram to predict!
        img_log_mel = np.expand_dims(padded_log_mel, axis=-1)
        batch_log_mel = np.expand_dims(img_log_mel, axis=0)
        recap = f"img_log_mel.shape: {img_log_mel.shape}"
        recap += f" batch_log_mel.shape: {batch_log_mel.shape}"
        logg.debug(recap)

        pred, att_weights = self.att_weight_model.predict(batch_log_mel)
        # move the all_pred and append the latest
        # self.all_pred

        time_003 = timer()
        recap = f" pred.shape: {pred.shape}"
        recap += f"    preditions time: {time_003-time_002}"
        logg.debug(recap)
        logg.debug(f"pred: {pred}")

        return (self.im_spec,)

    def update_plot_weights(self, frame) -> ty.Any:
        """MAKEDOC: what is update_plot_weights doing?"""
        # logg = logging.getLogger(f"c.{__name__}.update_plot_weights")
        # logg.setLevel("INFO")
        # logg.debug("Start update_plot_weights")

        return (self.im_attw,)

    def update_plot_predictions(self, frame) -> ty.Any:
        """MAKEDOC: what is update_plot_predictions doing?"""
        # logg = logging.getLogger(f"c.{__name__}.update_plot_predictions")
        # logg.setLevel("INFO")
        # logg.debug("Start update_plot_predictions")

        return (self.im_pred,)

    def get_spec_aug_info(self) -> None:
        """MAKEDOC: what is get_spec_aug_info doing?"""
        logg = logging.getLogger(f"c.{__name__}.get_spec_aug_info")
        # logg.setLevel("INFO")
        logg.debug("Start get_spec_aug_info")

        self.p2d_kwargs = {"ref": np.max}

        if self.train_dataset.startswith("me"):
            spec_dict = get_spec_dict()
            self.mel_kwargs = spec_dict[self.train_dataset]
            spec_shape_dict = get_spec_shape_dict()
            self.spec_shape = spec_shape_dict[self.train_dataset]

        elif self.train_dataset.startswith("au"):
            aug_dict = get_aug_dict()
            self.mel_kwargs = aug_dict[self.train_dataset]["mel_kwargs"]
            self.spec_shape = aug_dict[self.train_dataset]["aug_shape"]

    def run(self) -> None:
        """MAKEDOC: what is run doing?"""
        logg = logging.getLogger(f"c.{__name__}.run")
        # logg.setLevel("INFO")
        logg.debug("Start run")

        with self.stream:
            plt.show()


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        "--path_input",
        type=str,
        default="hp.jpg",
        help="path to input image to use",
    )

    parser.add_argument(
        "-s", "--rand_seed", type=int, default=-1, help="random seed to use"
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel: str = "DEBUG") -> None:
    """Setup logger that outputs to console for the module"""
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    # log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # log_format_module = "%(name)s - %(levelname)s: %(message)s"
    # log_format_module = '%(levelname)s: %(message)s'
    log_format_module = "%(name)s: %(message)s"
    # log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logroot.log(5, 'Exceedingly verbose debug')


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")

    args = parse_arguments()

    # setup seed value
    if args.rand_seed == -1:
        myseed = 1
        myseed = int(timer() * 1e9 % 2 ** 32)
    else:
        myseed = args.rand_seed
    rseed(myseed)
    np.random.seed(myseed)

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 demo.py"
    for a, v in args._get_kwargs():
        if a == "rand_seed":
            recap += f" --rand_seed {myseed}"
        else:
            recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def run_demo(args: argparse.Namespace) -> None:
    """TODO: What is demo doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_demo")
    logg.debug("Starting run_demo")

    # magic to fix the GPUs
    setup_gpus()

    device = None

    device_info = sd.query_devices(device=device, kind="input")
    logg.debug(f"device_info: {device_info}")

    window = 1000
    # window = 200
    # interval = 1000
    # interval = 30
    interval = 100
    # blocksize = 0
    samplerate_input = device_info["default_samplerate"]
    channels = [1]
    # block_duration = 50

    the_demo = Demo(
        device,
        window,
        interval,
        samplerate_input,
        channels,
    )

    the_demo.run()


if __name__ == "__main__":
    args = setup_env()
    run_demo(args)
