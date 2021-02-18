from pathlib import Path
from timeit import default_timer as timer
import argparse
import logging
import queue
import typing as ty

from tensorflow.keras import models as tf_models  # type: ignore
from tensorflow.keras import backend as K  # type: ignore
import librosa  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore

from augment_data import get_aug_dict
from preprocess_data import get_spec_dict
from preprocess_data import get_spec_shape_dict
from train_area import build_area_name
from train_attention import build_attention_name
from utils import setup_gpus
from utils import words_types


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
        train_dataset,
        train_words_type,
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
        self.train_dataset = train_dataset
        self.train_words_type = train_words_type

        self.mapping = [c - 1 for c in self.channels]  # Channel numbers start with 1

        self.plot_downsample = 1
        self.samplerate_train = 16000

        # self.train_dataset = "mel04"
        # self.train_dataset = "aug07"
        # self.train_dataset = "auA07"
        # self.train_dataset = "aug14"

        # the trained model with additional outputs
        # self.att_weight_model = self.load_trained_model_area(self.train_dataset)
        self.load_trained_model_area()
        # self.load_trained_model_att()

        # grab the output dimentions
        # logg.debug(f"self.att_weight_model.outputs: {self.att_weight_model.outputs}")
        self.pred_output = self.att_weight_model.outputs[0]
        # logg.debug(f"self.weight_output.shape: {self.weight_output.shape}")
        self.weight_output = self.att_weight_model.outputs[1]
        # logg.debug(f"self.weight_output.shape: {self.weight_output.shape}")

        # info on the spectrogram
        self.get_spec_aug_info()
        self.spec = np.zeros(self.spec_shape)
        logg.debug(f"self.spec.shape: {self.spec.shape}")
        logg.debug(f"self.spec_shape: {self.spec_shape}")

        # info on the predictions
        self.max_old_pred = 50
        # self.num_labels = 11
        self.num_labels = self.pred_output.shape[1]
        self.all_pred = np.zeros((self.max_old_pred, self.num_labels))

        # info on the att_weights
        # self.att_weight_shape = (16, 1)
        self.att_weight_shape = self.weight_output.shape[1 : 2 + 1]
        logg.debug(f"self.att_weight_shape: {self.att_weight_shape}")
        if len(self.att_weight_shape) == 1:
            self.att_weight_shape = (self.att_weight_shape[0], 1)
        logg.debug(f"self.att_weight_shape: {self.att_weight_shape}")
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
        # self.im_attw = ax_attw.imshow(self.att_weights, origin="lower", vmin=0, vmax=1)
        self.im_attw = ax_attw.imshow(
            self.att_weights, origin="lower", vmin=0, vmax=0.005
        )
        # self.im_attw = ax_attw.imshow(self.att_weights, origin="lower")
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
            self.fig, self.update_plots, interval=self.interval, blit=True
        )

    def audio_callback(self, indata, frames, time, status) -> None:
        """MAKEDOC: what is audio_callback doing?"""
        # logg = logging.getLogger(f"c.{__name__}.audio_callback")
        # # logg.setLevel("INFO")
        # logg.debug("Start audio_callback")

        # a hard copy must be made, be careful to not just use a reference
        self.audio_queue.put(indata[::, self.mapping])

    def update_plots(self, frame) -> ty.List[ty.Any]:
        """MAKEDOC: what is update_plots doing?

        This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.
        """
        logg = logging.getLogger(f"c.{__name__}.update_plots")
        # logg.setLevel("INFO")
        logg.debug("--------------- Start update_plots ---------------")

        #################################################################
        #   aquire data from queue
        #################################################################

        while True:
            try:
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.audio_signal = np.roll(self.audio_signal, -shift, axis=0)
            self.audio_signal[-shift:, :] = data

        #################################################################
        #   plot the waveform of the channel you are using
        #################################################################

        for column, line in enumerate(self.lines_wave):
            line.set_ydata(self.audio_signal[:: self.plot_downsample, column])

        #################################################################
        #   spectrogram
        #################################################################

        time_001 = timer()

        # compute it
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

        # plot it
        self.im_spec.set_data(self.spec)

        # time it
        time_002 = timer()
        recap = f"self.spec.shape: {self.spec.shape}"
        recap += f" self.spec.min() {self.spec.min()}"
        recap += f" self.spec.max() {self.spec.min()}"
        recap += f"    spectrogram time: {time_002-time_001}"
        logg.debug(recap)

        #################################################################
        #   predictions
        #################################################################

        ###### turn 2D image in a batch of a 3D image
        img_log_mel = np.expand_dims(padded_log_mel, axis=-1)
        batch_log_mel = np.expand_dims(img_log_mel, axis=0)
        recap = f"img_log_mel.shape: {img_log_mel.shape}"
        recap += f" batch_log_mel.shape: {batch_log_mel.shape}"
        logg.debug(recap)

        ###### get predictions and weights
        # we do a megatrick super magic:
        # get a batch size of 32 (or whatever is set by the hypas)
        shape_batch = (self.batch_size, *img_log_mel.shape)
        logg.debug(f"shape_batch: {shape_batch}")
        batch_log_mel_big = np.zeros(shape_batch, dtype=np.float32)
        # fill the entire big batch with the same spectrogram
        for i in range(self.batch_size):
            batch_log_mel_big[i, :, :, :] = batch_log_mel
        # and finally predict: now magically is correct!
        pred, att_weights = self.att_weight_model.predict(batch_log_mel_big)

        ###### time the prediction
        time_003 = timer()
        recap = f"pred.shape: {pred.shape}"
        recap += f"    preditions time: {time_003-time_002:.7f}"
        pred_f = ", ".join([f"{p:.3f}" for p in pred[0]])
        recap += f" pred_f: {pred_f}"
        logg.debug(recap)
        # logg.debug(f"pred: {pred}")

        ###### move the all_pred and append the latest
        self.all_pred = np.roll(self.all_pred, -1, axis=0)
        self.all_pred[-1, :] = pred[0]
        self.im_pred.set_data(self.all_pred)

        ###### plot the weights

        # if the att_weights are one dimensiona, add the dims
        logg.debug(f"att_weights.shape: {att_weights.shape}")
        this_weight = att_weights[0]
        logg.debug(f"this_weight.shape: {this_weight.shape}")
        # logg.debug(f"this_weight: {this_weight}")

        if len(this_weight.shape) == 1:
            self.att_weights = np.expand_dims(this_weight, axis=-1)

        else:
            # self.att_weights = att_weights[0][:, :, 0]
            self.att_weights = this_weight[:, :, 0]

        recap = f"self.att_weights.shape: {self.att_weights.shape}"
        recap += f" self.att_weights.max() {self.att_weights.max()}"
        recap += f" self.att_weights.min() {self.att_weights.min()}"
        logg.debug(recap)

        # update the data in the plot
        self.im_attw.set_data(self.att_weights)
        self.im_attw.set_clim(self.att_weights.min(), self.att_weights.max())

        ###### build the list of artist to update
        all_artists = []
        # we EXTEND, lines_wave is already an iterable of lines
        all_artists.extend(self.lines_wave)
        # we APPEND, im_spec is just one object
        all_artists.append(self.im_spec)
        all_artists.append(self.im_pred)
        all_artists.append(self.im_attw)

        return all_artists

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

    def load_trained_model_area(self) -> None:
        """MAKEDOC: what is load_trained_model_area doing?"""
        logg = logging.getLogger(f"c.{__name__}.load_trained_model_area")
        # logg.setLevel("INFO")
        logg.debug("Start load_trained_model_area")

        hypa = {
            "batch_size_type": "32",
            # "dataset_name": "aug07",
            "dataset_name": self.train_dataset,
            "epoch_num_type": "15",
            "learning_rate_type": "05",
            "net_type": "VAN",
            "optimizer_type": "a1",
            # "words_type": "LTnum",
            "words_type": self.train_words_type,
        }
        use_validation = True

        hypa = {
            "batch_size_type": "32",
            # "dataset_name": "mel04",
            "dataset_name": self.train_dataset,
            "epoch_num_type": "15",
            "learning_rate_type": "03",
            "net_type": "VAN",
            "optimizer_type": "a1",
            # "words_type": "LTBnum"
            "words_type": self.train_words_type,
        }

        # hypa = {
        #     "batch_size_type": "32",
        #     # "dataset_name": "auA07",
        #     "dataset_name": self.train_dataset,
        #     "epoch_num_type": "15",
        #     "learning_rate_type": "03",
        #     "net_type": "VAN",
        #     "optimizer_type": "a1",
        #     # "words_type": "LTnumLS",
        #     "words_type": self.train_words_type,
        # }
        # use_validation = False

        # hypa = {
        #     "batch_size_type": "32",
        #     # "dataset_name": "aug14",
        #     "dataset_name": train_dataset,
        #     "epoch_num_type": "15",
        #     "learning_rate_type": "03",
        #     "net_type": "AAN",
        #     "optimizer_type": "a1",
        #     "words_type": "LTnum",
        # }
        # use_validation = True

        self.batch_size = int(hypa["batch_size_type"])

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

        self.att_weight_model = att_weight_model

    def load_trained_model_att(self) -> None:
        """MAKEDOC: what is load_trained_model_att doing?"""
        logg = logging.getLogger(f"c.{__name__}.load_trained_model_att")
        # logg.setLevel("INFO")
        logg.debug("Start load_trained_model_att")

        # ATT_ct02_dr01_ks01_lu01_qt05_dw01_opa1_lr03_bs02_en02_dsaug07_wLTnum
        hypa = {
            "batch_size_type": "02",
            "conv_size_type": "02",
            # "dataset_name": "aug07",
            "dataset_name": self.train_dataset,
            "dense_width_type": "01",
            "dropout_type": "01",
            "epoch_num_type": "02",
            "kernel_size_type": "01",
            "learning_rate_type": "03",
            "lstm_units_type": "01",
            "optimizer_type": "a1",
            "query_style_type": "05",
            # "words_type": "LTnum",
            "words_type": self.train_words_type,
        }
        use_validation = True

        hypa = {
            "batch_size_type": "02",
            "conv_size_type": "02",
            # "dataset_name": "meL04",
            "dataset_name": self.train_dataset,
            "dense_width_type": "01",
            "dropout_type": "01",
            "epoch_num_type": "02",
            "kernel_size_type": "01",
            "learning_rate_type": "03",
            "lstm_units_type": "01",
            "optimizer_type": "a1",
            "query_style_type": "01",
            # "words_type": "LTnumLS"
            "words_type": self.train_words_type,
        }
        use_validation = False

        # this is valid for
        # hypa["dataset_name"] = "mel04"
        # hypa["words_type"] = "LTBnum"
        # hypa["words_type"] = "LTBall"
        # hypa["dataset_name"] = "mel04L"
        # hypa["words_type"] = "LTBnumLS"
        # hypa["words_type"] = "LTBallLS"
        hypa = {
            "batch_size_type": "02",
            "conv_size_type": "02",
            "dataset_name": self.train_dataset,
            "dense_width_type": "01",
            "dropout_type": "01",
            "epoch_num_type": "02",
            "kernel_size_type": "01",
            "learning_rate_type": "03",
            "lstm_units_type": "01",
            "optimizer_type": "a1",
            "query_style_type": "01",
            "words_type": self.train_words_type,
        }
        use_validation = True

        model_name = build_attention_name(hypa, use_validation)
        logg.debug(f"model_name: {model_name}")

        # load the model
        model_folder = Path("trained_models") / "attention"
        model_path = model_folder / f"{model_name}.h5"

        # model = tf.keras.models.load_model(model_path)
        # https://github.com/keras-team/keras/issues/5088#issuecomment-401498334
        model = tf_models.load_model(model_path, custom_objects={"backend": K})
        # model.summary()
        # logg.debug(f"ascii_model(model): {ascii_model(model)}")

        att_weight_model = tf_models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer("output").output,
                model.get_layer("att_softmax").output,
            ],
        )
        # att_weight_model.summary()
        # logg.debug(f"att_weight_model.outputs: {att_weight_model.outputs}")

        self.att_weight_model = att_weight_model

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

    tra_types = [w for w in words_types.keys() if not w.startswith("_")]
    parser.add_argument(
        "-tw",
        "--train_words_type",
        type=str,
        default="f2",
        choices=tra_types,
        help="Words the dataset was trained on",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="mel01",
        help="Name of the dataset folder",
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
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 demo.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def run_demo(args: argparse.Namespace) -> None:
    """TODO: What is demo doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_demo")
    logg.debug("Starting run_demo")

    train_words_type = args.train_words_type
    dataset_name = args.dataset_name

    # magic to fix the GPUs
    setup_gpus()

    device = None

    device_info = sd.query_devices(device=device, kind="input")
    logg.debug(f"device_info: {device_info}")

    if train_words_type.endswith("LS"):
        window = 500
    else:
        window = 1000

    # window = 200
    # interval = 1000
    interval = 30
    # interval = 100
    # interval = 500
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
        train_dataset=dataset_name,
        train_words_type=train_words_type,
    )

    the_demo.run()


if __name__ == "__main__":
    args = setup_env()
    run_demo(args)
