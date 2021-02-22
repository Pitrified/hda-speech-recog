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
        window: int,
        interval: int,
        samplerate_input: int,
        channels: ty.List[int],
        arch_type: str,
        train_words_type: str,
    ) -> None:
        """MAKEDOC: what is __init__ doing?"""
        logg = logging.getLogger(f"c.{__name__}.__init__")
        # logg.setLevel("INFO")
        logg.debug("Start __init__")

        # save params
        self.device = device
        self.window = window
        self.interval = interval
        self.samplerate_input = int(samplerate_input)
        self.channels = channels
        self.mapping = [c - 1 for c in self.channels]  # Channel numbers start with 1
        self.arch_type = arch_type
        self.train_words_type = train_words_type

        logg.debug(f"self.window: {self.window}")

        # define constants
        self.plot_downsample = 1
        self.samplerate_train = 16000

        # setup the model
        self.setup_model()

        # info on the spectrogram
        self.get_spec_aug_info()
        self.spec = np.zeros(self.spec_shape)
        logg.debug(f"self.spec.shape: {self.spec.shape}")
        logg.debug(f"self.spec_shape: {self.spec_shape}")

        # info on the predictions
        self.max_old_pred = 50
        self.all_pred = np.zeros((self.max_old_pred, self.num_labels))

        # info on the att_weights
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
        # self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        # self.fig = plt.figure(constrained_layout=True)
        self.fig = plt.figure(figsize=(12, 12))
        gs = self.fig.add_gridspec(3, 2)
        self.ax_wave = self.fig.add_subplot(gs[0, 0])
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_attw = self.fig.add_subplot(gs[2, 0])
        self.ax_pred = self.fig.add_subplot(gs[:, 1])

        # setup dynamic title for predictions
        self.ax_pred.set_title("Predictions")
        self.pred_title = self.ax_pred.text(
            0.5,
            0.04,
            "Predictions",
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=self.ax_pred.transAxes,
            bbox={"facecolor": "w", "alpha": 0.9, "pad": 5},
            family="monospace",
        )

        # the waveform ax
        self.ax_wave.set_title("Audio signal")
        self.lines_wave = self.ax_wave.plot(
            self.audio_signal[:: self.plot_downsample, 0]
        )
        logg.debug(f"self.lines_wave: {self.lines_wave}")
        self.ax_wave.axis((0, len(self.audio_signal // self.plot_downsample), -1, 1))
        # self.ax_wave.set_yticks([0])
        self.ax_wave.yaxis.grid(True)

        # the spectrogram ax
        self.ax_spec.set_title("Mel Spectrogram")
        self.im_spec = self.ax_spec.imshow(
            self.spec, origin="lower", vmin=-80, vmax=-10, aspect="auto"
        )
        logg.debug(f"self.im_spec: {self.im_spec}")

        # the attention weights ax
        self.ax_attw.set_title("Attention weights")
        self.im_attw = self.ax_attw.imshow(
            self.att_weights, origin="lower", vmin=0, vmax=0.005, aspect="auto"
        )
        logg.debug(f"self.im_attw: {self.im_attw}")

        # the prediction ax
        self.im_pred = self.ax_pred.imshow(
            self.all_pred, origin="lower", vmin=0, vmax=1, aspect="auto"
        )
        logg.debug(f"self.im_pred: {self.im_pred}")
        self.ax_pred.set_xticks(np.arange(self.num_labels))
        self.ax_pred.set_xticklabels(self.sorted_train_words, rotation=90)

        # fig setup
        self.fig.suptitle(f"Demo for {self.arch_type} {self.train_words_type}")
        self.fig.tight_layout()

        # setup the animations
        self.animation_waveform = FuncAnimation(
            self.fig, self.update_plots, interval=self.interval, blit=True
        )

    def setup_model(self) -> None:
        r"""MAKEDOC: what is setup_model doing?"""
        logg = logging.getLogger(f"c.{__name__}.setup_model")
        # logg.setLevel("INFO")
        logg.debug("Start setup_model")

        ###### save the training words, sorted so that they match the predictions

        train_words = words_types[self.train_words_type]
        self.sorted_train_words = sorted(train_words)

        self.max_label_len = 0
        for i, word in enumerate(self.sorted_train_words):

            # remove LS tags
            if word.startswith("loudest_"):
                clean_word = word[8:]
                self.sorted_train_words[i] = clean_word
            elif word.endswith("_loud"):
                clean_word = word[:-5]
                self.sorted_train_words[i] = clean_word

            # translate weird labels to readable names
            if self.sorted_train_words[i] == "_background":
                self.sorted_train_words[i] = "_silence"
            if self.sorted_train_words[i] == "_other_ltts":
                self.sorted_train_words[i] = "_conversation"

            if len(self.sorted_train_words[i]) > self.max_label_len:
                self.max_label_len = len(self.sorted_train_words[i])

        ###### build the trained model with additional outputs
        if self.arch_type.startswith("ATT"):
            self.load_trained_model_att()
        elif self.arch_type.startswith("VAN") or self.arch_type.startswith("AAN"):
            self.load_trained_model_area()

        ###### grab the output dimentions

        # get the number of labels
        # logg.debug(f"self.att_weight_model.outputs: {self.att_weight_model.outputs}")
        self.pred_output = self.att_weight_model.outputs[0]
        self.num_labels: int = self.pred_output.shape[1]

        # get the shape of the attention weights layer
        self.weight_output = self.att_weight_model.outputs[1]
        logg.debug(f"self.weight_output.shape: {self.weight_output.shape}")
        # self.att_weight_shape = (16, 1)
        att_weight_shape = self.weight_output.shape[1 : 2 + 1]
        logg.debug(f"att_weight_shape: {att_weight_shape}")

        self.att_weight_shape: ty.Tuple[int, int] = (0, 0)

        # if it is an att model you need to add a dimension
        if len(att_weight_shape) == 1:
            # self.att_weight_shape = (att_weight_shape[0], 1)
            self.att_weight_shape = (1, att_weight_shape[0])
        else:
            self.att_weight_shape = att_weight_shape
        logg.debug(f"self.att_weight_shape: {self.att_weight_shape}")

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
        logg.setLevel("INFO")
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
        logg.debug(f"log_mel.shape: {log_mel.shape}")
        pad_needed_ = self.spec_shape[1] - log_mel.shape[1]
        pad_needed = max(0, pad_needed_)
        pad_width = ((0, 0), (0, pad_needed))
        padded_log_mel = np.pad(log_mel, pad_width=pad_width)
        # padded_log_mel[:, -pad_needed_:] = -80
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

        # get the predicted word
        pred_index = np.argmax(pred[0])
        pred_word = self.sorted_train_words[pred_index]
        logg.debug(f"sorted pred_word: {pred_word} pred_index {pred_index}")
        self.pred_title.set_text(f"Predicted: {pred_word:{self.max_label_len}}")

        ###### plot the weights

        # if the att_weights are one dimensiona, add the dims
        logg.debug(f"att_weights.shape: {att_weights.shape}")
        this_weight = att_weights[0]
        logg.debug(f"this_weight.shape: {this_weight.shape}")
        # logg.debug(f"this_weight: {this_weight}")

        # MAYBE check if you are using ATT
        if len(this_weight.shape) == 1:
            self.att_weights = np.expand_dims(this_weight, axis=-1).T

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
        all_artists.append(self.pred_title)

        return all_artists

    def get_spec_aug_info(self) -> None:
        """MAKEDOC: what is get_spec_aug_info doing?"""
        logg = logging.getLogger(f"c.{__name__}.get_spec_aug_info")
        # logg.setLevel("INFO")
        logg.debug("Start get_spec_aug_info")

        self.p2d_kwargs = {"ref": np.max}

        if self.train_dataset.startswith("me"):
            spec_dict = get_spec_dict()
            self.mel_kwargs: ty.Dict[str, ty.Any] = spec_dict[self.train_dataset]
            spec_shape_dict = get_spec_shape_dict()
            self.spec_shape: ty.Tuple[int, int] = spec_shape_dict[self.train_dataset]

        elif self.train_dataset.startswith("au"):
            aug_dict = get_aug_dict()
            self.mel_kwargs = aug_dict[self.train_dataset]["mel_kwargs"]
            self.spec_shape = aug_dict[self.train_dataset]["aug_shape"]

    def load_trained_model_area(self) -> None:
        """MAKEDOC: what is load_trained_model_area doing?"""
        logg = logging.getLogger(f"c.{__name__}.load_trained_model_area")
        # logg.setLevel("INFO")
        logg.debug("Start load_trained_model_area")

        ###### get the hypas for this model/words combo

        if self.arch_type.startswith("VAN"):

            if self.train_words_type == "LTBnum":
                # VAN_opa1_lr03_bs32_en15_dsaug14_wLTBnum
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "aug14",
                    "epoch_num_type": "15",
                    "learning_rate_type": "03",
                    "net_type": "VAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBnum",
                }
                use_validation = True

            elif self.train_words_type == "LTBnumLS":
                # VAN_opa1_lr03_bs32_en15_dsmel04L_wLTBnumLS_noval
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04L",
                    "epoch_num_type": "15",
                    "learning_rate_type": "03",
                    "net_type": "VAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBnumLS",
                }
                use_validation = False

            elif self.train_words_type == "LTBall":
                # VAN_opa1_lr03_bs32_en15_dsmel04_wLTBall_noval
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04",
                    "epoch_num_type": "15",
                    "learning_rate_type": "03",
                    "net_type": "VAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBall",
                }
                use_validation = False

            elif self.train_words_type == "LTBallLS":
                # VAN_opa1_lr03_bs32_en15_dsmel04L_wLTBallLS
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04L",
                    "epoch_num_type": "15",
                    "learning_rate_type": "03",
                    "net_type": "VAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBallLS",
                }
                use_validation = True

            else:
                raise ValueError(f"Not available {self.train_words_type}")

        elif self.arch_type.startswith("AAN"):

            if self.train_words_type == "LTBall":
                # AAN_opa1_lr03_bs32_en15_dsmel04_wLTBall_noval
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04",
                    "epoch_num_type": "15",
                    "learning_rate_type": "03",
                    "net_type": "AAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBall",
                }
                use_validation = False

            elif self.train_words_type == "LTBallLS":
                # AAN_opa1_lr04_bs32_en15_dsmel04L_wLTBallLS
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04L",
                    "epoch_num_type": "15",
                    "learning_rate_type": "04",
                    "net_type": "AAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBallLS",
                }
                use_validation = True

            elif self.train_words_type == "LTBnum":
                # AAN_opa1_lr04_bs32_en15_dsmel04_wLTBnum_noval
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04",
                    "epoch_num_type": "15",
                    "learning_rate_type": "04",
                    "net_type": "AAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBnum",
                }
                use_validation = False

            elif self.train_words_type == "LTBnumLS":
                # AAN_opa1_lr03_bs32_en15_dsmel04L_wLTBnumLS_noval
                hypa = {
                    "batch_size_type": "32",
                    "dataset_name": "mel04L",
                    "epoch_num_type": "15",
                    "learning_rate_type": "03",
                    "net_type": "AAN",
                    "optimizer_type": "a1",
                    "words_type": "LTBnumLS",
                }
                use_validation = False

            else:
                raise ValueError(f"Not available {self.train_words_type}")

        ###### get info from the hypas used
        self.batch_size = int(hypa["batch_size_type"])
        self.train_dataset = hypa["dataset_name"]
        self.train_words_type = hypa["words_type"]

        ###### build the model
        model_name = build_area_name(hypa, use_validation)
        logg.debug(f"model_name: {model_name}")

        model_folder = Path("demo_models") / "area" / model_name
        model_path = model_folder / f"{model_name}.h5"
        model = tf_models.load_model(model_path)

        # grab the last layer that has no name because you were a fool
        name_output_layer = model.layers[-1].name
        logg.debug(f"name_output_layer: {name_output_layer}")

        self.att_weight_model = tf_models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(name_output_layer).output,
                model.get_layer("area_values").output,
            ],
        )

    def load_trained_model_att(self) -> None:
        """MAKEDOC: what is load_trained_model_att doing?"""
        logg = logging.getLogger(f"c.{__name__}.load_trained_model_att")
        # logg.setLevel("INFO")
        logg.debug("Start load_trained_model_att")

        if self.arch_type.startswith("ATT"):

            if self.train_words_type == "LTnum":

                # ATT_ct02_dr01_ks01_lu01_qt05_dw01_opa1_lr03_bs02_en02_dsaug07_wLTnum
                hypa = {
                    "batch_size_type": "02",
                    "conv_size_type": "02",
                    "dataset_name": "aug07",
                    "dense_width_type": "01",
                    "dropout_type": "01",
                    "epoch_num_type": "02",
                    "kernel_size_type": "01",
                    "learning_rate_type": "03",
                    "lstm_units_type": "01",
                    "optimizer_type": "a1",
                    "query_style_type": "05",
                    "words_type": "LTnum",
                }
                use_validation = True

            elif self.train_words_type == "LTnumLS":

                # which is this FIXME add the filename
                hypa = {
                    "batch_size_type": "02",
                    "conv_size_type": "02",
                    "dataset_name": "meL04",
                    "dense_width_type": "01",
                    "dropout_type": "01",
                    "epoch_num_type": "02",
                    "kernel_size_type": "01",
                    "learning_rate_type": "03",
                    "lstm_units_type": "01",
                    "optimizer_type": "a1",
                    "query_style_type": "01",
                    "words_type": "LTnumLS",
                }
                use_validation = False

            elif self.train_words_type == "LTBnum":
                hypa = {
                    "batch_size_type": "02",
                    "conv_size_type": "02",
                    "dataset_name": "mel04",
                    "dense_width_type": "01",
                    "dropout_type": "01",
                    "epoch_num_type": "02",
                    "kernel_size_type": "01",
                    "learning_rate_type": "03",
                    "lstm_units_type": "01",
                    "optimizer_type": "a1",
                    "query_style_type": "01",
                    "words_type": "LTBnum",
                }
                use_validation = True

            elif self.train_words_type == "LTBall":
                hypa = {
                    "batch_size_type": "02",
                    "conv_size_type": "02",
                    "dataset_name": "mel04",
                    "dense_width_type": "01",
                    "dropout_type": "01",
                    "epoch_num_type": "02",
                    "kernel_size_type": "01",
                    "learning_rate_type": "03",
                    "lstm_units_type": "01",
                    "optimizer_type": "a1",
                    "query_style_type": "01",
                    "words_type": "LTBall",
                }
                use_validation = True

            elif self.train_words_type == "LTBnumLS":
                hypa = {
                    "batch_size_type": "02",
                    "conv_size_type": "02",
                    "dataset_name": "mel04L",
                    "dense_width_type": "01",
                    "dropout_type": "01",
                    "epoch_num_type": "02",
                    "kernel_size_type": "01",
                    "learning_rate_type": "03",
                    "lstm_units_type": "01",
                    "optimizer_type": "a1",
                    "query_style_type": "01",
                    "words_type": "LTBnumLS",
                }
                use_validation = True

            elif self.train_words_type == "LTBallLS":
                hypa = {
                    "batch_size_type": "02",
                    "conv_size_type": "02",
                    "dataset_name": "mel04L",
                    "dense_width_type": "01",
                    "dropout_type": "01",
                    "epoch_num_type": "02",
                    "kernel_size_type": "01",
                    "learning_rate_type": "03",
                    "lstm_units_type": "01",
                    "optimizer_type": "a1",
                    "query_style_type": "01",
                    "words_type": "LTBallLS",
                }
                use_validation = True

            else:
                raise ValueError(f"Not available {self.train_words_type}")

        self.batch_size = int(hypa["batch_size_type"])
        self.train_dataset = hypa["dataset_name"]
        self.train_words_type = hypa["words_type"]

        model_name = build_attention_name(hypa, use_validation)
        logg.debug(f"model_name: {model_name}")

        # load the model
        model_folder = Path("demo_models") / "attention"
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

    parser.add_argument(
        "-at",
        "--arch_type",
        type=str,
        default="VAN",
        help="Which model to use",
    )

    parser.add_argument(
        "-twt",
        "--train_words_type",
        type=str,
        default="LTBallLS",
        help="Which set of words to predict ",
    )

    parser.add_argument(
        "-in",
        "--interval",
        type=int,
        default=30,
        help="Milliseconds between frame updates",
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

    arch_type = args.arch_type
    train_words_type = args.train_words_type

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
    # interval = 30
    interval = args.interval
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
        arch_type=arch_type,
        train_words_type=train_words_type,
    )

    the_demo.run()


if __name__ == "__main__":
    args = setup_env()
    run_demo(args)
