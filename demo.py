from random import seed as rseed
from timeit import default_timer as timer
import argparse
import logging
import queue

from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore


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
        samplerate,
        downsample,
        channels,
    ) -> None:
        """MAKEDOC: what is __init__ doing?"""
        logg = logging.getLogger(f"c.{__name__}.__init__")
        # logg.setLevel("INFO")
        logg.debug("Start __init__")

        self.device = device
        self.window = window
        self.interval = interval
        self.samplerate = int(samplerate)
        self.downsample = downsample
        self.channels = channels
        self.mapping = [c - 1 for c in self.channels]  # Channel numbers start with 1

        self.stream = sd.InputStream(
            device=self.device,
            channels=max(self.channels),
            samplerate=self.samplerate,
            callback=self.audio_callback,
        )

        # just a timer
        self.last_update = timer()

        # the audio signal to be plotted
        self.length = int(self.window * self.samplerate / (1000 * self.downsample))
        self.audio_signal = np.zeros((self.length, len(self.channels)))

        # this is filled by sounddevice and emptied by matplotlib
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # the figs and artists
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        ax_wave = self.axes[0][0]
        self.lines_wave = ax_wave.plot(self.audio_signal)
        ax_wave.axis((0, len(self.audio_signal), -1, 1))
        ax_wave.set_yticks([0])
        ax_wave.yaxis.grid(True)

        # self.fig.tight_layout(pad=0)
        self.fig.tight_layout()

        # the animation
        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=self.interval, blit=True
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
        # start_update = timer()
        # from_last_update = start_update - self.last_update
        # recap = f"now: {start_update:6f}"
        # recap += f"   from_last_update: {from_last_update:6f}"
        # recap += f"   indata.shape: {indata.shape}"
        # logg.debug(recap)

        self.audio_queue.put(indata[:: self.downsample, self.mapping])

    def update_plot(self, frame) -> None:
        """MAKEDOC: what is update_plot doing?"""
        # logg = logging.getLogger(f"c.{__name__}.update_plot")
        # logg.setLevel("INFO")
        # logg.debug("Start update_plot")

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

        for column, line in enumerate(self.lines_wave):
            line.set_ydata(self.audio_signal[:, column])

        # logg.debug(f"self.audio_signal: {self.audio_signal.T}")
        # logg.debug(f"self.audio_signal.shape: {self.audio_signal.shape}")
        # logg.debug(f"self.audio_signal.min(): {self.audio_signal.min()}")
        # logg.debug(f"self.audio_signal.max(): {self.audio_signal.max()}")

        return self.lines_wave

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

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = '%(name)s: %(message)s'
    log_format_module = "%(message)s"

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

    device = None

    device_info = sd.query_devices(device=device, kind="input")
    logg.debug(f"device_info: {device_info}")

    # window = 1000
    window = 200
    # interval = 1000
    interval = 30
    # blocksize = 0
    samplerate = device_info["default_samplerate"]
    downsample = 10
    channels = [1]
    # block_duration = 50

    the_demo = Demo(
        device,
        window,
        interval,
        samplerate,
        downsample,
        channels,
    )

    the_demo.run()


if __name__ == "__main__":
    args = setup_env()
    run_demo(args)
