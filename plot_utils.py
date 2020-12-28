import numpy as np  # type: ignore
from math import floor
from librosa import display as ld  # type: ignore

# import logging


def plot_waveform(sample_sig, ax, title="Sample waveform", sample_rate=16000):
    """TODO: what is plot_waveform doing?"""
    ax.plot(sample_sig)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    # one tick every tenth of a second
    num_ticks = floor(sample_sig.shape[0] / (sample_rate / 10))
    ax.set_xticks(np.linspace(0, sample_sig.shape[0], num_ticks + 1))
    # length of sample in seconds
    sig_len = sample_sig.shape[0] / sample_rate
    ax.set_xticklabels(np.linspace(0, sig_len, num_ticks + 1).round(2))
    return ax


def plot_spec(log_spec, ax, title="Spectrogram", sample_rate=16000):
    """TODO: what is plot_spec doing?"""
    # logg = logging.getLogger(f"c.{__name__}.plot_spec")
    # logg.debug("Start plot_spec")
    ld.specshow(
        log_spec, sr=sample_rate, x_axis="time", y_axis="mel", cmap="viridis", ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_xticks(np.linspace(0, 1, 11))
    return ax
