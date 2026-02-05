# %%
"""Wire capture -> squelch -> demod -> user callback."""

import threading
import time
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

from . import squelch


class DemodPacket(NamedTuple):
    timestamp: float
    iq_gated: np.ndarray


def phase_velocity(packet: DemodPacket) -> np.ndarray:
    product = packet.iq_gated[1:] * np.conj(packet.iq_gated[:-1])
    return np.angle(product) / np.pi


def amplitude(packet: DemodPacket) -> np.ndarray:
    return np.abs(packet.iq_gated[:-1])


def fsk_filtered(
    packet: DemodPacket,
    cutoff_hz: float = 50_000,
    sample_rate: float = 2.4e6,
    order: int = 5,
) -> np.ndarray:
    """FSK decode with low-pass filtering."""
    return filtfilt(
        *butter(order, cutoff_hz / (sample_rate / 2), btype="low"),
        x=phase_velocity(packet),
    )


def save_packets(packets: list[DemodPacket], path: str | Path) -> None:
    """Save DemodPackets to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(packets, f)
    print(f"Saved {len(packets)} packets to {path}")


def load_packets(path: str | Path) -> list[DemodPacket]:
    """Load DemodPackets from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_complex_ndarray(samples) -> np.ndarray:
    """Convert pyrtlsdr callback samples to 1D complex ndarray."""
    arr = np.asarray(samples, dtype=np.complex64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def run_decoder_loop() -> list[DemodPacket]:
    """Run the pipeline at 915 MHz, 2.4 Msps.

    Press Ctrl+C to stop.
    """
    packets: list[DemodPacket] = []

    def packet_callback(samples, ctx) -> None:
        iq = _ensure_complex_ndarray(samples)
        if iq.size < 2:
            return

        squelch_open, iq_gated = squelch.squelch(iq, 0.6, zero_when_closed=True)
        if not squelch_open:
            return

        timestamp = time.time()
        packets.append(DemodPacket(timestamp, iq_gated))
        print("Packet received", timestamp)

    def run(sdr: RtlSdr) -> None:
        try:
            sdr.read_samples_async(
                packet_callback, num_samples=16384
            )  # align with rtl_fm
        finally:
            print("Closing SDR...")
            sdr.cancel_read_async()
            sdr.close()

    sdr = RtlSdr(device_index=0)
    sdr.sample_rate = 2.4e6
    sdr.center_freq = 915e6
    sdr.gain = "auto"

    thread = threading.Thread(target=run, args=(sdr,), daemon=False)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        sdr.cancel_read_async()
        thread.join(timeout=5.0)
    print("Stopped.")
    return packets


# audio = run_decoder_loop()
# save_packets(audio, "capture.pkl")


# %%
def plot_audio_samples(
    samples: list[DemodPacket],
    figsize_per_plot: tuple[float, float] = (12, 3),
) -> None:
    """Plot DemodPacket samples in an n×3 grid: amplitude ts, phase ts, phase hist."""
    import matplotlib.pyplot as plt

    n = len(samples)
    if n == 0:
        return
    t0 = samples[0].timestamp
    fig, axes = plt.subplots(
        n, 3, figsize=(figsize_per_plot[0] * 1.5, figsize_per_plot[1] * n), sharex="col"
    )
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (row_axes, sample) in enumerate(zip(axes, samples)):
        offset_s = sample.timestamp - t0

        # Use DemodPacket helpers
        phase_arr = phase_velocity(sample)
        amplitude_arr = amplitude(sample)

        i_high_amplitude = np.nonzero(amplitude_arr > 0.5 * amplitude_arr.max())[0][
            10:-10
        ]
        row_axes[0].plot(amplitude_arr)
        row_axes[0].set_ylabel("Amplitude")
        row_axes[0].set_title(f"Sample {i} (t+{offset_s:.3f}s)")
        row_axes[1].plot(
            np.arange(len(phase_arr))[i_high_amplitude], phase_arr[i_high_amplitude]
        )
        row_axes[1].set_ylabel("Phase")
        row_axes[1].set_title(f"Sample {i} (t+{offset_s:.3f}s)")
        mean_phase = phase_arr[i_high_amplitude].mean()
        row_axes[2].hist(
            phase_arr[i_high_amplitude],
            np.linspace(mean_phase - 0.2, mean_phase + 0.2, 100),
        )
        row_axes[2].set_ylabel("Density")
        row_axes[2].set_title(f"Sample {i} (t+{offset_s:.3f}s)")
    axes[-1, 0].set_xlabel("Sample index")
    axes[-1, 1].set_xlabel("Sample index")
    axes[-1, 2].set_xlabel("Phase")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.close("all")
    audio = load_packets("capture.pkl")
    plot_audio_samples(audio)


# %%

#%%
class Digitized:
    def __init__(self, packet: DemodPacket, num_samples_in_preamble: int = 1000):
        self.phase_velocity = phase_velocity(packet)
        self.fsk = fsk_filtered(packet)
        self.pll = PhaseLockedLoop(self.phase_velocity)


def show_sample(
    sample: DemodPacket,
    range: tuple[int, int] = (),
    figsize: tuple[float, float] = (12, 12),
) -> None:
    """Plot a single DemodPacket in a 3×1 grid: amplitude, amplitude (DC removed), high-amp phase."""
    amplitude_arr = amplitude(sample)

    # Time steps where the amplitude is high
    i_amplitude_high = np.nonzero(amplitude_arr > 0.5 * amplitude_arr.max())[0]
    i_amplitude_high = i_amplitude_high[10:-10]

    if range == ():
        range = (i_amplitude_high[0], i_amplitude_high[-1])

    iq_slice = sample.iq_gated[range[0] : range[1]]

    # Ensure there is exactly one contiguous region of high amplitude. If there is more than one, we need to
    # combine packets. That functionality isn't implemented yet.
    if np.any(np.diff(i_amplitude_high) > 1):
        print(
            "Warning: Multiple contiguous regions of high amplitude found. This packet can't be digidized."
        )
    else:
        digidized = Digitized(DemodPacket(sample.timestamp, iq_slice))
        phase_velocity = digidized.phase_velocity
        fsk_arr = digidized.fsk
        fsk_bits = fsk_arr > np.median(fsk_arr)

    _, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)

    # The entire amplitude time series, with the high amplitude region highlighted
    axes[0].plot(amplitude_arr)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Amplitude (t={sample.timestamp:.3f}s)")
    axes[0].set_xlabel("Sample index")
    axes[0].add_patch(
        mpl_patches.Rectangle(
            (range[0], amplitude_arr.min()),
            range[1] - range[0],
            amplitude_arr.max() - amplitude_arr.min(),
            facecolor="red",
            alpha=0.4,
            zorder=0,
        )
    )

    # The I and Q components, and magnitude.
    x = np.arange(range[0], range[1])
    axes[1].plot(x, iq_slice.real, label="I", color="blue", lw=0.5)
    axes[1].plot(x, iq_slice.imag, label="Q", color="orange", lw=0.5)
    axes[1].plot(x, np.abs(iq_slice), label="Magnitude", color="green", lw=2)
    axes[1].set_ylabel("I / Q / Magnitude")
    axes[1].set_title(f"I & Q samples, high region (t={sample.timestamp:.3f}s)")
    axes[1].set_xlabel("Sample index")
    axes[1].legend()
    ax1_twin = axes[1].twinx()
    (h,) = ax1_twin.plot(x, np.angle(iq_slice), color="red", marker=".")
    ax1_twin.set_ylabel("Phase (rad)", color=h.get_color())
    ax1_twin.tick_params(axis="y", labelcolor=h.get_color())

    # The phase velocity time series, along with its smoothed and thresholded versions.
    axes[2].plot(x[:-1], phase_velocity)
    axes[2].plot(x[:-1], fsk_arr, color="red")
    axes[2].set_ylabel("Phase velocity")
    axes[2].set_title(
        f"Phase velocity, high-amplitude region (t={sample.timestamp:.3f}s)"
    )
    axes[2].set_xlabel("Sample index")
    ax2_twin = axes[2].twinx()
    (h,) = ax2_twin.plot(x[:-1], fsk_bits, color="black")
    ax2_twin.set_ylabel("FSK bits", color=h.get_color())
    ax2_twin.tick_params(axis="y", labelcolor=h.get_color())

    plt.tight_layout()
    plt.show()


    show_sample(audio[2], ())
