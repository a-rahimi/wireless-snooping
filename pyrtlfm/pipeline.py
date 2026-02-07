# %%
"""Wire capture -> squelch -> demod -> user callback."""

import logging
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

import fit_sinusoid
import decode
import packets
from importlib import reload

reload(decode)
reload(fit_sinusoid)
reload(packets)

logger = logging.getLogger(__name__)


def smoothed_phase_velocity(
    iq: np.ndarray,
    cutoff_hz: float = 50_000,
    sample_rate: float = 2.4e6,
    order: int = 5,
) -> np.ndarray:
    return filtfilt(
        *butter(order, cutoff_hz / (sample_rate / 2), btype="low"),
        x=packets.phase_velocity(iq),
    )


def _ensure_complex_ndarray(samples) -> np.ndarray:
    """Convert pyrtlsdr callback samples to 1D complex ndarray."""
    arr = np.asarray(samples, dtype=np.complex64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def run_decoder_loop() -> list[packets.DemodPacket]:
    """Run the pipeline at 915 MHz, 2.4 Msps.

    Press Ctrl+C to stop.
    """
    packets_list: list[packets.DemodPacket] = []

    def packet_callback(samples, ctx) -> None:
        iq = _ensure_complex_ndarray(samples)
        if iq.size < 2:
            return

        squelch_open, iq_gated = packets.squelch(iq, 0.6, zero_when_closed=True)
        if not squelch_open:
            return

        timestamp = time.time()
        packets_list.append(packets.DemodPacket(timestamp, iq_gated))
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
    return packets_list


# audio = run_decoder_loop()
# packets.save_packets(audio, "capture.pkl")


# %%
def plot_audio_samples(
    samples: list[packets.DemodPacket],
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
        phase_arr = packets.phase_velocity(sample.iq_gated)
        amplitude_arr = packets.amplitude(sample.iq_gated)

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


plt.close("all")
audio = packets.load_packets("capture.pkl")
plot_audio_samples(audio)


# %%


def decode_packet(
    sample: packets.DemodPacket,
    range: tuple[int, int] = (),
    show_plot: bool = True,
    num_samples_in_preamble: int = 1200,
    trim_sample: int = 40,
) -> None:
    """Plot a single DemodPacket in a 3×1 grid: amplitude, amplitude (DC removed), high-amp phase."""
    # Find time steps where the amplitude is high
    amplitude_arr = packets.amplitude(sample.iq_gated)

    if range == ():
        i_amplitude_high = np.nonzero(amplitude_arr > 0.5 * amplitude_arr.max())[0]
        i_amplitude_high = i_amplitude_high[trim_sample:-trim_sample]

        range = (i_amplitude_high[0], i_amplitude_high[-1])

        # Ensure there is exactly one contiguous region of high amplitude. If there is more than one, we need to
        # combine packets. That functionality isn't implemented yet.
        if range[0] <= 100 or np.any(np.diff(i_amplitude_high) > 1):
            logger.warning(
                "Multiple contiguous regions of high amplitude found. This packet can't be digidized."
            )
            return

    iq = sample.iq_gated[range[0] : range[1]]

    phase_vel = packets.phase_velocity(iq)
    preamble_params = fit_sinusoid.fit_sinusoid(phase_vel[:num_samples_in_preamble])
    print(
        "  bit_width: %.1f samples, range duration: %d samples (%.2f ms)"
        % (
            0.5 / preamble_params.frequency,
            range[1] - range[0],
            ((range[1] - range[0]) / 2.4e6 * 1e3),
        )
    )

    binarized = fit_sinusoid.binarize(phase_vel, preamble_params)
    data_bits = decode.trim_preamble(binarized.bits)
    hex_str = decode.bits_to_hex(data_bits)
    print(hex_str)

    if not show_plot:
        return

    _, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

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
    axes[1].plot(x, iq.real, label="I", color="blue", lw=0.5)
    axes[1].plot(x, iq.imag, label="Q", color="orange", lw=0.5)
    axes[1].plot(x, np.abs(iq), label="Magnitude", color="green", lw=2)
    axes[1].set_ylabel("I / Q / Magnitude")
    axes[1].set_title(f"I & Q samples, high region (t={sample.timestamp:.3f}s)")
    axes[1].set_xlabel("Sample index")
    axes[1].legend()
    ax1_twin = axes[1].twinx()
    (h,) = ax1_twin.plot(x, np.angle(iq), color="red", marker=".")
    ax1_twin.set_ylabel("Phase (rad)", color=h.get_color())
    ax1_twin.tick_params(axis="y", labelcolor=h.get_color())

    # The phase velocity time series, along with its smoothed and thresholded versions.
    axes[2].plot(x[:-1], phase_vel, label="Phase velocity")
    # Overlay the recovered sinusoid from fit_sinusoid (fitted on first 1000 samples)
    t_fit = np.arange(num_samples_in_preamble)
    axes[2].plot(
        x[: len(t_fit)],
        preamble_params.values(t_fit),
        color="magenta",
        lw=0.5,
        alpha=0.8,
        label="Preamble sinusoid",
    )
    # Mark the end of the preamble with a vertical line
    n_preamble_bits = max(0, len(binarized.bits) - len(data_bits) - 1)
    axes[2].axvline(
        range[0] + binarized.boundaries[n_preamble_bits][1],
        color="red",
        linestyle="--",
        lw=1.5,
        label="Preamble end",
    )
    axes[2].legend()
    axes[2].set_ylabel("Phase velocity")
    axes[2].set_title(
        f"Phase velocity, high-amplitude region (t={sample.timestamp:.3f}s)"
    )
    axes[2].set_xlabel("Sample index")
    ax2_twin = axes[2].twinx()
    for val, (b_start, b_end) in zip(binarized.values, binarized.boundaries):
        ax2_twin.plot(
            [range[0] + b_start, range[0] + b_end],
            [val, val],
            color="black",
            lw=2,
        )
    # Draw hex nibbles (every 4 bits) at the top of the graph.
    for i in range(0, len(binarized.bits) - 3, 4):
        nibble = binarized.bits[i : i + 4].astype(int)
        hex_val = (nibble[0] << 3) | (nibble[1] << 2) | (nibble[2] << 1) | nibble[3]
        # Centre the label between the start of the first bit and end of the fourth.
        x_center = range[0] + (binarized.boundaries[i][0] + binarized.boundaries[i + 3][1]) / 2
        ax2_twin.text(
            x_center,
            1.0,
            f"{hex_val:X}",
            transform=ax2_twin.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            fontfamily="monospace",
            color="black",
        )
    ax2_twin.set_ylabel("FSK bits", color="black")
    ax2_twin.tick_params(axis="y", labelcolor="black")

    plt.tight_layout()
    plt.show()


reload(decode)
decode_packet(audio[25])


# %%
reload(fit_sinusoid)
for ipacket, packet in enumerate(audio):
    elapsed = packet.timestamp - audio[0].timestamp
    print("\n=====", ipacket, "t+%.3fs: " % elapsed)
    decode_packet(packet, show_plot=False)
