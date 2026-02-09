import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

import fit_sinusoid
import decode
import packets


def decode_frame(
    sample: packets.DemodPacket,
    srange: tuple[int, int],
    show_plot: bool = True,
    num_samples_in_preamble: int = 2000,
) -> np.ndarray:
    # Just a few lines to actually decode.
    phase_vel = sample.phase_velocity()[srange[0] : srange[1]]
    decoding_params = fit_sinusoid.fit_sinusoid(phase_vel[:num_samples_in_preamble])
    binarized = fit_sinusoid.binarize(phase_vel, decoding_params)
    data_bits = decode.trim_preamble(binarized.bits)

    if not show_plot:
        return data_bits

    # Everything below is for plotting and printng.

    # Print some useful information about the packet
    freq_deviation_hz = np.mean(phase_vel) * sample.sample_rate_hz / 2
    print(
        "  freq deviation from baseband: %.1f kHz, bit_width: %.1f samples, range duration: %d samples (%.2f ms)"
        % (
            freq_deviation_hz / 1e3,
            0.5 / decoding_params.frequency,
            srange[1] - srange[0],
            ((srange[1] - srange[0]) / sample.sample_rate_hz * 1e3),
        )
    )
    transition = len(binarized.bits) - len(data_bits)
    pre = "".join(
        str(int(b)) for b in binarized.bits[max(0, transition - 8) : transition]
    )
    post = "".join(str(int(b)) for b in binarized.bits[transition : transition + 8])
    print(
        f"  preamble -> data: ...({transition} bits) {pre} | {post} ({len(data_bits)} bits)..."
    )

    # Create 2 or 3 panels, depending on how long the sample is. For very long
    # samples, the IQ plot will become a jumble.
    srange_len = srange[1] - srange[0]
    n_panels = 2 if srange_len > 1000 else 3
    _, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=False)

    # The entire amplitude time series, with the high amplitude region highlighted
    amplitude_arr = sample.amplitude()
    axes[0].plot(amplitude_arr)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Amplitude (t={sample.timestamp:.3f}s)")
    axes[0].set_xlabel("Sample index")
    axes[0].add_patch(
        mpl_patches.Rectangle(
            (srange[0], amplitude_arr.min()),
            srange[1] - srange[0],
            amplitude_arr.max() - amplitude_arr.min(),
            facecolor="red",
            alpha=0.4,
            zorder=0,
        )
    )

    x = np.arange(srange[0], srange[1])

    if n_panels == 3:
        # The I and Q components, and magnitude.
        iq = sample.iq[srange[0] : srange[1]]
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
    ax_pv = axes[-1]
    ax_pv.plot(x, phase_vel, label="Phase velocity")
    # Overlay the recovered sinusoid from fit_sinusoid (fitted on first 1000 samples)
    t_fit = np.arange(num_samples_in_preamble)
    ax_pv.plot(
        x[: len(t_fit)],
        decoding_params.values(t_fit),
        color="magenta",
        lw=0.5,
        alpha=0.8,
        label="Preamble sinusoid",
    )
    # Mark the end of the preamble with a vertical line
    n_preamble_bits = max(0, len(binarized.bits) - len(data_bits) - 1)
    ax_pv.axvline(
        srange[0] + binarized.boundaries[n_preamble_bits][1],
        color="red",
        linestyle="--",
        lw=1.5,
        label="Preamble end",
    )
    for val, (b_start, b_end) in zip(binarized.values, binarized.boundaries):
        ax_pv.plot(
            [srange[0] + b_start, srange[0] + b_end],
            [val, val],
            color="black",
            lw=2,
        )
    # Draw hex nibbles (every 4 data bits, skipping preamble) at the top of the graph.
    data_bit_offset = n_preamble_bits + 1  # index into binarized where data starts
    for i in range(0, len(data_bits) - 3, 4):
        bi = data_bit_offset + i  # index into binarized.boundaries
        nibble = binarized.bits[bi : bi + 4].astype(int)
        hex_val = (nibble[0] << 3) | (nibble[1] << 2) | (nibble[2] << 1) | nibble[3]
        x_center = (
            srange[0]
            + (binarized.boundaries[bi][0] + binarized.boundaries[bi + 3][1]) / 2
        )
        ax_pv.text(
            x_center,
            0.98,
            f"{hex_val:X}",
            transform=ax_pv.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            fontfamily="monospace",
            color="black",
        )
        # Thin black vertical line at the end of each nibble
        x_nibble_end = srange[0] + binarized.boundaries[bi + 3][1]
        ax_pv.axvline(x_nibble_end, color="black", lw=0.5)
    ax_pv.legend()
    ax_pv.set_ylabel("Phase velocity / FSK bits")
    ax_pv.set_title(
        f"Phase velocity, high-amplitude region (t={sample.timestamp:.3f}s)"
    )
    ax_pv.set_xlabel("Sample index")

    plt.tight_layout()
    plt.show()

    return data_bits


def find_run_of_high_amplitude(
    sample: packets.DemodPacket, trim_sample: int = 10, min_samples: int = 6000
):
    amplitude = sample.amplitude()
    is_high = amplitude > 0.1

    # label contiguous regions where amplitude is high
    diff = np.diff(is_high.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1  # index where high region begins
    ends = np.where(diff == -1)[0] + 1  # index where high region ends

    # handle edge cases where signal is high at the boundaries
    if is_high[0]:
        starts = np.concatenate(([0], starts))
    if is_high[-1]:
        ends = np.concatenate((ends, [len(amplitude)]))

    trimmed_starts = starts + trim_sample
    trimmed_ends = ends - trim_sample
    mask = (trimmed_ends - trimmed_starts) >= min_samples
    yield from zip(trimmed_starts[mask], trimmed_ends[mask])
