import os

from IPython import display
import matplotlib.animation
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_sinusoid
import packets


def draw_and_fit_sinusoid(
    fig: matplotlib.figure.Figure,
    t: np.ndarray,
    noisy_signal: np.ndarray,
    h_curve: matplotlib.lines.Line2D,
    h_inliners: matplotlib.lines.Line2D,
    gif_name: str,
    **fit_kwargs,
) -> fit_sinusoid.SinusoidParams:
    """Fit sinusoid iteratively, animating the fit on the given axes."""
    popts = [
        (
            fit_sinusoid.SinusoidParams(
                frequency=0,
                phase=0,
                amplitude=0,
                vertical_offset=noisy_signal.mean(),
            ),
            np.ones_like(noisy_signal, bool),
        )
    ] + list(fit_sinusoid.fit_sinusoid_iteratively(noisy_signal, **fit_kwargs))

    def update(frame: int):
        params, ingroup = popts[frame]
        h_curve.set_ydata(params.values(t))
        h_inliners.set_data(t[ingroup], noisy_signal[ingroup])
        return (h_curve, h_inliners)

    anim = matplotlib.animation.FuncAnimation(
        fig, update, frames=len(popts), interval=500, blit=True
    )

    # Save animated GIF to data/ and embed in notebook for Jupyter/nbviewer.
    os.makedirs("data", exist_ok=True)
    gif_path = os.path.join("data", gif_name)
    writer = matplotlib.animation.PillowWriter(fps=2)
    anim.save(gif_path, writer=writer, dpi=100)
    with open(gif_path, "rb") as f:
        gif_data = f.read()
    display.display(display.Image(data=gif_data, format="gif"))
    return popts[-1][0]


def demo_fit_random_signal():
    # 1. Generate synthetic signal
    true_params = fit_sinusoid.SinusoidParams(
        frequency=0.05,
        phase=5.0,
        amplitude=1.0,
        vertical_offset=0.5,
    )

    # Create time array
    t = np.arange(300)

    # Ground truth signal
    clean_signal = true_params.values(t)

    # Add noise
    np.random.seed(42)
    noisy_signal = clean_signal + np.random.normal(0, 0.2, len(t))

    # Plot the true and noisy signals
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    (h_clean,) = ax.plot(t, clean_signal, label="Clean Signal", linewidth=0.5)
    ax.plot(
        t, noisy_signal, label="Noisy Signal", color="red", linestyle="none", marker="."
    )
    (h_curve,) = ax.plot(t, 0 * t, label="Recovered Signal", linewidth=4, alpha=0.6)
    (h_inliners,) = ax.plot(
        t, noisy_signal, label="Inliners", linestyle="none", marker=".", color="green"
    )
    ax.set_title("Original vs Noisy Signal")
    ax.legend()
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    # Fit the model (animates on the same figure)
    popt = draw_and_fit_sinusoid(
        fig,
        t,
        noisy_signal,
        h_curve,
        h_inliners,
        gif_name="demo_fit_random_signal.gif",
        inlier_threshold=0.3,
    )

    # Calculate and print % difference for all parameters using a DataFrame
    rows = []
    for name in fit_sinusoid.SinusoidParams._fields:
        true_val = getattr(true_params, name)
        fitted_val = getattr(popt, name)
        pct_diff = abs((fitted_val - true_val) / true_val) * 100 if true_val != 0 else 0
        rows.append(
            {
                "parameter": name,
                "true": true_val,
                "fitted": fitted_val,
                "pct_diff": pct_diff,
            }
        )
    results_df = pd.DataFrame(rows)
    print(results_df)
    plt.close(fig)


def demo_fit_radio(frame: packets.DemodPacket, num_sample_in_preamble: int = 600):
    amplitude = frame.amplitude()
    i_high = np.nonzero(amplitude > 0.5 * amplitude.max())[0]
    preamble = frame.phase_velocity()[i_high[0] : i_high[-1]][:num_sample_in_preamble]

    t = np.arange(num_sample_in_preamble)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(t, preamble, label="Preamble (phase velocity)", alpha=0.7)
    (h_curve,) = ax.plot(t, 0 * t, label="Fitted sinusoid", linewidth=2, linestyle="--")
    (h_inliners,) = ax.plot(
        t, preamble, label="Inliners", linestyle="none", marker=".", color="green"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Phase velocity")
    ax.legend()
    plt.tight_layout()
    draw_and_fit_sinusoid(
        fig, t, preamble, h_curve, h_inliners, gif_name="demo_fit_radio.gif"
    )

    plt.close(fig)
