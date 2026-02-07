# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fit_sinusoid
import packets

from importlib import reload

reload(fit_sinusoid)

_HERE = Path(__file__).resolve().parent


def run_demo():
    # 1. Generate synthetic signal
    true_params = fit_sinusoid.SinusoidParams(
        frequency=0.05,
        phase=5.0,
        amplitude=2.0,
        vertical_offset=0.5,
    )

    # Create time array
    t = np.arange(100)

    # Ground truth signal
    clean_signal = true_params.values(t)

    # Add noise
    np.random.seed(42)
    noisy_signal = clean_signal + np.random.normal(0, 0.1, len(t))

    # 2. Fit the model
    popt = fit_sinusoid.fit_sinusoid(noisy_signal)

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

    # 3. Reconstruct fitted signal
    fitted_signal = popt.values(t)

    # 4. Plot results
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(t, clean_signal, label="Clean Signal", linewidth=3)
    ax.plot(t, noisy_signal, label="Noisy Signal", linestyle="none", marker=".")
    ax.plot(t, fitted_signal, label="Recovered Signal", linewidth=2, linestyle="--")
    ax.set_title("Original vs Noisy Signal")
    ax.legend()
    ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def run_demo_audio(sample_index: int = 24, num_sample_in_preamble: int = 1000):
    """Fit a sinusoid to the preamble of audio packet 24 from the capture file."""
    # 1. Load the captured packets and pick packet 24
    sample = packets.load_packets(_HERE / "capture.pkl")[sample_index]

    # 2. Extract the high-amplitude region (same logic as decode_packet)
    amplitude_arr = packets.amplitude(sample.iq_gated)
    i_high = np.nonzero(amplitude_arr > 0.5 * amplitude_arr.max())[0]
    i_high = i_high[40:-40]
    iq = sample.iq_gated[i_high[0] : i_high[-1]]

    # 3. Compute phase velocity and isolate the preamble (first 1000 samples)
    phase_vel = packets.phase_velocity(iq)
    preamble = phase_vel[:num_sample_in_preamble]

    # 4. Fit a sinusoid to the preamble
    popt = fit_sinusoid.fit_sinusoid(preamble)
    print("Fitted preamble params:")
    for name in fit_sinusoid.SinusoidParams._fields:
        print(f"  {name}: {getattr(popt, name):.6f}")

    # 5. Plot
    t = np.arange(num_sample_in_preamble)
    fitted_signal = popt.values(t)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(t, preamble, label="Preamble (phase velocity)", alpha=0.7)
    ax.plot(t, fitted_signal, label="Fitted sinusoid", linewidth=2, linestyle="--")
    ax.set_title(f"Sinusoid fit to preamble of audio[{sample_index}]")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Phase velocity")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo_audio()
