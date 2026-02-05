# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fit_preamble

from importlib import reload

reload(fit_preamble)


def run_demo():
    # 1. Generate synthetic signal
    true_params = fit_preamble.PreambleParams(
        frequency=0.05,
        frequency_offset=0.01,
        time_offset=25.0,
        amplitude=2.0,
        vertical_offset=0.5,
        bit_width=80.0,
    )

    t = np.arange(1000)

    # Ground truth signal
    clean_signal = fit_preamble.signal_model(t, *true_params)
    true_bits = fit_preamble.pulse_train(t, true_params.time_offset, true_params.bit_width)

    # Add noise
    np.random.seed(42)
    noisy_signal = clean_signal + np.random.normal(0, 0.1, len(t))

    # 2. Fit the model
    popt = fit_preamble.fit_preamble(noisy_signal)

    # Calculate and print % difference for all parameters using a DataFrame
    rows = []
    for name in fit_preamble.PreambleParams._fields:
        true_val = getattr(true_params, name)
        fitted_val = getattr(popt, name)
        if true_val != 0:
            diff_pct = (fitted_val - true_val) / true_val * 100
            abs_diff = fitted_val - true_val
        else:
            diff_pct = np.nan
            abs_diff = fitted_val - true_val
        rows.append(
            {
                "parameter": name,
                "true": true_val,
                "fitted": fitted_val,
                "pct_diff": diff_pct,
                "abs_diff": abs_diff,
            }
        )
    df = pd.DataFrame(
        rows, columns=["parameter", "true", "fitted", "pct_diff", "abs_diff"]
    )
    print(df.to_string(index=False))

    # 3. Reconstruct fitted signal
    fitted_signal = fit_preamble.signal_model(t, *popt)
    fitted_bits = fit_preamble.pulse_train(t, popt.time_offset, popt.bit_width)

    # 4. Plot results
    _, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(t, noisy_signal, "k.", label="Noisy Input", alpha=0.3, markersize=3)
    ax1.plot(
        t, clean_signal, "b-", label="Ground Truth Signal", alpha=0.6, linewidth=0.5
    )
    ax1.plot(
        t, fitted_signal, "r-", label="Fitted Model Signal", linewidth=2, alpha=0.8
    )

    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Create a secondary axis for the bits
    ax2 = ax1.twinx()
    # Plot true bits with a slight offset or different style
    ax2.plot(t, true_bits, "b-", label="True Bits", linewidth=2, alpha=0.7)
    ax2.plot(t, fitted_bits, "r-", label="Fitted Bits", linewidth=1.5, alpha=0.7)

    ax2.set_ylabel("Bit Value", color="k")
    ax2.tick_params(axis="y", labelcolor="k")
    ax2.set_ylim(-1.5, 1.5)  # Keep bits nicely centered
    ax2.legend(loc="upper right")

    plt.title("Fit Preamble Demo")
    plt.tight_layout()


if __name__ == "__main__":
    run_demo()
    plt.show()

# %%
