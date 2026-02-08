from typing import NamedTuple
import numpy as np
import scipy.optimize


class SinusoidParams(NamedTuple):
    frequency: float
    phase: float
    amplitude: float
    vertical_offset: float

    def values(self, t):
        return self.vertical_offset + self.amplitude * np.sin(
            2 * np.pi * self.frequency * (t - self.phase)
        )


def fit_sinusoid(signal: np.ndarray, num_annealing_passes: int = 10) -> SinusoidParams:

    # Initial parameter guesses using FFT-based estimation
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))

    # Find dominant frequency (excluding DC component)
    magnitude = np.abs(fft_vals)
    magnitude[0] = 0  # Remove DC component
    peak_idx = np.argmax(magnitude)

    freq_estimate = np.abs(freqs[peak_idx])
    p0 = SinusoidParams(
        frequency=freq_estimate,
        phase=-np.angle(fft_vals[peak_idx]) / (2 * np.pi * freq_estimate),
        amplitude=np.std(signal) * np.sqrt(2),
        vertical_offset=np.mean(signal),
    )

    t = np.arange(len(signal))
    params = p0

    # Fit the sinusoid using nonlinear least squares. Repeat the process a few times,
    # initially giving more weight to the early samples, until eventually all samples get equal weight.
    for i in range(num_annealing_passes + 1):
        popt, _ = scipy.optimize.curve_fit(
            lambda t, *args: SinusoidParams(*args).values(t),
            t,
            signal,
            p0=params,
            sigma=(t + 1.0) ** (2 * (-1 + i / num_annealing_passes)),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            maxfev=50000,
        )
        params = SinusoidParams(*popt)

    return params


class BinarizeResult(NamedTuple):
    values: np.ndarray
    bits: np.ndarray
    boundaries: list[tuple[int, int]]


def binarize(x: np.ndarray, params: SinusoidParams) -> BinarizeResult:
    bit_start = params.phase
    bit_width = 0.5 / params.frequency

    # Establish boundaries in floating point
    n_bits = int((len(x) - bit_start) / bit_width)
    edges = bit_start + np.arange(n_bits + 1) * bit_width
    int_edges = np.rint(edges).astype(int)

    # Compute means between rounded boundaries
    averages = np.array(
        [np.mean(x[int_edges[i] : int_edges[i + 1]]) for i in range(n_bits)]
    )
    boundaries = [(int_edges[i], int_edges[i + 1]) for i in range(n_bits)]

    bits = averages > params.vertical_offset
    return BinarizeResult(averages, bits, boundaries)
