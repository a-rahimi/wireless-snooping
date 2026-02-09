from typing import NamedTuple, Generator
import collections

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


def fit_sinusoid_iteratively(
    signal: np.ndarray, num_annealing_passes: int = 10, inlier_threshold: float = 0.1
) -> Generator[tuple[SinusoidParams, np.ndarray], None, None]:

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
    yield params, np.ones_like(signal, bool)

    # Anneal with a found rounds of nonlinear least squares. In each round, we add more of the sample.
    # We also find outliers from the previous round and don't include them.
    for n in np.linspace(len(t) // num_annealing_passes, len(t), num_annealing_passes):
        n = int(n)
        inliers = (t < n) & (np.abs(params.values(t) - signal) < inlier_threshold)

        def inlier_residuals(x: np.ndarray) -> np.ndarray:
            return SinusoidParams(*x).values(t[inliers]) - signal[inliers]

        result = scipy.optimize.least_squares(
            inlier_residuals,
            x0=list(params),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=50000,
        )
        params = SinusoidParams(*result.x)
        yield params, inliers


def fit_sinusoid(signal: np.ndarray, **fit_kwargs) -> SinusoidParams:
    return collections.deque(fit_sinusoid_iteratively(signal, **fit_kwargs), maxlen=1)[
        0
    ][0]


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
