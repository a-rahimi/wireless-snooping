from typing import NamedTuple
import numpy as np
import scipy.optimize


class PreambleParams(NamedTuple):
    frequency: float
    frequency_offset: float
    time_offset: float
    amplitude: float
    vertical_offset: float
    bit_width: float


def pulse_train(t, time_offset, bit_width):
    return np.sign(np.sin(np.pi * (t - time_offset) / bit_width))


def signal_model(
    t,
    frequency,
    frequency_offset,
    time_offset,
    amplitude,
    vertical_offset,
    bit_width,
):
    freq = frequency + frequency_offset * pulse_train(t, time_offset, bit_width)
    return vertical_offset + amplitude * np.sin(2 * np.pi * freq * (t - time_offset))


def signal_model_jac(
    t,
    frequency,
    frequency_offset,
    time_offset,
    amplitude,
    vertical_offset,
    bit_width,
):
    value = pulse_train(t, time_offset, bit_width)
    freq = frequency + frequency_offset * value
    phase = 2 * np.pi * freq * (t - time_offset)
    cos_phase = np.cos(phase)

    # partial derivatives
    d_frequency = amplitude * cos_phase * 2 * np.pi * (t - time_offset)
    d_frequency_offset = amplitude * cos_phase * 2 * np.pi * value * (t - time_offset)
    d_time_offset = amplitude * cos_phase * (-2 * np.pi * freq)
    d_amplitude = np.sin(phase)
    d_vertical_offset = np.ones_like(t)
    d_bit_width = np.zeros_like(t)

    return np.stack(
        [
            d_frequency,
            d_frequency_offset,
            d_time_offset,
            d_amplitude,
            d_vertical_offset,
            d_bit_width,
        ],
        axis=-1,
    )


def fit_preamble(signal: np.ndarray) -> PreambleParams:
    t = np.arange(len(signal))

    # Initial parameter guesses
    p0 = PreambleParams(
        frequency=0.03,
        frequency_offset=0.0,
        time_offset=0.0,
        amplitude=1.0,
        vertical_offset=0.0,
        bit_width=200.0,
    )

    # RANSAC parameters
    min_samples = 100
    if len(signal) < min_samples:
        min_samples = len(signal)
    
    residual_threshold = 0.2
    max_trials = 20
    
    best_inlier_count = 0
    best_popt = None
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_trials):
        # Sample random points
        sample_indices = rng.choice(len(signal), min_samples, replace=False)
        t_sample = t[sample_indices]
        signal_sample = signal[sample_indices]
        
        try:
            # Fast fit on subset
            popt, _ = scipy.optimize.curve_fit(
                signal_model,
                t_sample,
                signal_sample,
                p0=p0,
                maxfev=2000,
            )
            
            # Check inliers
            model_y = signal_model(t, *popt)
            residuals = np.abs(signal - model_y)
            inliers = residuals < residual_threshold
            n_inliers = np.sum(inliers)
            
            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_popt = popt
                
        except (RuntimeError, ValueError):
            continue

    # If RANSAC found a good model, refine it using all inliers
    if best_popt is not None and best_inlier_count > min_samples:
        model_y = signal_model(t, *best_popt)
        inliers = np.abs(signal - model_y) < residual_threshold
        
        try:
            popt, _ = scipy.optimize.curve_fit(
                signal_model,
                t[inliers],
                signal[inliers],
                p0=best_popt,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                maxfev=10000,
            )
            return PreambleParams(*popt)
        except (RuntimeError, ValueError):
            return PreambleParams(*best_popt)

    # Fallback to original fit on all data
    popt, _ = scipy.optimize.curve_fit(
        signal_model,
        t,
        signal,
        p0=p0,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        maxfev=10000,
    )
    return PreambleParams(*popt)
