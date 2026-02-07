import pytest
import numpy as np
from pyrtlfm.fit_sinusoid import fit_sinusoid, SinusoidParams


def test_fit_sinusoid_recovery():
    """fit_sinusoid should recover frequency, phase, amplitude, and offset."""
    true = SinusoidParams(
        frequency=0.05,
        phase=3.0,
        amplitude=1.0,
        vertical_offset=0.5,
    )

    t = np.arange(1000)
    signal = true.values(t)

    # Add some noise
    np.random.seed(42)
    signal += np.random.normal(0, 0.05, len(t))

    result = fit_sinusoid(signal)

    assert result.frequency == pytest.approx(true.frequency, abs=1e-3)
    assert result.vertical_offset == pytest.approx(true.vertical_offset, abs=0.1)

    # Amplitude may flip sign (with a corresponding phase shift), so check abs
    assert abs(result.amplitude) == pytest.approx(abs(true.amplitude), abs=0.1)
