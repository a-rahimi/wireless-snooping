"""Power-based squelch detector for IQ chunks."""

from __future__ import annotations

import numpy as np


def rms_complex(iq: np.ndarray) -> float:
    """RMS of complex IQ (magnitude), with DC correction similar to rtl_fm.

    rtl_fm uses RMS of interleaved I,Q with DC correction in squares.
    Here we use sqrt(mean(|iq - dc|^2)) where dc = mean(iq).
    """
    iq = np.asarray(iq, dtype=np.complex128)
    dc = np.mean(iq)
    centered = iq - dc
    power = np.mean(np.real(centered) ** 2 + np.imag(centered) ** 2)
    return float(np.sqrt(power))


def squelch(iq: np.ndarray, threshold: float, zero_when_closed: bool = True) -> tuple[bool, np.ndarray]:
    """Run power squelch on an IQ chunk.

    Parameters
    ----------
    iq : np.ndarray
        1D complex IQ chunk.
    threshold : float
        Squelch threshold. If RMS of the chunk is below this, squelch is closed.
        Use 0 to disable (squelch always open).
    zero_when_closed : bool, optional
        If True and squelch is closed, return a zeroed copy of iq so downstream
        (demod) does not process noise. If False, return iq unchanged.

    Returns
    -------
    squelch_open : bool
        True if power >= threshold (or threshold is 0).
    iq_out : np.ndarray
        Same shape as iq. Zeroed when squelch closed and zero_when_closed=True;
        otherwise a copy of iq (so caller can mutate without affecting original).
    """
    iq = np.asarray(iq)
    if threshold <= 0:
        return True, np.copy(iq)
    rms = rms_complex(iq)
    open_ = rms >= threshold
    if open_:
        return True, np.copy(iq)
    if zero_when_closed:
        return False, np.zeros_like(iq)
    return False, np.copy(iq)
