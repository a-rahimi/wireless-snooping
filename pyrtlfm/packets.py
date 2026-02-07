"""DemodPacket type, IQ analysis helpers, squelch, and serialization."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np


# TODO: fold the phase_velocity and amplitude methods functions into methods.
# TODO: rename iq_gated to iq.
class DemodPacket(NamedTuple):
    timestamp: float
    iq_gated: np.ndarray


def phase_velocity(iq: np.ndarray) -> np.ndarray:
    product = iq[1:] * np.conj(iq[:-1])
    return np.angle(product) / np.pi


def amplitude(iq: np.ndarray) -> np.ndarray:
    return np.abs(iq[:-1])


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


def save_packets(packets: list[DemodPacket], path: str | Path) -> None:
    """Save DemodPackets to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(packets, f)
    print(f"Saved {len(packets)} packets to {path}")


class _CompatUnpickler(pickle.Unpickler):
    """Remap pickled class references so captures saved from ``__main__`` load correctly."""

    def find_class(self, module: str, name: str):
        if name == "DemodPacket":
            module = __name__          # always resolve to this module
        return super().find_class(module, name)


def load_packets(path: str | Path) -> list[DemodPacket]:
    """Load DemodPackets from a pickle file."""
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()
