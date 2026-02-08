from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np


class DemodPacket(NamedTuple):
    timestamp: float
    iq: np.ndarray
    sample_rate_hz: float

    def phase_velocity(self) -> np.ndarray:
        """Phase velocity (normalised to [-1, 1]) between consecutive IQ samples."""
        product = self.iq[1:] * np.conj(self.iq[:-1])
        return np.angle(product) / np.pi

    def amplitude(self) -> np.ndarray:
        """Per-sample amplitude (length = len(iq) - 1)."""
        return np.abs(self.iq[:-1])


def squelch(iq: np.ndarray, threshold: float) -> bool:
    """Return True if the IQ chunk passes the power squelch.

    Parameters
    ----------
    iq : np.ndarray
        1D complex IQ chunk.
    threshold : float
        Squelch threshold. If RMS of the chunk is below this, squelch is closed.
        Use 0 to disable (squelch always open).
    """
    iq = np.asarray(iq, dtype=np.complex128)
    centered = iq - np.mean(iq)
    rms = np.sqrt(np.mean(np.real(centered) ** 2 + np.imag(centered) ** 2))
    return rms >= threshold


def save_packets(packets: list[DemodPacket], path: str | Path) -> None:
    """Save DemodPackets to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(packets, f)
    print(f"Saved {len(packets)} packets to {path}")


def load_packets(path: str | Path) -> list[DemodPacket]:
    """Load DemodPackets from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
