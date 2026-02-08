"""Pure-data helpers for decoding binarised FSK frames."""

import numpy as np


def trim_preamble(bits: np.ndarray) -> np.ndarray:
    """Remove the leading preamble from a bit sequence.

    The first bit is always part of the preamble.  The remaining preamble
    consists of the longest run of alternating bits starting from bit 1
    (i.e. each consecutive pair in ``bits[1:]`` differs).  The first index
    in ``bits[1:]`` where two adjacent bits are equal marks the end of the
    preamble; everything from that index onward is returned.

    Returns an empty array if the entire sequence is preamble.
    """
    # ignore the first bit. it's usually a partial bit and often wrong.
    bits = bits[1:]

    # advance until we encounter a pair of equal bits.
    for i in range(len(bits) - 1):
        if bits[i] == bits[i + 1]:
            return bits[i + 1 :]

    # We ate all the bits.
    return np.array([], dtype=np.uint8)


def bits_to_uint8(bits: np.ndarray) -> np.ndarray:
    """Convert an array of 0/1 values to a uint8 array (MSB-first bytes).

    Bits that don't fill a complete byte at the end are ignored.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n_bytes = len(bits) // 8
    if n_bytes == 0:
        return np.empty(0, dtype=np.uint8)
    # Reshape into (n_bytes, 8) and pack each row MSB-first
    bit_matrix = bits[: n_bytes * 8].reshape(n_bytes, 8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return (bit_matrix @ weights).astype(np.uint8)


def find_after(bits: np.ndarray, target: int) -> np.ndarray:
    """Return all bits after the first occurrence of *target* in *bits*.

    *target* is an integer (e.g. ``0x2dd4``).  It is converted to its
    binary representation and searched for in *bits*.  If found, everything
    after the last bit of the match is returned.  If the target does not
    appear in *bits*, an empty array is returned.

    Parameters
    ----------
    bits : array-like of 0/1
        The bit sequence to search in.
    target : int
        The pattern to look for, expressed as an integer (e.g. ``0x2dd4``).
        The number of bits is derived from the minimal byte-aligned
        representation (e.g. ``0x2dd4`` → 16 bits).

    Returns
    -------
    np.ndarray
        The remaining bits after the match, or an empty uint8 array.
    """
    bits = np.asarray(bits, dtype=np.uint8)

    # Convert integer to a bit pattern (byte-aligned)
    n_bytes = max((target.bit_length() + 7) // 8, 1)
    target_bytes = target.to_bytes(n_bytes, byteorder="big")
    target_bits = np.unpackbits(np.frombuffer(target_bytes, dtype=np.uint8))

    n = len(target_bits)
    if n == 0 or len(bits) < n:
        return np.empty(0, dtype=np.uint8)

    # Slide through bits looking for the pattern
    for i in range(len(bits) - n + 1):
        if np.array_equal(bits[i : i + n], target_bits):
            return bits[i + n :]

    return np.empty(0, dtype=np.uint8)


def uint8_to_hex(data: np.ndarray) -> str:
    """Convert a uint8 array to a space-separated hex string.

    Each byte is rendered as two lowercase hex digits, e.g. ``"a3 f0 12"``.
    """
    return " ".join(f"{b:02x}" for b in np.asarray(data, dtype=np.uint8))
