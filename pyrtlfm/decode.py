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
    if len(bits) < 2:
        return bits
    # The first bit is always part of the preamble; check alternation
    # starting from bit 1.
    rest = bits[1:]
    if len(rest) < 2:
        return bits[0:0]
    same = np.where(rest[:-1] == rest[1:])[0]
    if len(same) == 0:
        return bits[0:0]  # whole thing is preamble
    return bits[same[0] + 1 :]


def bits_to_hex(bits: np.ndarray) -> str:
    """Convert an array of 0/1 values to a hex string (MSB-first nibbles).

    Bits that don't fill a complete nibble at the end are ignored.
    """
    bits = np.asarray(bits, dtype=int)
    n_nibbles = len(bits) // 4
    hex_chars = []
    for i in range(n_nibbles):
        nibble = bits[i * 4 : i * 4 + 4]
        value = (nibble[0] << 3) | (nibble[1] << 2) | (nibble[2] << 1) | nibble[3]
        hex_chars.append(f"{value:x}" + (" " if i % 2 == 1 else ""))
    return "".join(hex_chars).rstrip()
