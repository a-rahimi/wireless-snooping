import numpy as np
import pyrtlfm.decode


def bits(s: str) -> np.ndarray:
    """Convert a string of '0's and '1's to a numpy int array.

    >>> bits("0101")
    array([0, 1, 0, 1])
    """
    return np.array([int(c) for c in s], dtype=int)


def bitstr(arr) -> str:
    """Convert a numpy array (or list) of 0/1 values to a string.

    >>> bitstr(np.array([0, 1, 0, 1]))
    '0101'
    """
    return "".join(str(int(v)) for v in arr)


def test_trims_alternating_prefix():
    result = pyrtlfm.decode.trim_preamble(bits("0101110"))
    assert bitstr(result) == "110"


def test_trims_alternating_prefix_starting_with_one():
    # First bit (1) always preamble, then 0,1,0 alternate, then 0==0 stops.
    result = pyrtlfm.decode.trim_preamble(bits("1010001"))
    assert bitstr(result) == "001"


def test_all_alternating_returns_empty():
    result = pyrtlfm.decode.trim_preamble(bits("010101"))
    assert len(result) == 0


def test_no_preamble():
    # The first bit is ignored.
    result = pyrtlfm.decode.trim_preamble(bits("11010"))
    assert len(result) == 0


def test_short_preamble():
    result = pyrtlfm.decode.trim_preamble(bits("01101"))
    assert bitstr(result) == "101"


def test_boolean_input():
    bits_arr = np.array([True, False, True, False, False, True])
    result = pyrtlfm.decode.trim_preamble(bits_arr)
    assert bitstr(result) == "01"
