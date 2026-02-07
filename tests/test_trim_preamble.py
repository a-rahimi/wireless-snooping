import numpy as np
import pyrtlfm.decode


def test_trims_alternating_prefix():
    # First bit (0) always preamble, then 1,0,1 alternate, then 1==1 stops.
    bits = np.array([0, 1, 0, 1, 1, 1, 0])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert list(result) == [1, 1, 1, 0]


def test_trims_alternating_prefix_starting_with_one():
    # First bit (1) always preamble, then 0,1,0 alternate, then 0==0 stops.
    bits = np.array([1, 0, 1, 0, 0, 0, 1])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert list(result) == [0, 0, 0, 1]


def test_all_alternating_returns_empty():
    bits = np.array([0, 1, 0, 1, 0, 1])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert len(result) == 0


def test_no_preamble():
    # First bit is always preamble; after it [1, 0, 1, 0] alternates,
    # so the whole array is preamble.
    bits = np.array([1, 1, 0, 1, 0])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert len(result) == 0


def test_short_preamble():
    # First bit is preamble, then 1==1 immediately stops alternation,
    # so only the first bit is trimmed.
    bits = np.array([0, 1, 1, 0, 1])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert list(result) == [1, 1, 0, 1]


def test_single_bit_returned_as_is():
    bits = np.array([1])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert list(result) == [1]


def test_two_bits_always_preamble():
    # First bit is always preamble; only one bit after, so whole thing is preamble.
    bits = np.array([0, 1])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert len(result) == 0


def test_empty_input():
    bits = np.array([], dtype=int)
    result = pyrtlfm.decode.trim_preamble(bits)
    assert len(result) == 0


def test_boolean_input():
    bits = np.array([True, False, True, False, False, True])
    result = pyrtlfm.decode.trim_preamble(bits)
    assert list(result) == [False, False, True]
