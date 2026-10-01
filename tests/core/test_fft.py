"""The FFT thread setting: results do not depend on it."""

import numpy as np
import pytest

import sonore as so
from sonore.core.fft import fast_padding


def test_set_fft_workers_as_call_and_block():
    before = so.fft_workers()
    assert before >= 1
    with so.set_fft_workers(1):
        assert so.fft_workers() == 1
        with so.set_fft_workers(2):
            assert so.fft_workers() == 2
        assert so.fft_workers() == 1
    assert so.fft_workers() == before
    so.set_fft_workers(1)
    try:
        assert so.fft_workers() == 1
    finally:
        so.set_fft_workers(before)
    with pytest.raises(ValueError):
        so.set_fft_workers(0)


def test_results_are_identical_for_any_number_of_workers():
    s = so.gaussian_noise(0.5, 16000, rng=0)
    fb = so.ERBFilterbank(20, 80.0, 6000.0)
    out = {}
    for n in (1, 3):
        with so.set_fft_workers(n):
            sb = fb.analyze(s)
            out[n] = (sb.data, sb.envelopes().data, sb.synthesize().data)
    for a, b in zip(out[1], out[3], strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("n", [1000, 1001, 48000, 48001, 213222])
def test_fast_padding(n):
    p = fast_padding(n, 37)
    m = n + 2 * p
    for q in (2, 3, 5, 7, 11):
        while m % q == 0:
            m //= q
    assert p >= 37 and m == 1
    assert fast_padding(n, p) == p and all(fast_padding(n, k) == p for k in range(37, p))
