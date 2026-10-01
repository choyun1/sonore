"""List-level processing and filtering."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestProcessing:
    def test_bandpass_filters_each_channel_independently(self):
        n = so.correlated_noise(1, FS, corr=0, rng=0)
        y = so.bandpass(n, 500, 1000)
        assert abs(np.corrcoef(y.data.T)[0, 1]) < 0.1

    def test_pad_uses_zeros(self):
        a, b = so.pad([so.Sound(np.ones(3), FS), so.Sound(np.ones(5), FS)], align="center")
        np.testing.assert_array_equal(a.data[:, 0], [0, 1, 1, 1, 0])

    def test_concat_upmixes_mono(self):
        s = so.concat([so.correlated_noise(0.1, FS, rng=0), so.silence(0.1, FS)])
        assert s.n_channels == 2

    def test_mix_different_lengths(self):
        m = so.mix([so.silence(0.1, FS), so.pure_tone(0.2, FS, 100)])
        assert m.duration == pytest.approx(0.2)
