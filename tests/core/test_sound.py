"""The Sound container."""

import numpy as np
import pytest
from helpers import FS, peak_freq

import sonore as so


class TestSound:
    def test_numpy_scalar_on_left(self):
        x = so.pure_tone(0.1, FS, 440)
        assert isinstance(np.float64(2) * x, so.Sound)
        assert isinstance(np.int64(2) * x, so.Sound)
        np.testing.assert_allclose((np.float64(2) * x).data, 2 * x.data)

    def test_reflected_division_and_subtraction(self):
        x = so.Sound(np.array([1.0, 2.0, 4.0]), FS)
        np.testing.assert_allclose((2 / x).data[:, 0], [2, 1, 0.5])
        np.testing.assert_allclose((x - x).data, 0)
        y = so.Sound(np.ones(3), FS)
        np.testing.assert_allclose((y - x).data[:, 0], [0, -1, -3])

    def test_adding_number_is_an_error_but_sum_works(self):
        x = so.pure_tone(0.1, FS, 440)
        with pytest.raises(TypeError, match=r"6\*dB"):
            x + 6
        np.testing.assert_allclose(sum([x, x]).data, 2 * x.data)

    def test_gain_db(self):
        x = so.pure_tone(0.1, FS, 440)
        assert so.amp_to_db(x.gain_db(6).rms / x.rms) == pytest.approx(6)

    def test_immutable(self):
        x = so.pure_tone(0.1, FS, 440)
        before = x.data.copy()
        x.ramp(0.01)
        np.testing.assert_array_equal(x.data, before)
        with pytest.raises(ValueError):
            x.data[0] = 1

    def test_channels_first_is_rejected(self):
        with pytest.raises(ValueError, match="channels-first"):
            so.Sound(np.zeros((2, 1000)), FS)

    def test_mono_broadcasts_against_stereo(self):
        env = so.Sound(np.linspace(0, 1, 100), FS)
        s = so.correlated_noise(100 / FS, FS, corr=0, rng=0)
        assert (env * s).n_channels == 2

    def test_mismatched_fs_is_an_error(self):
        with pytest.raises(ValueError, match="sampling rates"):
            so.silence(0.1, 44100) + so.silence(0.1, 48000)

    def test_time_slicing(self):
        x = so.pure_tone(1.0, FS, 440)
        assert len(x[0.25:0.5]) == FS // 4
        with pytest.raises(TypeError):
            x[0]

    @pytest.mark.parametrize("d", [10 / FS, 10.37 / FS])
    def test_fractional_delay(self, d):
        x = so.gaussian_noise(0.1, FS, band=(100, 15000), rng=1)
        y = x.delay(d)
        # compare in the frequency domain: phase ramp of exactly d
        n = len(y)
        X, Y = np.fft.rfft(x.data[:, 0], n), np.fft.rfft(y.data[:, 0], n)
        f = np.fft.rfftfreq(n, 1 / FS)
        band = (f > 200) & (f < 10000)
        phase = np.unwrap(np.angle(Y[band] / X[band]))
        slope = np.polyfit(f[band], phase, 1)[0]
        assert -slope / (2 * np.pi) == pytest.approx(d, rel=1e-3)

    def test_resample(self):
        x = so.pure_tone(0.5, 48000, 1000).resample(44100)
        assert x.fs == 44100 and len(x) == 22050
        assert peak_freq(x) == pytest.approx(1000, abs=2)
