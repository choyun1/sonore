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


class TestResonator:
    """Klatt's (1980) resonator and antiresonator."""

    def test_unit_gain_at_0_hz_peak_at_f_and_width_bw(self):
        fs = 16000
        imp = so.Sound(np.r_[1.0, np.zeros(2**17 - 1)], fs)
        h = np.abs(np.fft.rfft(so.resonator(imp, 500, 60).data[:, 0]))
        f = np.fft.rfftfreq(2**17, 1 / fs)
        assert h[0] == pytest.approx(1.0, abs=1e-9)
        assert f[np.argmax(h)] == pytest.approx(500, abs=1)
        band = f[h >= h.max() / np.sqrt(2)]
        assert band.max() - band.min() == pytest.approx(60, abs=1)

    def test_antiresonator_inverts_resonator(self):
        x = so.gaussian_noise(0.2, FS, rng=0)
        f = ([0, 0.2], [300, 2000])
        y = so.antiresonator(so.resonator(x, f, 80), f, 80)
        np.testing.assert_allclose(y.data, x.data, atol=1e-10)

    def test_constant_track_equals_number(self):
        x = so.gaussian_noise(0.1, FS, rng=0)
        a = so.resonator(x, 700, 100)
        b = so.resonator(x, ([0, 1], [700, 700]), ([0.05], [100]))
        np.testing.assert_allclose(b.data, a.data, atol=1e-10)

    def test_moving_formant_has_no_clicks(self):
        # F1 300 -> 700 Hz over 40 ms on a low-pass voiced source: nothing
        # reaches 5 kHz, as there would if the filter restarted at each step.
        fs = 16000
        k = np.arange(1, 20)
        x = so.harmonic_complex(0.06, fs, 120, harmonics=k, amplitudes=1 / k**2)
        y = so.resonator(x, ([0, 0.04], [300, 700]), 60).data[:, 0] * np.hanning(len(x))
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / fs)
        assert 10 * np.log10(s[f > 5000].sum() / s.sum()) < -80

    def test_each_channel_filtered(self):
        x = so.correlated_noise(0.1, FS, corr=0, rng=0)
        y = so.resonator(x, ([0, 0.1], [500, 900]), 100)
        z = so.resonator(so.Sound(x.data[:, 1], FS), ([0, 0.1], [500, 900]), 100)
        np.testing.assert_allclose(y.data[:, 1], z.data[:, 0])

    @pytest.mark.parametrize("f, bw", [(-1, 50), (FS / 2, 50), (500, 0)])
    def test_rejects_bad_values(self, f, bw):
        with pytest.raises(ValueError):
            so.resonator(so.silence(0.01, FS), f, bw)
