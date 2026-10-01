"""Stimulus generators."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestGenerators:
    def test_tone_frequency_exact_for_short_tones(self):
        x = so.pure_tone(0.01, FS, 1000)
        t = np.arange(len(x)) / FS
        np.testing.assert_allclose(x.data[:, 0], np.sqrt(2) * np.cos(2 * np.pi * 1000 * t), atol=1e-2)

    def test_harmonics_above_nyquist_dropped(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            x = so.harmonic_complex(0.1, FS, 5000, np.arange(1, 6))
        assert x.rms == pytest.approx(1)

    def test_bandlimited_square_has_no_aliases(self):
        x = so.square_wave(1.0, FS, 1000)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        off = np.abs(f / 1000 - np.round(f / 1000)) * 1000 > 5
        assert X[off].max() < 1e-6 * X.max()

    def test_pulse_train_period(self):
        x = so.pulse_train(0.1, FS, 441, bandlimited=False)
        idx = np.flatnonzero(x.data[:, 0])
        np.testing.assert_array_equal(np.diff(idx), 100)

    def test_noise_reproducible(self):
        a = so.gaussian_noise(0.1, FS, rng=42)
        b = so.gaussian_noise(0.1, FS, rng=42)
        np.testing.assert_array_equal(a.data, b.data)

    def test_noise_tilt_is_db_per_octave(self):
        x = so.gaussian_noise(20, FS, tilt=-3, rng=0)
        spec = so.long_term_spectrum(x)
        lo, hi = spec.level_at(np.array([1000, 2000]))
        assert hi - lo == pytest.approx(-3, abs=0.5)

    def test_noise_band(self):
        x = so.gaussian_noise(1, FS, band=(500, 1000), rng=0)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        assert X[(f < 490) | (f > 1010)].max() < 1e-9

    def test_correlated_noise(self):
        for c in (-0.5, 0, 0.8):
            x = so.correlated_noise(5, FS, corr=c, rng=0)
            assert np.corrcoef(x.data.T)[0, 1] == pytest.approx(c, abs=0.02)

    def test_irn_pitch(self):
        x = so.iterated_ripple_noise(1, FS, delay=5e-3, iterations=8, rng=0)
        ac = np.correlate(x.data[:4000, 0], x.data[:4000, 0], "full")[3999:]
        lag = np.argmax(ac[50:400]) + 50
        assert lag == pytest.approx(5e-3 * FS, abs=1)
