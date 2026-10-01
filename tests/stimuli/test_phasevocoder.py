"""Phase vocoder."""

import numpy as np
import pytest
from helpers import FS, cents, dominant_freq

import sonore as so


class TestPhaseVocoder:
    @pytest.mark.parametrize("factor", [0.5, 1.5, 2.0, 3.0])
    def test_time_stretch_keeps_pitch_and_scales_duration(self, factor):
        x = so.pure_tone(1.0, FS, 440.0).ramp(20e-3)
        y = so.time_stretch(x, factor)
        assert len(y) == round(len(x) * factor)
        mid = y[0.2 * y.duration : 0.8 * y.duration]
        assert abs(cents(dominant_freq(mid), 440.0)) < 2

    def test_time_stretch_preserves_level(self):
        x = so.harmonic_complex(1.0, FS, 200, np.arange(1, 10)).ramp(20e-3)
        y = so.time_stretch(x, 1.7)
        mid_x = x[0.2:0.8].rms
        mid_y = y[0.2 * y.duration : 0.8 * y.duration].rms
        assert so.amp_to_db(mid_y / mid_x) == pytest.approx(0, abs=0.5)

    def test_identity_stretch_reconstructs(self):
        x = so.harmonic_complex(1.0, FS, 150, np.arange(1, 20), phases="random", rng=0)
        y = so.time_stretch(x, 1.0)
        np.testing.assert_allclose(y.data, x.data, atol=1e-8)

    @pytest.mark.parametrize("semitones", [-7, -1, 1, 4, 12])
    def test_pitch_shift(self, semitones):
        x = so.pure_tone(1.0, FS, 330.0).ramp(20e-3)
        y = so.pitch_shift(x, semitones)
        assert len(y) == len(x)
        target = 330.0 * 2 ** (semitones / 12)
        assert abs(cents(dominant_freq(y[0.2:0.8]), target)) < 2

    @staticmethod
    def crest_db(s):
        """Median crest factor over 10 ms blocks in the middle of the sound."""
        d = s[0.3 * s.duration : 0.7 * s.duration].data[:, 0]
        B = int(0.01 * FS)
        d = d[: len(d) // B * B].reshape(-1, B)
        return np.median(20 * np.log10(np.abs(d).max(axis=1) / np.sqrt(np.mean(d**2, axis=1))))

    def test_stationary_complex_keeps_phase_coherence(self):
        # a cosine-phase complex is peaky; losing inter-partial phase coherence flattens it
        x = so.harmonic_complex(1.0, FS, 200, np.arange(1, 15)).ramp(20e-3)
        for factor in (0.7, 2.5):
            assert self.crest_db(so.time_stretch(x, factor)) == pytest.approx(self.crest_db(x), abs=0.3)

    def test_phase_locking_reduces_phasiness(self):
        t = np.arange(2 * FS) / FS
        phase = 2 * np.pi * np.cumsum(200 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / FS
        x = so.Sound(sum(np.cos(k * phase) for k in range(1, 15)), FS).normalize().ramp(20e-3)
        locked = self.crest_db(so.time_stretch(x, 1.5, phase_lock=True))
        unlocked = self.crest_db(so.time_stretch(x, 1.5, phase_lock=False))
        assert locked == pytest.approx(self.crest_db(x), abs=0.5)
        assert unlocked < locked - 3

    def test_analysis_instantaneous_frequency(self):
        x = so.pure_tone(0.5, FS, 1234.5)
        a = so.pv_analyze(x)
        k = np.argmax(a.magnitude[0].mean(axis=1))
        f = np.median(a.freq[0, k, 3:-3])
        assert f == pytest.approx(1234.5, abs=0.1)  # sub-bin: bins are ~21.5 Hz apart

    def test_oscillator_bank_identity(self):
        x = so.harmonic_complex(1.0, FS, 220, np.arange(1, 8)).ramp(20e-3)
        y = so.pv_analyze(x).resynthesize()
        sl = slice(int(0.1 * FS), int(0.9 * FS))
        assert np.corrcoef(x.data[sl, 0], y.data[sl, 0])[0, 1] > 0.99
        assert so.amp_to_db(so.rms(y.data[sl]) / so.rms(x.data[sl])) == pytest.approx(0, abs=0.3)

    def test_oscillator_bank_frequency_map(self):
        x = so.pure_tone(1.0, FS, 500.0).ramp(20e-3)
        a = so.pv_analyze(x)
        assert abs(cents(dominant_freq(a.resynthesize(freq_map=1.5)[0.2:0.8]), 750)) < 2
        assert abs(cents(dominant_freq(a.resynthesize(freq_map=lambda f: f + 100)[0.2:0.8]), 600)) < 2
        stretched = a.resynthesize(time_scale=2.0)
        assert len(stretched) == 2 * len(x)
        assert abs(cents(dominant_freq(stretched[0.4:1.6]), 500)) < 2
