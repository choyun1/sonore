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
        x = so.harmonic_complex(1.0, FS, 200, harmonics=np.arange(1, 10)).ramp(20e-3)
        y = so.time_stretch(x, 1.7)
        mid_x = x[0.2:0.8].rms
        mid_y = y[0.2 * y.duration : 0.8 * y.duration].rms
        assert so.amp_to_db(mid_y / mid_x) == pytest.approx(0, abs=0.5)

    def test_identity_stretch_reconstructs(self):
        x = so.harmonic_complex(1.0, FS, 150, harmonics=np.arange(1, 20), phases="random", rng=0)
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
        x = so.harmonic_complex(1.0, FS, 200, harmonics=np.arange(1, 15)).ramp(20e-3)
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
        x = so.harmonic_complex(1.0, FS, 220, harmonics=np.arange(1, 8)).ramp(20e-3)
        y = so.pv_analyze(x).to_sound()
        sl = slice(int(0.1 * FS), int(0.9 * FS))
        assert np.corrcoef(x.data[sl, 0], y.data[sl, 0])[0, 1] > 0.99
        assert so.amp_to_db(so.rms(y.data[sl]) / so.rms(x.data[sl])) == pytest.approx(0, abs=0.3)

    def test_oscillator_bank_frequency_map(self):
        x = so.pure_tone(1.0, FS, 500.0).ramp(20e-3)
        a = so.pv_analyze(x)
        assert abs(cents(dominant_freq(a.to_sound(freq_map=1.5)[0.2:0.8]), 750)) < 2
        assert abs(cents(dominant_freq(a.to_sound(freq_map=lambda f: f + 100)[0.2:0.8]), 600)) < 2
        stretched = a.to_sound(time_scale=2.0)
        assert len(stretched) == 2 * len(x)
        assert abs(cents(dominant_freq(stretched[0.4:1.6]), 500)) < 2


def _onsets(sound, threshold=0.5):
    """Times where the envelope rises through `threshold` of its peak after
    being below it for at least 0.1 s."""
    envelope = np.abs(sound.data[:, 0])
    width = int(0.01 * sound.fs)
    envelope = np.convolve(envelope, np.ones(width) / width, "same")
    above = envelope > threshold * envelope.max()
    rises = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    quiet_before = [not above[max(0, rise - int(0.1 * sound.fs)) : rise].any() for rise in rises]
    return rises[quiet_before] / sound.fs


def _two_bursts():
    """A 440 Hz tone on from 0.2 to 0.4 s and from 0.7 to 0.9 s, with 10 ms ramps."""
    t = np.arange(FS) / FS
    ramp = 0.01
    gate = sum(np.clip(np.minimum(t - on, off - t) / ramp, 0, 1) for on, off in ((0.2, 0.4), (0.7, 0.9)))
    return so.Sound(np.sin(2 * np.pi * 440 * t) * gate, FS)


class TestStretchTiming:
    @pytest.mark.parametrize("factor", [0.6, 3.0, 7.7])
    def test_events_land_at_factor_times_their_time(self, factor):
        # 7.7 is where rounding the analysis hop (to 7.68) and truncating it
        # (to 7.80) differ by 1.5%.
        onsets = _onsets(so.time_stretch(_two_bursts(), factor))
        assert len(onsets) == 2
        assert onsets[1] - onsets[0] == pytest.approx(0.5 * factor, rel=0.005)
        assert onsets[0] == pytest.approx(0.2 * factor, abs=0.01 * factor + 0.005)

    def test_length_rounds(self):
        x = so.Sound(so.pure_tone(1.1, FS, 440.0).data[:44101], FS)  # 44101 * 1.5 = 66151.5
        assert len(so.time_stretch(x, 1.5)) == 66152
        assert len(so.pv_analyze(x).to_sound(time_scale=1.5)) == 66152

    def test_factor_must_be_positive_and_silence_stays_silent(self):
        with pytest.raises(ValueError, match="positive"):
            so.time_stretch(so.pure_tone(0.1, FS, 440.0), 0)
        silence = so.time_stretch(so.Sound(np.zeros(4410), FS), 2.0)
        assert len(silence) == 8820 and not silence.data.any()


class TestAnalysisOptions:
    def test_window_is_even_and_hop_is_as_asked(self):
        x = so.pure_tone(0.5, FS, 1234.5)
        a = so.pv_analyze(x, win_dur=0.023, hop_dur=0.002)  # 1014.3 samples: rounds to 1014
        assert a.n_win % 2 == 0 and a.hop == round(0.002 * FS)
        assert so.pv_analyze(x, win_dur=0.0231).n_win % 2 == 0  # 1018.7 samples, odd once rounded
        k = np.argmax(a.magnitude[0].mean(axis=1))
        assert np.median(a.freq[0, k, 3:-3]) == pytest.approx(1234.5, abs=0.1)

    def test_oscillator_bank_time_scale_moves_events(self):
        stretched = so.pv_analyze(_two_bursts()).to_sound(time_scale=2.0)
        onsets = _onsets(stretched)
        assert len(onsets) == 2
        assert onsets[1] - onsets[0] == pytest.approx(1.0, rel=0.01)

    def test_oscillator_bank_drops_partials_above_nyquist_and_quiet_bins(self):
        t = np.arange(FS) / FS
        x = so.Sound(np.sin(2 * np.pi * 15000 * t) + 1e-3 * np.sin(2 * np.pi * 1000 * t), FS).ramp(20e-3)
        a = so.pv_analyze(x)
        # 15 kHz doubled is 30 kHz, above Nyquist: only the quiet 2 kHz partial is left
        doubled = a.to_sound(freq_map=2.0)[0.2:0.8]
        assert so.rms(doubled.data) < 2e-3
        assert abs(cents(dominant_freq(doubled), 2000)) < 5
        # a floor of -40 dB skips the 1 kHz partial, 60 dB down
        quiet = a.to_sound(floor_db=-40.0)[0.2:0.8]
        assert abs(cents(dominant_freq(quiet), 15000)) < 2
        spectrum = np.abs(np.fft.rfft(quiet.data[:, 0]))
        frequencies = np.fft.rfftfreq(len(quiet), 1 / FS)
        assert spectrum[np.abs(frequencies - 1000) < 20].max() < 1e-5 * spectrum.max()
