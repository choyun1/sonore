"""Stimulus generators."""

import numpy as np
import pytest
from helpers import FAST, FS

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


HOP = 0.005  # frame period of the contours below [s], as in so.f0_track


def frames(f, dur):
    t = np.arange(0, dur + HOP / 2, HOP)
    return t, f(t)


def power_above(x, f_cut, fs=FAST, n_win=512):
    """Power above f_cut over total from a Hann STFT [dB]."""
    segs = np.lib.stride_tricks.sliding_window_view(x, n_win)[:: n_win // 4] * np.hanning(n_win)
    p = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    f = np.fft.rfftfreq(n_win, 1 / fs)
    return 10 * np.log10(p[:, f > f_cut].sum() / p.sum())


class TestHarmonicContours:
    """harmonic_complex with an F0 contour (docs/design/harmonic-source.md)."""

    def test_constant_contour_is_the_fixed_complex(self):
        t, f = frames(lambda t: np.full_like(t, 220.0), 0.5)
        a = so.harmonic_complex(0.5, FAST, (t, f), np.arange(1, 11), phases="random", rng=0)
        b = so.harmonic_complex(0.5, FAST, 220.0, np.arange(1, 11), phases="random", rng=0)
        np.testing.assert_allclose(a.data, b.data, atol=1e-10)

    def test_linear_glide_is_exact(self):
        t, f = frames(lambda t: 100 + 200 * t, 1.0)
        x = so.harmonic_complex(1.0, FAST, (t, f), [1])
        ts = np.arange(len(x)) / FAST
        ref = np.cos(2 * np.pi * (100 * ts + 100 * ts**2))
        np.testing.assert_allclose(x.data[:, 0], ref / np.sqrt(np.mean(ref**2)), atol=1e-9)

    def test_harmonics_follow_the_contour(self):
        from scipy.signal import hilbert

        def vib(t):
            return 150 * (1 + 0.04 * np.sin(2 * np.pi * 5.5 * t))

        # 2 s: the analytic signal's own error at the ends of a shorter excerpt is larger than the bound
        t, f = frames(vib, 2.0)
        mid = (np.arange(2 * FAST - 1) + 0.5) / FAST
        inner = (mid > 0.1) & (mid < 1.9)
        for k in (1, 10):
            x = so.harmonic_complex(2.0, FAST, (t, f), [k]).data[:, 0]
            inst = np.diff(np.unwrap(np.angle(hilbert(x)))) * FAST / (2 * np.pi)
            assert np.max(np.abs(inst[inner] / (k * vib(mid[inner])) - 1)) < 2e-4

    def test_gaps_are_filled_and_silent(self):
        # voiced 0-0.2 s at 100 Hz, unvoiced 0.2-0.4 s, voiced 0.4-0.6 s at 200 Hz
        t, f = frames(lambda t: np.where(t < 0.2, 100.0, np.where(t < 0.4, 0.0, 200.0)), 0.6)
        x = so.harmonic_complex(0.6, FAST, (t, f), [1]).data[:, 0]
        gap = slice(int(0.21 * FAST), int(0.39 * FAST))
        assert np.max(np.abs(x[gap])) == 0
        # filled, not swept through zero: the period never exceeds 1/100 s while sounding
        crossings = np.flatnonzero(np.diff(np.signbit(x)))
        assert np.max(np.diff(crossings)) <= FAST / 100 / 2 + 2

    def test_rising_glide_does_not_alias(self):
        t, f = frames(lambda t: 100 * 10**t, 1.0)
        x = so.harmonic_complex(1.0, FAST, (t, f))
        assert power_above(x.data[:, 0], 0.45 * FAST + 400) < -90

    def test_f0_track_pair_and_channels(self):
        track = so.f0_track(so.harmonic_complex(0.3, FAST, 150.0, np.arange(1, 10)))
        assert track.voiced.any()
        a = so.harmonic_complex(0.3, FAST, track)
        b = so.harmonic_complex(0.3, FAST, (track.t, track.f0[0]))
        np.testing.assert_array_equal(a.data, b.data)
        two = so.harmonic_complex(0.3, FAST, (track.t, np.vstack([track.f0[0], 1.5 * track.f0[0]])))
        assert two.n_channels == 2

    def test_unvoiced_noise_is_reproducible(self):
        t, f = frames(lambda t: np.where(t < 0.2, 0.0, 120.0), 0.4)
        a = so.harmonic_complex(0.4, FAST, (t, f), unvoiced="noise", rng=3)
        b = so.harmonic_complex(0.4, FAST, (t, f), unvoiced="noise", rng=3)
        np.testing.assert_array_equal(a.data, b.data)
        assert np.std(a.data[: int(0.15 * FAST)]) > 0.1

    def test_phases_need_one_per_harmonic(self):
        t, f = frames(lambda t: np.full_like(t, 1000.0), 0.1)
        n = int(0.45 * FAST // 1000)
        so.harmonic_complex(0.1, FAST, (t, f), phases=np.zeros(n))
        with pytest.raises(ValueError, match="needs one starting phase"):
            so.harmonic_complex(0.1, FAST, (t, f), phases=np.zeros(n + 1))

    def test_amplitudes_as_a_spectral_envelope(self):
        t, f = frames(lambda t: np.full_like(t, 200.0), 0.2)
        x = so.harmonic_complex(0.2, FAST, (t, f), [1, 2], amplitudes=lambda t, f: (f < 300).astype(float))
        y = so.harmonic_complex(0.2, FAST, 200.0, [1])
        np.testing.assert_allclose(x.data, y.data, atol=1e-10)

    def test_complex_amplitudes_by_harmonic_number(self):
        """A complex gain shifts each harmonic's phase by its angle, and the
        function may take the harmonic number as a third argument."""
        amplitudes = np.array([1.0, 0.5, 0.25])
        phases = np.array([0.3, -1.2, 2.0])

        def gains(t, f, number):
            return np.full_like(t, amplitudes[number - 1] * np.exp(1j * phases[number - 1]), dtype=complex)

        want = so.harmonic_complex(0.2, FAST, 200.0, [1, 2, 3], amplitudes, phases)
        got = so.harmonic_complex(0.2, FAST, 200.0, [1, 2, 3], gains)
        np.testing.assert_allclose(got.data, want.data, atol=1e-12)
        t, f = frames(lambda t: np.full_like(t, 200.0), 0.2)
        want = so.harmonic_complex(0.2, FAST, (t, f), [1, 2, 3], amplitudes, phases)
        got = so.harmonic_complex(0.2, FAST, (t, f), [1, 2, 3], gains)
        np.testing.assert_allclose(got.data, want.data, atol=1e-12)


class TestNamedWaveforms:
    """Square, sawtooth and pulse train go through harmonic_complex."""

    @staticmethod
    def previous(f0, phase, kind):
        """The band-limited waveforms as they were computed before."""
        t = np.arange(int(round(0.3 * FS))) / FS
        wt = 2 * np.pi * f0 * t + phase
        r = int(np.ceil(FS / 2 / f0)) - 1
        n, a, trig = {
            "square": (np.arange(1, r + 1, 2), lambda n: 1 / n, np.sin),
            "sawtooth": (np.arange(1, r + 1), lambda n: (-1.0) ** (n + 1) / n, np.sin),
            "pulse": (np.arange(1, r + 1), np.ones_like, np.cos),
        }[kind]
        x = sum(ak * trig(nk * wt) for nk, ak in zip(n, a(n.astype(float)), strict=True))
        return x / np.sqrt(np.mean(x**2))

    def test_fixed_f0_unchanged(self):
        for f0 in (97.3, 1000.0):
            for kind, fn in (
                ("square", so.square_wave),
                ("sawtooth", so.sawtooth_wave),
                ("pulse", so.pulse_train),
            ):
                np.testing.assert_allclose(
                    fn(0.3, FS, f0, phase=0.4).data[:, 0], self.previous(f0, 0.4, kind), atol=1e-10
                )

    def test_contours_do_not_alias(self):
        t, f = frames(lambda t: 100 * 10**t, 1.0)
        for fn in (so.square_wave, so.sawtooth_wave, so.pulse_train, so.schroeder_complex):
            assert power_above(fn(1.0, FAST, (t, f)).data[:, 0], 0.45 * FAST + 400) < -90

    def test_unbandlimited_needs_a_number(self):
        with pytest.raises(ValueError, match="fixed f0"):
            so.square_wave(0.1, FS, (np.array([0.0, 0.1]), np.array([100.0, 200.0])), bandlimited=False)
