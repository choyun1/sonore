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

    def test_linear_chirp_phase(self):
        """cos(2 pi (f0 t + (f1 - f0) t^2 / 2T) + phase), scaled to RMS 1."""
        duration, f0, f1, phase = 0.05, 200.0, 2000.0, 0.3
        x = so.linear_chirp(duration, FS, f0, f1, phase).data[:, 0]
        t = np.arange(len(x)) / FS
        expected = np.cos(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)) + phase)
        np.testing.assert_allclose(x, expected / np.sqrt(np.mean(expected**2)), atol=1e-9)

    def test_harmonics_above_nyquist_dropped(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            x = so.harmonic_complex(0.1, FS, 5000, np.arange(1, 6))
        assert x.rms == pytest.approx(1)

    def test_fixed_f0_honors_f_max(self):
        """A fixed F0 with f_max stops below it and fades as a contour does; without
        f_max it is unchanged."""
        f0, f_max = 120.0, 5000.0
        x = so.harmonic_complex(1.0, FS, f0, f_max=f_max).data[:, 0]
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / FS)
        assert spectrum[f >= f_max].sum() < 1e-12 * spectrum.sum()
        top = np.argmin(np.abs(f - 41 * f0))  # 4920 Hz, in the fade
        first = np.argmin(np.abs(f - f0))
        fade = np.cos(np.pi / 2 * (41 * f0 - 0.9 * f_max) / (0.1 * f_max)) ** 2
        assert spectrum[top] / spectrum[first] == pytest.approx(fade**2, rel=1e-6)
        contour = so.harmonic_complex(1.0, FS, ([0.0, 1.0], [f0, f0]), f_max=f_max, ramp=0)
        np.testing.assert_allclose(x / np.std(x), contour.data[:, 0] / contour.data[:, 0].std(), atol=1e-6)
        unchanged = so.harmonic_complex(1.0, FS, f0, harmonics=np.arange(1, int(np.ceil(FS / 2 / f0))))
        np.testing.assert_array_equal(so.harmonic_complex(1.0, FS, f0).data, unchanged.data)

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

    def test_long_term_spectrum_window_is_in_seconds(self):
        x = so.gaussian_noise(1, FS, rng=0)
        assert np.diff(so.long_term_spectrum(x).f)[0] == pytest.approx(10.0)
        assert np.diff(so.long_term_spectrum(x, win_dur=0.05).f)[0] == pytest.approx(20.0)

    def test_noise_band(self):
        x = so.gaussian_noise(1, FS, band=(500, 1000), rng=0)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        assert X[(f < 490) | (f > 1010)].max() < 1e-9

    def test_correlated_noise_is_exact(self):
        """Exact correlation and RMS 1 per channel, even with few components
        (9 here: 0.1 s, 500-590 Hz), where a plain mix misses by about 0.2."""
        for c in (-1, -0.5, 0, 0.2, 0.8, 1):
            for kwargs in ({}, {"band": (500, 590)}):
                x = so.correlated_noise(0.1, FS, corr=c, rng=0, **kwargs).data
                zero_lag = (x[:, 0] @ x[:, 1]) / np.sqrt((x[:, 0] @ x[:, 0]) * (x[:, 1] @ x[:, 1]))
                assert zero_lag == pytest.approx(c, abs=1e-12)
                np.testing.assert_allclose(np.sqrt(np.mean(x**2, axis=0)), 1, atol=1e-12)
        with pytest.raises(ValueError, match="corr must be"):
            so.correlated_noise(0.1, FS, corr=1.5)

    def test_irn_pitch(self):
        x = so.iterated_ripple_noise(1, FS, delay=5e-3, iterations=8, rng=0)
        ac = np.correlate(x.data[:4000, 0], x.data[:4000, 0], "full")[3999:]
        lag = np.argmax(ac[50:400]) + 50
        assert lag == pytest.approx(5e-3 * FS, abs=1)


HOP = 0.005  # hop of the contours below [s], as in so.f0_track


def time_windows(f, dur):
    t = np.arange(0, dur + HOP / 2, HOP)
    return t, f(t)


def power_above(x, f_cut, fs=FAST, n_win=512):
    """Power above f_cut over total from a Hann STFT [dB]."""
    segs = np.lib.stride_tricks.sliding_window_view(x, n_win)[:: n_win // 4] * np.hanning(n_win)
    p = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    f = np.fft.rfftfreq(n_win, 1 / fs)
    return 10 * np.log10(p[:, f > f_cut].sum() / p.sum())


class TestHarmonicContours:
    """harmonic_complex with an F0 contour (docs/design/sources/harmonic-source.md)."""

    def test_constant_contour_is_the_fixed_complex(self):
        t, f = time_windows(lambda t: np.full_like(t, 220.0), 0.5)
        a = so.harmonic_complex(0.5, FAST, (t, f), np.arange(1, 11), phases="random", rng=0)
        b = so.harmonic_complex(0.5, FAST, 220.0, np.arange(1, 11), phases="random", rng=0)
        np.testing.assert_allclose(a.data, b.data, atol=1e-10)

    def test_linear_glide_is_exact(self):
        t, f = time_windows(lambda t: 100 + 200 * t, 1.0)
        x = so.harmonic_complex(1.0, FAST, (t, f), [1])
        ts = np.arange(len(x)) / FAST
        ref = np.cos(2 * np.pi * (100 * ts + 100 * ts**2))
        np.testing.assert_allclose(x.data[:, 0], ref / np.sqrt(np.mean(ref**2)), atol=1e-9)

    def test_harmonics_follow_the_contour(self):
        from scipy.signal import hilbert

        def vib(t):
            return 150 * (1 + 0.04 * np.sin(2 * np.pi * 5.5 * t))

        # 2 s: the analytic signal's own error at the ends of a shorter excerpt is larger than the bound
        t, f = time_windows(vib, 2.0)
        mid = (np.arange(2 * FAST - 1) + 0.5) / FAST
        inner = (mid > 0.1) & (mid < 1.9)
        for k in (1, 10):
            x = so.harmonic_complex(2.0, FAST, (t, f), [k]).data[:, 0]
            inst = np.diff(np.unwrap(np.angle(hilbert(x)))) * FAST / (2 * np.pi)
            assert np.max(np.abs(inst[inner] / (k * vib(mid[inner])) - 1)) < 2e-4

    def test_gaps_are_filled_and_silent(self):
        # voiced 0-0.2 s at 100 Hz, unvoiced 0.2-0.4 s, voiced 0.4-0.6 s at 200 Hz
        t, f = time_windows(lambda t: np.where(t < 0.2, 100.0, np.where(t < 0.4, 0.0, 200.0)), 0.6)
        x = so.harmonic_complex(0.6, FAST, (t, f), [1]).data[:, 0]
        gap = slice(int(0.21 * FAST), int(0.39 * FAST))
        assert np.max(np.abs(x[gap])) == 0
        # filled, not swept through zero: the period never exceeds 1/100 s while sounding
        crossings = np.flatnonzero(np.diff(np.signbit(x)))
        assert np.max(np.diff(crossings)) <= FAST / 100 / 2 + 2

    def test_rising_glide_does_not_alias(self):
        t, f = time_windows(lambda t: 100 * 10**t, 1.0)
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
        t, f = time_windows(lambda t: np.where(t < 0.2, 0.0, 120.0), 0.4)
        a = so.harmonic_complex(0.4, FAST, (t, f), unvoiced="noise", rng=3)
        b = so.harmonic_complex(0.4, FAST, (t, f), unvoiced="noise", rng=3)
        np.testing.assert_array_equal(a.data, b.data)
        assert np.std(a.data[: int(0.15 * FAST)]) > 0.1

    def test_phases_need_one_per_harmonic(self):
        t, f = time_windows(lambda t: np.full_like(t, 1000.0), 0.1)
        n = int(0.45 * FAST // 1000)
        so.harmonic_complex(0.1, FAST, (t, f), phases=np.zeros(n))
        with pytest.raises(ValueError, match="needs one starting phase"):
            so.harmonic_complex(0.1, FAST, (t, f), phases=np.zeros(n + 1))

    def test_amplitudes_as_a_spectral_envelope(self):
        t, f = time_windows(lambda t: np.full_like(t, 200.0), 0.2)
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
        t, f = time_windows(lambda t: np.full_like(t, 200.0), 0.2)
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
        t, f = time_windows(lambda t: 100 * 10**t, 1.0)
        for fn in (so.square_wave, so.sawtooth_wave, so.pulse_train, so.schroeder_complex):
            assert power_above(fn(1.0, FAST, (t, f)).data[:, 0], 0.45 * FAST + 400) < -90

    def test_unbandlimited_needs_a_number(self):
        with pytest.raises(ValueError, match="fixed f0"):
            so.square_wave(0.1, FS, (np.array([0.0, 0.1]), np.array([100.0, 200.0])), bandlimited=False)


class TestAuditSitting4:
    """Claims the sitting 4 audit found untested."""

    def test_silence_has_its_channels(self):
        s = so.silence(0.01, FS, n_channels=3)
        assert s.n_channels == 3 and s.rms == 0

    def test_phase_presets(self):
        t = np.arange(int(0.01 * FS)) / FS
        sine = so.harmonic_complex(0.01, FS, 1000.0, [1], phases="sine").data[:, 0]
        np.testing.assert_allclose(sine, np.sqrt(2) * np.sin(2 * np.pi * 1000 * t), atol=1e-9)
        alternating = so.harmonic_complex(0.01, FS, 1000.0, [1, 2], phases="alternating").data[:, 0]
        expected = np.cos(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 2000 * t)
        np.testing.assert_allclose(alternating, expected / np.sqrt(np.mean(expected**2)), atol=1e-9)

    def test_random_phases_cover_the_circle(self):
        from sonore.sources.waveforms import _harmonic_phases

        phases = _harmonic_phases(np.arange(1, 2001), "random", 0)
        assert phases.min() < -0.99 * np.pi and phases.max() > 0.99 * np.pi

    @pytest.mark.parametrize("sign", [1, -1])
    def test_schroeder_phases(self, sign):
        n = np.arange(1, 21)
        want = so.harmonic_complex(0.05, FS, 200.0, n, phases=sign * np.pi * n * (n + 1) / 20)
        got = so.schroeder_complex(0.05, FS, 200.0, n_harmonics=20, sign=sign)
        np.testing.assert_allclose(got.data, want.data, atol=1e-10)
        # the point of Schroeder phases: a far lower peak than cosine phases
        assert got.peak < 0.5 * so.harmonic_complex(0.05, FS, 200.0, n).peak

    @pytest.mark.parametrize(
        "times, values, message",
        [
            ([0, 0.2, 0.1], [100, 100, 100], "times must increase"),
            ([0, 0.1, 0.2], [100, -5, 100], "finite and >= 0"),
            ([0, 0.1], [100, 100, 100], "one value per time"),
        ],
    )
    def test_contour_checks(self, times, values, message):
        with pytest.raises(ValueError, match=message):
            so.harmonic_complex(0.2, FS, (np.array(times, float), np.array(values, float)))

    def test_voicing_gate_switches_halfway_with_hann_ramps(self):
        from sonore.sources.waveforms import _voicing_gate

        fs, ramp = 10000, 0.01
        t = np.arange(int(0.2 * fs)) / fs
        gate = _voicing_gate(t, np.array([0.0, 0.1, 0.2]), np.array([True, False, False]), ramp, fs)
        # nearest window: voiced up to 0.05 s, ramped over 0.01 s centered there
        assert gate[int(0.04 * fs)] == pytest.approx(1)
        assert gate[int(0.06 * fs)] == pytest.approx(0, abs=1e-12)
        assert gate[int(0.05 * fs)] == pytest.approx(0.5, abs=0.05)
        falling = gate[int(0.044 * fs) : int(0.056 * fs)]
        assert np.all(np.diff(falling) <= 1e-12)
        assert 0.6 < gate[int(0.0475 * fs)] < 0.95 and 0.05 < gate[int(0.0525 * fs)] < 0.4

    def test_unvoiced_noise_matches_the_voiced_level(self):
        t, f = time_windows(lambda t: np.where(t < 0.5, 0.0, 150.0), 1.0)
        x = so.harmonic_complex(1.0, FAST, (t, f), unvoiced="noise", rng=0).data[:, 0]
        noise, voiced = x[: int(0.45 * FAST)], x[int(0.55 * FAST) :]
        assert np.std(noise) / np.std(voiced) == pytest.approx(1, abs=0.05)

    def test_harmonic_at_nyquist_is_dropped(self):
        with pytest.warns(UserWarning, match="dropping 1 harmonic"):
            x = so.harmonic_complex(0.01, FS, FS / 4, [1, 2])
        np.testing.assert_allclose(x.data, so.harmonic_complex(0.01, FS, FS / 4, [1]).data)

    def test_naive_square_and_sawtooth(self):
        from scipy.signal import sawtooth

        t = np.arange(int(0.05 * FS)) / FS
        square = so.square_wave(0.05, FS, 100.0, phase=0.3, bandlimited=False).data[:, 0]
        np.testing.assert_array_equal(square, np.sign(np.sin(2 * np.pi * 100 * t + 0.3)))
        saw = so.sawtooth_wave(0.05, FS, 100.0, bandlimited=False).data[:, 0]
        expected = sawtooth(2 * np.pi * 100 * t + np.pi)
        np.testing.assert_allclose(saw, expected / np.sqrt(np.mean(expected**2)))
        # rising, and 0 at t = 0 like the band-limited sawtooth
        assert saw[0] == pytest.approx(0, abs=1e-12) and saw[10] > saw[5] > 0

    def test_naive_pulse_train_phase_shifts_earlier(self):
        # phase pi/2 moves the pulses a quarter period earlier: 441 Hz at 44.1 kHz has
        # 100-sample periods, so the first pulse is at sample 75
        x = so.pulse_train(0.1, FS, 441, phase=np.pi / 2, bandlimited=False).data[:, 0]
        assert np.flatnonzero(x)[0] == 75

    def test_exponential_chirp_phase(self):
        duration, f0, f1, phase = 0.05, 200.0, 3200.0, 0.3
        x = so.exponential_chirp(duration, FS, f0, f1, phase).data[:, 0]
        t = np.arange(len(x)) / FS
        k = f1 / f0
        expected = np.cos(2 * np.pi * f0 * duration * (k ** (t / duration) - 1) / np.log(k) + phase)
        np.testing.assert_allclose(x, expected / np.sqrt(np.mean(expected**2)), atol=1e-9)

    def test_brown_noise_is_minus_6_db_per_octave(self):
        spec = so.long_term_spectrum(so.gaussian_noise(20, FS, tilt=-6, rng=0))
        lo, hi = spec.level_at(np.array([1000, 2000]))
        assert hi - lo == pytest.approx(-6, abs=0.5)

    def test_noise_mean_is_removed(self):
        assert abs(so.gaussian_noise(0.1, FS, rng=0).data.mean()) < 1e-12

    def test_spectrum_table_is_a_level_in_db_interpolated_in_hz(self):
        # 0 dB at 100 Hz to -40 dB at 10 kHz: linear in Hz gives -20 dB at 5050 Hz
        # (on a log-frequency axis it would be about -35 dB)
        table = ([100.0, 10000.0], [0.0, -40.0])
        spec = so.long_term_spectrum(so.gaussian_noise(20, FS, spectrum=table, rng=0))
        at_100, at_5050 = spec.level_at(np.array([100.0, 5050.0]))
        assert at_5050 - at_100 == pytest.approx(-20, abs=1)
        same = so.gaussian_noise(1, FS, spectrum=lambda f: np.interp(f, *table), rng=1)
        np.testing.assert_allclose(same.data, so.gaussian_noise(1, FS, spectrum=table, rng=1).data)

    @staticmethod
    def irn_by_iteration(x, delay_samples, gain, iterations, network):
        """Iterated rippled noise in the time domain, delaying circularly as an FFT does."""
        y = x.copy()
        for _ in range(iterations):
            y = (y if network == "add-same" else x) + gain * np.roll(y, delay_samples)
        return y

    @pytest.mark.parametrize("network", ["add-same", "add-original"])
    def test_irn_networks_gain_iterations_and_warm_up(self, network):
        fs, delay, gain, iterations = 8000, 0.004, 0.5, 4
        x = so.iterated_ripple_noise(0.1, fs, delay, gain, iterations, network, rng=0).data[:, 0]
        warmup = int(np.ceil(iterations * delay * fs))
        source = so.gaussian_noise((len(x) + warmup) / fs, fs, rng=0).data[:, 0]
        y = self.irn_by_iteration(source, int(round(delay * fs)), gain, iterations, network)[warmup:]
        np.testing.assert_allclose(x, y / np.sqrt(np.mean(y**2)), atol=1e-9)
