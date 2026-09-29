import numpy as np
import pytest

import audstim as au

FS = 44100


def peak_freq(s: au.Sound) -> float:
    X = np.abs(np.fft.rfft(s.data[:, 0]))
    return np.fft.rfftfreq(len(s), 1 / s.fs)[np.argmax(X)]


# ------------------------------------------------------------------ Sound
class TestSound:
    def test_numpy_scalar_on_left(self):
        x = au.pure_tone(0.1, FS, 440)
        assert isinstance(np.float64(2) * x, au.Sound)
        assert isinstance(np.int64(2) * x, au.Sound)
        np.testing.assert_allclose((np.float64(2) * x).data, 2 * x.data)

    def test_reflected_division_and_subtraction(self):
        x = au.Sound(np.array([1.0, 2.0, 4.0]), FS)
        np.testing.assert_allclose((2 / x).data[:, 0], [2, 1, 0.5])
        np.testing.assert_allclose((x - x).data, 0)
        y = au.Sound(np.ones(3), FS)
        np.testing.assert_allclose((y - x).data[:, 0], [0, -1, -3])

    def test_adding_number_is_an_error_but_sum_works(self):
        x = au.pure_tone(0.1, FS, 440)
        with pytest.raises(TypeError, match=r"6\*dB"):
            x + 6
        np.testing.assert_allclose(sum([x, x]).data, 2 * x.data)

    def test_gain_db(self):
        x = au.pure_tone(0.1, FS, 440)
        assert au.amp_to_db(x.gain_db(6).rms / x.rms) == pytest.approx(6)


class TestDecibels:
    x = au.pure_tone(0.1, FS, 440)

    def level_change(self, y):
        return float(au.amp_to_db(y.rms / self.x.rms))

    @pytest.mark.parametrize(
        "expr, expected",
        [
            (lambda x, dB: x + 6 * dB, 6),
            (lambda x, dB: x - 3 * dB, -3),
            (lambda x, dB: 6 * dB + x, 6),
            (lambda x, dB: x + np.float64(2.5) * dB, 2.5),
            (lambda x, dB: x + dB * 6, 6),
            (lambda x, dB: x + (6 * dB + 3 * dB), 9),
            (lambda x, dB: x + -(4 * dB), -4),
            (lambda x, dB: x + (12 * dB) / 2, 6),
        ],
    )
    def test_level_arithmetic(self, expr, expected):
        assert self.level_change(expr(self.x, au.dB)) == pytest.approx(expected)

    def test_matches_gain_db(self):
        np.testing.assert_allclose((self.x + 6 * au.dB).data, self.x.gain_db(6).data)
        np.testing.assert_allclose(self.x.gain_db(6 * au.dB).data, self.x.gain_db(6).data)

    def test_mixing_at_snr(self):
        target, masker = au.pure_tone(1, FS, 500), au.gaussian_noise(1, FS, rng=0)
        mix = target + (masker - 10 * au.dB)
        noise = mix - target
        assert au.amp_to_db(target.rms / noise.rms) == pytest.approx(10)

    def test_traps_raise(self):
        with pytest.raises(TypeError, match="ambiguous"):
            self.x * (6 * au.dB)
        with pytest.raises(TypeError):
            3 * au.dB - self.x

    def test_repr_and_value(self):
        assert repr(6 * au.dB) == "6 dB"
        assert (6 * au.dB).gain == pytest.approx(10**0.3)
        assert 3 * au.dB < 6 * au.dB

    def test_immutable(self):
        x = au.pure_tone(0.1, FS, 440)
        before = x.data.copy()
        x.ramp(0.01)
        np.testing.assert_array_equal(x.data, before)
        with pytest.raises(ValueError):
            x.data[0] = 1

    def test_channels_first_is_rejected(self):
        with pytest.raises(ValueError, match="channels-first"):
            au.Sound(np.zeros((2, 1000)), FS)

    def test_mono_broadcasts_against_stereo(self):
        env = au.Sound(np.linspace(0, 1, 100), FS)
        s = au.correlated_noise(100 / FS, FS, corr=0, rng=0)
        assert (env * s).n_channels == 2

    def test_mismatched_fs_is_an_error(self):
        with pytest.raises(ValueError, match="sampling rates"):
            au.silence(0.1, 44100) + au.silence(0.1, 48000)

    def test_time_slicing(self):
        x = au.pure_tone(1.0, FS, 440)
        assert len(x[0.25:0.5]) == FS // 4
        with pytest.raises(TypeError):
            x[0]

    @pytest.mark.parametrize("d", [10 / FS, 10.37 / FS])
    def test_fractional_delay(self, d):
        x = au.gaussian_noise(0.1, FS, band=(100, 15000), rng=1)
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
        x = au.pure_tone(0.5, 48000, 1000).resample(44100)
        assert x.fs == 44100 and len(x) == 22050
        assert peak_freq(x) == pytest.approx(1000, abs=2)


# -------------------------------------------------------------- generators
class TestGenerators:
    def test_tone_frequency_exact_for_short_tones(self):
        x = au.pure_tone(0.01, FS, 1000)
        t = np.arange(len(x)) / FS
        np.testing.assert_allclose(x.data[:, 0], np.sqrt(2) * np.cos(2 * np.pi * 1000 * t), atol=1e-2)

    def test_harmonics_above_nyquist_dropped(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            x = au.harmonic_complex(0.1, FS, 5000, np.arange(1, 6))
        assert x.rms == pytest.approx(1)

    def test_bandlimited_square_has_no_aliases(self):
        x = au.square_wave(1.0, FS, 1000)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        off = np.abs(f / 1000 - np.round(f / 1000)) * 1000 > 5
        assert X[off].max() < 1e-6 * X.max()

    def test_pulse_train_period(self):
        x = au.pulse_train(0.1, FS, 441, bandlimited=False)
        idx = np.flatnonzero(x.data[:, 0])
        np.testing.assert_array_equal(np.diff(idx), 100)

    def test_noise_reproducible(self):
        a = au.gaussian_noise(0.1, FS, rng=42)
        b = au.gaussian_noise(0.1, FS, rng=42)
        np.testing.assert_array_equal(a.data, b.data)

    def test_noise_tilt_is_db_per_octave(self):
        x = au.gaussian_noise(20, FS, tilt=-3, rng=0)
        spec = au.long_term_spectrum(x)
        lo, hi = spec.level_at(np.array([1000, 2000]))
        assert hi - lo == pytest.approx(-3, abs=0.5)

    def test_noise_band(self):
        x = au.gaussian_noise(1, FS, band=(500, 1000), rng=0)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        assert X[(f < 490) | (f > 1010)].max() < 1e-9

    def test_correlated_noise(self):
        for c in (-0.5, 0, 0.8):
            x = au.correlated_noise(5, FS, corr=c, rng=0)
            assert np.corrcoef(x.data.T)[0, 1] == pytest.approx(c, abs=0.02)

    def test_irn_pitch(self):
        x = au.iterated_ripple_noise(1, FS, delay=5e-3, iterations=8, rng=0)
        ac = np.correlate(x.data[:4000, 0], x.data[:4000, 0], "full")[3999:]
        lag = np.argmax(ac[50:400]) + 50
        assert lag == pytest.approx(5e-3 * FS, abs=1)


# ------------------------------------------------------------- processing
class TestProcessing:
    def test_bandpass_filters_each_channel_independently(self):
        n = au.correlated_noise(1, FS, corr=0, rng=0)
        y = au.bandpass(n, 500, 1000)
        assert abs(np.corrcoef(y.data.T)[0, 1]) < 0.1

    def test_pad_uses_zeros(self):
        a, b = au.pad([au.Sound(np.ones(3), FS), au.Sound(np.ones(5), FS)], align="center")
        np.testing.assert_array_equal(a.data[:, 0], [0, 1, 1, 1, 0])

    def test_concat_upmixes_mono(self):
        s = au.concat([au.correlated_noise(0.1, FS, rng=0), au.silence(0.1, FS)])
        assert s.n_channels == 2

    def test_mix_different_lengths(self):
        m = au.mix([au.silence(0.1, FS), au.pure_tone(0.2, FS, 100)])
        assert m.duration == pytest.approx(0.2)


# ------------------------------------------------------ representations
class TestRepresentations:
    def test_spectrum_db_scale(self):
        g = au.gaussian_noise(0.5, FS, rng=0)
        d = au.Spectrum.from_sound(2 * g).level - au.Spectrum.from_sound(g).level
        assert np.median(d[1:]) == pytest.approx(20 * np.log10(2), abs=1e-6)

    @pytest.mark.parametrize("win", [20e-3, 20.1e-3])
    def test_stft_perfect_reconstruction(self, win):
        g = au.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(au.STFT(g, win).to_sound().data, g.data, atol=1e-10)

    def test_stft_stereo(self):
        s = au.correlated_noise(0.2, FS, corr=0, rng=0)
        S = au.STFT(s)
        assert S.data.shape[0] == 2
        np.testing.assert_allclose(S.to_sound().data, s.data, atol=1e-10)

    def test_stft_time_axis(self):
        S = au.STFT(au.gaussian_noise(1, FS, rng=0), 20e-3, 10e-3)
        assert np.diff(S.t) == pytest.approx(10e-3)

    def test_ibm_recovers_target_when_well_separated(self):
        target = au.pure_tone(1, FS, 500)
        masker = au.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
        mixture = target + masker
        mask = au.ideal_binary_mask(au.STFT(target), au.STFT(masker))
        out = (au.STFT(mixture) * mask).to_sound()
        snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
        assert snr > 20

    def test_griffin_lim_converges(self):
        x = au.harmonic_complex(0.5, 16000, 200, np.arange(1, 10))
        S = au.STFT(x, 32e-3)
        y = S.griffin_lim(n_iter=50, rng=0)
        err = np.linalg.norm(au.STFT(y, 32e-3).magnitude - S.magnitude) / np.linalg.norm(S.magnitude)
        assert err < 0.1

    def test_modulation_spectrum_peak(self):
        x = au.amplitude_modulate(au.gaussian_noise(2, FS, rng=0), 8, depth=1)
        ms = au.ModulationSpectrum(au.STFT(x, 20e-3))
        row = ms.level[0]
        pos = ms.w_t > 2
        assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)


# -------------------------------------------------------------- filterbank
class TestFilterbank:
    def test_power_complementary(self):
        fb = au.ERBFilterbank(30, 50, 8000)
        H = fb.response(np.linspace(0, 22050, 5000))
        np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)

    def test_perfect_reconstruction(self):
        g = au.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(au.subbands(g).synthesize().data, g.data, atol=1e-10)

    def test_vocoder_runs_and_keeps_level(self):
        x = au.harmonic_complex(0.5, FS, 150, np.arange(1, 20))
        v = au.noise_vocode(x, 8, rng=0)
        assert len(v) == len(x) and v.rms == pytest.approx(x.rms)


# ---------------------------------------------------------------- binaural
class TestBinaural:
    @pytest.mark.parametrize("itd", [300e-6, -500e-6, 123.4e-6])
    def test_itd_ild_roundtrip(self, itd):
        g = au.gaussian_noise(1, FS, band=(100, 8000), rng=0)
        b = au.apply_itd_ild(g, itd=itd, ild=6)
        c = au.interaural_cues(b, 50e-3)
        assert np.nanmedian(c.itd) == pytest.approx(itd, abs=3e-6)
        assert np.nanmedian(c.ild) == pytest.approx(6, abs=0.05)
        # fractional lags: the sampled CCF peak sits slightly below its true height
        assert np.nanmedian(c.iac) > (0.995 if abs(itd * FS - round(itd * FS)) < 1e-6 else 0.99)

    def test_simple_bir_is_level_independent_of_itd(self):
        a, b = au.simple_bir(FS, 100e-6), au.simple_bir(FS, 500e-6)
        assert np.sum(a.data**2) == pytest.approx(np.sum(b.data**2), rel=2e-2)

    def test_per_band_cues(self):
        b = au.apply_itd_ild(au.gaussian_noise(0.5, FS, rng=0), itd=200e-6)
        c = au.interaural_cues(b, 20e-3, filterbank=au.ERBFilterbank(8, 200, 1500))
        assert c.itd.shape[1] == 10
        assert np.nanmedian(c.itd[:, 2:-2]) == pytest.approx(200e-6, abs=10e-6)

    def test_oscor_and_phasewarp_correlation_oscillates(self):
        for make, fn in [(au.oscor, np.sin), (au.phasewarp, np.cos)]:
            s = make(2, FS, 2, rng=0)
            L, R = s.data.T
            w = int(0.02 * FS)
            centers = np.arange(w, len(s) - w, w)
            r = [np.corrcoef(L[c - w // 2 : c + w // 2], R[c - w // 2 : c + w // 2])[0, 1] for c in centers]
            expected = fn(2 * np.pi * 2 * centers / FS)
            assert np.corrcoef(r, expected)[0, 1] > 0.95


# ------------------------------------------------------------------ reverb
class TestReverb:
    def test_rt60_and_drr(self):
        ir = au.synth_ir(0.5, FS, drr_db=5, rng=0)
        direct = ir.data[0, 0] ** 2
        tail = np.sum(ir.data[1:, 0] ** 2)
        assert 10 * np.log10(direct / tail) == pytest.approx(5, abs=0.5)
        # Schroeder backward integration on the broadband tail
        e = np.cumsum(ir.data[::-1, 0] ** 2)[::-1]
        edc = 10 * np.log10(e / e[0])
        t = np.arange(len(e)) / FS
        sel = (edc < -5) & (edc > -25)
        slope = np.polyfit(t[sel], edc[sel], 1)[0]
        assert -60 / slope == pytest.approx(0.5, rel=0.25)

    def test_binaural_tails_are_decorrelated(self):
        ir = au.synth_ir(0.5, FS, n_channels=2, rng=0)
        assert abs(np.corrcoef(ir.data.T)[0, 1]) < 0.1


# ---------------------------------------------------------- spatialization
def toy_hrirs(fs=48000, taps=256):
    """Spherical-head-ish toy set: ITD from Woodworth, ILD from azimuth."""
    az = np.arange(0, 360, 10)
    el = np.arange(-40, 91, 20)
    hcc = np.array([(100, e, a) for e in el for a in az])
    pos = np.column_stack(au.hcc_to_rect(*hcc.T))
    theta = np.radians(hcc[:, 2])
    lat = np.arcsin(np.sin(theta) * np.cos(np.radians(hcc[:, 1])))
    itd = 0.0875 / 343 * (lat + np.sin(lat))
    irs = np.zeros((len(hcc), 2, taps))
    for i, d in enumerate(itd):
        base = 20
        irs[i, 0, base + int(round(max(d, 0) * fs))] = 10 ** (-np.sin(lat[i]) * 5 / 20)
        irs[i, 1, base + int(round(max(-d, 0) * fs))] = 10 ** (np.sin(lat[i]) * 5 / 20)
    return au.HRIRSet(irs, pos, fs), itd


class TestSpatialization:
    def test_coordinate_roundtrip(self):
        h = (150.0, 20.0, 250.0)
        np.testing.assert_allclose(au.rect_to_hcc(*au.hcc_to_rect(*h)), h)
        x, y, z = au.hcc_to_rect(100, 0, 90)
        np.testing.assert_allclose([x, y, z], [1, 0, 0], atol=1e-12)  # 90 deg = right

    def test_distance_gain(self):
        assert au.distance_gain_db(2.0) == pytest.approx(-6.02, abs=0.01)

    def test_interpolation_at_measured_point_is_exact(self):
        hs, _ = toy_hrirs()
        h = hs.at(hs.positions[5], fs=hs.fs)[0]
        n = hs.irs.shape[-1]
        np.testing.assert_allclose(h[:, :n], hs.irs[5], atol=1e-6)

    def test_moving_sound_itd_sweeps(self):
        hs, _ = toy_hrirs()
        x = au.gaussian_noise(2, 48000, band=(200, 3000), rng=0)
        traj = au.circular_trajectory((100, 0, 90), (100, 0, -90), 200)
        y = au.move_sound(x, traj, hs)
        c = au.interaural_cues(y, 50e-3)
        itd = c.itd[np.isfinite(c.itd)]
        assert itd[:3].mean() > 400e-6  # starts right
        assert itd[-3:].mean() < -400e-6  # ends left
        assert abs(np.median(itd[len(itd) // 2 - 2 : len(itd) // 2 + 2])) < 100e-6

    def test_move_sound_crossfade_preserves_level(self):
        hs, _ = toy_hrirs()
        x = au.gaussian_noise(1, 48000, rng=0)
        still = au.spatialize(x, au.hcc_to_rect(100, 0, 0), hs)
        moving = au.move_sound(x, np.repeat([au.hcc_to_rect(100, 0, 0)], 50, axis=0), hs)
        np.testing.assert_allclose(moving.data[: len(still)], still.data, atol=1e-9)


def test_zero_lag_correlation_tracks_oscor():
    s = au.oscor(2, FS, 2, rng=0)
    c = au.interaural_cues(s, 10e-3)
    ok = np.isfinite(c.corr0)
    r = np.corrcoef(c.corr0[ok], np.sin(2 * np.pi * 2 * c.t[ok]))[0, 1]
    assert r > 0.95


# ------------------------------------------------------------ phase vocoder
def dominant_freq(s: au.Sound) -> float:
    """Peak frequency with parabolic interpolation on a zero-padded spectrum."""
    x = s.data[:, 0] * np.hanning(len(s))
    n = 8 * len(x)
    X = np.abs(np.fft.rfft(x, n))
    k = np.argmax(X[1:-1]) + 1
    a, b, c = np.log(X[k - 1 : k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * s.fs / n


def cents(f, ref):
    return 1200 * np.log2(f / ref)


class TestPhaseVocoder:
    @pytest.mark.parametrize("factor", [0.5, 1.5, 2.0, 3.0])
    def test_time_stretch_keeps_pitch_and_scales_duration(self, factor):
        x = au.pure_tone(1.0, FS, 440.0).ramp(20e-3)
        y = au.time_stretch(x, factor)
        assert len(y) == round(len(x) * factor)
        mid = y[0.2 * y.duration : 0.8 * y.duration]
        assert abs(cents(dominant_freq(mid), 440.0)) < 2

    def test_time_stretch_preserves_level(self):
        x = au.harmonic_complex(1.0, FS, 200, np.arange(1, 10)).ramp(20e-3)
        y = au.time_stretch(x, 1.7)
        mid_x = x[0.2:0.8].rms
        mid_y = y[0.2 * y.duration : 0.8 * y.duration].rms
        assert au.amp_to_db(mid_y / mid_x) == pytest.approx(0, abs=0.5)

    def test_identity_stretch_reconstructs(self):
        x = au.harmonic_complex(1.0, FS, 150, np.arange(1, 20), phases="random", rng=0)
        y = au.time_stretch(x, 1.0)
        np.testing.assert_allclose(y.data, x.data, atol=1e-8)

    @pytest.mark.parametrize("semitones", [-7, -1, 1, 4, 12])
    def test_pitch_shift(self, semitones):
        x = au.pure_tone(1.0, FS, 330.0).ramp(20e-3)
        y = au.pitch_shift(x, semitones)
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
        x = au.harmonic_complex(1.0, FS, 200, np.arange(1, 15)).ramp(20e-3)
        for factor in (0.7, 2.5):
            assert self.crest_db(au.time_stretch(x, factor)) == pytest.approx(self.crest_db(x), abs=0.3)

    def test_phase_locking_reduces_phasiness(self):
        t = np.arange(2 * FS) / FS
        phase = 2 * np.pi * np.cumsum(200 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / FS
        x = au.Sound(sum(np.cos(k * phase) for k in range(1, 15)), FS).normalize().ramp(20e-3)
        locked = self.crest_db(au.time_stretch(x, 1.5, phase_lock=True))
        unlocked = self.crest_db(au.time_stretch(x, 1.5, phase_lock=False))
        assert locked == pytest.approx(self.crest_db(x), abs=0.5)
        assert unlocked < locked - 3

    def test_analysis_instantaneous_frequency(self):
        x = au.pure_tone(0.5, FS, 1234.5)
        a = au.pv_analyze(x)
        k = np.argmax(a.magnitude[0].mean(axis=1))
        f = np.median(a.freq[0, k, 3:-3])
        assert f == pytest.approx(1234.5, abs=0.1)  # sub-bin: bins are ~21.5 Hz apart

    def test_oscillator_bank_identity(self):
        x = au.harmonic_complex(1.0, FS, 220, np.arange(1, 8)).ramp(20e-3)
        y = au.pv_analyze(x).resynthesize()
        sl = slice(int(0.1 * FS), int(0.9 * FS))
        assert np.corrcoef(x.data[sl, 0], y.data[sl, 0])[0, 1] > 0.99
        assert au.amp_to_db(au.rms(y.data[sl]) / au.rms(x.data[sl])) == pytest.approx(0, abs=0.3)

    def test_oscillator_bank_frequency_map(self):
        x = au.pure_tone(1.0, FS, 500.0).ramp(20e-3)
        a = au.pv_analyze(x)
        assert abs(cents(dominant_freq(a.resynthesize(freq_map=1.5)[0.2:0.8]), 750)) < 2
        assert abs(cents(dominant_freq(a.resynthesize(freq_map=lambda f: f + 100)[0.2:0.8]), 600)) < 2
        stretched = a.resynthesize(time_scale=2.0)
        assert len(stretched) == 2 * len(x)
        assert abs(cents(dominant_freq(stretched[0.4:1.6]), 500)) < 2
