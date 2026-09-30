import numpy as np
import pytest

import sonore as so

FS = 44100


def peak_freq(s: so.Sound) -> float:
    X = np.abs(np.fft.rfft(s.data[:, 0]))
    return np.fft.rfftfreq(len(s), 1 / s.fs)[np.argmax(X)]


# ------------------------------------------------------------------ Sound
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


class TestDecibels:
    x = so.pure_tone(0.1, FS, 440)

    def level_change(self, y):
        return float(so.amp_to_db(y.rms / self.x.rms))

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
        assert self.level_change(expr(self.x, so.dB)) == pytest.approx(expected)

    def test_matches_gain_db(self):
        np.testing.assert_allclose((self.x + 6 * so.dB).data, self.x.gain_db(6).data)
        np.testing.assert_allclose(self.x.gain_db(6 * so.dB).data, self.x.gain_db(6).data)

    def test_mixing_at_snr(self):
        target, masker = so.pure_tone(1, FS, 500), so.gaussian_noise(1, FS, rng=0)
        mix = target + (masker - 10 * so.dB)
        noise = mix - target
        assert so.amp_to_db(target.rms / noise.rms) == pytest.approx(10)

    def test_traps_raise(self):
        with pytest.raises(TypeError, match="ambiguous"):
            self.x * (6 * so.dB)
        with pytest.raises(TypeError):
            3 * so.dB - self.x

    def test_repr_and_value(self):
        assert repr(6 * so.dB) == "6 dB"
        assert (6 * so.dB).gain == pytest.approx(10**0.3)
        assert 3 * so.dB < 6 * so.dB

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


# -------------------------------------------------------------- generators
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


# ------------------------------------------------------------- processing
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


# ------------------------------------------------------ representations
class TestRepresentations:
    def test_spectrum_db_scale(self):
        g = so.gaussian_noise(0.5, FS, rng=0)
        d = so.Spectrum.from_sound(2 * g).level - so.Spectrum.from_sound(g).level
        assert np.median(d[1:]) == pytest.approx(20 * np.log10(2), abs=1e-6)

    @pytest.mark.parametrize("win", [20e-3, 20.1e-3])
    def test_stft_perfect_reconstruction(self, win):
        g = so.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(so.STFT(g, win).to_sound().data, g.data, atol=1e-10)

    def test_stft_stereo(self):
        s = so.correlated_noise(0.2, FS, corr=0, rng=0)
        S = so.STFT(s)
        assert S.data.shape[0] == 2
        np.testing.assert_allclose(S.to_sound().data, s.data, atol=1e-10)

    def test_stft_time_axis(self):
        S = so.STFT(so.gaussian_noise(1, FS, rng=0), 20e-3, 10e-3)
        assert np.diff(S.t) == pytest.approx(10e-3)

    def test_ibm_recovers_target_when_well_separated(self):
        target = so.pure_tone(1, FS, 500)
        masker = so.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
        mixture = target + masker
        mask = so.ideal_binary_mask(so.STFT(target), so.STFT(masker))
        out = (so.STFT(mixture) * mask).to_sound()
        snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
        assert snr > 20

    def test_griffin_lim_converges(self):
        x = so.harmonic_complex(0.5, 16000, 200, np.arange(1, 10))
        S = so.STFT(x, 32e-3)
        y = S.griffin_lim(n_iter=50, rng=0)
        err = np.linalg.norm(so.STFT(y, 32e-3).magnitude - S.magnitude) / np.linalg.norm(S.magnitude)
        assert err < 0.1

    def test_modulation_spectrum_peak(self):
        x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 8, depth=1)
        ms = so.ModulationSpectrum(so.STFT(x, 20e-3))
        row = ms.level[0]
        pos = ms.w_t > 2
        assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)


# -------------------------------------------------------------- filterbank
class TestFilterbank:
    def test_power_complementary(self):
        fb = so.ERBFilterbank(30, 50, 8000)
        H = fb.response(np.linspace(0, 22050, 5000))
        np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)

    def test_perfect_reconstruction(self):
        g = so.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(so.subbands(g).synthesize().data, g.data, atol=1e-10)

    def test_vocoder_runs_and_keeps_level(self):
        x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 20))
        v = so.noise_vocode(x, 8, rng=0)
        assert len(v) == len(x) and v.rms == pytest.approx(x.rms)


# ---------------------------------------------------------------- binaural
class TestBinaural:
    @pytest.mark.parametrize("itd", [300e-6, -500e-6, 123.4e-6])
    def test_itd_ild_roundtrip(self, itd):
        g = so.gaussian_noise(1, FS, band=(100, 8000), rng=0)
        b = so.apply_itd_ild(g, itd=itd, ild=6)
        c = so.interaural_cues(b, 50e-3)
        assert np.nanmedian(c.itd) == pytest.approx(itd, abs=3e-6)
        assert np.nanmedian(c.ild) == pytest.approx(6, abs=0.05)
        # fractional lags: the sampled CCF peak sits slightly below its true height
        assert np.nanmedian(c.iac) > (0.995 if abs(itd * FS - round(itd * FS)) < 1e-6 else 0.99)

    def test_simple_bir_is_level_independent_of_itd(self):
        a, b = so.simple_bir(FS, 100e-6), so.simple_bir(FS, 500e-6)
        assert np.sum(a.data**2) == pytest.approx(np.sum(b.data**2), rel=2e-2)

    def test_per_band_cues(self):
        b = so.apply_itd_ild(so.gaussian_noise(0.5, FS, rng=0), itd=200e-6)
        c = so.interaural_cues(b, 20e-3, filterbank=so.ERBFilterbank(8, 200, 1500))
        assert c.itd.shape[1] == 10
        assert np.nanmedian(c.itd[:, 2:-2]) == pytest.approx(200e-6, abs=10e-6)

    def test_oscor_and_phasewarp_correlation_oscillates(self):
        for make, fn in [(so.oscor, np.sin), (so.phasewarp, np.cos)]:
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
        ir = so.synth_ir(0.5, FS, drr_db=5, rng=0)
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
        ir = so.synth_ir(0.5, FS, n_channels=2, rng=0)
        assert abs(np.corrcoef(ir.data.T)[0, 1]) < 0.1


# ---------------------------------------------------------- spatialization
def toy_hrirs(fs=48000, taps=256):
    """Spherical-head-ish toy set: ITD from Woodworth, ILD from azimuth."""
    az = np.arange(0, 360, 10)
    el = np.arange(-40, 91, 20)
    hcc = np.array([(100, e, a) for e in el for a in az])
    pos = np.column_stack(so.hcc_to_rect(*hcc.T))
    theta = np.radians(hcc[:, 2])
    lat = np.arcsin(np.sin(theta) * np.cos(np.radians(hcc[:, 1])))
    itd = 0.0875 / 343 * (lat + np.sin(lat))
    irs = np.zeros((len(hcc), 2, taps))
    for i, d in enumerate(itd):
        base = 20
        irs[i, 0, base + int(round(max(d, 0) * fs))] = 10 ** (-np.sin(lat[i]) * 5 / 20)
        irs[i, 1, base + int(round(max(-d, 0) * fs))] = 10 ** (np.sin(lat[i]) * 5 / 20)
    return so.HRIRSet(irs, pos, fs), itd


class TestSpatialization:
    def test_coordinate_roundtrip(self):
        h = (150.0, 20.0, 250.0)
        np.testing.assert_allclose(so.rect_to_hcc(*so.hcc_to_rect(*h)), h)
        x, y, z = so.hcc_to_rect(100, 0, 90)
        np.testing.assert_allclose([x, y, z], [1, 0, 0], atol=1e-12)  # 90 deg = right

    def test_distance_gain(self):
        assert so.distance_gain_db(2.0) == pytest.approx(-6.02, abs=0.01)

    def test_interpolation_at_measured_point_is_exact(self):
        hs, _ = toy_hrirs()
        h = hs.at(hs.positions[5], fs=hs.fs)[0]
        n = hs.irs.shape[-1]
        np.testing.assert_allclose(h[:, :n], hs.irs[5], atol=1e-6)

    def test_moving_sound_itd_sweeps(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(2, 48000, band=(200, 3000), rng=0)
        traj = so.circular_trajectory((100, 0, 90), (100, 0, -90), 200)
        y = so.move_sound(x, traj, hs)
        c = so.interaural_cues(y, 50e-3)
        itd = c.itd[np.isfinite(c.itd)]
        assert itd[:3].mean() > 400e-6  # starts right
        assert itd[-3:].mean() < -400e-6  # ends left
        assert abs(np.median(itd[len(itd) // 2 - 2 : len(itd) // 2 + 2])) < 100e-6

    def test_move_sound_crossfade_preserves_level(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(1, 48000, rng=0)
        still = so.spatialize(x, so.hcc_to_rect(100, 0, 0), hs)
        moving = so.move_sound(x, np.repeat([so.hcc_to_rect(100, 0, 0)], 50, axis=0), hs)
        np.testing.assert_allclose(moving.data[: len(still)], still.data, atol=1e-9)


def test_zero_lag_correlation_tracks_oscor():
    s = so.oscor(2, FS, 2, rng=0)
    c = so.interaural_cues(s, 10e-3)
    ok = np.isfinite(c.corr0)
    r = np.corrcoef(c.corr0[ok], np.sin(2 * np.pi * 2 * c.t[ok]))[0, 1]
    assert r > 0.95


# ------------------------------------------------------------ phase vocoder
def dominant_freq(s: so.Sound) -> float:
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


def test_subband_plot_labels_and_scale():
    import matplotlib

    matplotlib.use("Agg")
    sb = so.subbands(so.exponential_chirp(0.2, FS, 100, 6000), n_bands=6, f_lo=100, f_hi=6000)
    axes = sb.plot()
    assert len(axes) == 8
    labels = [ax.get_ylabel() for ax in axes]
    assert labels[0] == "> 6000" and labels[-1] == "< 100"
    assert len({ax.get_ylim() for ax in axes}) == 1  # shared amplitude scale
    with pytest.raises(ValueError, match="need 8 axes"):
        sb.plot(axes[:6])


# ------------------------------------------------------------------ ripples
class TestRipples:
    @pytest.mark.parametrize("carrier", ["tones", "harmonic", "noise", "low-noise"])
    @pytest.mark.parametrize("rate, density", [(4, 1), (-8, 2), (16, 0)])
    def test_modulation_spectrum_peak(self, carrier, rate, density):
        # f0 = 40 Hz: harmonics are dense enough near 250 Hz for 2 cyc/oct
        s = so.ripple_sound(so.Ripple(rate, density), 1.0, FS, carrier=carrier, f0=40, rng=0)
        ms = so.ModulationSpectrum.octave(s, f_lo=250, f_hi=8000)
        got_rate, got_density = ms.peak()
        bin_width = ms.w_f[1] - ms.w_f[0]
        assert got_rate == pytest.approx(abs(rate) if density == 0 else rate, abs=1.0)
        assert got_density == pytest.approx(density, abs=bin_width / 2 + 1e-9)

    def test_depth_is_exact_for_tone_carrier(self):
        # with density 0 the pattern is a pure AM: output / unmodulated = 1 + m sin(...)
        m, rate = 0.7, 5.0
        mod = so.ripple_sound(so.Ripple(rate, 0, depth=m), 0.5, FS, rng=1)
        flat = so.ripple_sound(so.Ripple(rate, 0, depth=0), 0.5, FS, rng=1)
        ratio = mod.data[:, 0] / flat.data[:, 0]
        ok = np.abs(flat.data[:, 0]) > 0.05 * flat.peak
        expected = 1 + m * np.sin(2 * np.pi * rate * mod.t)
        k = np.median(ratio[ok] / expected[ok])  # normalization constant
        np.testing.assert_allclose(ratio[ok] / k, expected[ok], rtol=1e-6)

    def test_direction_convention(self):
        # positive rate & density: the envelope maximum moves down in frequency
        r = so.Ripple(2, 1, depth=1)
        x = np.linspace(0, 1, 2001)
        peak_x = [x[np.argmax(r.envelope(np.array([t]), x)[:, 0])] for t in (0.0, 0.1)]
        assert peak_x[1] < peak_x[0]
        assert r.direction == "downward" and so.Ripple(-2, 1).direction == "upward"

    def test_carriers_share_long_term_spectrum(self):
        levels = []
        for carrier in ("tones", "harmonic", "noise", "low-noise"):
            s = so.ripple_sound(so.Ripple(4, 1, depth=0), 2.0, FS, carrier=carrier, rng=0)
            spec = so.long_term_spectrum(s, nperseg=8192).smooth(1)
            f = np.array([500, 1000, 2000, 4000])
            levels.append(spec.level_at(f) - spec.level_at(np.array([1000]))[0])
        spread = np.ptp(np.array(levels), axis=0)
        assert np.all(spread < 3)  # all within 3 dB of each other, octave by octave

    def test_sums_and_validation(self):
        both = so.Ripple(4, 1, depth=0.5) + so.Ripple(-8, 2, depth=0.4)
        assert isinstance(both, so.RippleSum) and len(both.components) == 2
        assert sum([so.Ripple(4, 1, depth=0.3), so.Ripple(8, 2, depth=0.3)]).components[1].rate == 8
        with pytest.raises(ValueError, match="negative"):
            so.Ripple(4, 1, depth=0.6) + so.Ripple(8, 1, depth=0.6)
        with pytest.raises(ValueError, match="linear-scale and dB"):
            so.Ripple(4, 1) + so.Ripple(8, 1, depth=20, scale="db")
        with pytest.raises(ValueError):
            so.Ripple(4, 1, depth=1.5)
        s = so.ripple_sound(both, 1.0, FS, rng=0)
        assert s.rms == pytest.approx(1)

    def test_db_scale_depth(self):
        r = so.Ripple(0, 1, depth=30, scale="db")
        env = r.envelope(np.array([0.0]), np.linspace(0, 1, 1001))
        assert 20 * np.log10(env.max() / env.min()) == pytest.approx(30, abs=0.01)

    def test_resolution_warnings(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            so.ripple_sound(so.Ripple(4, 12), 0.2, FS, tones_per_octave=20, rng=0)
        with pytest.warns(UserWarning, match="lowest harmonics"):
            so.ripple_sound(so.Ripple(4, 2), 0.2, FS, carrier="harmonic", f0=200, rng=0)

    def test_callable_pattern(self):
        s = so.ripple_sound(lambda t, x: 1 + 0.9 * np.sin(2 * np.pi * (6 * t + 1.0 * x)), 1.0, FS, rng=0)
        rate, density = so.ModulationSpectrum.octave(s, f_lo=250).peak()
        assert rate == pytest.approx(6, abs=1) and density == pytest.approx(1, abs=0.15)

    def test_sound_carrier(self):
        speechlike = so.harmonic_complex(1.0, FS, 120, np.arange(1, 60), phases="random", rng=0)
        s = so.ripple_sound(so.Ripple(-4, 1), 1.0, FS, carrier=speechlike, rng=0)
        rate, density = so.ModulationSpectrum.octave(s, f_lo=250).peak()
        assert rate == pytest.approx(-4, abs=1) and density == pytest.approx(1, abs=0.15)


class TestDynamicRipple:
    def test_trajectories_stay_in_range_and_vary_slowly(self):
        d = so.DynamicRipple(seed=0)
        t = np.arange(0, 10, 1e-3)
        rate, density = d.trajectories(t)
        assert rate.min() >= -350 and rate.max() <= 350 and np.ptp(rate) > 400
        assert density.min() >= 0 and density.max() <= 4 and np.ptp(density) > 2
        # slow: little power above the specified rate of change
        R = np.abs(np.fft.rfft(rate - rate.mean())) ** 2
        f = np.fft.rfftfreq(len(rate), 1e-3)
        assert R[f > 4 * d.rate_change].sum() < 0.01 * R.sum()

    def test_reproducible_and_fs_independent(self):
        d = so.DynamicRipple(seed=7)
        a = so.ripple_sound(d, 0.5, FS, rng=1)
        b = so.ripple_sound(d, 0.5, FS, rng=1)
        np.testing.assert_array_equal(a.data, b.data)
        r44, _ = d.trajectories(np.arange(0, 0.5, 1 / 44100))
        r48, _ = d.trajectories(np.arange(0, 0.5, 1 / 48000))
        assert np.interp(0.25, np.arange(0, 0.5, 1 / 48000), r48) == pytest.approx(
            np.interp(0.25, np.arange(0, 0.5, 1 / 44100), r44), abs=0.5
        )
        assert so.DynamicRipple().seed is not None  # a random seed is drawn and kept

    def test_envelope_depth(self):
        d = so.DynamicRipple(depth=45, seed=0)
        env = d.envelope(np.arange(0, 0.2, 1e-3), np.linspace(0, 5, 200))
        assert 20 * np.log10(env.max() / env.min()) == pytest.approx(45, abs=0.5)


def test_octave_filterbank_reconstructs():
    fb = so.OctaveFilterbank.per_octave(12, 125, 8000)
    H = fb.response(np.linspace(0, FS / 2, 5000))
    np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)
    g = so.gaussian_noise(0.5, FS, rng=0)
    np.testing.assert_allclose(fb.analyze(g).synthesize().data, g.data, atol=1e-10)


# ---------------------------------------------------------------- envelopes
class TestEnvelopes:
    x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 30), phases="random", rng=0)

    def test_types_follow_the_concepts(self):
        sb = so.subbands(self.x, n_bands=8)
        assert isinstance(sb[3], so.Sound) and all(isinstance(b, so.Sound) for b in sb)
        assert isinstance(sb.tfs(), so.Subbands)  # fine structure is audible
        assert isinstance(sb.envelopes(), so.Envelopes)  # envelopes are not
        assert isinstance(sb.envelopes()[3], so.Envelope)
        assert isinstance(self.x.envelope(), so.Envelope)

    def test_hilbert_decomposition_is_exact(self):
        sb = so.subbands(self.x, n_bands=8)
        rebuilt = sb.envelopes() * sb.tfs()
        np.testing.assert_allclose(rebuilt.data, sb.data, atol=1e-10)
        np.testing.assert_allclose(rebuilt.synthesize().data, self.x.data, atol=1e-10)
        band = sb[4]
        np.testing.assert_allclose((band.envelope() * (band / band.envelope())).data, band.data, atol=1e-10)

    def test_low_rate_envelopes_upsample_automatically(self):
        sb = so.subbands(self.x, n_bands=8)
        coarse = sb.envelopes(lowpass=100, fs=1000)
        assert coarse.fs == 1000 and coarse.n_samples == 500
        # compare with the same lowpassed envelopes kept at the full rate:
        # the only difference is the automatic upsampling
        full = (sb.envelopes(lowpass=100) * sb.tfs()).synthesize()
        upsampled = (coarse * sb.tfs()).synthesize()
        assert (upsampled - full).rms / full.rms < 0.001
        env = so.gaussian_noise(0.5, FS, rng=0).envelope().lowpass(20).resample(500)
        assert len(env * self.x) == len(self.x)

    def test_resampling_keeps_edges(self):
        # a steady tone has a flat envelope; downsampling must not drag its ends to zero
        env = so.pure_tone(0.5, FS, 1000).envelope().resample(1000)
        mid = np.median(env.data)
        np.testing.assert_allclose(env.data[[0, 1, -2, -1], 0], mid, rtol=0.02)

    def test_envelope_arithmetic(self):
        t = np.arange(FS // 2) / FS
        am = so.Envelope(1 + 0.5 * np.sin(2 * np.pi * 4 * t), FS)
        np.testing.assert_allclose((am * self.x).data[:, 0], am.data[:, 0] * self.x.data[:, 0])
        np.testing.assert_allclose((self.x * am).data, (am * self.x).data)
        assert isinstance(1 + 0.5 * am, so.Envelope) and isinstance(am * am, so.Envelope)
        with pytest.raises(TypeError):
            self.x + am  # adding an envelope to a sound means nothing
        with pytest.raises(ValueError, match="non-negative"):
            so.Envelope(np.sin(2 * np.pi * 4 * t), FS)
        with pytest.raises(ValueError, match="durations differ"):
            so.Envelope(np.ones(FS), FS) * self.x

    def test_modulation_spectrum_from_envelopes(self):
        fb = so.OctaveFilterbank.per_octave(12, 125, 8000)
        direct = so.ModulationSpectrum.octave(self.x)
        via = fb.analyze(self.x.mono()).envelopes(fs=1000).modulation_spectrum()
        np.testing.assert_allclose(via.level, direct.level)
        erb = so.subbands(self.x, 20).envelopes(fs=1000).modulation_spectrum()
        assert erb.spectral_unit == "cyc/ERB" and via.spectral_unit == "cyc/oct"

    def test_rendered_pattern_matches_its_parameters(self):
        fb = so.OctaveFilterbank.per_octave(12, 250, 8000)
        env = so.Ripple(-6, 1.5).render(fb, 2.0, 1000)
        assert isinstance(env, so.Envelopes)
        rate, density = env.modulation_spectrum().peak()
        assert rate == pytest.approx(-6, abs=0.5) and density == pytest.approx(1.5, abs=0.1)

    def test_band_count_mismatch(self):
        env = so.subbands(self.x, n_bands=8).envelopes()
        with pytest.raises(ValueError, match="band counts"):
            env * so.subbands(self.x, n_bands=10)

    def test_plots(self):
        import matplotlib

        matplotlib.use("Agg")
        sb = so.subbands(self.x, n_bands=8)
        assert sb.envelopes().plot().get_title() == "Envelopes (cochleagram)"
        assert sb[2].envelope().plot().get_title() == "Envelope"
