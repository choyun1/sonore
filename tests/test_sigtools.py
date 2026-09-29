import numpy as np
import pytest

import sigtools as st

FS = 44100


def peak_freq(s: st.Sound) -> float:
    X = np.abs(np.fft.rfft(s.data[:, 0]))
    return np.fft.rfftfreq(len(s), 1 / s.fs)[np.argmax(X)]


# ------------------------------------------------------------------ Sound
class TestSound:
    def test_numpy_scalar_on_left(self):
        x = st.pure_tone(0.1, FS, 440)
        assert isinstance(np.float64(2) * x, st.Sound)
        assert isinstance(np.int64(2) * x, st.Sound)
        np.testing.assert_allclose((np.float64(2) * x).data, 2 * x.data)

    def test_reflected_division_and_subtraction(self):
        x = st.Sound(np.array([1.0, 2.0, 4.0]), FS)
        np.testing.assert_allclose((2 / x).data[:, 0], [2, 1, 0.5])
        np.testing.assert_allclose((x - x).data, 0)
        y = st.Sound(np.ones(3), FS)
        np.testing.assert_allclose((y - x).data[:, 0], [0, -1, -3])

    def test_adding_number_is_an_error_but_sum_works(self):
        x = st.pure_tone(0.1, FS, 440)
        with pytest.raises(TypeError, match="gain_db"):
            x + 6
        np.testing.assert_allclose(sum([x, x]).data, 2 * x.data)

    def test_gain_db(self):
        x = st.pure_tone(0.1, FS, 440)
        assert st.amp_to_db(x.gain_db(6).rms / x.rms) == pytest.approx(6)

    def test_immutable(self):
        x = st.pure_tone(0.1, FS, 440)
        before = x.data.copy()
        x.ramp(0.01)
        np.testing.assert_array_equal(x.data, before)
        with pytest.raises(ValueError):
            x.data[0] = 1

    def test_channels_first_is_rejected(self):
        with pytest.raises(ValueError, match="channels-first"):
            st.Sound(np.zeros((2, 1000)), FS)

    def test_mono_broadcasts_against_stereo(self):
        env = st.Sound(np.linspace(0, 1, 100), FS)
        s = st.correlated_noise(100 / FS, FS, corr=0, rng=0)
        assert (env * s).n_channels == 2

    def test_mismatched_fs_is_an_error(self):
        with pytest.raises(ValueError, match="sampling rates"):
            st.silence(0.1, 44100) + st.silence(0.1, 48000)

    def test_time_slicing(self):
        x = st.pure_tone(1.0, FS, 440)
        assert len(x[0.25:0.5]) == FS // 4
        with pytest.raises(TypeError):
            x[0]

    @pytest.mark.parametrize("d", [10 / FS, 10.37 / FS])
    def test_fractional_delay(self, d):
        x = st.gaussian_noise(0.1, FS, band=(100, 15000), rng=1)
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
        x = st.pure_tone(0.5, 48000, 1000).resample(44100)
        assert x.fs == 44100 and len(x) == 22050
        assert peak_freq(x) == pytest.approx(1000, abs=2)


# -------------------------------------------------------------- generators
class TestGenerators:
    def test_tone_frequency_exact_for_short_tones(self):
        x = st.pure_tone(0.01, FS, 1000)
        t = np.arange(len(x)) / FS
        np.testing.assert_allclose(x.data[:, 0], np.sqrt(2) * np.cos(2 * np.pi * 1000 * t), atol=1e-2)

    def test_harmonics_above_nyquist_dropped(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            x = st.harmonic_complex(0.1, FS, 5000, np.arange(1, 6))
        assert x.rms == pytest.approx(1)

    def test_bandlimited_square_has_no_aliases(self):
        x = st.square_wave(1.0, FS, 1000)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        off = np.abs(f / 1000 - np.round(f / 1000)) * 1000 > 5
        assert X[off].max() < 1e-6 * X.max()

    def test_pulse_train_period(self):
        x = st.pulse_train(0.1, FS, 441, bandlimited=False)
        idx = np.flatnonzero(x.data[:, 0])
        np.testing.assert_array_equal(np.diff(idx), 100)

    def test_noise_reproducible(self):
        a = st.gaussian_noise(0.1, FS, rng=42)
        b = st.gaussian_noise(0.1, FS, rng=42)
        np.testing.assert_array_equal(a.data, b.data)

    def test_noise_tilt_is_db_per_octave(self):
        x = st.gaussian_noise(20, FS, tilt=-3, rng=0)
        spec = st.long_term_spectrum(x)
        lo, hi = spec.level_at(np.array([1000, 2000]))
        assert hi - lo == pytest.approx(-3, abs=0.5)

    def test_noise_band(self):
        x = st.gaussian_noise(1, FS, band=(500, 1000), rng=0)
        X = np.abs(np.fft.rfft(x.data[:, 0]))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        assert X[(f < 490) | (f > 1010)].max() < 1e-9

    def test_correlated_noise(self):
        for c in (-0.5, 0, 0.8):
            x = st.correlated_noise(5, FS, corr=c, rng=0)
            assert np.corrcoef(x.data.T)[0, 1] == pytest.approx(c, abs=0.02)

    def test_irn_pitch(self):
        x = st.iterated_ripple_noise(1, FS, delay=5e-3, iterations=8, rng=0)
        ac = np.correlate(x.data[:4000, 0], x.data[:4000, 0], "full")[3999:]
        lag = np.argmax(ac[50:400]) + 50
        assert lag == pytest.approx(5e-3 * FS, abs=1)


# ------------------------------------------------------------- processing
class TestProcessing:
    def test_bandpass_filters_each_channel_independently(self):
        n = st.correlated_noise(1, FS, corr=0, rng=0)
        y = st.bandpass(n, 500, 1000)
        assert abs(np.corrcoef(y.data.T)[0, 1]) < 0.1

    def test_pad_uses_zeros(self):
        a, b = st.pad([st.Sound(np.ones(3), FS), st.Sound(np.ones(5), FS)], align="center")
        np.testing.assert_array_equal(a.data[:, 0], [0, 1, 1, 1, 0])

    def test_concat_upmixes_mono(self):
        s = st.concat([st.correlated_noise(0.1, FS, rng=0), st.silence(0.1, FS)])
        assert s.n_channels == 2

    def test_mix_different_lengths(self):
        m = st.mix([st.silence(0.1, FS), st.pure_tone(0.2, FS, 100)])
        assert m.duration == pytest.approx(0.2)


# ------------------------------------------------------ representations
class TestRepresentations:
    def test_spectrum_db_scale(self):
        g = st.gaussian_noise(0.5, FS, rng=0)
        d = st.Spectrum.from_sound(2 * g).level - st.Spectrum.from_sound(g).level
        assert np.median(d[1:]) == pytest.approx(20 * np.log10(2), abs=1e-6)

    @pytest.mark.parametrize("win", [20e-3, 20.1e-3])
    def test_stft_perfect_reconstruction(self, win):
        g = st.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(st.STFT(g, win).to_sound().data, g.data, atol=1e-10)

    def test_stft_stereo(self):
        s = st.correlated_noise(0.2, FS, corr=0, rng=0)
        S = st.STFT(s)
        assert S.data.shape[0] == 2
        np.testing.assert_allclose(S.to_sound().data, s.data, atol=1e-10)

    def test_stft_time_axis(self):
        S = st.STFT(st.gaussian_noise(1, FS, rng=0), 20e-3, 10e-3)
        assert np.diff(S.t) == pytest.approx(10e-3)

    def test_ibm_recovers_target_when_well_separated(self):
        target = st.pure_tone(1, FS, 500)
        masker = st.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
        mixture = target + masker
        mask = st.ideal_binary_mask(st.STFT(target), st.STFT(masker))
        out = (st.STFT(mixture) * mask).to_sound()
        snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
        assert snr > 20

    def test_griffin_lim_converges(self):
        x = st.harmonic_complex(0.5, 16000, 200, np.arange(1, 10))
        S = st.STFT(x, 32e-3)
        y = S.griffin_lim(n_iter=50, rng=0)
        err = np.linalg.norm(st.STFT(y, 32e-3).magnitude - S.magnitude) / np.linalg.norm(S.magnitude)
        assert err < 0.1

    def test_modulation_spectrum_peak(self):
        x = st.amplitude_modulate(st.gaussian_noise(2, FS, rng=0), 8, depth=1)
        ms = st.ModulationSpectrum(st.STFT(x, 20e-3))
        row = ms.level[0]
        pos = ms.w_t > 2
        assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)


# -------------------------------------------------------------- filterbank
class TestFilterbank:
    def test_power_complementary(self):
        fb = st.ERBFilterbank(30, 50, 8000)
        H = fb.response(np.linspace(0, 22050, 5000))
        np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)

    def test_perfect_reconstruction(self):
        g = st.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(st.subbands(g).synthesize().data, g.data, atol=1e-10)

    def test_vocoder_runs_and_keeps_level(self):
        x = st.harmonic_complex(0.5, FS, 150, np.arange(1, 20))
        v = st.noise_vocode(x, 8, rng=0)
        assert len(v) == len(x) and v.rms == pytest.approx(x.rms)


# ---------------------------------------------------------------- binaural
class TestBinaural:
    @pytest.mark.parametrize("itd", [300e-6, -500e-6, 123.4e-6])
    def test_itd_ild_roundtrip(self, itd):
        g = st.gaussian_noise(1, FS, band=(100, 8000), rng=0)
        b = st.apply_itd_ild(g, itd=itd, ild=6)
        c = st.interaural_cues(b, 50e-3)
        assert np.nanmedian(c.itd) == pytest.approx(itd, abs=3e-6)
        assert np.nanmedian(c.ild) == pytest.approx(6, abs=0.05)
        # fractional lags: the sampled CCF peak sits slightly below its true height
        assert np.nanmedian(c.iac) > (0.995 if abs(itd * FS - round(itd * FS)) < 1e-6 else 0.99)

    def test_simple_bir_is_level_independent_of_itd(self):
        a, b = st.simple_bir(FS, 100e-6), st.simple_bir(FS, 500e-6)
        assert np.sum(a.data**2) == pytest.approx(np.sum(b.data**2), rel=2e-2)

    def test_per_band_cues(self):
        b = st.apply_itd_ild(st.gaussian_noise(0.5, FS, rng=0), itd=200e-6)
        c = st.interaural_cues(b, 20e-3, filterbank=st.ERBFilterbank(8, 200, 1500))
        assert c.itd.shape[1] == 10
        assert np.nanmedian(c.itd[:, 2:-2]) == pytest.approx(200e-6, abs=10e-6)

    def test_oscor_and_phasewarp_correlation_oscillates(self):
        for make, fn in [(st.oscor, np.sin), (st.phasewarp, np.cos)]:
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
        ir = st.synth_ir(0.5, FS, drr_db=5, rng=0)
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
        ir = st.synth_ir(0.5, FS, n_channels=2, rng=0)
        assert abs(np.corrcoef(ir.data.T)[0, 1]) < 0.1


# ---------------------------------------------------------- spatialization
def toy_hrirs(fs=48000, taps=256):
    """Spherical-head-ish toy set: ITD from Woodworth, ILD from azimuth."""
    az = np.arange(0, 360, 10)
    el = np.arange(-40, 91, 20)
    hcc = np.array([(100, e, a) for e in el for a in az])
    pos = np.column_stack(st.hcc_to_rect(*hcc.T))
    theta = np.radians(hcc[:, 2])
    lat = np.arcsin(np.sin(theta) * np.cos(np.radians(hcc[:, 1])))
    itd = 0.0875 / 343 * (lat + np.sin(lat))
    irs = np.zeros((len(hcc), 2, taps))
    for i, d in enumerate(itd):
        base = 20
        irs[i, 0, base + int(round(max(d, 0) * fs))] = 10 ** (-np.sin(lat[i]) * 5 / 20)
        irs[i, 1, base + int(round(max(-d, 0) * fs))] = 10 ** (np.sin(lat[i]) * 5 / 20)
    return st.HRIRSet(irs, pos, fs), itd


class TestSpatialization:
    def test_coordinate_roundtrip(self):
        h = (150.0, 20.0, 250.0)
        np.testing.assert_allclose(st.rect_to_hcc(*st.hcc_to_rect(*h)), h)
        x, y, z = st.hcc_to_rect(100, 0, 90)
        np.testing.assert_allclose([x, y, z], [1, 0, 0], atol=1e-12)  # 90 deg = right

    def test_distance_gain(self):
        assert st.distance_gain_db(2.0) == pytest.approx(-6.02, abs=0.01)

    def test_interpolation_at_measured_point_is_exact(self):
        hs, _ = toy_hrirs()
        h = hs.at(hs.positions[5], fs=hs.fs)[0]
        n = hs.irs.shape[-1]
        np.testing.assert_allclose(h[:, :n], hs.irs[5], atol=1e-6)

    def test_moving_sound_itd_sweeps(self):
        hs, _ = toy_hrirs()
        x = st.gaussian_noise(2, 48000, band=(200, 3000), rng=0)
        traj = st.circular_trajectory((100, 0, 90), (100, 0, -90), 200)
        y = st.move_sound(x, traj, hs)
        c = st.interaural_cues(y, 50e-3)
        itd = c.itd[np.isfinite(c.itd)]
        assert itd[:3].mean() > 400e-6  # starts right
        assert itd[-3:].mean() < -400e-6  # ends left
        assert abs(np.median(itd[len(itd) // 2 - 2 : len(itd) // 2 + 2])) < 100e-6

    def test_move_sound_crossfade_preserves_level(self):
        hs, _ = toy_hrirs()
        x = st.gaussian_noise(1, 48000, rng=0)
        still = st.spatialize(x, st.hcc_to_rect(100, 0, 0), hs)
        moving = st.move_sound(x, np.repeat([st.hcc_to_rect(100, 0, 0)], 50, axis=0), hs)
        np.testing.assert_allclose(moving.data[: len(still)], still.data, atol=1e-9)
