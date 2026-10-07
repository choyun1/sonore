"""Envelope and Envelopes types."""

import numpy as np
import pytest
from helpers import FAST, FAST_HI, FS

import sonore as so


class TestEnvelopes:
    x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 30), phases="random", rng=0)

    def test_types_follow_the_concepts(self):
        sb = so.cosine_filterbank(8).analyze(self.x)
        assert isinstance(sb[3], so.Sound) and all(isinstance(b, so.Sound) for b in sb)
        assert isinstance(sb.tfs(), so.Subbands)  # fine structure is audible
        assert isinstance(sb.envelopes(), so.Envelopes)  # envelopes are not
        assert isinstance(sb.envelopes()[3], so.Envelope)
        assert isinstance(self.x.envelope(), so.Envelope)

    def test_hilbert_decomposition_is_exact(self):
        sb = so.cosine_filterbank(8).analyze(self.x)
        rebuilt = sb.envelopes() * sb.tfs()
        np.testing.assert_allclose(rebuilt.data, sb.data, atol=1e-10)
        np.testing.assert_allclose(rebuilt.to_sound().data, self.x.data, atol=1e-10)
        band = sb[4]
        np.testing.assert_allclose((band.envelope() * (band / band.envelope())).data, band.data, atol=1e-10)

    def test_low_rate_envelopes_upsample_automatically(self):
        sb = so.cosine_filterbank(8).analyze(self.x)
        coarse = sb.envelopes(lowpass=100, fs=1000)
        assert coarse.fs == 1000 and coarse.n_samples == 500
        # compare with the same lowpassed envelopes kept at the full rate:
        # the only difference is the automatic upsampling
        full = (sb.envelopes(lowpass=100) * sb.tfs()).to_sound()
        upsampled = (coarse * sb.tfs()).to_sound()
        assert (upsampled - full).rms / full.rms < 0.001
        env = so.gaussian_noise(0.5, FS, rng=0).envelope().lowpass(20).resample(500)
        assert len(env * self.x) == len(self.x)

    def test_resampling_keeps_edges(self):
        # A periodic tone's circular envelope is flat (pad=0: treat it as
        # periodic, not as gated on at t=0, which genuinely rings at the edges).
        # Downsampling must not drag the ends of that flat envelope toward zero.
        env = so.pure_tone(0.5, FS, 1000).envelope(pad=0).resample(1000)
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
        x = so.harmonic_complex(0.5, FAST, 150, np.arange(1, 30), phases="random", rng=0)
        fb = so.cosine_filterbank(f_lo=125, f_hi=FAST_HI, spacing=1 / 12, scale="octave")
        direct = so.ModulationSpectrum.octave(x, f_hi=FAST_HI)
        via = fb.analyze(x).envelopes(fs=1000).modulation_spectrum()
        np.testing.assert_allclose(via.level, direct.level)
        erb = so.cosine_filterbank(20).analyze(x).envelopes(fs=1000).modulation_spectrum()
        assert erb.spectral_unit == "cyc/ERB" and via.spectral_unit == "cyc/oct"

    def test_rendered_pattern_matches_its_parameters(self):
        fb = so.cosine_filterbank(f_lo=250, f_hi=8000, spacing=1 / 12, scale="octave")
        env = so.Ripple(-6, 1.5).render(fb, 2.0, 1000)
        assert isinstance(env, so.Envelopes)
        rate, density = env.modulation_spectrum().peak()
        assert rate == pytest.approx(-6, abs=0.5) and density == pytest.approx(1.5, abs=0.1)

    def test_band_count_mismatch(self):
        env = so.cosine_filterbank(8).analyze(self.x).envelopes()
        with pytest.raises(ValueError, match="band counts"):
            env * so.cosine_filterbank(10).analyze(self.x)

    def test_plots(self):
        import matplotlib

        matplotlib.use("Agg")
        sb = so.cosine_filterbank(8).analyze(self.x)
        assert sb.envelopes().plot().get_title() == "Envelopes (cochleagram)"
        assert sb[2].envelope().plot().get_title() == "Envelope"


def test_gammatone_peak_delay_aligns_a_click():
    """Shifting each band by its envelope peak lines a click up to within 0.2 ms
    (the group delay would leave a sweep of several ms)."""
    import matplotlib

    matplotlib.use("Agg")
    fs = 16000
    x = np.zeros(int(0.1 * fs))
    x[int(0.03 * fs)] = 1.0
    fb = so.gammatone_filterbank(n_bands=16, f_lo=100, f_hi=5000)
    env = fb.analyze(so.Sound(x, fs)).envelopes()
    delay = fb.envelope_peak_delay
    assert delay[0] == delay[-1] == 0 and len(delay) == fb.n_filters
    # (order - 1) / (2 pi b) is the peak of t**3 exp(-2 pi b t) alone; the
    # measured peak of the true envelope agrees to within 1% at these centers
    b = 1.019 * 24.7 * (4.37e-3 * fb.band_cfs + 1)
    np.testing.assert_allclose(delay[1:-1], 3 / (2 * np.pi * b), rtol=1e-2)
    peaks = np.argmax(env.data[:, 1:-1, 0], axis=0) / fs
    assert np.ptp(peaks) > 5e-3
    assert np.ptp(peaks - delay[1:-1]) < 0.2e-3
    ax = env.plot(align="peak", fscale="linear", fmax=5000)
    mesh = ax.collections[0].get_coordinates()  # (rows + 1, cols + 1, 2) cell corners
    row_left = mesh[:-1, 0, 0]
    assert np.ptp(row_left) > 5e-3  # each row starts at its own, shifted time
    assert ax.get_ylim() == (0, 5.0)
    zero = so.gammatone_filterbank(n_bands=16, f_lo=100, f_hi=5000, phase="zero")
    assert not np.any(zero.envelope_peak_delay)
    morlet = so.morlet_filterbank(n_bands=8, f_lo=100, f_hi=5000)
    assert not np.any(morlet.envelope_peak_delay)  # zero-phase: nothing to shift
    morlet.analyze(so.Sound(x, fs)).envelopes().plot(align="peak")


class TestToSound:
    x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 30), phases="random", rng=0)
    carrier = so.gaussian_noise(0.5, FS, rng=4)

    def test_envelope_takes_only_the_carriers_fine_structure(self):
        env = self.x.envelope()
        out = env.to_sound(self.carrier)
        fine = self.carrier / self.carrier.envelope()
        np.testing.assert_allclose(out.data, env.data * fine.data)
        # the carrier's own envelope is gone: any level of carrier gives the same sound
        np.testing.assert_allclose(env.to_sound(3 * self.carrier).data, out.data, atol=1e-12)
        with pytest.raises(TypeError, match="Sound"):
            env.to_sound("noise")

    def test_envelopes_are_envelope_to_sound_in_every_band(self):
        fb = so.cosine_filterbank(8)
        env = fb.analyze(self.x).envelopes(lowpass=50)
        whole = env.to_sound(self.carrier)
        carrier_bands = fb.analyze(self.carrier)
        bands = np.stack([env[i].to_sound(carrier_bands[i]).data for i in range(len(env))], axis=1)
        per_band = fb.synthesize(so.Subbands(bands, FS, fb))
        # away from the ends, where only the bands' padding differs, they agree to 0.1%
        middle = slice(FS // 10, 4 * FS // 10)
        error = whole.data[middle] - per_band.data[middle]
        assert np.sqrt(np.mean(error**2) / np.mean(whole.data[middle] ** 2)) < 2e-3

    def test_own_fine_structure_gives_the_sound_back(self):
        sb = so.cosine_filterbank(8).analyze(self.x)
        np.testing.assert_allclose(sb.envelopes().to_sound(self.x).data, self.x.data, atol=1e-9)

    def test_carriers(self):
        fb = so.cosine_filterbank(8)
        env = fb.analyze(self.x).envelopes(lowpass=50, fs=1000)
        noise = env.to_sound("noise", fs=FS, rng=1)
        assert noise.fs == FS and len(noise) == len(self.x)
        np.testing.assert_array_equal(noise.data, env.to_sound("noise", fs=FS, rng=1).data)
        assert not np.array_equal(noise.data, env.to_sound("noise", fs=FS, rng=2).data)
        assert env.to_sound("noise", rng=1).fs == 1000  # the envelopes' own rate by default
        # bands of noise scaled to a mean-square envelope of 1 keep the envelopes' level
        tone_level = env.to_sound("tone", fs=FS).rms
        assert noise.rms == pytest.approx(tone_level, rel=0.1)
        tone = env.without_edges().to_sound("tone", fs=FS)
        power = np.abs(np.fft.rfft(tone.data[:, 0])) ** 2
        freqs = np.fft.rfftfreq(len(tone), 1 / FS)
        # the power sits within the 50 Hz envelope bandwidth of the band centres,
        # not at the harmonics of 150 Hz
        near_centre = np.min(np.abs(freqs[:, None] - fb.cfs[None, 1:-1]), axis=1) < 60
        assert power[near_centre].sum() / power.sum() > 0.95
        with pytest.raises(TypeError, match="rng"):
            env.to_sound("tone", rng=0)
        with pytest.raises(TypeError, match="fs and rng"):
            env.to_sound(self.x, rng=0)
        with pytest.raises(ValueError, match="carrier must be"):
            env.to_sound("pink")
        with pytest.raises(ValueError, match="carrier must be"):
            env.to_sound(np.zeros(10))

    def test_stereo_noise_has_a_noise_per_channel(self):
        stereo = so.Sound(np.column_stack([self.x.data[:, 0], self.x.data[:, 0]]), FS)
        out = so.cosine_filterbank(8).analyze(stereo).envelopes().to_sound("noise", rng=0)
        assert out.n_channels == 2 and not np.allclose(out.data[:, 0], out.data[:, 1])


class TestEnvelope:
    t = np.arange(FS // 2) / FS
    am = so.Envelope(1 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(FS // 2) / FS), FS)

    def test_round_off_negatives_are_clipped_and_real_ones_refused(self):
        values = np.ones(10)
        values[3] = -1e-10  # within 1e-9 of the largest value: round-off
        env = so.Envelope(values, FS)
        assert env.data[3, 0] == 0.0
        values[3] = -1e-6
        with pytest.raises(ValueError, match="non-negative"):
            so.Envelope(values, FS)

    def test_non_finite_values_are_refused(self):
        with pytest.raises(ValueError, match="finite"):
            so.Envelope([1.0, 0.0, 1.0], FS) / 0
        with pytest.raises(ValueError, match="finite"):
            so.Envelope([1.0, np.nan], FS)

    def test_shape_and_axes(self):
        assert self.am.data.shape == (FS // 2, 1) and not self.am.data.flags.writeable
        assert self.am.t[0] == 0 and self.am.t[-1] == pytest.approx((FS // 2 - 1) / FS)
        assert self.am.duration == pytest.approx(0.5)
        np.testing.assert_allclose(self.am.db, 20 * np.log10(self.am.data))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            so.Envelope(np.ones((4, 2, 2)), FS)

    def test_arithmetic_values(self):
        other = so.Envelope(np.full(FS // 2, 2.0), FS)
        np.testing.assert_allclose((self.am / 2).data, self.am.data / 2)
        np.testing.assert_allclose((self.am / other).data, self.am.data / 2)
        np.testing.assert_allclose((self.am + 1).data, self.am.data + 1)
        np.testing.assert_allclose((1 + self.am).data, self.am.data + 1)
        np.testing.assert_allclose((self.am * other).data, 2 * self.am.data)
        with pytest.raises(ValueError, match="share fs and length"):
            self.am * so.Envelope(np.ones(FS // 4), FS)
        with pytest.raises(TypeError):
            self.am * True

    def test_lowpass_is_zero_phase_butterworth(self):
        from scipy.signal import butter, sosfiltfilt

        noisy = so.gaussian_noise(0.5, FS, rng=0).envelope()
        expected = sosfiltfilt(butter(4, 30, fs=FS, output="sos"), noisy.data, axis=0)
        np.testing.assert_allclose(noisy.lowpass(30).data, np.maximum(expected, 0), atol=1e-12)
        # a zero-phase filter leaves a slow modulation where it was
        # (a causal 4th-order filter at 20 Hz would delay it by about 30 ms)
        smoothed = self.am.lowpass(20)
        shift = np.argmax(smoothed.data[: FS // 4, 0]) - np.argmax(self.am.data[: FS // 4, 0])
        assert abs(shift) < 1e-3 * FS

    def test_resample_keeps_the_shape(self):
        coarse = self.am.resample(1000)
        assert coarse.fs == 1000 and len(coarse) == 500
        expected = 1 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(500) / 1000)
        np.testing.assert_allclose(coarse.data[:, 0], expected, atol=2e-3)

    def test_upsampled_envelope_is_band_limited_to_its_last_sample(self):
        coarse = self.am.resample(1000)
        sound = so.Sound(np.ones(FS // 2), FS)
        applied = (coarse * sound).data[:, 0]
        before_end = slice(0, -FS // 200)  # the last few ms rest on the last coarse samples
        np.testing.assert_allclose(applied[before_end], self.am.data[before_end, 0], atol=2e-3)
        assert applied[-1] == pytest.approx(self.am.data[-1, 0], abs=0.02)
        # an envelope one sample shorter at the same rate is held at its end
        short = so.Envelope(self.am.data[:-1], FS)
        assert (short * sound).data[-1, 0] == pytest.approx(self.am.data[-2, 0])

    def test_duration_must_agree_within_a_sample_and_a_half(self):
        sound = so.Sound(np.ones(FS // 2), FS)
        coarse = self.am.resample(1000)
        so.Envelope(np.ones(501), 1000) * sound  # one sample over: fine
        with pytest.raises(ValueError, match="durations differ"):
            so.Envelope(np.ones(502), 1000) * sound
        assert len(coarse * sound) == len(sound)

    def test_dividing_a_silent_envelope_out_stays_finite(self):
        silent = so.Envelope(np.zeros(FS // 2), FS)
        divided = so.Sound(np.ones(FS // 2), FS) / silent
        assert np.all(np.isfinite(divided.data))

    def test_odd_rates_are_quick_and_band_limited(self):
        import time

        # 44100 / 1234.567 is 29400/823 to 1e-9: a polyphase filter that size took seconds
        fs_odd = 1234.567
        n = int(round(0.5 * fs_odd))
        k = np.arange(n)
        # different values at the two ends, which the FFT would otherwise wrap
        ramp = so.Envelope(1 + k / (n - 1) + 0.5 * np.sin(2 * np.pi * 4 * k / fs_odd), fs_odd)
        sound = so.Sound(np.ones(FS // 2), FS)
        start = time.perf_counter()
        applied = (ramp * sound).data[:, 0]
        assert time.perf_counter() - start < 0.5
        t = np.arange(FS // 2) / FS
        expected = 1 + t * fs_odd / (n - 1) + 0.5 * np.sin(2 * np.pi * 4 * t)
        middle = slice(FS // 10, 4 * FS // 10)
        np.testing.assert_allclose(applied[middle], expected[middle], atol=1e-3)


class TestEnvelopesBank:
    x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 30), phases="random", rng=0)
    sb = so.cosine_filterbank(8).analyze(x)

    def test_axes_and_levels(self):
        env = self.sb.envelopes()
        assert env.duration == pytest.approx(0.5) and env.t[0] == 0
        assert env.t[-1] == pytest.approx((len(self.x) - 1) / FS)
        np.testing.assert_allclose(env.db, 20 * np.log10(np.maximum(env.data, 1e-300)), atol=1e-9)
        assert env.resample(FS) is env

    def test_lowpass_is_the_single_envelope_lowpass_in_every_band(self):
        env = so.cosine_filterbank(8).analyze(self.x, pad=0).envelopes()
        for order in (2, 4):
            np.testing.assert_allclose(env.lowpass(40, order).data[:, 3], env[3].lowpass(40, order).data)

    def test_products(self):
        env = self.sb.envelopes()
        np.testing.assert_allclose((env * 2).data, 2 * env.data)
        np.testing.assert_allclose((2 * env).data, 2 * env.data)
        for other in (True, env, so.Envelope(np.ones(len(self.x)), FS)):
            with pytest.raises(TypeError):
                env * other
            with pytest.raises(TypeError):
                other * env

    def test_modulation_spectrum_averages_channels(self):
        stereo = so.Sound(np.column_stack([self.x.data[:, 0], 0.5 * self.x.data[:, 0]]), FS)
        fb = so.cosine_filterbank(8)
        both = fb.analyze(stereo).envelopes(fs=1000).modulation_spectrum()
        mean = fb.analyze(0.75 * self.x).envelopes(fs=1000).modulation_spectrum()
        np.testing.assert_allclose(both.level, mean.level, atol=1e-6)
        with pytest.raises(ValueError, match="'linear' or 'db'"):
            fb.analyze(self.x).envelopes().modulation_spectrum(scale="log")

    def test_bank_without_edges_keeps_every_band(self):
        x = so.harmonic_complex(0.5, FAST, 150, np.arange(1, 30), phases="random", rng=0)
        bank = so.cosine_filterbank(f_lo=125, f_hi=FAST_HI, spacing=1 / 12, scale="octave", edges=False)
        with_edges = so.cosine_filterbank(f_lo=125, f_hi=FAST_HI, spacing=1 / 12, scale="octave")
        kept = bank.analyze(x).envelopes(fs=1000).modulation_spectrum()
        dropped = with_edges.analyze(x).envelopes(fs=1000).modulation_spectrum()
        # the same inner bands either way
        np.testing.assert_allclose(kept.level, dropped.level, atol=1e-9)


class TestNoiseVocode:
    x = so.harmonic_complex(0.5, FS, 120, np.arange(1, 50), phases="sine")

    def test_noise_bands_keep_their_own_fluctuations(self):
        fb = so.cosine_filterbank(16, 80, 7600)

        def fluctuation(sound):
            env = fb.analyze(sound).envelopes().data[FS // 10 : -FS // 10, 1:-1, 0]
            return np.median(env.std(0) / env.mean(0))

        noise = so.channel_vocode(self.x, 16, 80, 7600, rng=1)
        tone = so.channel_vocode(self.x, 16, 80, 7600, carrier="tone")
        # a steady input: tones carry its steady envelopes, noise bands add their own
        # fluctuations (measured 0.52; the noise's fine structure alone gave 0.32)
        assert fluctuation(tone) < 0.01 and fluctuation(noise) > 0.45

    def test_options(self):
        a = so.channel_vocode(self.x, 8, rng=1)
        np.testing.assert_array_equal(a.data, so.channel_vocode(self.x, 8, rng=1).data)
        assert not np.array_equal(a.data, so.channel_vocode(self.x, 8, rng=2).data)
        # f_hi above Nyquist is the same as Nyquist
        above = so.channel_vocode(self.x, 8, 80, 2 * FS, rng=1)
        np.testing.assert_array_equal(above.data, so.channel_vocode(self.x, 8, 80, FS / 2, rng=1).data)
        # the envelope lowpass smooths the envelopes the tones carry
        fb = so.cosine_filterbank(8)
        smooth = so.channel_vocode(self.x, 8, carrier="tone", env_lowpass=20)
        rough = so.channel_vocode(self.x, 8, carrier="tone", env_lowpass=None)
        fast = [np.abs(np.fft.rfft(fb.analyze(s).envelopes().data[:, 4, 0])) for s in (smooth, rough)]
        freqs = np.fft.rfftfreq(len(self.x), 1 / FS)
        band = (freqs > 60) & (freqs < 200)  # the 120 Hz periodicity
        assert fast[0][band].sum() < 0.3 * fast[1][band].sum()  # measured 0.15

    def test_carrier_checks(self):
        with pytest.raises(ValueError, match="shorter"):
            so.channel_vocode(self.x, 8, carrier=so.gaussian_noise(0.2, FS, rng=0))
        with pytest.raises(ValueError, match="sample rates differ"):
            so.channel_vocode(self.x, 8, carrier=so.gaussian_noise(1.0, 22050, rng=0))
        with pytest.raises(ValueError, match="carrier must be"):
            so.channel_vocode(self.x, 8, carrier="pink")
        with pytest.raises(ValueError, match="carrier must be"):
            so.channel_vocode(self.x, 8, carrier=np.zeros(10))
