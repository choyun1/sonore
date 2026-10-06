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
        stereo = so.Sound(np.column_stack([self.x.data[:, 0], self.x.data[::-1, 0]]), FS)
        out = so.cosine_filterbank(8).analyze(stereo).envelopes().to_sound("noise", rng=0)
        assert out.n_channels == 2 and not np.allclose(out.data[:, 0], out.data[:, 1])
