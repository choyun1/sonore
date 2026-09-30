"""Envelope and Envelopes types."""

import numpy as np
import pytest
from helpers import FAST, FAST_HI, FS

import sonore as so


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
        fb = so.OctaveFilterbank.per_octave(12, 125, FAST_HI)
        direct = so.ModulationSpectrum.octave(x, f_hi=FAST_HI)
        via = fb.analyze(x).envelopes(fs=1000).modulation_spectrum()
        np.testing.assert_allclose(via.level, direct.level)
        erb = so.subbands(x, 20).envelopes(fs=1000).modulation_spectrum()
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
