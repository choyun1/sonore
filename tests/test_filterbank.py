"""Cosine filterbanks and subbands."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


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


def test_octave_filterbank_reconstructs():
    fb = so.OctaveFilterbank.per_octave(12, 125, 6000)
    H = fb.response(np.linspace(0, FS / 2, 5000))
    np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)
    g = so.gaussian_noise(0.5, 16000, rng=0)
    np.testing.assert_allclose(fb.analyze(g).synthesize().data, g.data, atol=1e-10)


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
    chosen = sb.plot(bands=[1, 3, 5])
    assert [ax.get_ylabel() for ax in chosen] == [f"{sb.cfs[i]:.0f}" for i in (5, 3, 1)]
