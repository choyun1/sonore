"""Binaural manipulation and cue analysis."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestBinaural:
    def test_sign_convention_written_by_hand(self):
        """Positive ITD and ILD point right (module docstring): the right ear leads and is louder.
        Checked on the channels directly, so a sign flipped in both apply_itd_ild and
        interaural_cues cannot pass the round trip below unnoticed."""
        click = so.Sound(np.r_[1.0, np.zeros(99)], FS)
        b = so.apply_itd_ild(click, itd=10 / FS, ild=6)
        left, right = b.data[:, 0], b.data[:, 1]
        assert np.argmax(right) == 0 and np.argmax(left) == 10
        assert 20 * np.log10(right.max() / left.max()) == pytest.approx(6, abs=1e-12)

    def test_interaural_cues_sign_on_a_hand_made_sound(self):
        """The right channel is the left one 8 samples earlier and twice as large: ITD +8/FS, ILD +6.02 dB."""
        g = so.gaussian_noise(1, FS, band=(100, 8000), rng=0).data[:, 0]
        left, right = g[:-8], 2 * g[8:]
        c = so.interaural_cues(so.Sound(np.column_stack([left, right]), FS), 50e-3)
        assert np.nanmedian(c.itd) == pytest.approx(8 / FS, abs=1e-6)
        # each window sees slightly different noise samples in the two ears, hence not exact
        assert np.nanmedian(c.ild) == pytest.approx(20 * np.log10(2), abs=0.01)

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


def test_zero_lag_correlation_tracks_oscor():
    s = so.oscor(2, FS, 2, rng=0)
    c = so.interaural_cues(s, 10e-3)
    ok = np.isfinite(c.corr0)
    r = np.corrcoef(c.corr0[ok], np.sin(2 * np.pi * 2 * c.t[ok]))[0, 1]
    assert r > 0.95
