"""Frames: the Frame/Filterbank contract (docs/design/frames.md).

The dense-matrix oracle checks (bounds as eigenvalues, masked coefficients vs.
canonical least squares) come with the oracle helper; these tests cover the
interface, the tight/general equivalence and the non-frame behaviour.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from helpers import GaussianFilterbank

import sonore as so

FS = 8000
N = 96


def _noise(n_channels=1, seed=0):
    return so.Sound(np.random.default_rng(seed).standard_normal((N, n_channels)), FS)


@dataclass(frozen=True)
class _UntightERB(so.ERBFilterbank):
    """An ERB bank forced through the general (divide-by-s) synthesis path."""

    tight = False


def test_cosine_banks_are_tight_filterbank_frames():
    for fb in [so.ERBFilterbank(10, 50, 3000), so.OctaveFilterbank(6, 125, 3000)]:
        assert isinstance(fb, so.Filterbank) and isinstance(fb, so.Frame)
        assert fb.tight and fb.n_filters == fb.n_bands + 2
        assert fb.frame_bounds(N, FS) == (1.0, 1.0)
        s = fb.frame_power(N, FS)
        assert np.allclose(s, 1.0, atol=1e-12)  # the flag is backed by the responses


@pytest.mark.parametrize("pad", ["auto", 0])
@pytest.mark.parametrize("n_channels", [1, 2])
def test_tight_path_equals_general_path(pad, n_channels):
    x = _noise(n_channels)
    tight, general = so.ERBFilterbank(10, 50, 3000), _UntightERB(10, 50, 3000)
    sb = tight.analyze(x, pad=pad)
    masked = so.Subbands(sb._full * np.linspace(0, 1, sb._full.shape[1])[None, :, None], FS, tight, sb.pad)
    for coefs in (sb, masked):
        assert np.allclose(tight.synthesize(coefs).data, general.synthesize(coefs).data, rtol=0, atol=1e-12)
    lo, hi = general.frame_bounds(N, FS, pad=pad)
    assert abs(lo - 1) < 1e-12 and abs(hi - 1) < 1e-12


@pytest.mark.parametrize("pad", ["auto", 0])
@pytest.mark.parametrize("n_channels", [1, 2])
def test_nontight_filterbank_reconstructs_exactly(pad, n_channels):
    fb = GaussianFilterbank()
    x = _noise(n_channels, seed=1)
    lo, hi = fb.frame_bounds(N, FS, pad=pad)
    assert 0 < lo < 0.9 * hi  # genuinely non-tight
    sb = fb.analyze(x, pad=pad)
    assert sb.data.shape == (N, fb.n_filters, n_channels)
    assert np.allclose(sb.synthesize().data, x.data, rtol=0, atol=1e-12)


@pytest.mark.parametrize("pad", ["auto", 0])
def test_energy_lies_within_frame_bounds(pad):
    fb = GaussianFilterbank()
    lo, hi = fb.frame_bounds(N, FS, pad=pad)
    for seed in range(5):
        x = _noise(2, seed)
        e = fb.energy(fb.analyze(x, pad=pad))
        signal = np.sum(x.data**2, axis=0)  # zero padding adds no signal energy
        assert e.shape == (2,)
        assert np.all(lo * signal <= e * (1 + 1e-12)) and np.all(e <= hi * signal * (1 + 1e-12))


def test_cosine_energy_is_signal_energy():
    x = _noise(2, 3)
    for pad in ["auto", 0]:
        e = so.ERBFilterbank(10, 50, 3000).energy(so.ERBFilterbank(10, 50, 3000).analyze(x, pad=pad))
        assert np.allclose(e, np.sum(x.data**2, axis=0), rtol=1e-12)


def test_coverage_gap_analyzes_but_refuses_to_synthesize():
    fb = GaussianFilterbank(width=0.1)  # supports reach 0.3 spacings each side: gaps between filters
    lo, hi = fb.frame_bounds(N, FS)
    assert lo == 0.0 and hi > 0
    sb = fb.analyze(_noise())  # D4: analysis always works
    env = sb.envelopes()
    assert env.data.shape == (N, fb.n_filters, 1)
    with pytest.raises(ValueError, match="not a frame"):
        sb.synthesize()


def test_modulation_spectrum_needs_a_scale():
    env = GaussianFilterbank().analyze(_noise()).envelopes()
    with pytest.raises(TypeError, match="spacing"):
        env.modulation_spectrum()


def test_shape_checks_use_n_filters():
    fb = GaussianFilterbank(n=5)
    with pytest.raises(ValueError, match=r"\(n_samples, 5, n_channels\)"):
        so.Subbands(np.zeros((N, 7, 1)), FS, fb)
    with pytest.raises(ValueError, match=r"\(n_samples, 5, n_channels\)"):
        so.Envelopes(np.zeros((N, 7, 1)), FS, fb)
