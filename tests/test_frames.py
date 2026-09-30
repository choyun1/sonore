"""Frames: the Frame/Filterbank contract (docs/design/frames.md).

The dense-matrix oracle checks (bounds as eigenvalues, masked coefficients vs.
canonical least squares) come with the oracle helper; these tests cover the
interface, the tight/general equivalence and the non-frame behaviour.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from helpers import GaussianFilterbank
from scipy.signal.windows import hann

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


# ------------------------------------------------------------------- Gabor
WIN, HOP = 32 / FS, 8 / FS  # 32-sample window, 8-sample hop


def _hann15(n):
    return hann(n, sym=False) ** 1.5


# Hann^1.5 is tight at hop = win/4 too (sin^6 summed over 4 equal shifts is
# constant); a 10-sample hop on the 32-sample window makes it genuinely non-tight.
GABORS = {
    "hann": so.GaborFrame(WIN, HOP),
    "hann^1.5": so.GaborFrame(WIN, 10 / FS, window=_hann15),
    "hann^1.5, n_fft 48": so.GaborFrame(WIN, 10 / FS, window=_hann15, n_fft=48),
    "hann^1.5, n_fft 45": so.GaborFrame(WIN, 5 / FS, window=_hann15, n_fft=45),
}


def _scipy_frame_power(frame, n):
    """s(t) from SciPy's own frame range p_min..p_max (tools/check_frame_claims.py)."""
    sft, w = frame.sft(FS), frame.window_samples(FS)
    s = np.zeros(n)
    for q in range(sft.p_min, sft.p_max(n)):
        t = q * sft.hop - sft.m_num_mid + np.arange(len(w))
        ok = (t >= 0) & (t < n)
        s[t[ok]] += sft.mfft * w[ok] ** 2
    return s


@pytest.mark.parametrize("name", GABORS)
@pytest.mark.parametrize("n_channels", [1, 2])
def test_gabor_reconstructs_exactly(name, n_channels):
    frame, x = GABORS[name], _noise(n_channels, seed=2)
    S = frame.analyze(x)
    assert isinstance(S, so.STFT) and S.frame is frame
    assert S.data.shape[:2] == (n_channels, frame.sft(FS).f_pts)
    assert np.allclose(frame.synthesize(S).data, x.data, rtol=0, atol=1e-12)
    assert np.allclose(S.to_sound().data, x.data, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", GABORS)
def test_gabor_frame_power_matches_scipy_frame_range(name):
    frame = GABORS[name]
    for n in (64, 97, 128):
        assert np.allclose(frame.frame_power(n, FS), _scipy_frame_power(frame, n), rtol=1e-13, atol=0)


def test_hann_quarter_hop_is_tight_and_hann15_is_not():
    lo, hi = GABORS["hann"].frame_bounds(N, FS)
    assert abs(hi / lo - 1) < 1e-12
    lo, hi = GABORS["hann^1.5"].frame_bounds(N, FS)
    assert lo < 0.9 * hi


@pytest.mark.parametrize("name", GABORS)
def test_gabor_energy_lies_within_frame_bounds(name):
    frame = GABORS[name]
    lo, hi = frame.frame_bounds(N, FS)
    for seed in range(5):
        x = _noise(2, seed)
        e, signal = frame.energy(frame.analyze(x)), np.sum(x.data**2, axis=0)
        assert np.all(lo * signal <= e * (1 + 1e-12)) and np.all(e <= hi * signal * (1 + 1e-12))


def test_gabor_bin_weights():
    assert GABORS["hann^1.5, n_fft 48"].bin_weights(FS).tolist() == [1.0] + [2.0] * 23 + [1.0]
    assert GABORS["hann^1.5, n_fft 45"].bin_weights(FS).tolist() == [1.0] + [2.0] * 22


def test_stft_is_the_gabor_frame():
    x = _noise(2, 4)
    S = so.STFT(x, WIN, HOP)
    assert S.frame == so.GaborFrame(WIN, HOP) and S.sft is S.frame.sft(FS)
    assert np.array_equal(S.data, so.GaborFrame(WIN, HOP).analyze(x).data)
    assert so.STFT(x).frame == so.GaborFrame(20e-3)


def test_gabor_coverage_gap_raises_with_bounds():
    gap = so.GaborFrame(WIN, WIN)  # periodic Hann is 0 at its first sample: hop = window leaves gaps
    lo, hi = gap.frame_bounds(N, FS)  # computed without SciPy's (refused) object
    assert lo == 0.0 and hi > 0
    with pytest.raises(ValueError, match=r"not a frame \(bounds A=0,"):
        gap.analyze(_noise())
    with pytest.raises(ValueError, match="leaves gaps"):
        so.GaborFrame(WIN, 1.25 * WIN)


def test_gabor_rejects_bad_parameters():
    with pytest.raises(ValueError, match="n_fft"):
        so.GaborFrame(WIN, HOP, n_fft=16).analyze(_noise())
    S = GABORS["hann"].analyze(_noise())
    with pytest.raises(ValueError, match="frequency bins"):
        GABORS["hann^1.5, n_fft 48"].synthesize(S)
