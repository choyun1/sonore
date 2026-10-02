"""The reassigned spectrogram."""

import numpy as np
import pytest

import sonore as so

_RFS = 16000


_RT = np.arange(_RFS) / _RFS


_REASSIGN_FRAMES = {
    "hann": so.GaborFrame(0.032, 0.004),
    "gaussian": so.GaborFrame(0.032, 0.004, window=("gaussian", 64)),
}


def _reassign(x, frame, threshold_db):
    r = so.reassigned_spectrogram(so.Sound(x, _RFS), frame, threshold_db=threshold_db)
    return r, r.keep[0] & (r.t_hat[0] > 0.1) & (r.t_hat[0] < 0.9)


@pytest.mark.parametrize("window", _REASSIGN_FRAMES)
def test_reassignment_moves_a_tone_to_its_frequency(window):
    """0.37 bins off the grid; a sign slip in the correction would miss by about 100 Hz."""
    f_tone = 1000 + 0.37 * _RFS / 512
    r, keep = _reassign(np.cos(2 * np.pi * f_tone * _RT), _REASSIGN_FRAMES[window], -20)
    assert np.abs(r.f_hat[0][keep] - f_tone).max() < 0.2


@pytest.mark.parametrize("window", _REASSIGN_FRAMES)
def test_reassignment_moves_an_impulse_to_its_time(window):
    x = np.zeros(_RFS)
    x[8050] = 1.0
    r = so.reassigned_spectrogram(so.Sound(x, _RFS), _REASSIGN_FRAMES[window], threshold_db=-40)
    assert np.abs(r.t_hat[0][r.keep[0]] - 8050 / _RFS).max() < 1e-9


@pytest.mark.parametrize("window", _REASSIGN_FRAMES)
def test_reassignment_puts_a_chirp_on_its_line(window):
    r, keep = _reassign(np.cos(2 * np.pi * (500 * _RT + 1500 * _RT**2)), _REASSIGN_FRAMES[window], -20)
    assert np.abs(r.f_hat[0][keep] - (500 + 3000 * r.t_hat[0][keep])).max() < 0.2


def test_reassignment_binning_and_limits():
    s = so.correlated_noise(0.2, _RFS, corr=0, rng=0)
    r = so.reassigned_spectrogram(s, so.GaborFrame(0.016, 0.002), threshold_db=-200)
    assert r.t_hat.shape == r.power.shape and r.power.shape[0] == 2
    g = r.binned(np.linspace(-1, 2, 4), np.linspace(-1e5, 1e5, 3))
    # nearly all kept power lands in these wide cells; a few near-silent cells are thrown far outside
    np.testing.assert_allclose(g.power.sum(axis=(1, 2)), r.power.sum(axis=(1, 2), where=r.keep), rtol=1e-8)
    assert g.power.shape == (2, 2, 3)
    with pytest.raises(ValueError, match="hann"):
        so.reassigned_spectrogram(s, so.GaborFrame(0.016, window="hamming"))
