"""STFTs: the Gabor frame's coefficients."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


@pytest.mark.parametrize("win", [20e-3, 20.1e-3])
def test_stft_perfect_reconstruction(win):
    g = so.gaussian_noise(0.5, FS, rng=0)
    np.testing.assert_allclose(so.STFT(g, win).to_sound().data, g.data, atol=1e-10)


def test_stft_stereo():
    s = so.correlated_noise(0.2, FS, corr=0, rng=0)
    S = so.STFT(s)
    assert S.data.shape[0] == 2
    np.testing.assert_allclose(S.to_sound().data, s.data, atol=1e-10)


def test_stft_time_axis():
    S = so.STFT(so.gaussian_noise(1, FS, rng=0), 20e-3, 10e-3)
    assert np.diff(S.t) == pytest.approx(10e-3)


def test_griffin_lim_converges():
    x = so.harmonic_complex(0.5, 16000, 200, np.arange(1, 10))
    S = so.STFT(x, 32e-3)
    y = S.griffin_lim(n_iter=50, rng=0)
    err = np.linalg.norm(so.STFT(y, 32e-3).magnitude - S.magnitude) / np.linalg.norm(S.magnitude)
    assert err < 0.1


def test_default_hop_is_a_quarter_window():
    """32 ms at 8 kHz: a 256-sample window, a 64-sample hop, and n_fft = the window."""
    assert so.GaborFrame(32e-3).lengths(8000) == (256, 64, 256)


def test_time_varying_window_centers_round_to_the_nearest_sample():
    """At 1 kHz, 2.6 ms is 2.6 samples, which rounds to 3; 2.4 ms rounds to 2."""
    frame = so.TVGaborFrame([0.0026, 0.0054], [0.004, 0.004])
    assert frame.layout(1000).centers.tolist() == [3, 5]
