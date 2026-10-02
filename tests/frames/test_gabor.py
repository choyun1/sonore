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
