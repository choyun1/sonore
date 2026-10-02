"""Modulation spectra."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


def test_modulation_spectrum_peak():
    x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 8, depth=1)
    ms = so.ModulationSpectrum(so.STFT(x, 20e-3))
    row = ms.level[0]
    pos = ms.w_t > 2
    assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)
