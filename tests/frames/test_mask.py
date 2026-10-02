"""Time-frequency masks."""

import numpy as np
from helpers import FS

import sonore as so


def test_ibm_recovers_target_when_well_separated():
    target = so.pure_tone(1, FS, 500)
    masker = so.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
    mixture = target + masker
    mask = so.ideal_binary_mask(so.STFT(target), so.STFT(masker))
    out = (so.STFT(mixture) * mask).to_sound()
    snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
    assert snr > 20


def test_ideal_ratio_mask_known_cases():
    """(|T|^2 / (|T|^2 + |M|^2))**beta: equal powers give 0.5**beta, a silent masker 1, silence 0."""
    noise = so.STFT(so.gaussian_noise(0.2, FS, rng=0))
    silent = so.STFT(so.silence(0.2, FS))
    for beta in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(so.ideal_ratio_mask(noise, noise, beta).values, 0.5**beta, rtol=1e-12)
    np.testing.assert_allclose(so.ideal_ratio_mask(noise, silent).values, 1.0, rtol=1e-12)
    np.testing.assert_array_equal(so.ideal_ratio_mask(silent, noise).values, 0.0)
    np.testing.assert_array_equal(so.ideal_ratio_mask(silent, silent).values, 0.0)
