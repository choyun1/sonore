"""Spectra, STFTs, masks and modulation spectra."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestRepresentations:
    def test_spectrum_db_scale(self):
        g = so.gaussian_noise(0.5, FS, rng=0)
        d = so.Spectrum.from_sound(2 * g).level - so.Spectrum.from_sound(g).level
        assert np.median(d[1:]) == pytest.approx(20 * np.log10(2), abs=1e-6)

    @pytest.mark.parametrize("win", [20e-3, 20.1e-3])
    def test_stft_perfect_reconstruction(self, win):
        g = so.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(so.STFT(g, win).to_sound().data, g.data, atol=1e-10)

    def test_stft_stereo(self):
        s = so.correlated_noise(0.2, FS, corr=0, rng=0)
        S = so.STFT(s)
        assert S.data.shape[0] == 2
        np.testing.assert_allclose(S.to_sound().data, s.data, atol=1e-10)

    def test_stft_time_axis(self):
        S = so.STFT(so.gaussian_noise(1, FS, rng=0), 20e-3, 10e-3)
        assert np.diff(S.t) == pytest.approx(10e-3)

    def test_ibm_recovers_target_when_well_separated(self):
        target = so.pure_tone(1, FS, 500)
        masker = so.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
        mixture = target + masker
        mask = so.ideal_binary_mask(so.STFT(target), so.STFT(masker))
        out = (so.STFT(mixture) * mask).to_sound()
        snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
        assert snr > 20

    def test_griffin_lim_converges(self):
        x = so.harmonic_complex(0.5, 16000, 200, np.arange(1, 10))
        S = so.STFT(x, 32e-3)
        y = S.griffin_lim(n_iter=50, rng=0)
        err = np.linalg.norm(so.STFT(y, 32e-3).magnitude - S.magnitude) / np.linalg.norm(S.magnitude)
        assert err < 0.1

    def test_modulation_spectrum_peak(self):
        x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 8, depth=1)
        ms = so.ModulationSpectrum(so.STFT(x, 20e-3))
        row = ms.level[0]
        pos = ms.w_t > 2
        assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)
