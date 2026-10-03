"""Modulation spectra."""

import warnings

import numpy as np
import pytest
from helpers import FAST, FAST_HI, FS

import sonore as so


def test_modulation_spectrum_peak():
    x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 8, depth=1)
    ms = so.ModulationSpectrum(so.STFT(x, 20e-3))
    row = ms.level[0]
    pos = ms.w_t > 2
    assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)


# -- going back to a sound -------------------------------------------------
def _speech_like(rng=0):
    """Noise with a slow (4 Hz) and a fast (16 Hz) amplitude modulation, 2 s at FAST."""
    noise = so.gaussian_noise(2, FAST, rng=rng)
    return so.amplitude_modulate(so.amplitude_modulate(noise, 4, depth=0.9), 16, depth=0.5)


def _octave_spectrum(x, scale="linear"):
    bank = so.cosine_filterbank(f_lo=250, f_hi=FAST_HI, spacing=1 / 6, scale="octave")
    envelopes = bank.analyze(x).envelopes(fs=1000)
    return envelopes, envelopes.modulation_spectrum(scale)


def _share_above(spectrum, rate_hz):
    """Share of the untapered modulation power faster than rate_hz."""
    rate = np.abs(np.fft.fftfreq(spectrum._magnitude.shape[1], 1 / spectrum._analysis.fs))[None, :]
    power = spectrum._magnitude**2
    return power[np.broadcast_to(rate > rate_hz, power.shape)].sum() / power.sum()


@pytest.mark.parametrize("scale", ["linear", "db"])
def test_own_sound_as_carrier_rebuilds_its_envelopes(scale):
    x = _speech_like()
    envelopes, spectrum = _octave_spectrum(x, scale)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to clip
        rebuilt = spectrum.to_envelopes(carrier=x)
    inner = slice(1, -1)
    np.testing.assert_allclose(rebuilt.data[:, inner], envelopes.data[:, inner], rtol=0, atol=1e-9)
    assert np.all(rebuilt.data[:, [0, -1]] == 0)  # the spectrum dropped the edge bands


def test_own_sound_as_carrier_gives_the_sound_back_inside_the_bands():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    y = spectrum.to_sound(carrier=x)
    bank = spectrum._analysis.filterbank
    envelopes = bank.analyze(x).envelopes(fs=1000)
    unpadded = so.Envelopes(envelopes.data, 1000, bank).without_edges()  # rebuilt envelopes carry no padding
    expected = (unpadded * bank.analyze(x).tfs()).to_sound()
    np.testing.assert_allclose(y.data, expected.normalize().data, atol=1e-9)


def test_random_phase_keeps_the_magnitudes():
    _, spectrum = _octave_spectrum(_speech_like())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # random phase clips some cells
        rebuilt = spectrum.to_envelopes(rng=1)
    unclipped = rebuilt.data[:, 1:-1, 0].T
    assert (
        np.corrcoef(np.abs(np.fft.fft2(unclipped - unclipped.mean())).ravel(), spectrum._magnitude.ravel())[
            0, 1
        ]
        > 0.9
    )


def test_clipping_is_reported():
    _, spectrum = _octave_spectrum(_speech_like())
    with pytest.warns(UserWarning, match="were clipped"):
        spectrum.to_envelopes(rng=1)


def test_with_gain_removes_fast_modulations_on_tones():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    slow = spectrum.with_gain(lambda rate, density: np.abs(rate) <= 8)
    assert _share_above(slow, 8) == 0
    assert np.all(slow.level <= spectrum.level + 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = slow.to_sound(carrier="tones", fs=FAST, rng=2)
    _, again = _octave_spectrum(y)
    assert _share_above(again, 12) < 0.25 * _share_above(spectrum, 12)


def test_with_gain_is_symmetric():
    _, spectrum = _octave_spectrum(_speech_like())
    downward_only = spectrum.with_gain(lambda rate, density: (rate * density > 0) | (density == 0))
    magnitude = downward_only._magnitude
    mirrored = np.roll(magnitude[::-1, ::-1], 1, axis=(0, 1))
    np.testing.assert_allclose(magnitude, mirrored, rtol=1e-12, atol=1e-12)


def test_to_sound_needs_envelopes_and_a_rate():
    stft_spectrum = so.ModulationSpectrum(so.STFT(_speech_like(), 20e-3))
    with pytest.raises(TypeError, match="made from Envelopes"):
        stft_spectrum.to_sound(carrier="tones", fs=FAST)
    _, spectrum = _octave_spectrum(_speech_like())
    with pytest.raises(TypeError, match="fs is needed"):
        spectrum.to_sound(carrier="tones")
    with pytest.raises(ValueError, match="carrier must be"):
        spectrum.to_sound(carrier="pink", fs=FAST)
    with pytest.raises(ValueError, match="shorter"):
        spectrum.to_sound(carrier=so.gaussian_noise(1, FAST, rng=0))


@pytest.mark.parametrize("carrier", ["tones", "noise"])
def test_random_carriers_are_seeded(carrier):
    _, spectrum = _octave_spectrum(_speech_like())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = spectrum.to_sound(carrier=carrier, fs=FAST, rng=3)
        b = spectrum.to_sound(carrier=carrier, fs=FAST, rng=3)
    assert a.duration == pytest.approx(2) and a.rms == pytest.approx(1)
    np.testing.assert_array_equal(a.data, b.data)
