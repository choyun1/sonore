"""The Klatt-style formant synthesizer."""

from types import SimpleNamespace

import numpy as np
import pytest

import sonore as so

FS = 16000


def _line_spectrum(sound, f0, n_harm, window=0.5):
    """Magnitudes at harmonics 1..n_harm over the last `window` seconds (a
    whole number of periods, so each harmonic falls on a bin)."""
    x = sound.data[-int(window * FS) :, 0]
    spec = np.abs(np.fft.rfft(x))
    return spec[(np.arange(1, n_harm + 1) * f0 * window).astype(int)]


def _gain(f_res, bw, f):
    c = -np.exp(-2 * np.pi * bw / FS)
    b = 2 * np.exp(-np.pi * bw / FS) * np.cos(2 * np.pi * f_res / FS)
    z = np.exp(-2j * np.pi * f / FS)
    return np.abs((1 - b - c) / (1 - b * z - c * z**2))


def test_vowel_is_source_times_formants_times_radiation():
    formants = [(730, 60), (1090, 100), (2440, 120), (3400, 175), (4500, 250)]
    p = {}
    for k, (f, b) in enumerate(formants, 1):
        p |= {f"F{k}": f, f"B{k}": b}
    y = so.klatt_synthesize(1.0, FS, F0=100, **p)
    f = np.arange(1, 61) * 100.0
    got = _line_spectrum(y, 100, 60)
    radiation = np.abs(1 - np.exp(-2j * np.pi * f / FS))
    want = _gain(0, 100, f) * radiation * np.prod([_gain(*fb, f) for fb in formants], axis=0)
    np.testing.assert_allclose(20 * np.log10(got / got[0]), 20 * np.log10(want / want[0]), atol=1e-6)


def test_parallel_formant_levels_follow_their_amplitudes():
    y = so.klatt_synthesize(1.0, FS, AV=0, AF=60, A2=60, A4=50, F2=1500, F4=3500, rng=0).data[:, 0]
    seg = y.reshape(-1, 1000) * np.hanning(1000)
    s = np.mean(np.abs(np.fft.rfft(seg, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(1000, 1 / FS)
    level = lambda fc: 10 * np.log10(s[np.abs(f - fc) <= 32].max())  # noqa: E731
    assert level(1500) - level(3500) == pytest.approx(10, abs=1.5)
    assert level(1500) - level(800) > 10


def test_equal_levels_mean_equal_sources():
    # Aspiration and voicing at the same dB give about the same output level.
    y = so.klatt_synthesize(
        0.5, FS, AV=([0, 0.25, 0.26], [60, 60, 0]), AH=([0, 0.25, 0.26], [0, 0, 60]), F1=730, F2=1090, rng=0
    ).data[:, 0]
    voiced, breath = np.mean(y[800:4000] ** 2), np.mean(y[4800:7200] ** 2)
    assert abs(10 * np.log10(breath / voiced)) < 3


def test_noise_is_reproducible_from_a_seed():
    a = so.klatt_synthesize(0.1, FS, AH=50, rng=3)
    b = so.klatt_synthesize(0.1, FS, AH=50, rng=3)
    c = so.klatt_synthesize(0.1, FS, AH=50, rng=4)
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.allclose(a.data, c.data)


def test_silent_when_every_source_is_off():
    assert so.klatt_synthesize(0.1, FS, AV=0).rms == 0
    assert so.klatt_synthesize(0.1, FS, F0=0).rms == 0


def test_low_rate_drops_default_formants_above_nyquist():
    assert so.klatt_synthesize(0.1, 8000).rms == pytest.approx(1)
    with pytest.raises(ValueError):
        so.klatt_synthesize(0.1, 8000, F5=4500)


def test_unknown_parameter():
    with pytest.raises(ValueError, match="unknown"):
        so.klatt_synthesize(0.1, FS, f1=500)


def test_continuum():
    start = {"F2": ([0, 0.05], [900, 1100]), "F1": 300}
    steps = so.klatt_continuum(start, {"F2": ([0, 0.05], [1800, 1100])}, 3)
    assert steps[0]["F1"] == 300 and steps[2]["F1"] == so.KLATT_DEFAULTS["F1"]
    np.testing.assert_allclose(steps[1]["F2"][1], [1350, 1100])
    with pytest.raises(ValueError):
        so.klatt_continuum({"F2": ([0, 1], [1, 2])}, {"F2": ([0, 2], [1, 2])}, 3)


def test_f0_track_is_accepted():
    t = np.arange(0, 0.2, 0.005)
    f0 = np.where(t < 0.1, 120.0, 0.0)
    a = so.klatt_synthesize(0.2, FS, F0=SimpleNamespace(t=t, f0=f0))  # what an F0Track carries
    b = so.klatt_synthesize(0.2, FS, F0=(t, f0))
    np.testing.assert_array_equal(a.data, b.data)
