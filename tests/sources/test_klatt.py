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


def test_lf_vowel_is_lf_flow_times_formants_times_radiation():
    formants = [(730, 60), (1090, 100), (2440, 120), (3400, 175), (4500, 250)]
    p = {"SS": 3, "RD": 1.2}
    for k, (f, b) in enumerate(formants, 1):
        p |= {f"F{k}": f, f"B{k}": b}
    y = so.klatt_synthesize(1.0, FS, F0=100, **p)
    numbers = np.arange(1, 61)
    f = numbers * 100.0
    got = _line_spectrum(y, 100, 60)
    radiation = np.abs(1 - np.exp(-2j * np.pi * f / FS))
    flow = np.abs(so.lf_harmonics(numbers, 1.2, flow=True))
    want = flow * radiation * np.prod([_gain(*fb, f) for fb in formants], axis=0)
    np.testing.assert_allclose(20 * np.log10(got / got[0]), 20 * np.log10(want / want[0]), atol=1e-6)


def test_lf_source_is_darker_than_the_impulse_source():
    impulses = _line_spectrum(so.klatt_synthesize(1.0, FS), 100, 30)
    lf = _line_spectrum(so.klatt_synthesize(1.0, FS, SS=3), 100, 30)
    drop = 20 * np.log10((lf[29] / lf[0]) / (impulses[29] / impulses[0]))
    assert drop == pytest.approx(-11.8, abs=0.1)  # 3 kHz, relative to the first harmonic


def test_rd_may_change_over_time():
    y = so.klatt_synthesize(0.3, FS, SS=3, RD=([0.0, 0.3], [0.4, 2.0]), F0=([0.0, 0.3], [100.0, 140.0]))
    assert np.isfinite(y.data).all() and y.rms == pytest.approx(1)


def test_source_switch_is_checked():
    with pytest.raises(ValueError, match="KLGLOTT88"):
        so.klatt_synthesize(0.1, FS, SS=2)
    with pytest.raises(ValueError, match="SS must be"):
        so.klatt_synthesize(0.1, FS, SS=([0.0, 0.1], [1, 3]))
    with pytest.raises(ValueError, match="same at both ends"):
        so.klatt_continuum({"SS": 1}, {"SS": 3}, 3)
    steps = so.klatt_continuum({"SS": 3, "RD": 0.5}, {"SS": 3, "RD": 1.5}, 3)
    assert [step["RD"] for step in steps] == [0.5, 1.0, 1.5] and all(step["SS"] == 3 for step in steps)


def _band_levels(y, n=1000):
    """Mean power spectrum (dB) of `y` over Hann-windowed blocks of `n` samples."""
    blocks = y[: len(y) // n * n].reshape(-1, n) * np.hanning(n)
    power = np.mean(np.abs(np.fft.rfft(blocks, axis=1)) ** 2, axis=0)
    return np.fft.rfftfreq(n, 1 / FS), 10 * np.log10(power)


def test_voice_level_does_not_depend_on_where_or_how_long_it_is_voiced():
    # Breath after voice, the voice switched off by F0 = 0 or by AV = 0. The
    # voice is scaled over its voiced samples, and the breath is modulated
    # only where both F0 and AV are on, so the two give the same balance. No
    # click at the start: the balance is the same for a short sound.
    def breath_over_voice(duration, **voicing):
        half, end = duration / 2, int(duration * FS)
        y = so.klatt_synthesize(
            duration, FS, AH=([0, half, half + 1 / FS], [0, 0, 60]), F1=730, F2=1090, rng=0, **voicing
        ).data[:, 0]
        voice, breath = y[end // 10 : end // 2 - end // 20], y[end // 2 + end // 10 : end - end // 20]
        return 10 * np.log10(np.mean(breath**2) / np.mean(voice**2))

    for duration in (0.1, 0.5):
        step = ([0, duration / 2, duration / 2 + 1 / FS], [1, 1, 0])
        f0_off = breath_over_voice(duration, F0=(step[0], 100 * np.array(step[1])))
        av_off = breath_over_voice(duration, AV=(step[0], 60 * np.array(step[1])))
        assert f0_off == pytest.approx(av_off, abs=0.5)
        assert abs(av_off) < 1


def test_noise_is_halved_in_the_second_half_of_each_period():
    # Frication straight through the bypass, with a voice too quiet to matter:
    # the second half of each 50 Hz period has a quarter of the first's power.
    y = so.klatt_synthesize(2.0, FS, F0=50, AV=1, AF=60, AB=60, rng=0).data[:, 0]
    power = y.reshape(-1, FS // 50) ** 2
    halves = 10 * np.log10(power[:, FS // 100 :].mean() / power[:, : FS // 100].mean())
    assert halves == pytest.approx(-6.02, abs=0.3)


def test_bypass_is_flat():
    f, level = _band_levels(so.klatt_synthesize(4.0, FS, AV=0, AF=60, AB=60, rng=0).data[:, 0])
    low, high = level[(f > 200) & (f < 1000)].mean(), level[(f > 5000) & (f < 7000)].mean()
    assert low == pytest.approx(high, abs=0.5)


def test_nasal_pole_and_zero():
    y = so.klatt_synthesize(1.0, FS, F0=100, FNP=270, BNP=100, FNZ=450, BNZ=100)
    f = np.arange(1, 61) * 100.0
    got = _line_spectrum(y, 100, 60)
    radiation = np.abs(1 - np.exp(-2j * np.pi * f / FS))
    formants = [(500, 60), (1500, 90), (2500, 150), (3500, 200), (4500, 250)]
    nasal = _gain(270, 100, f) / _gain(450, 100, f)
    want = _gain(0, 100, f) * radiation * nasal * np.prod([_gain(*fb, f) for fb in formants], axis=0)
    np.testing.assert_allclose(20 * np.log10(got / got[0]), 20 * np.log10(want / want[0]), atol=1e-6)


def test_parallel_formants_alternate_in_sign():
    # With alternating signs, two neighbors add between their peaks, as the
    # cascade does; with equal signs they would cancel there.
    y = so.klatt_synthesize(4.0, FS, AV=0, AF=60, A2=60, A3=60, F2=1500, F3=2500, rng=0).data[:, 0]
    f, level = _band_levels(y)
    at = lambda fc: level[np.argmin(np.abs(f - fc))]  # noqa: E731
    assert at(1500) - at(2000) < 15


def test_first_parallel_formant_input_is_not_differenced():
    y = so.klatt_synthesize(4.0, FS, AV=0, AF=60, A1=60, F1=500, B1=60, rng=0).data[:, 0]
    f, level = _band_levels(y)
    at = lambda fc: level[np.argmin(np.abs(f - fc))]  # noqa: E731
    want = 20 * np.log10(_gain(500, 60, 112) / _gain(500, 60, 500))
    assert at(112) - at(500) == pytest.approx(want, abs=1)


def test_parallel_formants_at_a_low_rate():
    # A formant that is off may lie above Nyquist; one that is on may not.
    assert so.klatt_synthesize(0.1, 8000, AV=0, AF=60, A2=60, rng=0).rms == pytest.approx(1)
    with pytest.raises(ValueError, match="A6 is on"):
        so.klatt_synthesize(0.1, 8000, AV=0, AF=60, A6=60, rng=0)


def test_negative_f0_is_refused():
    with pytest.raises(ValueError, match="F0 must be"):
        so.klatt_synthesize(0.1, FS, F0=([0, 0.1], [100, -10]))


def test_continuum_mixes_numbers_and_tracks():
    steps = so.klatt_continuum({"F1": 300}, {"F1": ([0, 0.1], [500, 700])}, 3)
    np.testing.assert_allclose(steps[1]["F1"][0], [0, 0.1])
    np.testing.assert_allclose(steps[1]["F1"][1], [400, 500])


def test_continuum_refusals():
    with pytest.raises(ValueError, match="at least 2"):
        so.klatt_continuum({}, {}, 1)
    with pytest.raises(ValueError, match="unknown"):
        so.klatt_continuum({"f1": 300}, {}, 3)
    for bad in ("ab", ([0, 1], [1, 2, 3]), ([[0, 1]], [[1, 2]])):
        with pytest.raises(ValueError, match="number or a"):
            so.klatt_continuum({"F1": bad}, {}, 3)
