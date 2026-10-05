import numpy as np
import pytest

import sonore as so

FS = 44100
F0 = 311.13


def tone(attack=0.02, duration=1.0, slope=1.0, f0=F0, n_harmonics=20):
    """A harmonic complex with amplitudes n^-slope, a linear attack and a 50 ms linear release."""
    t = np.arange(int(duration * FS)) / FS
    harmonics = np.arange(1.0, n_harmonics + 1)
    signal = (harmonics[:, None] ** -slope * np.sin(2 * np.pi * f0 * harmonics[:, None] * t)).sum(0)
    envelope = np.minimum(1, t / attack) * np.clip((duration - t) / 0.05, 0, 1)
    return so.Sound(signal * envelope, FS)


def test_attack_time_grows_with_the_attack():
    measured = [so.log_attack_time(tone(attack)) for attack in (0.005, 0.02, 0.08, 0.3)]
    assert np.all(np.diff(measured) > 0)
    # a 300 ms linear attack measures between 200 and 320 ms with the default envelope
    assert np.log10(0.2) < measured[-1] < np.log10(0.32)


def test_paper_envelope_smears_short_attacks():
    """With the paper's descriptor setting (5 Hz, one pass) a 5 ms attack measures
    longer than 60 ms; the default (20 Hz, zero phase) keeps it under 25 ms."""
    short = tone(0.005)
    assert so.log_attack_time(short, cutoff=5, zero_phase=False) > np.log10(0.06)
    assert so.log_attack_time(short) < np.log10(0.025)


def test_attack_segment_lies_in_the_ramp():
    start, end = so.attack_segment(tone(0.3))
    assert 0 <= start < end <= 0.31


def test_attack_of_silence_is_an_error():
    with pytest.raises(ValueError, match="silent"):
        so.attack_segment(so.Sound(np.zeros(FS), FS))


def test_power_centroid_matches_the_partials():
    """For a steady harmonic tone the power centroid is sum(n f0 a_n^2) / sum(a_n^2)."""
    amplitudes = np.arange(1, 21) ** -1.0
    expected = (np.arange(1, 21) * F0 * amplitudes**2).sum() / (amplitudes**2).sum()
    measured = so.spectral_centroid(tone()).median[0]
    assert measured == pytest.approx(expected, rel=0.02)


def test_power_centroid_of_a_dull_tone_ignores_the_sample_rate():
    """The window's sidelobes lift a dull tone's magnitude centroid, more at a higher sample
    rate; its power centroid stays on the partials."""
    amplitudes = np.arange(1, 21) ** -3.0
    expected = (np.arange(1, 21) * F0 * amplitudes**2).sum() / (amplitudes**2).sum()
    dull = tone(slope=3)
    assert so.spectral_centroid(dull).median[0] == pytest.approx(expected, rel=0.01)
    assert so.spectral_centroid(dull.resample(22050)).median[0] == pytest.approx(expected, rel=0.01)
    assert so.spectral_centroid(dull, scale="magnitude").median[0] > 1.5 * expected


def test_magnitude_centroid_sits_a_little_above_the_partials():
    amplitudes = np.arange(1, 21) ** -1.0
    expected = (np.arange(1, 21) * F0 * amplitudes).sum() / amplitudes.sum()
    measured = so.spectral_centroid(tone(), scale="magnitude").median[0]
    assert expected < measured < 1.08 * expected


def test_brighter_tones_have_higher_centroids():
    centroids = [so.spectral_centroid(tone(slope=slope)).median[0] for slope in (3, 2, 1, 0.5)]
    assert np.all(np.diff(centroids) > 0)


def test_silent_time_windows_are_nan_and_skipped():
    sound = so.concat([so.silence(0.2, FS), tone(duration=0.5)])
    track = so.spectral_centroid(sound)
    assert np.isnan(track.values[0, 0])
    assert np.isfinite(track.median[0]) and np.isfinite(track.iqr[0])


def test_flux_tells_a_gliding_spectrum_from_a_steady_one():
    t = np.arange(FS) / FS
    harmonics = np.arange(1, 21)[:, None]
    gliding = so.Sound((harmonics ** -(2.0 - 1.5 * t) * np.sin(2 * np.pi * F0 * harmonics * t)).sum(0), FS)
    steady = tone()
    middle = slice(40, -40)
    gliding_flux = np.median(so.spectral_flux(gliding).values[0, middle])
    steady_flux = np.median(so.spectral_flux(steady).values[0, middle])
    assert gliding_flux > 10 * steady_flux
    # one hop apart, as in the paper, the two are much closer
    gliding_hop = np.median(so.spectral_flux(gliding, spacing=None).values[0, middle])
    steady_hop = np.median(so.spectral_flux(steady, spacing=None).values[0, middle])
    assert gliding_hop < 3 * steady_hop


def test_flux_track_starts_one_spacing_late():
    centroid = so.spectral_centroid(tone())
    flux = so.spectral_flux(tone(), spacing=0.1)
    lag = round(0.1 / 0.0058)
    assert flux.t[0] == pytest.approx(centroid.t[lag])
    assert flux.values.shape[1] == centroid.values.shape[1] - lag


def test_channels_are_measured_separately():
    left, right = tone(slope=2), tone(slope=0.5)
    track = so.spectral_centroid(so.Sound.from_channels(left, right))
    assert track.values.shape[0] == 2
    assert track.median[0] < track.median[1]


def test_descriptor_track_is_a_one_way_view():
    with pytest.raises(so.NotInvertibleError):
        so.spectral_centroid(tone()).to_sound()


def test_invalid_arguments():
    with pytest.raises(ValueError, match="scale"):
        so.spectral_centroid(tone(), scale="decibels")
    with pytest.raises(ValueError, match="shorter"):
        so.spectral_centroid(so.Sound(np.ones(100), FS))
