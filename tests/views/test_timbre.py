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


def test_track_plot_draws_one_channel_against_time():
    import matplotlib

    matplotlib.use("Agg")
    track = so.spectral_centroid(tone())
    ax = track.plot()
    line = ax.get_lines()[0]
    np.testing.assert_array_equal(line.get_xdata(), track.t)
    np.testing.assert_array_equal(line.get_ydata(), track.values[0])
    assert ax.get_ylabel() == "spectral centroid [Hz]" and ax.get_xlabel() == "Time [s]"
    assert so.spectral_flux(tone()).plot().get_ylabel() == "spectral flux"


def test_attack_of_a_sound_that_ends_loud():
    """Silence, a 10 ms rise, then a tone to the very end: the end must not wrap round to the start."""
    t = np.arange(FS) / FS
    envelope = np.interp(t, [0, 0.5, 0.51, 1.0], [0, 0, 1, 1])
    start, end = so.attack_segment(so.Sound(envelope * np.sin(2 * np.pi * F0 * t), FS))
    assert 0.45 < start < 0.51 and end < 0.55


def _shaped_sine(times, levels, f_carrier=2000.0):
    """A sine whose amplitude follows the piecewise-linear (times, levels), so the envelope is known."""
    t = np.arange(FS) / FS
    return so.Sound(np.interp(t, times, levels) * np.sin(2 * np.pi * f_carrier * t), FS)


def test_attack_follows_the_weakest_effort_rule():
    """A first effort of 38 ms against a mean of 15 ms is weak (at most three times the mean) but
    would not be at twice; the attack starts at the envelope's dip inside it, 125 ms, not at the
    0.1 crossing (108 ms) or the 0.2 crossing (146 ms), and ends at the peak, 242 ms, not at the
    0.9 crossing (230 ms)."""
    rise = [0.146 + 0.012 * step for step in range(1, 9)]
    sound = _shaped_sine(
        [0, 0.1, 0.110, 0.125, 0.146, *rise, 0.35, 0.9, 1.0],
        [0, 0, 0.13, 0.05, 0.2, *(0.2 + 0.1 * np.arange(1, 9)), 0.8, 0.8, 0],
    )
    start, end = so.attack_segment(sound, cutoff=200)
    assert start == pytest.approx(0.125, abs=0.002)
    assert end == pytest.approx(0.242, abs=0.002)


def test_attack_skips_a_slow_lead_in():
    """A 400 ms swell to 15% before a 10 ms rise: the first effort is far above three times the
    mean, so the attack starts after it, not at the 0.1 crossing near 270 ms."""
    t = np.arange(FS) / FS
    envelope = np.interp(t, [0, 0.4, 0.41, 0.95, 1.0], [0, 0.15, 1, 1, 0])
    start, end = so.attack_segment(so.Sound(envelope * np.sin(2 * np.pi * F0 * t), FS))
    assert 0.38 < start < 0.41 and end < 0.45


def test_attack_time_is_at_least_one_sample():
    impulse = np.zeros(FS)
    impulse[0] = 1.0
    assert so.log_attack_time(so.Sound(impulse, FS)) == pytest.approx(np.log10(1 / FS))


def test_envelope_is_a_third_order_butterworth_of_the_padded_hilbert_amplitude():
    from scipy.signal import butter, filtfilt, hilbert

    from sonore.views.timbre import _energy_envelope

    sound = tone(0.05, duration=0.3)
    samples = sound.data[:, 0]
    b, a = butter(3, 20 / (FS / 2))
    expected = filtfilt(b, a, np.abs(hilbert(samples, N=2 * len(samples)))[: len(samples)])
    np.testing.assert_allclose(_energy_envelope(sound, 0, 20.0, True), expected)


def test_track_summaries_are_median_and_interquartile_range():
    track = so.spectral_flux(tone(slope=1.5))
    np.testing.assert_allclose(track.median, np.nanmedian(track.values, axis=1))
    upper, lower = np.nanpercentile(track.values, [75, 25], axis=1)
    np.testing.assert_allclose(track.iqr, upper - lower)
    assert track.median[0] != pytest.approx(np.nanmean(track.values))


def test_track_times_are_window_centres():
    centroid = so.spectral_centroid(tone())
    n_window, n_hop = round(0.0232 * FS), round(0.0058 * FS)
    assert centroid.t[0] == pytest.approx(n_window / 2 / FS)
    assert np.diff(centroid.t)[0] == pytest.approx(n_hop / FS)
    # the paper's flux, one hop apart, starts one window later
    assert so.spectral_flux(tone(), spacing=None).t[0] == pytest.approx(centroid.t[1])


def test_flux_needs_two_windows_spacing_apart():
    with pytest.raises(ValueError, match="too short"):
        so.spectral_flux(so.Sound(np.ones(round(0.05 * FS)), FS))
