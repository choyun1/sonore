"""so.source_filter_synthesize given known views: the level and noise share it
rebuilds, its channels and the inputs it refuses. The analyses' own errors
are checked in tests/views/test_spectral_envelope.py and test_aperiodicity.py,
and the whole chain on recordings in tools/check_voice_recipe_claims.py."""

from types import SimpleNamespace

import numpy as np
import pytest

import sonore as so

FS = 16000
HOP = 0.005


def flat_views(f0, share, duration=0.5, n_channels=1):
    """An envelope of 1 at every frequency (what white noise of variance 1
    reads as) and a noise share that is the same everywhere."""
    times = np.arange(int(duration / HOP)) * HOP
    power = np.ones((n_channels, 513, len(times)))
    envelope = so.SpectralEnvelope(power, times, np.arange(513) * FS / 1024, fs=FS)
    aperiodicity = so.Aperiodicity(np.full_like(power, np.sqrt(share)), times, FS, "truth")
    return (times, np.full(len(times), f0)), envelope, aperiodicity


@pytest.mark.parametrize("share_db", [-20.0, -6.0])
def test_it_rebuilds_a_known_level_and_noise_share(share_db):
    track, envelope, aperiodicity = flat_views(150.0, 10 ** (share_db / 10))
    y = so.source_filter_synthesize(track, envelope, aperiodicity, rng=1)
    inside = (track[0] > 0.1) & (track[0] < 0.4)
    rebuilt = so.pitch_adaptive_envelope(y, track)
    band = (rebuilt.f > 300) & (rebuilt.f < 6000)
    # the envelope reads the rebuilt sound at the level it was given, within
    # 1 dB (one noise draw, and the analysis's own error)
    assert abs(np.mean(rebuilt.db[0][band][:, inside])) < 1.0
    # harmonic_aperiodicity reads a known share within about 1.5 dB
    share = so.harmonic_aperiodicity(y, track).bands([300, 6000], envelope=rebuilt)[0, 0, inside]
    assert abs(10 * np.log10(np.mean(share)) - share_db) < 2.0


def test_unvoiced_time_windows_are_all_noise():
    track, envelope, aperiodicity = flat_views(0.0, 1.0)
    y = so.source_filter_synthesize(track, envelope, aperiodicity, rng=1)
    # white noise of variance 1, shaped by an envelope of 1, stays variance 1
    middle = y.data[int(0.05 * FS) : int(0.45 * FS), 0]
    assert abs(np.var(middle) - 1) < 0.1


def test_length_and_noise():
    track, envelope, aperiodicity = flat_views(150.0, 0.1)
    y = so.source_filter_synthesize(track, envelope, aperiodicity, rng=1)
    assert y.fs == FS
    assert y.n_samples == int(round((track[0][-1] + HOP) * FS))
    assert (
        so.source_filter_synthesize(track, envelope, aperiodicity, duration=0.3, rng=1).n_samples == 0.3 * FS
    )
    assert np.array_equal(y.data, so.source_filter_synthesize(track, envelope, aperiodicity, rng=1).data)
    assert not np.allclose(y.data, so.source_filter_synthesize(track, envelope, aperiodicity, rng=2).data)


def test_channels():
    track, envelope, aperiodicity = flat_views(150.0, 0.1, n_channels=2)
    # one F0 row and an envelope function giving one channel serve every channel
    y = so.source_filter_synthesize(track, lambda t, f: envelope(t, f)[:1], aperiodicity, rng=1)
    assert y.n_channels == 2
    # anything with .t and .f0 is a track
    track_object = SimpleNamespace(t=track[0], f0=np.vstack([track[1], 2 * track[1]]))
    assert so.source_filter_synthesize(track_object, envelope, aperiodicity, rng=1).n_channels == 2
    with pytest.raises(ValueError, match="channels"):
        so.source_filter_synthesize((track[0], np.vstack([track[1]] * 3)), envelope, aperiodicity)


def test_bad_inputs_are_refused():
    track, envelope, aperiodicity = flat_views(150.0, 0.1)
    for not_a_track in (track[1], 150.0):
        with pytest.raises(TypeError, match="f0 must be"):
            so.source_filter_synthesize(not_a_track, envelope, aperiodicity)
    with pytest.raises(TypeError, match="envelope"):
        so.source_filter_synthesize(track, envelope.data, aperiodicity)
    with pytest.raises(ValueError, match="gave shape"):
        so.source_filter_synthesize(track, lambda t, f: envelope(t, f)[0], aperiodicity)


def flat_power(f0):
    """The power the harmonics of a flat envelope of 1 ask for: ``a_n^2 / 2 =
    2 F0 / fs`` per harmonic, faded above 0.9 of 0.45 fs as harmonic_complex
    fades them."""
    f_max = 0.45 * FS
    harmonic_freqs = np.arange(1, int(f_max / f0) + 1) * f0
    position = (harmonic_freqs - 0.9 * f_max) / (0.1 * f_max)
    fade = np.cos(np.pi / 2 * np.clip(position, 0, 1)) ** 2
    return np.sum(2 * f0 / FS * fade**2)


def grid_envelope(power_over_time, duration=0.5):
    """An envelope flat in frequency whose power follows ``power_over_time(t)``."""
    times = np.arange(int(duration / HOP)) * HOP
    power = np.ones((1, 513, len(times))) * power_over_time(times)
    return times, so.SpectralEnvelope(power, times, np.arange(513) * FS / 1024, fs=FS)


def test_the_harmonic_level_is_the_formula():
    track, envelope, aperiodicity = flat_views(150.0, 0.0)
    y = so.source_filter_synthesize(track, envelope, aperiodicity, rng=1)
    assert abs(10 * np.log10(np.mean(y.data**2) / flat_power(150.0))) < 0.05
    # with a noise share, the harmonics carry S (1 - a) and the noise S a
    track, envelope, aperiodicity = flat_views(150.0, 0.1)
    y = so.source_filter_synthesize(track, envelope, aperiodicity, rng=1)
    assert abs(10 * np.log10(np.mean(y.data**2) / (0.9 * flat_power(150.0) + 0.1))) < 0.1
    # an envelope 20 dB lower in the second half: the mean of the two halves
    times, envelope = grid_envelope(lambda t: np.where(t < 0.25, 1.0, 0.01))
    aperiodicity = so.Aperiodicity(np.zeros(envelope.data.shape), times, FS, "truth")
    y = so.source_filter_synthesize((times, np.full(len(times), 150.0)), envelope, aperiodicity, rng=1)
    assert abs(10 * np.log10(np.mean(y.data**2) / (flat_power(150.0) * 0.505))) < 0.1


def test_harmonics_fade_over_harmonic_complexs_gate_at_a_voicing_edge():
    # all noise where unvoiced, as an aperiodicity measures it, but with the
    # envelope's power near 0 there, so that the harmonics are heard alone
    times, envelope = grid_envelope(lambda t: np.where(t < 0.2, 1.0, 1e-300), duration=0.4)
    f0 = np.where(times < 0.2, 150.0, 0.0)  # the last voiced time window at 0.195 s
    unvoiced = np.ones(envelope.data.shape) * (f0 == 0)
    aperiodicity = so.Aperiodicity(unvoiced, times, FS, "truth")
    y = so.source_filter_synthesize((times, f0), envelope, aperiodicity, rng=1).data[:, 0]

    def level_db(start):
        piece = y[int(round(start * FS)) : int(round((start + 0.0005) * FS))]
        return 10 * np.log10(np.mean(piece**2) + 1e-30)

    # held through the unvoiced windows, the harmonics fade over the 5 ms gate
    # centered between 0.195 and 0.2 s, not within half a millisecond
    assert level_db(0.1955) > -25
    assert -60 < level_db(0.1990) < -30
    assert level_db(0.2010) < -200


def test_without_an_aperiodicity_voicing_is_all_or_nothing():
    times, envelope = grid_envelope(lambda t: 1.0)
    f0 = np.where(times < 0.25, 160.0, 0.0)  # a period of 100 samples
    y = so.source_filter_synthesize((times, f0), envelope, rng=1).data[:, 0]
    voiced, unvoiced = y[int(0.05 * FS) : int(0.2 * FS)], y[int(0.3 * FS) : int(0.45 * FS)]

    def one_period_apart(x):
        return np.corrcoef(x[:-100], x[100:])[0, 1]

    assert one_period_apart(voiced) > 0.99
    assert abs(one_period_apart(unvoiced)) < 0.1
    # the unvoiced half is white noise at the envelope's level, variance 1
    assert abs(np.var(unvoiced) - 1) < 0.1


def test_a_time_window_halfway_between_two_track_times_takes_the_earlier():
    from sonore.sources.source_filter import _all_or_nothing

    share = _all_or_nothing(np.array([0.0, 0.01]), np.array([[150.0, 0.0]]), np.array([0.005]), 3)
    assert share.shape == (1, 3, 1)
    assert np.all(share == 0)
    share = _all_or_nothing(np.array([0.0, 0.01]), np.array([[0.0, 150.0]]), np.array([0.005]), 3)
    assert np.all(share == 1)


def test_a_sampling_rate_that_disagrees_is_refused():
    track, envelope, aperiodicity = flat_views(150.0, 0.1)
    with pytest.raises(ValueError, match="aperiodicity's"):
        so.source_filter_synthesize(track, envelope, aperiodicity, fs=2 * FS)
    with pytest.raises(ValueError, match="envelope's"):
        so.source_filter_synthesize(track, envelope, fs=2 * FS)
    no_fs = SimpleNamespace(t=aperiodicity.t, f=aperiodicity.f, data=aperiodicity.data)
    with pytest.raises(TypeError, match="no fs"):
        so.source_filter_synthesize(track, envelope, no_fs)
    with pytest.raises(TypeError, match="pass fs"):
        so.source_filter_synthesize(track, SimpleNamespace(t=envelope.t, f=envelope.f, data=envelope.data))


def test_the_hop_is_the_grids():
    times = np.arange(50) * 0.010
    envelope = so.SpectralEnvelope(np.ones((1, 513, 50)), times, np.arange(513) * FS / 1024, fs=FS)
    y = so.source_filter_synthesize((times, np.full(50, 150.0)), envelope, rng=1)
    assert y.n_samples == int(round((times[-1] + 0.010) * FS))


def test_a_longer_noise_window_spreads_a_step_in_level_further():
    times, envelope = grid_envelope(lambda t: np.where(t < 0.25, 1.0, 0.01))
    unvoiced = (times, np.zeros(len(times)))

    def just_after_the_step_db(win_dur):
        y = so.source_filter_synthesize(unvoiced, envelope, rng=1, win_dur=win_dur).data[:, 0]
        return 10 * np.log10(np.mean(y[int(0.25 * FS) : int(0.26 * FS)] ** 2))

    # measured: -20.2 dB with a 10 ms window, -9.2 dB with an 80 ms one
    assert just_after_the_step_db(0.080) > just_after_the_step_db(0.010) + 6
