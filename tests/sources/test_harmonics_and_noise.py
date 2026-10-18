"""so.harmonics_and_noise given known views: the level and noise share it
rebuilds, its channels and the inputs it refuses. The analyses' own errors
are checked in tests/views/test_world.py, and the whole chain on recordings
in tools/check_voice_recipe_claims.py."""

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
    envelope = so.SpectralEnvelope(power, times, FS, np.nan)
    aperiodicity = so.Aperiodicity(np.full_like(power, np.sqrt(share)), times, FS, "truth")
    return (times, np.full(len(times), f0)), envelope, aperiodicity


@pytest.mark.parametrize("share_db", [-20.0, -6.0])
def test_it_rebuilds_a_known_level_and_noise_share(share_db):
    track, envelope, aperiodicity = flat_views(150.0, 10 ** (share_db / 10))
    y = so.harmonics_and_noise(track, envelope, aperiodicity, rng=1)
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
    y = so.harmonics_and_noise(track, envelope, aperiodicity, rng=1)
    # white noise of variance 1, shaped by an envelope of 1, stays variance 1
    middle = y.data[int(0.05 * FS) : int(0.45 * FS), 0]
    assert abs(np.var(middle) - 1) < 0.1


def test_length_and_noise():
    track, envelope, aperiodicity = flat_views(150.0, 0.1)
    y = so.harmonics_and_noise(track, envelope, aperiodicity, rng=1)
    assert y.fs == FS
    assert y.n_samples == int(round((track[0][-1] + HOP) * FS))
    assert so.harmonics_and_noise(track, envelope, aperiodicity, duration=0.3, rng=1).n_samples == 0.3 * FS
    assert np.array_equal(y.data, so.harmonics_and_noise(track, envelope, aperiodicity, rng=1).data)
    assert not np.allclose(y.data, so.harmonics_and_noise(track, envelope, aperiodicity, rng=2).data)


def test_channels():
    track, envelope, aperiodicity = flat_views(150.0, 0.1, n_channels=2)
    # one F0 row and an envelope function giving one channel serve every channel
    y = so.harmonics_and_noise(track, lambda t, f: envelope(t, f)[:1], aperiodicity, rng=1)
    assert y.n_channels == 2
    # anything with .t and .f0 is a track
    track_object = SimpleNamespace(t=track[0], f0=np.vstack([track[1], 2 * track[1]]))
    assert so.harmonics_and_noise(track_object, envelope, aperiodicity, rng=1).n_channels == 2
    with pytest.raises(ValueError, match="channels"):
        so.harmonics_and_noise((track[0], np.vstack([track[1]] * 3)), envelope, aperiodicity)


def test_bad_inputs_are_refused():
    track, envelope, aperiodicity = flat_views(150.0, 0.1)
    for not_a_track in (track[1], 150.0):
        with pytest.raises(TypeError, match="f0 must be"):
            so.harmonics_and_noise(not_a_track, envelope, aperiodicity)
    with pytest.raises(TypeError, match="envelope"):
        so.harmonics_and_noise(track, envelope.data, aperiodicity)
    with pytest.raises(ValueError, match="gave shape"):
        so.harmonics_and_noise(track, lambda t, f: envelope(t, f)[0], aperiodicity)
