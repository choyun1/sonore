"""Copies of one event scattered in time (docs/design/sources/event-texture.md)."""

import numpy as np
import pytest
from scipy import stats

import sonore as so

FS = 1000
CLICK = so.Sound([1.0], FS)  # a one-sample event: the output is the impulse train itself


def _onsets(sound: so.Sound) -> np.ndarray:
    return np.flatnonzero(np.abs(sound.data[:, 0]) > 1e-9)


def _counts(sound: so.Sound) -> float:
    return float(np.round(sound.data[:, 0].sum()))


def _chirp(fs: float) -> so.Sound:
    n = int(0.030 * fs)
    return so.exponential_chirp(n / fs, fs, 2000, 6000) * np.hanning(n)[:, None]


def test_poisson_counts_have_poisson_mean_and_variance():
    counts = [_counts(so.event_train(CLICK, 2.0, stats.expon(scale=1 / 20), rng=seed)) for seed in range(400)]
    assert np.mean(counts) == pytest.approx(40, rel=0.03)
    assert np.var(counts) / np.mean(counts) == pytest.approx(1, abs=0.15)


def test_constant_interval_is_regular():
    sound = so.event_train(CLICK, 1.0, 0.1)
    np.testing.assert_array_equal(_onsets(sound), np.arange(100, 1000, 100))
    assert sound.duration == pytest.approx(1.0)


def test_reproducible_from_rng():
    kwargs = dict(level_db=stats.norm(0, 3), pitch_cents=stats.norm(0, 50), rng=7)
    first = so.event_train(_chirp(16000), 0.5, stats.expon(scale=0.02), **kwargs)
    second = so.event_train(_chirp(16000), 0.5, stats.expon(scale=0.02), **kwargs)
    np.testing.assert_array_equal(first.data, second.data)


def test_rate_track_follows_its_integral():
    # rate 10/s rising linearly to 50/s over 4 s: 40 events expected in the first half, 80 in the second
    halves = []
    for seed in range(300):
        sound = so.event_train(CLICK, 4.0, stats.expon(scale=0.1), rate_scale=([0, 4], [1, 5]), rng=seed)
        halves.append([sound.data[:2000, 0].sum(), sound.data[2000:, 0].sum()])
    np.testing.assert_allclose(np.mean(halves, axis=0), [40, 80], rtol=0.04)


def test_shot_noise_kurtosis_and_level_spread():
    # Campbell's theorem: excess kurtosis integral(h^4) / (rate integral(h^2)^2), times exp(4 s^2)
    # for levels normal in dB with s = sd ln(10) / 20
    fs, rate = 16000, 100
    chirp = _chirp(fs)
    h = chirp.data[:, 0]
    predicted = np.sum(h**4) * fs / (rate * np.sum(h**2) ** 2)
    for sd_db, seeds in ((0.0, range(4)), (3.0, range(8))):
        values = [
            stats.kurtosis(
                so.event_train(
                    chirp,
                    20.0,
                    stats.expon(scale=1 / rate),
                    level_db=stats.norm(0, sd_db) if sd_db else 0.0,
                    rng=seed,
                ).data[:, 0]
            )
            for seed in seeds
        ]
        factor = np.exp(4 * (sd_db * np.log(10) / 20) ** 2)
        assert np.mean(values) == pytest.approx(factor * predicted, rel=0.15)


def test_convolution_and_one_by_one_agree():
    # an event function that ignores the pitch ratio forces the one-by-one path with the same waveform
    chirp = _chirp(16000)
    by_convolution = so.event_train(chirp, 0.5, stats.expon(scale=0.01), level_db=stats.norm(0, 3), rng=3)
    one_by_one = so.event_train(
        lambda ratio: chirp,
        0.5,
        stats.expon(scale=0.01),
        level_db=stats.norm(0, 3),
        pitch_cents=stats.norm(0, 100),
        rng=3,
    )
    np.testing.assert_allclose(one_by_one.data, by_convolution.data, atol=1e-9)


def test_resampled_copy_is_higher_and_shorter():
    event = so.Sound(np.hanning(202)[1:-1], FS)  # 200 nonzero samples
    octave_up = _onsets(so.event_train(event, 1.0, 0.5, pitch_cents=1200))
    assert octave_up[0] == 500
    assert octave_up[-1] - octave_up[0] + 1 == pytest.approx(100, abs=3)


def test_lead_brings_in_events_already_under_way():
    event = so.Sound(np.ones(300), FS)
    assert abs(so.event_train(event, 1.0, 0.25).data[0, 0]) < 1e-9  # first onset at 0.25 s
    with_lead = so.event_train(event, 1.0, 0.25, lead=0.3)  # onsets at -0.05, 0.2, 0.45 s, ...
    assert with_lead.data[0, 0] == pytest.approx(1)
    assert with_lead.duration == pytest.approx(1.0)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="not be negative"):
        so.event_train(CLICK, 1.0, 0.1, rate_scale=([0, 1], [1, -1]))
    with pytest.raises(TypeError, match="rvs"):
        so.event_train(CLICK, 1.0, "poisson")
    with pytest.raises(ValueError, match="positive"):
        so.event_train(CLICK, 1.0, 0.0)
    with pytest.raises(ValueError, match="not negative"):
        so.event_train(CLICK, 1.0, stats.norm(0, 1), rng=0)
