"""Copies of one event scattered in time (docs/design/sources/event-texture.md)."""

import numpy as np
import pytest
from scipy import stats

import sonore as so
from sonore.sources.events import _transposed as transposed

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


def test_transposed_copy_is_higher_and_shorter():
    event = so.Sound(np.hanning(202)[1:-1], FS)  # 200 nonzero samples
    octave_up = _onsets(so.event_train(event, 1.0, 0.5, pitch_cents=1200))
    assert octave_up[0] == 500
    assert octave_up[-1] - octave_up[0] + 1 == pytest.approx(100, abs=3)


@pytest.mark.parametrize("cents", [-400, -7, 100, 400])
def test_transposed_copies_match_resampling(cents):
    """The shared windowed sinc and SciPy's polyphase resampling agree, below the top
    of the spectrum; a noise burst also checks that going up nothing folds back
    (docs/design/sources/event-train-speed.md)."""
    fs = 44100
    ratio = 2 ** (cents / 1200)
    noise = so.gaussian_noise(0.03, fs, rng=0)
    for event, tolerance in ((_chirp(fs), 1e-3), (noise * np.hanning(noise.n_samples)[:, None], 0.03)):
        copy = transposed(event, ratio, fs)[:, 0]
        resampled = event.resample(fs / ratio).data[:, 0]
        n = min(len(copy), len(resampled))
        assert abs(len(copy) - len(resampled)) <= 1
        assert so.rms(copy[:n] - resampled[:n]) < tolerance * so.rms(resampled)


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


def test_each_copy_carries_its_own_pitch_ratio():
    # a one-sample event whose value is the pitch ratio it was asked for: every copy shows
    # its ratio, a whole number of cents, spread as the cents were drawn (uniform 0-1200)
    sound = so.event_train(
        lambda ratio: so.Sound([ratio], FS), 20.0, 0.01, pitch_cents=stats.uniform(0, 1200), rng=0
    )
    cents = 1200 * np.log2(sound.data[_onsets(sound), 0])
    np.testing.assert_allclose(cents, np.round(cents), atol=1e-9)
    assert cents.min() >= 0 and cents.max() <= 1200
    assert np.std(cents) == pytest.approx(1200 / np.sqrt(12), rel=0.05)


def test_pitch_is_rounded_to_the_nearest_cent():
    chirp = _chirp(16000)
    exact = so.event_train(chirp, 0.3, 0.05, pitch_cents=100)
    np.testing.assert_array_equal(so.event_train(chirp, 0.3, 0.05, pitch_cents=99.6).data, exact.data)


def test_lead_works_with_a_pitch_spread():
    # the copies added one by one start before time 0 too
    event = so.Sound(np.ones(300), FS)
    sound = so.event_train(lambda ratio: event, 1.0, 0.25, lead=0.3, pitch_cents=stats.norm(0, 100), rng=0)
    assert sound.data[0, 0] == pytest.approx(1)
    assert sound.duration == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", [2 ** (100 / 1200), 1.5, 2.0, 0.5, 2 ** (-7 / 1200)])
def test_transposed_copy_lasts_the_event_over_the_ratio(ratio):
    event = so.Sound(np.hanning(301), FS)
    assert len(transposed(event, ratio, FS)) == int(np.ceil(301 / ratio))


class _NotANumber:
    def rvs(self, size=None, random_state=None):
        return np.full(size, np.nan)


def test_rejects_more_bad_arguments():
    with pytest.raises(ValueError, match="duration must be positive"):
        so.event_train(CLICK, 0.0, 0.1)
    with pytest.raises(ValueError, match="lead not negative"):
        so.event_train(CLICK, 1.0, 0.1, lead=-0.1)
    with pytest.raises(ValueError, match="finite"):
        so.event_train(CLICK, 1.0, _NotANumber())
    with pytest.raises(ValueError, match="one sampling rate"):
        so.event_train(
            lambda ratio: so.Sound([1.0], FS if ratio == 1 else 2 * FS), 1.0, 0.1, pitch_cents=stats.norm(0, 50), rng=0
        )
