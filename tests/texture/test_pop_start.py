import numpy as np
import pytest
from scipy import stats

import sonore as so
from sonore.texture.synth import POP_START_CAP, pop_rate, pop_start

FS = 20000


@pytest.fixture(scope="module")
def crackle():
    """Shot noise with power-law sizes: 30 bursts a second over quiet noise, 8 s."""
    burst = so.gaussian_noise(0.003, FS, rng=0)
    burst = burst * np.hanning(len(burst))[:, None]
    levels = stats.truncexpon(b=40 / (20 / (1.5 * np.log(10))), scale=20 / (1.5 * np.log(10)))
    pops = so.event_train(burst, 8.0, stats.expon(scale=1 / 30), level_db=levels, lead=0.01, rng=0)
    return pops.normalize(rms=1.0) + so.gaussian_noise(pops.duration, FS, rng=1) * 0.02


def test_pop_rate_counts_crossings_once(crackle):
    """A pop that spans two 5 ms bins is counted once, and quiet noise has no pops."""
    rate = pop_rate(crackle)
    assert 10 < rate < 40  # 30 a second asked for; overlaps and bursts under 10 dB lose some
    assert pop_rate(so.gaussian_noise(5.0, FS, rng=2)) < 2


def test_pop_start_shape_and_defaults(crackle):
    start = pop_start(crackle, 3.0, rng=0)
    assert start.fs == crackle.fs and start.duration == pytest.approx(3.0)
    assert start.n_channels == 1
    assert np.array_equal(start.data, pop_start(crackle, 3.0, rng=0).data)
    assert not np.array_equal(start.data, pop_start(crackle, 3.0, rng=1).data)
    assert pop_start(crackle, 3.0, fs=16000, rng=0).fs == 16000


def test_pop_start_is_noise_times_sparse_bumps(crackle):
    """Dividing out the noise (the generator's first draw) leaves the gain: 1 between bumps,
    at most the cap, about ``rate * duration`` bumps, each at most 4 ms long."""
    duration, rate = 10.0, 20.0
    start = pop_start(crackle, duration, rate=rate, exponent=1.5, rng=0)
    noise = np.random.default_rng(0).standard_normal(len(start))
    gain = start.data[:, 0] / noise
    assert gain.min() >= 1 - 1e-9 and gain.max() <= POP_START_CAP + 1e-9
    raised = gain > 1 + 1e-9
    starts = np.flatnonzero(raised[1:] & ~raised[:-1]) + 1
    assert 0.8 * rate * duration < len(starts) <= rate * duration  # overlaps merge a few
    ends = np.flatnonzero(~raised[1:] & raised[:-1]) + 1
    assert np.max(ends[: len(starts)] - starts[: len(ends)]) <= 2 * 0.004 * FS  # two can touch
    assert np.mean(raised) < 0.1  # sparse


def test_pop_start_rejects_bad_options(crackle):
    with pytest.raises(ValueError):
        pop_start(crackle, 1.0, rate=0.0)
    with pytest.raises(ValueError):
        pop_start(crackle, 1.0, exponent=-1.0)
