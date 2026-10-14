"""Copies of one event scattered in time, from single events to a texture.

One short sound, the event, is copied many times. The time between onsets,
each copy's level and each copy's pitch are drawn from distributions the
caller gives, and a rate track can speed the onsets up or slow them down
over time. At a low rate the events stand apart; at a high one they
overlap many deep and the sum approaches Gaussian noise with the event's
spectrum (docs/design/sources/event-texture.md).
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Protocol

import numpy as np

from sonore.core.processing import Track, _track
from sonore.core.sound import Sound
from sonore.core.utils import RNG, _read_between_samples, as_rng, db_to_amp, n_samples, time_axis

__all__ = ["event_train"]


class Distribution(Protocol):
    """Anything with SciPy's ``rvs``, such as a frozen ``scipy.stats`` distribution."""

    def rvs(self, size=None, random_state=None): ...


#: A number every event shares, or a distribution each event draws from.
Draw = float | Distribution


def _draw(value: Draw, size: int, rng: np.random.Generator, name: str) -> np.ndarray:
    if isinstance(value, numbers.Real):
        return np.full(size, float(value))
    rvs = getattr(value, "rvs", None)
    if rvs is None:
        raise TypeError(f"{name} must be a number or a distribution with an rvs method, not {value!r}")
    return np.asarray(rvs(size=size, random_state=rng), dtype=float).reshape(size)


def _warped_onsets(intervals: Draw, span: float, rng: np.random.Generator) -> np.ndarray:
    """Onsets on the warped clock, from 0 up to ``span``: running sums of intervals,
    drawn in growing batches until they pass the end."""
    if isinstance(intervals, numbers.Real) and intervals <= 0:
        raise ValueError("a constant interval must be positive")
    batches, last, batch = [], 0.0, 64
    while last < span:
        drawn = _draw(intervals, batch, rng, "intervals")
        if np.any(drawn < 0) or not np.all(np.isfinite(drawn)):
            raise ValueError("intervals must be finite and not negative")
        times = last + np.cumsum(drawn)
        batches.append(times)
        last, batch = times[-1], 2 * batch
    onsets = np.concatenate(batches) if batches else np.zeros(0)
    return onsets[onsets < span]


def _transposed(event: Sound | Callable[[float], Sound], ratio: float, fs: float) -> np.ndarray:
    """The event ``ratio`` times higher, as (samples, channels). A sound is read every
    ``ratio`` samples, so it is also ``ratio`` times shorter, as a tape played faster
    is; a function of the ratio makes its own transposed copy."""
    if callable(event):
        copy = event(ratio)
        if copy.fs != fs:
            raise ValueError("the event function must return sounds at one sampling rate")
        return copy.data
    if ratio == 1:
        return event.data
    positions = np.arange(int(np.ceil(event.n_samples / ratio))) * ratio
    return np.stack(
        [_read_between_samples(channel, positions, stretch=max(ratio, 1.0)) for channel in event.data.T],
        axis=1,
    )


def event_train(
    event: Sound | Callable[[float], Sound],
    duration: float,
    intervals: Draw,
    *,
    rate_scale: Track = 1.0,
    level_db: Draw = 0.0,
    pitch_cents: Draw = 0.0,
    lead: float = 0.0,
    rng: RNG = None,
) -> Sound:
    """Copies of ``event`` at random times, levels and pitches.

    The intervals between onsets are drawn independently from ``intervals``
    (seconds), so ``scipy.stats.expon(scale=1/30)`` gives Poisson timing at
    30 events per second, the number ``1/30`` gives regular timing at the
    same rate, and ``scipy.stats.gamma(16, scale=1/(30*16))`` lies between.
    Each copy's level (dB, relative to the event as given) and pitch shift
    (cents) are drawn from ``level_db`` and ``pitch_cents``. Each of the
    three takes a number, which every event shares, or anything with SciPy's
    ``rvs(size=, random_state=)``.

    ``rate_scale`` (a number, or ``(times, values)`` interpolated linearly)
    multiplies the rate over time by warping time: onsets are drawn on a
    clock that runs ``rate_scale`` times as fast as real time, so where it
    is 2 events come twice as often. With exponential intervals this is a
    Poisson process with that rate at each moment. It must not be negative.

    With every pitch the same, the result is the event convolved with a
    train of impulses, one per onset with the copy's amplitude (shot noise,
    for Poisson timing); otherwise the copies are added one by one. Pitch
    shifts are rounded to the nearest cent, and each distinct one is made
    once: a ``Sound`` event is read every ``ratio`` samples with the
    windowed sinc that :func:`move_sound` reads between samples with,
    widened by the ratio going up so nothing folds back (so a higher copy
    is also shorter), while an event given as a function ``event(ratio) -> Sound`` makes its
    own copy at that pitch ratio, so a synthetic event can keep its length.

    Onsets are also drawn over ``lead`` seconds before the start, so that
    with ``lead`` at the event's length an event already under way at time
    0 is heard; the result is ``duration`` long and copies running past the
    end are cut. The overall level is left to the caller: the expected power
    grows in proportion to the rate.

    docs/design/sources/event-texture.md measures what each distribution
    does to the sound's statistics (``tools/check_event_texture_claims.py``).
    """
    if duration <= 0 or lead < 0:
        raise ValueError("duration must be positive and lead not negative")
    rng = as_rng(rng)
    fs = event(1.0).fs if callable(event) else event.fs
    n_out, n_lead = n_samples(duration, fs), n_samples(lead, fs)

    # The warped clock: Lambda(t), the integral of rate_scale from -lead, on the sample grid.
    t = time_axis(n_lead + n_out + 1, fs) - n_lead / fs
    scale = np.broadcast_to(_track(rate_scale, t, "rate_scale"), t.shape)
    if np.any(scale < 0):
        raise ValueError("rate_scale must not be negative")
    warped = np.concatenate([[0.0], np.cumsum((scale[1:] + scale[:-1]) / 2) / fs])
    onsets = np.interp(_warped_onsets(intervals, warped[-1], rng), warped, t)
    n_total = n_lead + n_out
    onset_samples = np.round(onsets * fs).astype(int) + n_lead
    onset_samples = onset_samples[onset_samples < n_total]

    amplitudes = db_to_amp(_draw(level_db, len(onset_samples), rng, "level_db"))
    cents = np.round(_draw(pitch_cents, len(onset_samples), rng, "pitch_cents")).astype(int)
    distinct = np.unique(cents)

    if len(distinct) <= 1:
        impulses = np.zeros(n_total)
        np.add.at(impulses, onset_samples, amplitudes)
        shape = Sound(_transposed(event, 2 ** (distinct[0] / 1200) if len(distinct) else 1.0, fs), fs)
        return Sound(Sound(impulses, fs).convolve(shape).data[n_lead:n_total], fs)

    copies = {c: _transposed(event, 2 ** (c / 1200), fs) for c in distinct}
    n_channels = copies[distinct[0]].shape[1]
    total = np.zeros((n_total, n_channels))
    for start, amplitude, c in zip(onset_samples, amplitudes, cents, strict=True):
        waveform = copies[c][: len(total) - start]
        total[start : start + len(waveform)] += amplitude * waveform
    return Sound(total[n_lead:], fs)
