"""Sound quilts (Overath, McDermott, Zarate & Poeppel, 2015): a sound cut
into segments of one length and strung together in a new order, so that it
keeps its structure within a segment and across each join, and loses it
beyond the segment length.

Written from the paper's Online Methods; McDermott's Sound Quilting Toolbox
(MATLAB) was consulted for behavior only. Where the two differ, this follows
the paper (docs/design/views/quilting.md lists the differences).
"""

from __future__ import annotations

import math

import numpy as np

from sonore.core.processing import splice
from sonore.core.sound import Sound
from sonore.core.utils import RNG, as_rng
from sonore.frames.filterbank import Filterbank, cosine_filterbank

__all__ = ["quilt"]


def quilt(
    sound: Sound,
    segment_dur: float,
    duration: float,
    *,
    rng: RNG = None,
    filterbank: Filterbank | None = None,
    border_dur: float = 0.03,
    crossfade: float = 0.03,
    max_shift: float = 0.015,
) -> Sound:
    """A sound quilt of ``sound``, ``duration`` [s] long, made of segments
    ``segment_dur`` [s] long (Overath et al., 2015).

    The sound is cut into whole segments, and the quilt uses each one at
    most once. The first segment is drawn at random from ``rng``. After
    segment ``k``, the next is the one whose join with ``k`` changes the
    cochleagram by as much as the sound itself changed after ``k``: the
    ``n`` that minimizes ``|d(k, n) - d(k, k + 1)|``, where ``d`` is the
    mean squared difference between the last ``border_dur`` of one segment
    and the first ``border_dur`` of another. The sound's own next segment
    ``k + 1`` is never chosen, so every join is a new one, and its last
    segment, which has no change of its own to match, may only end the
    quilt.

    The cochleagram is the Hilbert envelope of each band of ``filterbank``,
    raised to the power 0.3 (default: the paper's 30 ERB-spaced cosine
    filters from 20 Hz to 10 kHz, capped at Nyquist, with their edge
    filters). The segments are joined by :func:`splice`, with raised-cosine
    cross-fades ``crossfade`` long, each new segment moved by up to
    ``max_shift`` to align its waveform with what precedes it.

    The quilt keeps the sound's level and is trimmed to ``duration``; the
    paper's band-limiting, pause removal and final fade are left to the
    caller. The same ``rng`` seed and sound give the same quilt. Raises if
    the sound has too few segments for the quilt. Mono only.
    """
    if sound.n_channels != 1:
        raise ValueError(f"quilt needs a mono sound, not {sound.n_channels} channels")
    fs = sound.fs
    n_segment = int(round(segment_dur * fs))
    n_border = int(round(border_dur * fs))
    if n_segment < 1:
        raise ValueError("segment_dur is shorter than one sample")
    if not 1 <= n_border <= n_segment:
        raise ValueError("border_dur must be at least one sample and at most segment_dur")
    n_segments = len(sound) // n_segment
    n_out = int(round(duration * fs))
    n_needed = math.ceil(n_out / n_segment)
    if n_out < 1:
        raise ValueError("duration is shorter than one sample")
    if n_needed > n_segments:
        raise ValueError(
            f"a {duration:g} s quilt of {segment_dur:g} s segments needs {n_needed} segments, "
            f"but the sound ({len(sound) / fs:g} s) has {n_segments}"
        )

    bank = filterbank if filterbank is not None else cosine_filterbank(30, 20, min(10000, fs / 2))
    cochleagram = bank.analyze(sound).envelopes().data[: n_segments * n_segment, :, 0] ** 0.3
    by_segment = cochleagram.reshape(n_segments, n_segment, -1)
    heads = by_segment[:, :n_border].reshape(n_segments, -1)
    tails = by_segment[:, -n_border:].reshape(n_segments, -1)
    # change[k, n]: from the end of segment k into the start of segment n,
    # for every pair at once from one matrix product
    change = (
        np.sum(tails**2, axis=1)[:, None] + np.sum(heads**2, axis=1)[None, :] - 2 * tails @ heads.T
    ) / heads.shape[1]

    generator = as_rng(rng)
    last = n_segments - 1
    current = int(generator.integers(n_segments if n_needed == 1 else last))
    order = [current]
    used = np.zeros(n_segments, dtype=bool)
    used[current] = True
    for position in range(1, n_needed):
        mismatch = np.abs(change[current] - change[current, current + 1])
        blocked = used.copy()
        blocked[current + 1] = True
        if position < n_needed - 1:
            blocked[last] = True
        if blocked.all():
            raise ValueError(
                f"the sound's {n_segments} segments ran out after {position} of the {n_needed} "
                "the quilt needs; use a longer sound or a shorter quilt"
            )
        mismatch[blocked] = np.inf
        current = int(np.argmin(mismatch))
        order.append(current)
        used[current] = True

    starts = np.array(order) * n_segment / fs
    joined = splice(sound, starts, n_segment / fs, crossfade=crossfade, max_shift=max_shift)
    return Sound(joined.data[:n_out], fs)
