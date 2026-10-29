"""F0 tracking: candidates from a difference function, refinement by the
instantaneous frequency of harmonics, a periodicity score, and a Viterbi pass
that decides voicing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from sonore.core.sound import Sound
from sonore.core.utils import _parabola_vertex
from sonore.views.view import View

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from sonore.sources.waveforms import ContourLike

__all__ = ["F0Track", "F0TrackStream", "F0State", "f0_track", "scale_f0"]


MAX_CANDIDATES = 8


N_HARMONICS = 6  # the refinement uses at least this many harmonics ...


HARMONIC_CEILING = 1500.0  # ... and every harmonic up to here [Hz] ...


MAX_HARMONICS = 24  # ... but no more than this many


PERIODS = 3.0  # window length for refinement, in periods of the candidate


SCORE_PERIODS = 2.0  # stretch length for the score, in periods of the candidate ...


SCORE_MIN_WINDOW = 1 / 60.0  # ... and at least this long [s]: one period of the default f_lo


NEAR_DEEPEST = 0.1  # a dip this close to the deepest one counts as nearly as deep


SINC_HALF = 32  # half-length of the fractional-delay kernel [samples]


SINC_TAPER = np.kaiser(2 * SINC_HALF, 8.0)  # the kernel's taper
SINC_TAPER.flags.writeable = False


CHUNK = 512  # candidates processed at once


ROW_STEP = 32  # rows of samples are padded to a multiple of this many, set by the candidate alone


@dataclass(frozen=True)
class F0Track(View):
    """An F0 track: one estimate per channel every ``hop`` seconds. A view:
    it keeps only the pitch and the voicing.

    ``f0`` and ``score`` have shape ``(n_channels, n_windows)``; ``f0`` is 0
    where the time window is unvoiced, and ``score`` is the periodicity score of
    the chosen candidate (0 where unvoiced). ``candidates`` and
    ``candidate_scores`` have shape ``(n_channels, n_windows, 8)`` and hold
    every refined candidate the tracker weighed, NaN where a time window had fewer.
    ``t`` and a row of ``f0`` go straight into
    :meth:`~sonore.frames.gabor.TVGaborFrame.pitch_adaptive` and
    :meth:`~sonore.views.cepstrum.Cepstrum.lifter`.
    """

    discards = "F0Track keeps only the pitch and the voicing of each time window."
    back_to_sound = (
        "so.source_filter_synthesize rebuilds an approximation of the voice from it together with a spectral "
        "envelope and an aperiodicity."
    )

    t: np.ndarray
    f0: np.ndarray
    score: np.ndarray
    candidates: np.ndarray
    candidate_scores: np.ndarray
    fs: float
    threshold: float
    first_window: int = 0  # a part of a stream starts later

    @classmethod
    def concatenate(cls, parts: Sequence[F0Track]) -> F0Track:
        """Join the parts of a stream (:class:`F0TrackStream`) in order:
        exactly ``f0_track(sound, delay=...)`` when they cover the whole sound."""
        parts = list(parts)
        if not parts:
            raise ValueError("no parts to join")
        first = parts[0]
        position = first.first_window
        for part in parts:
            if not isinstance(part, F0Track):
                raise TypeError(f"concatenate joins F0Tracks, not {type(part).__name__}")
            if (part.fs, part.threshold) != (first.fs, first.threshold) or part.f0.shape[0] != first.f0.shape[
                0
            ]:
                raise ValueError("the parts come from different tracks")
            if part.first_window != position:
                raise ValueError(
                    f"parts are not consecutive: time window {part.first_window} after {position}"
                )
            position += part.f0.shape[1]
        return replace(
            first,
            t=np.concatenate([part.t for part in parts]),
            f0=np.concatenate([part.f0 for part in parts], axis=1),
            score=np.concatenate([part.score for part in parts], axis=1),
            candidates=np.concatenate([part.candidates for part in parts], axis=1),
            candidate_scores=np.concatenate([part.candidate_scores for part in parts], axis=1),
        )

    @property
    def voiced(self) -> np.ndarray:
        """True where the time window is voiced, shape ``(n_channels, n_windows)``."""
        return self.f0 > 0

    def __repr__(self) -> str:
        n_channels, n_windows = self.f0.shape
        voiced = self.voiced
        share = f"{voiced.mean():.0%}" if voiced.size else "0%"
        f0_range = f", F0 {self.f0[voiced].min():.0f}-{self.f0[voiced].max():.0f} Hz" if voiced.any() else ""
        return f"F0Track({n_windows} time windows, {n_channels} ch, {share} voiced{f0_range})"

    def plot(self, ax: Axes | None = None, *, channel: int = 0, candidates: bool = False, **kwargs) -> Axes:
        """F0 against time (see :func:`~sonore.plotting.plot_f0_track`)."""
        from sonore.plotting import plot_f0_track

        return plot_f0_track(self, ax=ax, channel=channel, candidates=candidates, **kwargs)


def f0_track(
    sound: Sound,
    *,
    f_lo: float = 60.0,
    f_hi: float = 500.0,
    hop: float = 0.005,
    threshold: float = 0.5,
    octave_cost: float = 2.0,
    switch_cost: float = 0.5,
    subharmonic_margin: float | None = 0.05,
    delay: float | None = None,
) -> F0Track:
    """Track the F0 of each channel of a sound.

    Four stages, every ``hop`` seconds from time 0:

    1. **Candidates.** Up to eight local minima of the cumulative-mean-normalized
       difference function (de Cheveigné & Kawahara, 2002) over a 25 ms window,
       at lags between ``1/f_hi`` and ``1/f_lo``, each refined by a parabola.
       This is the autocorrelation idea with YIN's corrections: it finds the
       period from all the harmonics, so it works when the fundamental is weak
       or filtered out. Every minimum below 1 is offered, so that the score
       alone decides voicing. A very regular voice has a minimum at every
       multiple of its period, so eight leave room for the period itself
       (with four, a steady 300 Hz vowel could lose it), and, as in YIN,
       the shortest-lag minimum within 0.1 of the deepest is always among
       them (with ``f_hi`` raised, a sung 800 Hz vowel has more multiples in
       range than there are candidates).
    2. **Refinement.** For a candidate f, a Blackman window three periods long;
       the instantaneous frequency of each harmonic is its phase advance over
       one sample, and the new F0 is the power-weighted mean of each
       harmonic's frequency divided by its number, over the harmonics up to
       1.5 kHz (at least six, at most 24). Done twice. This is the idea of
       WORLD's StoneMask, written from its description; StoneMask uses the
       first six harmonics, which below 100 Hz hold a few percent of a
       vowel's power and in noise put the refined F0 a percent off, enough
       for the score of a low voice to collapse.
    3. **Score.** The normalized correlation between a stretch two periods
       long, and at least 17 ms (one period of 60 Hz), and the same stretch
       one period later, the fractional part of the period done with a
       windowed sinc. For a periodic sound in independent noise it is about
       ``s / (1 + s)``, ``s`` the ratio of periodic to noise power, so 0.5
       means the two are equally strong. Two periods rather than three so
       that a low voice whose F0 moves (a vibrato at 70 Hz) still lines up
       with its copy one period later; the floor so that a short-period
       candidate is judged across a whole glottal cycle of the lowest voice
       searched, where a formant's ringing no longer passes for a period.
    4. **Tracking.** A Viterbi pass over (unvoiced, candidates) per time window. A
       candidate costs ``1 - score`` and the unvoiced state ``1 - threshold``,
       so a time window leans voiced when its best score exceeds ``threshold``;
       moving between candidates costs ``octave_cost`` per octave, and a
       voicing change costs ``switch_cost``. A candidate is never chosen if
       another one at a whole multiple of its frequency (an octave, a
       twelfth, ...) scores at least as well, less ``subharmonic_margin``:
       anything periodic with period T is also periodic with period 2T, 3T,
       ..., and without this rule a female voice is tracked low on about
       1.4% of the time windows where it and the reference both say voiced
       (measured on the FDA database), and a steady synthetic vowel at
       300 Hz at a half or a third of its F0. No smoothing follows.

    Where each stage comes from: the candidates are YIN's (de Cheveigné &
    Kawahara, 2002); the refinement is WORLD's StoneMask (Morise et al.,
    2009); the score is the normalized correlation one period apart, as in
    RAPT (Talkin, 1995) and Praat (Boersma, 1993); the tracking is the
    dynamic programming of RAPT and Praat, with an unvoiced state and costs
    for octave jumps and voicing changes. The rule against whole multiples
    is sonore's own. The combination is no single published tracker; pYIN
    (Mauch & Dixon, 2014), YIN candidates followed by a Viterbi pass over
    pitch and voicing, is the closest. Design and measurements in
    ``docs/design/views/f0.md``.

    On synthetic vowels the refined F0 is within 0.1% of the truth on a one
    octave per second glide and a 5.5 Hz vibrato. Against laryngograph
    reference F0 (a male and a female speaker), it gets the voicing of 4.0%
    and 1.6% of time windows wrong, where WORLD's Harvest, which leans toward
    calling time windows voiced, gets about 21% wrong; where both it and the
    reference say voiced, 98% and 95% of time windows are within 5%. Lower ``threshold`` to voice
    more weak or creaky stretches, for example before resynthesis.

    Parameters
    ----------
    sound
        The sound; each channel is tracked separately.
    f_lo, f_hi
        The search range [Hz]. The defaults cover speaking voices; raise
        ``f_hi`` for singing (on speech that costs about 0.3 points of
        voicing error, mostly false voicings). Refined candidates are kept
        from 0.9 ``f_lo`` to 1.1 ``f_hi``, so the F0 can come out up to 10%
        outside the range. The lowest voices are the least reliable: below
        about 70 Hz a vowel whose F0 moves by more than a few percent within
        the score's stretch (a wide vibrato) can come out unvoiced or near a
        formant's frequency, and lowering ``f_lo`` below the default voices
        period-doubled stretches, which the default leaves unvoiced, an
        octave low.
    hop
        Spacing between time windows [s].
    threshold
        The periodicity score above which a time window leans voiced.
    octave_cost, switch_cost
        Tracking costs, as above.
    subharmonic_margin
        How much lower a candidate at a whole multiple may score and still
        rule out a candidate. ``None`` turns the rule off, to see the raw choice.
    delay
        ``None`` (the default) chooses the cheapest path through the whole
        track. A number of seconds decides each time window that long
        after it, as a live tracker must: the time window takes the state
        that the cheapest path to the time window ``delay`` later passes
        through (the last ``delay`` of the track take the whole path).
        :class:`F0TrackStream` gives this result block by block. At 50 ms
        it chose the same as the whole track on all 12,505 time windows of
        16 recordings (tools/check_streaming_claims.py, C8).
    """
    fs = float(sound.fs)
    if not 0 < f_lo < f_hi < fs / 2:
        raise ValueError(f"need 0 < f_lo < f_hi < fs/2, got f_lo={f_lo:g}, f_hi={f_hi:g}")
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop:g}")
    if subharmonic_margin is not None and not 0 <= subharmonic_margin < 1:
        raise ValueError(f"subharmonic_margin must be in [0, 1) or None, got {subharmonic_margin:g}")
    lag = _lag(delay, hop)
    t = np.arange(0.0, sound.duration, hop)
    n_ch = sound.n_channels
    shape = (n_ch, len(t))
    f0 = np.zeros(shape)
    score = np.zeros(shape)
    all_candidates = np.full(shape + (MAX_CANDIDATES,), np.nan)
    all_candidate_scores = np.full(shape + (MAX_CANDIDATES,), np.nan)
    for ch in range(n_ch):
        samples = np.asarray(sound.data[:, ch], dtype=float)
        all_candidates[ch], all_candidate_scores[ch] = _scored_candidates(samples, t * fs, fs, f_lo, f_hi)
        f0[ch], score[ch] = _viterbi(
            all_candidates[ch],
            all_candidate_scores[ch],
            threshold,
            octave_cost,
            switch_cost,
            subharmonic_margin,
            lag,
        )
    return F0Track(t, f0, score, all_candidates, all_candidate_scores, fs, threshold)


def _lag(delay: float | None, hop: float) -> int | None:
    """``delay`` [s] in time windows (None stays None)."""
    if delay is None:
        return None
    if delay < 0:
        raise ValueError(f"delay must be 0 or more seconds, or None, got {delay:g}")
    return int(round(delay / hop))


def _scored_candidates(samples, positions, fs, f_lo, f_hi, origin=0):
    """The refined candidates of every time window at ``positions`` (times
    in samples; ``samples[0]`` is sample ``origin``) and their periodicity
    scores, each ``(n_windows, MAX_CANDIDATES)``, NaN where a time window
    has fewer. Each time window's result depends on its own samples only,
    so a stream that holds them gets the same numbers."""
    window_candidates = _candidates(samples, positions, fs, f_lo, f_hi, origin)
    found = np.isfinite(window_candidates)
    candidate_positions = np.repeat(positions, MAX_CANDIDATES)[found.ravel()]
    refined = _refine(samples, candidate_positions, window_candidates[found], fs, origin=origin)
    in_range = (refined > 0.9 * f_lo) & (refined < 1.1 * f_hi)
    refined_scores = np.zeros_like(refined)
    refined_scores[in_range] = _periodicity(
        samples, candidate_positions[in_range], refined[in_range], fs, origin=origin
    )
    window_candidates[found] = np.where(in_range, refined, np.nan)
    window_scores = np.full(window_candidates.shape, np.nan)
    window_scores[found] = np.where(in_range, refined_scores, np.nan)
    return window_candidates, window_scores


def _candidates(x, positions, fs, f_lo, f_hi, origin=0):
    """Up to MAX_CANDIDATES F0 candidates per time window, best first, NaN-padded:
    minima of YIN's normalized difference function below 1. ``positions``
    are the time windows' times in samples (``t * fs``) and ``x[0]`` is
    sample ``origin``; each time window's result depends on its own samples
    only."""
    n_window, tau_max, tau_min, n_span = _difference_layout(fs, f_lo, f_hi)
    padded = np.concatenate([np.zeros(n_window), x, np.zeros(n_span)])
    starts = np.round(positions).astype(int) - origin + n_window - n_window // 2
    n_fft = 1 << int(np.ceil(np.log2(2 * n_span)))
    out = np.full((len(positions), MAX_CANDIDATES), np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_span)
    lags = np.arange(tau_min, tau_max)
    for chunk_start in range(0, len(positions), 4 * CHUNK):
        segments = windows[starts[chunk_start : chunk_start + 4 * CHUNK]]
        reference = segments[:, :n_window]
        # d(tau) = sum a^2 + sum b_tau^2 - 2 sum a b_tau (a = reference, b_tau = the
        # stretch tau samples later), the cross term by FFT
        cross_term = np.fft.irfft(
            np.conj(np.fft.rfft(reference, n_fft)) * np.fft.rfft(segments, n_fft), n_fft
        )
        cumulative_energy = np.concatenate(
            [np.zeros((len(segments), 1)), np.cumsum(segments**2, axis=1)], axis=1
        )
        lagged_energy = (
            cumulative_energy[:, n_window : n_window + tau_max + 2] - cumulative_energy[:, : tau_max + 2]
        )
        difference = np.maximum(
            np.sum(reference * reference, axis=1, keepdims=True)
            + lagged_energy
            - 2 * cross_term[:, : tau_max + 2],
            0,
        )
        difference[:, 0] = 0
        cumulative_difference = np.cumsum(difference[:, 1:], axis=1)
        cmnd = np.ones_like(difference)
        with np.errstate(divide="ignore", invalid="ignore"):
            cmnd[:, 1:] = np.where(
                cumulative_difference > 0,
                difference[:, 1:] * np.arange(1, tau_max + 2) / cumulative_difference,
                1.0,
            )
        cmnd_before, cmnd_at_lag, cmnd_after = cmnd[:, lags - 1], cmnd[:, lags], cmnd[:, lags + 1]
        is_min = (cmnd_at_lag < cmnd_before) & (cmnd_at_lag <= cmnd_after) & (cmnd_at_lag < 1)
        ranked_pos = np.argsort(np.where(is_min, cmnd_at_lag, np.inf), axis=1)[:, :MAX_CANDIDATES]
        # A steady high voice has a dip at every multiple of its period, all
        # about equally deep, and there can be more of them than candidates.
        # As in YIN, the shortest-lag dip nearly as deep as the deepest is
        # always offered (in place of the last), so the period itself is.
        deepest = np.min(np.where(is_min, cmnd_at_lag, np.inf), axis=1, keepdims=True)
        near_deepest = is_min & (cmnd_at_lag <= deepest + NEAR_DEEPEST)
        first_near = np.argmax(near_deepest, axis=1)
        left_out = near_deepest.any(axis=1) & ~np.any(ranked_pos == first_near[:, None], axis=1)
        ranked_pos[left_out, -1] = first_near[left_out]
        rows = np.arange(len(segments))[:, None]
        is_valid = is_min[rows, ranked_pos]
        min_before, min_value, min_after = (
            cmnd_before[rows, ranked_pos],
            cmnd_at_lag[rows, ranked_pos],
            cmnd_after[rows, ranked_pos],
        )
        shift = _parabola_vertex(min_before, min_value, min_after, dip=True)
        out[chunk_start : chunk_start + 4 * CHUNK] = np.where(
            is_valid, fs / (lags[ranked_pos] + shift), np.nan
        )
    return out


def _difference_layout(fs, f_lo, f_hi):
    """``(n_window, tau_max, tau_min, n_span)`` of the difference function:
    a time window's stretch starts ``n_window // 2`` samples before it and
    is ``n_span`` long."""
    n_window = int(round(max(0.025, 1 / f_lo) * fs))
    tau_max = int(np.ceil(fs / f_lo)) + 1
    tau_min = max(int(np.floor(fs / f_hi)), 1)
    return n_window, tau_max, tau_min, n_window + tau_max + 1


def _row_length(n):
    """``n`` rounded up to a multiple of ROW_STEP: candidates are grouped by
    this length, so a sum over a row's samples (padded with zeros) is the
    same whichever candidates are computed with it."""
    return -(-np.asarray(n) // ROW_STEP) * ROW_STEP


def _groups(lengths):
    """The indices of each run of equal ``lengths``, shortest first, at most CHUNK at a time."""
    order = np.argsort(lengths, kind="stable")
    sorted_lengths = lengths[order]
    edges = np.flatnonzero(np.diff(sorted_lengths)) + 1
    for group in np.split(order, edges):
        for chunk_start in range(0, len(group), CHUNK):
            yield group[chunk_start : chunk_start + CHUNK], int(lengths[group[0]])


def _refine_half_length(f0s, fs):
    """Half the row length the refinement uses for candidates at ``f0s``
    (refinement can lower an F0 a little, so this leaves room)."""
    return _row_length((np.round(PERIODS * fs / (0.8 * f0s)).astype(int) | 1) // 2)


def _blackman(n, length):
    """A Blackman window of (odd) `length` samples evaluated at offsets n from
    its center; zero outside."""
    half = (length - 1) / 2
    phase = 2 * np.pi * (n + half) / (length - 1)
    window = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2 * phase)
    return np.where(np.abs(n) <= half, window, 0.0)


@lru_cache(maxsize=1024)
def _blackman_row(half_length, length):
    """:func:`_blackman` at the offsets ``-half_length .. half_length`` (cached, read-only)."""
    row = _blackman(np.arange(-half_length, half_length + 1), length)
    row.flags.writeable = False
    return row


def _refine(samples, candidate_positions, candidate_f0s, fs, iters=2, origin=0):
    """Instantaneous-frequency refinement of the candidate F0s at their
    positions (times in samples; ``samples[0]`` is sample ``origin``). Each
    candidate's result depends on itself and its samples only."""
    out = candidate_f0s.copy()
    if not len(candidate_f0s):
        return out
    half_lengths = _refine_half_length(candidate_f0s, fs)
    pad = int(half_lengths.max()) + 2
    padded = np.concatenate([np.zeros(pad), samples, np.zeros(pad)])
    harmonic_numbers = np.arange(1, MAX_HARMONICS + 1)
    for chunk, half_length in _groups(half_lengths):
        chunk_f0 = candidate_f0s[chunk]
        offsets = np.arange(-half_length, half_length + 1)
        centers = np.round(candidate_positions[chunk]).astype(int) - origin
        sample_index = centers[:, None] + pad + offsets[None, :]
        segment, segment_next = padded[sample_index], padded[sample_index + 1]
        for _ in range(iters):
            length = np.round(PERIODS * fs / chunk_f0).astype(int) | 1
            window = np.stack([_blackman_row(half_length, n) for n in length.tolist()])
            windowed, windowed_next = segment * window, segment_next * window
            rotation = np.exp(-2j * np.pi * chunk_f0[:, None] * offsets[None, :] / fs)
            harmonic_rotation = rotation.copy()
            # Every harmonic up to the ceiling, at least N_HARMONICS, below Nyquist. Harmonic k's
            # frequency divided by k is as exact an F0 as the fundamental's and has k times
            # less jitter for the same power, and below 100 Hz the first six hold little.
            # (more for some candidates of the group than they use, which get no weight below)
            up_to_ceiling = np.ceil(HARMONIC_CEILING / chunk_f0.min())
            n_harmonics = int(min(MAX_HARMONICS, max(N_HARMONICS, up_to_ceiling)))
            # both windowed stretches against each harmonic's rotation, as real and
            # imaginary parts: sums[:, i, j, part]
            both = np.stack([windowed, windowed_next], axis=1)
            sums = np.zeros((len(chunk_f0), MAX_HARMONICS, 2, 2))
            for i in range(n_harmonics):  # harmonic_rotation = rotation ** (i + 1)
                sums[:, i] = both @ harmonic_rotation.view(float).reshape(*rotation.shape, 2)
                harmonic_rotation *= rotation
            spectrum = sums[:, :, 0, 0] + 1j * sums[:, :, 0, 1]
            spectrum_next = sums[:, :, 1, 0] + 1j * sums[:, :, 1, 1]
            harmonic_freq = harmonic_numbers[None, :] * chunk_f0[:, None]
            used = (harmonic_freq < fs / 2) & (
                (harmonic_numbers[None, :] <= N_HARMONICS) | (harmonic_freq <= HARMONIC_CEILING)
            )
            power = np.abs(spectrum) ** 2 * used
            inst_freq = (
                np.angle(spectrum_next * np.conj(spectrum)) * fs / (2 * np.pi) / harmonic_numbers[None, :]
            )
            total_power = power.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                new_f0 = np.sum(power * inst_freq, axis=1) / total_power
            good = (total_power > 0) & np.isfinite(new_f0) & (new_f0 > 0)
            chunk_f0 = np.where(good, new_f0, chunk_f0)
        out[chunk] = chunk_f0
    return out


def _periodicity(samples, candidate_positions, candidate_f0s, fs, origin=0):
    """Normalized correlation between a stretch SCORE_PERIODS periods long (at
    least SCORE_MIN_WINDOW), centered half a period before the candidate time,
    and the same stretch one period later. The fractional part of the period
    is done with a Kaiser-windowed sinc. ``candidate_positions`` and
    ``origin`` as for :func:`_refine`."""
    out = np.zeros(len(candidate_f0s))
    if not len(candidate_f0s):
        return out
    min_length = SCORE_MIN_WINDOW * fs
    lengths = np.round(np.maximum(SCORE_PERIODS * fs / candidate_f0s, min_length)).astype(int)
    row_lengths = _row_length(lengths + 1)
    max_period = int(np.ceil(fs / candidate_f0s.min())) + 1
    pad = int(row_lengths.max()) + max_period + 2 * SINC_HALF + 2
    padded = np.concatenate([np.zeros(pad), samples, np.zeros(pad)])
    taps = np.arange(-SINC_HALF + 1, SINC_HALF + 1)
    for chunk, row_length in _groups(row_lengths):
        chunk_f0 = candidate_f0s[chunk]
        period = fs / chunk_f0
        length = lengths[chunk]
        offsets = np.arange(row_length)
        start = np.round(candidate_positions[chunk] - length / 2 - period / 2).astype(int) - origin + pad
        in_window = offsets[None, :] < length[:, None]
        stretch = padded[start[:, None] + offsets[None, :]] * in_window
        period_int = np.floor(period).astype(int)
        period_frac = period - period_int
        sinc_kernel = np.sinc(taps[None, :] - period_frac[:, None]) * SINC_TAPER[None, :]
        # stretch_later[j] = samples(start + j + period)
        #                  = sum_tap sinc_kernel[tap] samples[start + j + period_int + tap]
        base = start + period_int + taps[0]
        reach = np.arange(row_length + len(taps) - 1)
        later_windows = np.lib.stride_tricks.sliding_window_view(
            padded[base[:, None] + reach], len(taps), axis=1
        )
        stretch_later = (later_windows @ sinc_kernel[:, :, None])[..., 0] * in_window
        norm = np.sqrt(np.sum(stretch * stretch, axis=1) * np.sum(stretch_later * stretch_later, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[chunk] = np.where(norm > 0, np.sum(stretch * stretch_later, axis=1) / norm, 0.0)
    return out


@dataclass(frozen=True)
class F0TrackStream:
    """:func:`f0_track` a block of sound at a time, deciding each time window
    ``delay`` seconds after it (default 50 ms). The parameters are
    ``f0_track``'s; each step returns an :class:`F0Track` of the time
    windows it decided, and :meth:`F0Track.concatenate` joins them into
    exactly ``f0_track(sound, ..., delay=delay)``::

        tracker = so.F0TrackStream()
        so.F0Track.concatenate(list(so.stream(tracker, so.blocks(snd, 441))))

    A time window's candidates need the samples a little after it (the
    difference function's stretch, and the refinement's and score's
    windows at the lowest F0 searched), so a time window is decided once
    those samples and ``delay`` more have arrived. Its candidates are
    computed from its own samples only, as in batch, so they are the same
    numbers; the Viterbi pass carries its costs from one time window to the
    next and keeps the backpointers of the undecided time windows.
    """

    f_lo: float = 60.0
    f_hi: float = 500.0
    hop: float = 0.005
    threshold: float = 0.5
    octave_cost: float = 2.0
    switch_cost: float = 0.5
    subharmonic_margin: float | None = 0.05
    delay: float = 0.05

    def __post_init__(self):
        _lag(self.delay, self.hop)  # checks it
        if self.hop <= 0:
            raise ValueError(f"hop must be positive, got {self.hop:g}")

    def start(self, fs: float) -> F0State:
        """Nothing seen yet: zeros before the first sample, as in batch."""
        fs = float(fs)
        if not 0 < self.f_lo < self.f_hi < fs / 2:
            raise ValueError(f"need 0 < f_lo < f_hi < fs/2, got f_lo={self.f_lo:g}, f_hi={self.f_hi:g}")
        before, _ = _reach(fs, self.f_lo, self.f_hi)
        return F0State(fs, np.zeros((before, 0)), -before, 0, 0, None, None)

    def step(self, state: F0State, block: Sound) -> tuple[F0State, F0Track]:
        """Take ``block`` and return the time windows now decided."""
        if not isinstance(block, Sound):
            raise TypeError(f"a block is a Sound, not {type(block).__name__}")
        if block.fs != state.fs:
            raise ValueError(f"this stream runs at {state.fs:g} Hz; the block is at {block.fs:g} Hz")
        if state.history.shape[1] == 0:
            if not len(block):  # an empty block does not fix the channels
                return state, self._part(state, 0, block.n_channels)
            state = replace(state, history=np.zeros((len(state.history), block.n_channels)))
        if block.n_channels != state.history.shape[1]:
            raise ValueError(
                f"this stream has {state.history.shape[1]} channels; the block has {block.n_channels}"
            )
        state = replace(
            state, history=np.concatenate([state.history, block.data]), n_seen=state.n_seen + len(block)
        )
        _, after = _reach(state.fs, self.f_lo, self.f_hi)
        # every time window whose stretch of samples has all arrived
        ready = (
            int(np.floor((state.n_seen - 1 - after) / (self.hop * state.fs))) + 1
            if state.n_seen > after
            else 0
        )
        state = self._score(state, max(ready, state.next_window))
        lag = _lag(self.delay, self.hop)
        assert lag is not None
        return self._decide(state, max(state.cands.shape[1] - lag, 0), lag)

    def finish(self, state: F0State) -> F0Track:
        """The time windows left, with zeros after the last sample, the last
        ``delay`` of them decided by the whole path."""
        n_channels = max(state.history.shape[1], 1)
        _, after = _reach(state.fs, self.f_lo, self.f_hi)
        history = np.zeros((len(state.history) + after + 1, n_channels))
        history[: len(state.history), : state.history.shape[1]] = state.history
        n_windows = len(np.arange(0.0, state.n_seen / state.fs, self.hop))
        state = self._score(replace(state, history=history), n_windows)
        _, part = self._decide(state, state.cands.shape[1], None)
        return part

    def _score(self, state: F0State, ready: int) -> F0State:
        """Score time windows ``state.next_window .. ready - 1`` and carry the
        Viterbi pass over them."""
        fs, n_channels = state.fs, state.history.shape[1]
        new = np.arange(state.next_window, ready)
        if not len(new):
            return state
        cands = np.full((n_channels, len(new), MAX_CANDIDATES), np.nan)
        scores = np.full(cands.shape, np.nan)
        f_states = np.zeros((n_channels, len(new), 1 + MAX_CANDIDATES))
        local = np.zeros(f_states.shape)
        for ch in range(n_channels):
            cands[ch], scores[ch] = _scored_candidates(
                state.history[:, ch], new * self.hop * fs, fs, self.f_lo, self.f_hi, state.origin
            )
            f_states[ch], local[ch] = _local_costs(
                cands[ch], scores[ch], self.threshold, self.subharmonic_margin
            )
        cost = None if state.cost is None else state.cost.copy()  # the state is not changed
        last_states = state.last_states
        pointers = np.zeros(f_states.shape, int)
        cheapest = np.zeros((n_channels, len(new)), int)
        for i in range(len(new)):
            if cost is None:  # the first time window of the sound
                cost = local[:, 0].copy()
            else:
                assert last_states is not None
                for ch in range(n_channels):
                    cost[ch], pointers[ch, i] = _viterbi_step(
                        cost[ch],
                        last_states[ch],
                        f_states[ch, i],
                        local[ch, i],
                        self.octave_cost,
                        self.switch_cost,
                    )
            last_states = f_states[:, i]
            cheapest[:, i] = np.argmin(cost, axis=1)
        # keep the samples that the next time windows reach back to
        before, _ = _reach(fs, self.f_lo, self.f_hi)
        keep_from = int(np.floor(ready * self.hop * fs)) - before - state.origin
        keep_from = max(min(keep_from, len(state.history)), 0)
        return F0State(
            fs,
            state.history[keep_from:],
            state.origin + keep_from,
            state.n_seen,
            ready,
            cost,
            np.array(last_states),
            np.concatenate([state.cands, cands], axis=1) if state.cands.size else cands,
            np.concatenate([state.scores, scores], axis=1) if state.scores.size else scores,
            np.concatenate([state.f_states, f_states], axis=1) if state.f_states.size else f_states,
            np.concatenate([state.pointers, pointers], axis=1) if state.pointers.size else pointers,
            np.concatenate([state.cheapest, cheapest], axis=1) if state.cheapest.size else cheapest,
        )

    def _decide(self, state: F0State, n_decide: int, lag: int | None) -> tuple[F0State, F0Track]:
        """Decide the first ``n_decide`` undecided time windows: each takes the
        state of the cheapest path to the time window ``lag`` later, or with
        ``lag=None`` (the end) of the cheapest whole path."""
        n_channels = state.history.shape[1] if state.history.shape[1] else 1
        if not n_decide:
            return state, self._part(state, 0, n_channels)
        path = np.zeros((n_channels, n_decide), int)
        for ch in range(n_channels):
            if lag is None:
                last = state.cands.shape[1] - 1
                states = [int(state.cheapest[ch, last])] + _trace_back(
                    state.pointers[ch, last:0:-1], state.cheapest[ch, last]
                )
                path[ch] = states[::-1][:n_decide]
            else:
                for i in range(n_decide):
                    last = i + lag
                    if lag:
                        path[ch, i] = _trace_back(state.pointers[ch, last:i:-1], state.cheapest[ch, last])[-1]
                    else:
                        path[ch, i] = state.cheapest[ch, i]
        rows = np.arange(n_decide)
        f0 = np.stack(
            [np.where(path[ch] > 0, state.f_states[ch, rows, path[ch]], 0.0) for ch in range(n_channels)]
        )
        chosen = np.stack(
            [
                np.where(
                    path[ch] > 0,
                    np.nan_to_num(state.scores[ch], nan=0.0)[rows, np.maximum(path[ch] - 1, 0)],
                    0.0,
                )
                for ch in range(n_channels)
            ]
        )
        first_window = state.next_window - state.cands.shape[1]
        t = (first_window + rows) * self.hop
        part = F0Track(
            t,
            f0,
            chosen,
            state.cands[:, :n_decide],
            state.scores[:, :n_decide],
            state.fs,
            self.threshold,
            first_window,
        )
        kept = replace(
            state,
            cands=state.cands[:, n_decide:],
            scores=state.scores[:, n_decide:],
            f_states=state.f_states[:, n_decide:],
            pointers=state.pointers[:, n_decide:],
            cheapest=state.cheapest[:, n_decide:],
        )
        return kept, part

    def _part(self, state: F0State, n_windows: int, n_channels: int) -> F0Track:
        """No time windows, at the next time window to be decided."""
        first_window = state.next_window - state.cands.shape[1]
        empty = np.zeros((n_channels, 0))
        candidates = np.zeros((n_channels, 0, MAX_CANDIDATES))
        return F0Track(
            np.zeros(0), empty, empty, candidates, candidates, state.fs, self.threshold, first_window
        )


def _nothing() -> np.ndarray:
    return np.zeros((0, 0, 0))


@dataclass(frozen=True)
class F0State:
    """The state of an :class:`F0TrackStream`: the samples the next time
    windows need (``history[0]`` is sample ``origin``), how many samples
    have arrived and how many time windows have been scored, the Viterbi
    pass's costs and states at the last time window scored, and for the
    time windows scored but not yet decided, ``(n_channels, n, ...)``:
    their candidates and scores, states, backpointers and cheapest state."""

    fs: float
    history: np.ndarray
    origin: int
    n_seen: int
    next_window: int
    cost: np.ndarray | None
    last_states: np.ndarray | None
    cands: np.ndarray = field(default_factory=_nothing)
    scores: np.ndarray = field(default_factory=_nothing)
    f_states: np.ndarray = field(default_factory=_nothing)
    pointers: np.ndarray = field(default_factory=_nothing)
    cheapest: np.ndarray = field(default_factory=_nothing)


def _reach(fs: float, f_lo: float, f_hi: float) -> tuple[int, int]:
    """How many samples before and after its own a time window's candidates
    can use: the difference function's stretch, the refinement's window at
    the lowest candidate it can be offered, and the score's stretch and its
    copy one period later at the lowest F0 kept, with a few samples to spare."""
    n_window, tau_max, _, n_span = _difference_layout(fs, f_lo, f_hi)
    refine_half = int(_refine_half_length(np.array([fs / (tau_max + 1)]), fs)[0]) + 2
    period = fs / (0.9 * f_lo)
    score_length = int(np.round(max(SCORE_PERIODS * period, SCORE_MIN_WINDOW * fs)))
    score_row = int(_row_length(score_length + 1))
    before = max(n_window // 2, refine_half, score_length + int(period) + SINC_HALF) + 4
    after = max(n_span - n_window // 2, refine_half, score_row + int(period) + SINC_HALF) + 4
    return before, after


def _local_costs(cand, score, threshold, subharmonic_margin):
    """Each time window's states (0 for unvoiced, then the candidates, -1
    where missing) and what being in each costs, both ``(n_windows, 1 +
    MAX_CANDIDATES)``. A row depends on its own time window only."""
    n_windows = len(cand)
    f_states = np.concatenate([np.zeros((n_windows, 1)), np.nan_to_num(cand, nan=-1.0)], axis=1)
    local_cost = np.concatenate(
        [np.full((n_windows, 1), 1 - threshold), 1 - np.nan_to_num(score, nan=0.0)], axis=1
    )
    if subharmonic_margin is not None:
        # A candidate with another at a whole multiple of its frequency (2, 3,
        # ...), scoring nearly as well, is a subharmonic (anything periodic
        # at T is periodic at 2T, 3T, ...), so it is never chosen. Octaves
        # alone are not enough for very regular voices: on a steady 300 Hz
        # vowel the candidates at 100 and 60 Hz have none an octave above.
        cand_f0 = np.nan_to_num(cand, nan=-1.0)
        cand_score = np.nan_to_num(score, nan=-np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = cand_f0[:, None, :] / np.where(cand_f0 > 0, cand_f0, np.nan)[:, :, None]
            multiple = np.round(ratio)
            at_multiple = (multiple >= 2) & (np.abs(ratio / multiple - 1) < 0.05)
        scores_nearly_as_well = cand_score[:, None, :] >= cand_score[:, :, None] - subharmonic_margin
        is_subharmonic = np.any(at_multiple & scores_nearly_as_well, axis=2)
        local_cost[:, 1:] = np.where(is_subharmonic, np.inf, local_cost[:, 1:])
    local_cost[f_states < 0] = np.inf  # missing candidates
    return f_states, local_cost


def _viterbi_step(cost, f_from, f_to, local_cost, octave_cost, switch_cost):
    """The cheapest cost of reaching each state of a time window (states
    ``f_to``, own costs ``local_cost``) from the one before (states
    ``f_from``, path costs ``cost``), and which state each comes from."""
    voiced_to, voiced_from = (f_to > 0)[:, None], (f_from > 0)[None, :]
    f_to, f_from = f_to[:, None], f_from[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        jump = octave_cost * np.abs(np.log2(np.where(voiced_to & voiced_from, f_to / f_from, 1.0)))
    path_cost = np.where(voiced_to == voiced_from, jump, switch_cost) + cost[None, :]
    backpointer = np.argmin(path_cost, axis=1)
    return path_cost[np.arange(path_cost.shape[0]), backpointer] + local_cost, backpointer


def _trace_back(backpointers, state):
    """The states of the time windows before a path's last one, last first:
    from ``state`` back through ``backpointers`` (the last time window's
    first)."""
    states = []
    for backpointer in backpointers:
        state = int(backpointer[state])
        states.append(state)
    return states


def _viterbi(cand, score, threshold, octave_cost, switch_cost, subharmonic_margin, lag=None):
    """Cheapest path through (unvoiced, candidates...) per time window. With
    ``lag`` (in time windows), time window i takes the state the cheapest
    path to time window ``i + lag`` passes through, as a stream that decides
    ``lag`` time windows late does; the last ``lag`` take the whole path."""
    n_windows = len(cand)
    if not n_windows:  # an empty sound
        return np.zeros(0), np.zeros(0)
    f_states, local_cost = _local_costs(cand, score, threshold, subharmonic_margin)
    backpointer = np.zeros(f_states.shape, int)
    cost = local_cost[0].copy()
    cheapest_end = np.zeros(n_windows, int)
    cheapest_end[0] = int(np.argmin(cost))
    for i in range(1, n_windows):
        cost, backpointer[i] = _viterbi_step(
            cost, f_states[i - 1], f_states[i], local_cost[i], octave_cost, switch_cost
        )
        cheapest_end[i] = int(np.argmin(cost))
    path = np.zeros(n_windows, int)
    path[-1] = cheapest_end[-1]
    for i in range(n_windows - 1, 0, -1):
        path[i - 1] = backpointer[i][path[i]]
    if lag is not None:
        for i in range(n_windows - lag - 1):
            last = i + lag
            path[i] = _trace_back(backpointer[last:i:-1], cheapest_end[last])[-1] if lag else cheapest_end[i]
    rows = np.arange(n_windows)
    f0 = np.where(path > 0, f_states[rows, path], 0.0)
    chosen_score = np.where(path > 0, np.nan_to_num(score, nan=0.0)[rows, np.maximum(path - 1, 0)], 0.0)
    return f0, chosen_score


# ------------------------------------------------------------ the pitch
def _spread_around(values: np.ndarray, median: float, ratio: float, spread: float) -> np.ndarray:
    """The pitch-change map: ``values`` times ``ratio``, their distance from
    ``median`` on a log scale multiplied by ``spread``."""
    return median * ratio * (values / median) ** spread


def _scaled(values: ArrayLike, ratio: float, spread: float) -> np.ndarray:
    """Voiced values (> 0, one row per channel) times ratio, spread around
    each row's median on a log scale; 0 and NaN stay as they are."""
    values = np.asarray(values, dtype=float)
    if spread == 1:
        return values * ratio
    rows = np.atleast_2d(values)
    out = rows.copy()
    for row_in, row_out in zip(rows.reshape(len(rows), -1), out.reshape(len(out), -1), strict=True):
        voiced = row_in > 0
        if voiced.any():
            median = np.median(row_in[voiced])
            row_out[voiced] = _spread_around(row_in[voiced], median, ratio, spread)
    return out.reshape(values.shape)


def scale_f0(contour: ContourLike, ratio: float, *, range: float = 1.0) -> ContourLike:
    """An F0 contour with its pitch changed: every voiced value multiplied
    by ``ratio`` and, if ``range`` is not 1, spread around its median on a
    log scale, ``median * ratio * (f0 / median) ** range`` (``range=0`` is a
    monotone at ``ratio`` times the median, 2 doubles every interval from
    it). Unvoiced time windows (0) stay unvoiced.

    ``contour`` is an :class:`~sonore.views.f0.F0Track`, which comes back
    as an ``F0Track`` (its candidates changed the same way), a ``(times,
    f0)`` pair, which comes back as a pair, or anything with ``.t`` and
    ``.f0``, which comes back as a ``(times, f0)`` pair. A ratio in
    semitones ``st`` is ``2 ** (st / 12)``. With ``ratio=1`` and
    ``range=1`` the contour itself is returned.

    The envelope is not touched, so a voice resynthesized on the new contour
    keeps its formants (unlike :func:`~sonore.views.phasevocoder.pitch_shift`,
    which moves them with the pitch).
    """
    if not ratio > 0:
        raise ValueError(f"ratio must be positive, not {ratio}")
    if not range >= 0:
        raise ValueError(f"range must be 0 or more, not {range}")
    if ratio == 1 and range == 1:
        return contour
    match contour:
        case F0Track():
            return replace(
                contour,
                f0=_scaled(contour.f0, ratio, range),
                candidates=_scaled_candidates(contour, ratio, range),
            )
        case (times, values):
            return times, _scaled(values, ratio, range)
        case _ if hasattr(contour, "t") and hasattr(contour, "f0"):
            return contour.t, _scaled(contour.f0, ratio, range)
        case _:
            raise TypeError("contour must be an F0Track, a (times, f0) pair, or have .t and .f0")


def _scaled_candidates(track: F0Track, ratio: float, spread: float) -> np.ndarray:
    """The candidates moved by the same map as the chosen F0 (each channel's
    median of voiced F0 as the center). A channel with no voiced time window
    has no center, so its candidates are only multiplied by ``ratio``."""
    out = track.candidates * ratio
    if spread == 1:
        return out
    for channel, f0_row in enumerate(track.f0):
        voiced = f0_row > 0
        if voiced.any():
            median = np.median(f0_row[voiced])
            out[channel] = _spread_around(track.candidates[channel], median, ratio, spread)
    return out
