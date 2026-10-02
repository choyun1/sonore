"""F0 tracking: candidates from a difference function, refinement by the
instantaneous frequency of harmonics, a periodicity score, and a Viterbi pass
that decides voicing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sonore.core.sound import Sound

__all__ = ["F0Track", "f0_track", "scale_f0"]


MAX_CANDIDATES = 8


N_HARMONICS = 6


PERIODS = 3.0  # window length for refinement and scoring, in periods of the candidate


SINC_HALF = 32  # half-length of the fractional-delay kernel [samples]


CHUNK = 512  # candidates processed at once


@dataclass(frozen=True)
class F0Track:
    """An F0 track: one estimate per channel every ``hop`` seconds.

    ``f0`` and ``score`` have shape ``(n_channels, n_windows)``; ``f0`` is 0
    where the time window is unvoiced, and ``score`` is the periodicity score of
    the chosen candidate (0 where unvoiced). ``candidates`` and
    ``candidate_scores`` have shape ``(n_channels, n_windows, 4)`` and hold
    every refined candidate the tracker weighed, NaN where a time window had fewer.
    ``t`` and a row of ``f0`` go straight into
    :meth:`~sonore.frames.gabor.TVGaborFrame.pitch_adaptive` and
    :meth:`~sonore.views.cepstrum.Cepstrum.lifter`.
    """

    t: np.ndarray
    f0: np.ndarray
    score: np.ndarray
    candidates: np.ndarray
    candidate_scores: np.ndarray
    fs: float
    threshold: float

    @property
    def voiced(self) -> np.ndarray:
        """True where the time window is voiced, shape ``(n_channels, n_windows)``."""
        return self.f0 > 0

    def __repr__(self) -> str:
        n_channels, n_windows = self.f0.shape
        voiced = self.voiced
        f0_range = f", F0 {self.f0[voiced].min():.0f}-{self.f0[voiced].max():.0f} Hz" if voiced.any() else ""
        return f"F0Track({n_windows} time windows, {n_channels} ch, {voiced.mean():.0%} voiced{f0_range})"

    def plot(self, ax=None, channel: int = 0, candidates: bool = False, **kwargs):
        """F0 against time (see :func:`~sonore.plotting.plot_f0_track`)."""
        from sonore.plotting import plot_f0_track

        return plot_f0_track(self, ax=ax, channel=channel, candidates=candidates, **kwargs)


def f0_track(
    sound: Sound,
    f_lo: float = 60.0,
    f_hi: float = 500.0,
    hop: float = 0.005,
    threshold: float = 0.5,
    *,
    octave_cost: float = 2.0,
    switch_cost: float = 0.5,
    subharmonic_margin: float | None = 0.05,
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
       (with four, a steady 300 Hz vowel could lose it).
    2. **Refinement.** For a candidate f, a Blackman window three periods long;
       the instantaneous frequency of each of the first six harmonics is its
       phase advance over one sample, and the new F0 is the power-weighted
       mean of each harmonic's frequency divided by its number. Done twice.
       This is the idea of WORLD's StoneMask, written from its description.
    3. **Score.** The normalized correlation between a stretch three periods
       long and the same stretch one period later, the fractional part of the
       period done with a windowed sinc. For a periodic sound in independent
       noise it is about ``s / (1 + s)``, ``s`` the ratio of periodic to noise
       power, so 0.5 means the two are equally strong.
    4. **Tracking.** A Viterbi pass over (unvoiced, candidates) per time window. A
       candidate costs ``1 - score`` and the unvoiced state ``1 - threshold``,
       so a time window leans voiced when its best score exceeds ``threshold``;
       moving between candidates costs ``octave_cost`` per octave, and a
       voicing change costs ``switch_cost``. A candidate is never chosen if
       another one at a whole multiple of its frequency (an octave, a
       twelfth, ...) scores at least as well, less ``subharmonic_margin``:
       anything periodic with period T is also periodic with period 2T, 3T,
       ..., and without this rule a female voice is tracked an octave low on
       about 1% of time windows, and a steady synthetic vowel at 300 Hz at a
       third of its F0. No
       smoothing follows.

    On synthetic vowels the refined F0 is within 0.12% of the truth on a one
    octave per second glide and a 5.5 Hz vibrato. Against laryngograph
    reference F0 (a male and a female speaker), it gets the voicing of 5.6%
    and 1.5% of time windows wrong, where WORLD's Harvest, which leans towards
    calling time windows voiced, gets about 21% wrong; where both it and the
    reference say voiced, 98% and 95% of time windows are within 5%. Lower ``threshold`` to voice
    more weak or creaky stretches, for example before resynthesis.

    Parameters
    ----------
    sound
        The sound; each channel is tracked separately.
    f_lo, f_hi
        The search range [Hz]. The defaults cover speaking voices; raise
        ``f_hi`` for singing.
    hop
        Spacing between time windows [s].
    threshold
        The periodicity score above which a time window leans voiced.
    octave_cost, switch_cost
        Tracking costs, as above.
    subharmonic_margin
        How much lower a candidate at a whole multiple may score and still
        rule out a candidate. ``None`` turns the rule off, to see the raw choice.
    """
    fs = float(sound.fs)
    if not 0 < f_lo < f_hi < fs / 2:
        raise ValueError(f"need 0 < f_lo < f_hi < fs/2, got f_lo={f_lo:g}, f_hi={f_hi:g}")
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop:g}")
    if subharmonic_margin is not None and not 0 <= subharmonic_margin < 1:
        raise ValueError(f"subharmonic_margin must be in [0, 1) or None, got {subharmonic_margin:g}")
    t = np.arange(0.0, sound.duration, hop)
    n_ch = sound.n_channels
    shape = (n_ch, len(t))
    f0 = np.zeros(shape)
    score = np.zeros(shape)
    all_candidates = np.full(shape + (MAX_CANDIDATES,), np.nan)
    all_candidate_scores = np.full(shape + (MAX_CANDIDATES,), np.nan)
    for ch in range(n_ch):
        samples = np.asarray(sound.data[:, ch], dtype=float)
        window_candidates = _candidates(samples, t, fs, f_lo, f_hi)
        found = np.isfinite(window_candidates)
        found_f0 = window_candidates[found]
        refined = _refine(samples, np.repeat(t, MAX_CANDIDATES)[found.ravel()], found_f0, fs)
        in_range = (refined > 0.9 * f_lo) & (refined < 1.1 * f_hi)
        refined_scores = np.zeros_like(refined)
        refined_scores[in_range] = _periodicity(
            samples, np.repeat(t, MAX_CANDIDATES)[found.ravel()][in_range], refined[in_range], fs
        )
        window_candidates[found] = np.where(in_range, refined, np.nan)
        window_scores = np.full(window_candidates.shape, np.nan)
        window_scores[found] = np.where(in_range, refined_scores, np.nan)
        all_candidates[ch], all_candidate_scores[ch] = window_candidates, window_scores
        f0[ch], score[ch] = _viterbi(
            window_candidates, window_scores, threshold, octave_cost, switch_cost, subharmonic_margin
        )
    return F0Track(t, f0, score, all_candidates, all_candidate_scores, fs, threshold)


def _candidates(x, t, fs, f_lo, f_hi):
    """Up to MAX_CANDIDATES F0 candidates per time window, best first, NaN-padded:
    minima of YIN's normalized difference function below 1."""
    n_window = int(round(max(0.025, 1 / f_lo) * fs))
    tau_max = int(np.ceil(fs / f_lo)) + 1
    tau_min = max(int(np.floor(fs / f_hi)), 1)
    n_span = n_window + tau_max + 1
    padded = np.concatenate([np.zeros(n_window), x, np.zeros(n_span)])
    starts = np.round(t * fs).astype(int) + n_window - n_window // 2
    n_fft = 1 << int(np.ceil(np.log2(2 * n_span)))
    out = np.full((len(t), MAX_CANDIDATES), np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_span)
    lags = np.arange(tau_min, tau_max)
    for chunk_start in range(0, len(t), 4 * CHUNK):
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
        rows = np.arange(len(segments))[:, None]
        is_valid = is_min[rows, ranked_pos]
        min_before, min_value, min_after = (
            cmnd_before[rows, ranked_pos],
            cmnd_at_lag[rows, ranked_pos],
            cmnd_after[rows, ranked_pos],
        )
        curvature = min_before - 2 * min_value + min_after
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(curvature > 0, 0.5 * (min_before - min_after) / curvature, 0.0)
        out[chunk_start : chunk_start + 4 * CHUNK] = np.where(
            is_valid, fs / (lags[ranked_pos] + shift), np.nan
        )
    return out


def _blackman(n, length):
    """A Blackman window of (odd) `length` samples evaluated at offsets n from
    its centre; zero outside."""
    half = (length - 1) / 2
    phase = 2 * np.pi * (n + half) / (length - 1)
    window = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2 * phase)
    return np.where(np.abs(n) <= half, window, 0.0)


def _refine(x, tc, f, fs, iters=2):
    """Instantaneous-frequency refinement of candidates f at times tc."""
    out = f.copy()
    if not len(f):
        return out
    # Refinement can lower f a little, so leave room for a longer window.
    pad = int(round(PERIODS * fs / (0.8 * f.min()))) + 2
    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    harmonic_numbers = np.arange(1, N_HARMONICS + 1)
    order = np.argsort(f)  # similar window lengths in each chunk
    for chunk_start in range(0, len(f), CHUNK):
        chunk = order[chunk_start : chunk_start + CHUNK]
        chunk_f0 = f[chunk]
        max_length = int(round(PERIODS * fs / (0.8 * chunk_f0.min()))) | 1
        offsets = np.arange(max_length) - max_length // 2
        sample_index = np.round(tc[chunk] * fs).astype(int)[:, None] + pad + offsets[None, :]
        segment, segment_next = padded[sample_index], padded[sample_index + 1]
        for _ in range(iters):
            length = np.round(PERIODS * fs / chunk_f0).astype(int) | 1
            window = _blackman(offsets[None, :], length[:, None])
            windowed, windowed_next = segment * window, segment_next * window
            rotation = np.exp(-2j * np.pi * chunk_f0[:, None] * offsets[None, :] / fs)
            harmonic_rotation = rotation.copy()
            spectrum = np.empty((len(chunk_f0), N_HARMONICS), complex)
            spectrum_next = np.empty_like(spectrum)
            for i in range(N_HARMONICS):  # harmonic_rotation = rotation ** (i + 1)
                spectrum[:, i] = np.sum(windowed * harmonic_rotation, axis=1)
                spectrum_next[:, i] = np.sum(windowed_next * harmonic_rotation, axis=1)
                harmonic_rotation *= rotation
            power = np.abs(spectrum) ** 2 * (harmonic_numbers[None, :] * chunk_f0[:, None] < fs / 2)
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


def _periodicity(x, tc, f, fs):
    """Normalized correlation between a stretch PERIODS periods long, centred
    half a period before tc, and the same stretch one period later. The
    fractional part of the period is done with a Kaiser-windowed sinc."""
    out = np.zeros(len(f))
    if not len(f):
        return out
    max_length = int(np.ceil(PERIODS * fs / f.min())) + 1
    max_period = int(np.ceil(fs / f.min())) + 1
    pad = max_length + max_period + 2 * SINC_HALF + 2
    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    taps = np.arange(-SINC_HALF + 1, SINC_HALF + 1)
    kaiser = np.kaiser(2 * SINC_HALF, 8.0)
    order = np.argsort(f)
    for chunk_start in range(0, len(f), CHUNK):
        chunk = order[chunk_start : chunk_start + CHUNK]
        chunk_f0 = f[chunk]
        offsets = np.arange(int(np.ceil(PERIODS * fs / chunk_f0.min())) + 1)
        period = fs / chunk_f0
        length = np.round(PERIODS * period).astype(int)
        start = np.round(tc[chunk] * fs - length / 2 - period / 2).astype(int) + pad
        in_window = offsets[None, :] < length[:, None]
        stretch = padded[start[:, None] + offsets[None, :]] * in_window
        period_int = np.floor(period).astype(int)
        period_frac = period - period_int
        sinc_kernel = np.sinc(taps[None, :] - period_frac[:, None]) * kaiser[None, :]
        # stretch_later[j] = x(start + j + period)
        #                  = sum_tap sinc_kernel[tap] x[start + j + period_int + tap]
        base = start + period_int
        stretch_later = np.zeros_like(stretch)
        for i, tap in enumerate(taps):
            stretch_later += sinc_kernel[:, i : i + 1] * padded[base[:, None] + offsets[None, :] + tap]
        stretch_later *= in_window
        norm = np.sqrt(np.sum(stretch * stretch, axis=1) * np.sum(stretch_later * stretch_later, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[chunk] = np.where(norm > 0, np.sum(stretch * stretch_later, axis=1) / norm, 0.0)
    return out


def _viterbi(cand, score, threshold, octave_cost, switch_cost, subharmonic_margin):
    """Cheapest path through (unvoiced, candidates...) per time window."""
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
    backpointer = np.zeros(f_states.shape, int)
    cost = local_cost[0].copy()
    voiced = f_states > 0
    for i in range(1, n_windows):
        f_to, f_from = f_states[i][:, None], f_states[i - 1][None, :]
        voiced_to, voiced_from = voiced[i][:, None], voiced[i - 1][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = octave_cost * np.abs(np.log2(np.where(voiced_to & voiced_from, f_to / f_from, 1.0)))
        path_cost = np.where(voiced_to == voiced_from, jump, switch_cost) + cost[None, :]
        backpointer[i] = np.argmin(path_cost, axis=1)
        cost = path_cost[np.arange(path_cost.shape[0]), backpointer[i]] + local_cost[i]
    path = np.zeros(n_windows, int)
    if n_windows:
        path[-1] = int(np.argmin(cost))
        for i in range(n_windows - 1, 0, -1):
            path[i - 1] = backpointer[i][path[i]]
    rows = np.arange(n_windows)
    f0 = np.where(path > 0, f_states[rows, path], 0.0)
    chosen_score = np.where(path > 0, np.nan_to_num(score, nan=0.0)[rows, np.maximum(path - 1, 0)], 0.0)
    return f0, chosen_score


# ------------------------------------------------------------ the pitch
def _scaled(values: np.ndarray, ratio: float, spread: float) -> np.ndarray:
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
            row_out[voiced] = median * ratio * (row_in[voiced] / median) ** spread
    return out.reshape(values.shape)


def scale_f0(contour, ratio: float, *, range: float = 1.0):
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
    keeps its formants (unlike :func:`~sonore.stimuli.phasevocoder.pitch_shift`,
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
    median of voiced F0 as the centre)."""
    if spread == 1:
        return track.candidates * ratio
    out = track.candidates.copy()
    for channel, f0_row in enumerate(track.f0):
        voiced = f0_row > 0
        if voiced.any():
            median = np.median(f0_row[voiced])
            out[channel] = median * ratio * (track.candidates[channel] / median) ** spread
    return out
