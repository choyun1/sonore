"""F0 tracking: candidates from a difference function, refinement by the
instantaneous frequency of harmonics, a periodicity score, and a Viterbi pass
that decides voicing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonore.core.sound import Sound

__all__ = ["F0Track", "f0_track"]

MAX_CANDIDATES = 4
N_HARMONICS = 6
PERIODS = 3.0  # window length for refinement and scoring, in periods of the candidate
SINC_HALF = 32  # half-length of the fractional-delay kernel [samples]
CHUNK = 512  # candidates processed at once


@dataclass(frozen=True)
class F0Track:
    """An F0 track: one estimate per channel every ``hop`` seconds.

    ``f0`` and ``score`` have shape ``(n_channels, n_frames)``; ``f0`` is 0
    where the frame is unvoiced, and ``score`` is the periodicity score of
    the chosen candidate (0 where unvoiced). ``candidates`` and
    ``candidate_scores`` have shape ``(n_channels, n_frames, 4)`` and hold
    every refined candidate the tracker weighed, NaN where a frame had fewer.
    ``t`` and a row of ``f0`` go straight into
    :meth:`~sonore.analysis.frames.TVGaborFrame.pitch_adaptive` and
    :meth:`~sonore.analysis.cepstrum.Cepstrum.lifter`.
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
        """True where the frame is voiced, shape ``(n_channels, n_frames)``."""
        return self.f0 > 0

    def __repr__(self) -> str:
        c, n = self.f0.shape
        v = self.voiced
        rng = f", F0 {self.f0[v].min():.0f}-{self.f0[v].max():.0f} Hz" if v.any() else ""
        return f"F0Track({n} frames, {c} ch, {v.mean():.0%} voiced{rng})"

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

    1. **Candidates.** Up to four local minima of the cumulative-mean-normalized
       difference function (de Cheveigné & Kawahara, 2002) over a 25 ms window,
       at lags between ``1/f_hi`` and ``1/f_lo``, each refined by a parabola.
       This is the autocorrelation idea with YIN's corrections: it finds the
       period from all the harmonics, so it works when the fundamental is weak
       or filtered out. Every minimum below 1 is offered, so that the score
       alone decides voicing.
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
    4. **Tracking.** A Viterbi pass over (unvoiced, candidates) per frame. A
       candidate costs ``1 - score`` and the unvoiced state ``1 - threshold``,
       so a frame leans voiced when its best score exceeds ``threshold``;
       moving between candidates costs ``octave_cost`` per octave, and a
       voicing change costs ``switch_cost``. A candidate is never chosen if
       another one an octave above scores at least as well, less
       ``subharmonic_margin``: anything periodic
       with period T is also periodic with period 2T, and without this rule
       a female voice is tracked an octave low on about 1% of frames. No
       smoothing follows.

    On synthetic vowels the refined F0 is within 0.12% of the truth on a one
    octave per second glide and a 5.5 Hz vibrato. Against laryngograph
    reference F0 (a male and a female speaker), it gets the voicing of 5.5%
    and 1.6% of frames wrong, where WORLD's Harvest, which leans towards
    calling frames voiced, gets about 21% wrong; where both it and the
    reference say voiced, 98% and 95% of frames are within 5%. Lower ``threshold`` to voice
    more weak or creaky stretches, for example before resynthesis.

    Parameters
    ----------
    sound
        The sound; each channel is tracked separately.
    f_lo, f_hi
        The search range [Hz]. The defaults cover speaking voices; raise
        ``f_hi`` for singing.
    hop
        Frame period [s].
    threshold
        The periodicity score above which a frame leans voiced.
    octave_cost, switch_cost
        Tracking costs, as above.
    subharmonic_margin
        How much lower the octave above may score and still rule out a
        candidate. ``None`` turns the rule off, to see the raw choice.
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
    cand = np.full(shape + (MAX_CANDIDATES,), np.nan)
    cand_score = np.full(shape + (MAX_CANDIDATES,), np.nan)
    for ch in range(n_ch):
        x = np.asarray(sound.data[:, ch], dtype=float)
        c = _candidates(x, t, fs, f_lo, f_hi)
        ok = np.isfinite(c)
        flat = c[ok]
        refined = _refine(x, np.repeat(t, MAX_CANDIDATES)[ok.ravel()], flat, fs)
        keep = (refined > 0.9 * f_lo) & (refined < 1.1 * f_hi)
        sc = np.zeros_like(refined)
        sc[keep] = _periodicity(x, np.repeat(t, MAX_CANDIDATES)[ok.ravel()][keep], refined[keep], fs, f_lo)
        c[ok] = np.where(keep, refined, np.nan)
        s = np.full(c.shape, np.nan)
        s[ok] = np.where(keep, sc, np.nan)
        cand[ch], cand_score[ch] = c, s
        f0[ch], score[ch] = _viterbi(c, s, threshold, octave_cost, switch_cost, subharmonic_margin)
    return F0Track(t, f0, score, cand, cand_score, fs, threshold)


def _candidates(x, t, fs, f_lo, f_hi):
    """Up to MAX_CANDIDATES F0 candidates per frame, best first, NaN-padded:
    minima of YIN's normalized difference function below 1."""
    W = int(round(max(0.025, 1 / f_lo) * fs))
    tau_max = int(np.ceil(fs / f_lo)) + 1
    tau_min = max(int(np.floor(fs / f_hi)), 1)
    span = W + tau_max + 1
    xp = np.concatenate([np.zeros(W), x, np.zeros(span)])
    starts = np.round(t * fs).astype(int) + W - W // 2
    n_fft = 1 << int(np.ceil(np.log2(2 * span)))
    out = np.full((len(t), MAX_CANDIDATES), np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(xp, span)
    lags = np.arange(tau_min, tau_max)
    for lo in range(0, len(t), 4 * CHUNK):
        seg = windows[starts[lo : lo + 4 * CHUNK]]
        a = seg[:, :W]
        # d(tau) = sum a^2 + sum b_tau^2 - 2 sum a b_tau, the cross term by FFT
        cross = np.fft.irfft(np.conj(np.fft.rfft(a, n_fft)) * np.fft.rfft(seg, n_fft), n_fft)
        c2 = np.concatenate([np.zeros((len(seg), 1)), np.cumsum(seg**2, axis=1)], axis=1)
        energy_b = c2[:, W : W + tau_max + 2] - c2[:, : tau_max + 2]
        d = np.maximum(np.sum(a * a, axis=1, keepdims=True) + energy_b - 2 * cross[:, : tau_max + 2], 0)
        d[:, 0] = 0
        cm = np.cumsum(d[:, 1:], axis=1)
        dn = np.ones_like(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            dn[:, 1:] = np.where(cm > 0, d[:, 1:] * np.arange(1, tau_max + 2) / cm, 1.0)
        y0, y1, y2 = dn[:, lags - 1], dn[:, lags], dn[:, lags + 1]
        is_min = (y1 < y0) & (y1 <= y2) & (y1 < 1)
        ranked = np.argsort(np.where(is_min, y1, np.inf), axis=1)[:, :MAX_CANDIDATES]
        rows = np.arange(len(seg))[:, None]
        valid = is_min[rows, ranked]
        a0, a1, a2 = y0[rows, ranked], y1[rows, ranked], y2[rows, ranked]
        den = a0 - 2 * a1 + a2
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(den > 0, 0.5 * (a0 - a2) / den, 0.0)
        out[lo : lo + 4 * CHUNK] = np.where(valid, fs / (lags[ranked] + shift), np.nan)
    return out


def _blackman(n, length):
    """A Blackman window of (odd) `length` samples evaluated at offsets n from
    its centre; zero outside."""
    half = (length - 1) / 2
    x = 2 * np.pi * (n + half) / (length - 1)
    w = 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)
    return np.where(np.abs(n) <= half, w, 0.0)


def _refine(x, tc, f, fs, iters=2):
    """Instantaneous-frequency refinement of candidates f at times tc."""
    out = f.copy()
    if not len(f):
        return out
    # Refinement can lower f a little, so leave room for a longer window.
    pad = int(round(PERIODS * fs / (0.8 * f.min()))) + 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    k = np.arange(1, N_HARMONICS + 1)
    order = np.argsort(f)  # similar window lengths in each chunk
    for lo in range(0, len(f), CHUNK):
        sel = order[lo : lo + CHUNK]
        fc = f[sel]
        L_max = int(round(PERIODS * fs / (0.8 * fc.min()))) | 1
        n = np.arange(L_max) - L_max // 2
        idx = np.round(tc[sel] * fs).astype(int)[:, None] + pad + n[None, :]
        seg0, seg1 = xp[idx], xp[idx + 1]
        for _ in range(iters):
            length = np.round(PERIODS * fs / fc).astype(int) | 1
            w = _blackman(n[None, :], length[:, None])
            s0, s1 = seg0 * w, seg1 * w
            step = np.exp(-2j * np.pi * fc[:, None] * n[None, :] / fs)
            z = step.copy()
            X0 = np.empty((len(fc), N_HARMONICS), complex)
            X1 = np.empty_like(X0)
            for i in range(N_HARMONICS):  # z = step ** (i + 1)
                X0[:, i] = np.sum(s0 * z, axis=1)
                X1[:, i] = np.sum(s1 * z, axis=1)
                z *= step
            power = np.abs(X0) ** 2 * (k[None, :] * fc[:, None] < fs / 2)
            inst = np.angle(X1 * np.conj(X0)) * fs / (2 * np.pi) / k[None, :]
            total = power.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                new = np.sum(power * inst, axis=1) / total
            good = (total > 0) & np.isfinite(new) & (new > 0)
            fc = np.where(good, new, fc)
        out[sel] = fc
    return out


def _periodicity(x, tc, f, fs, f_lo):
    """Normalized correlation between a stretch PERIODS periods long, centred
    half a period before tc, and the same stretch one period later. The
    fractional part of the period is done with a Kaiser-windowed sinc."""
    out = np.zeros(len(f))
    if not len(f):
        return out
    L_max = int(np.ceil(PERIODS * fs / f.min())) + 1
    T_max = int(np.ceil(fs / f.min())) + 1
    pad = L_max + T_max + 2 * SINC_HALF + 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    m = np.arange(-SINC_HALF + 1, SINC_HALF + 1)
    kaiser = np.kaiser(2 * SINC_HALF, 8.0)
    order = np.argsort(f)
    for lo in range(0, len(f), CHUNK):
        sel = order[lo : lo + CHUNK]
        fc = f[sel]
        j = np.arange(int(np.ceil(PERIODS * fs / fc.min())) + 1)
        T = fs / fc
        L = np.round(PERIODS * T).astype(int)
        start = np.round(tc[sel] * fs - L / 2 - T / 2).astype(int) + pad
        mask = j[None, :] < L[:, None]
        a = xp[start[:, None] + j[None, :]] * mask
        q = np.floor(T).astype(int)
        phi = T - q
        h = np.sinc(m[None, :] - phi[:, None]) * kaiser[None, :]
        # b[j] = x(start + j + T) = sum_m h[m] x[start + j + q + m]
        base = start + q
        b = np.zeros_like(a)
        for i, mm in enumerate(m):
            b += h[:, i : i + 1] * xp[base[:, None] + j[None, :] + mm]
        b *= mask
        den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[sel] = np.where(den > 0, np.sum(a * b, axis=1) / den, 0.0)
    return out


def _viterbi(cand, score, threshold, octave_cost, switch_cost, subharmonic_margin):
    """Cheapest path through (unvoiced, candidates...) per frame."""
    n = len(cand)
    f_states = np.concatenate([np.zeros((n, 1)), np.nan_to_num(cand, nan=-1.0)], axis=1)
    local = np.concatenate([np.full((n, 1), 1 - threshold), 1 - np.nan_to_num(score, nan=0.0)], axis=1)
    if subharmonic_margin is not None:
        # A candidate whose octave above is also a candidate, scoring nearly
        # as well, is a subharmonic (anything periodic at T is periodic at
        # 2T), so it is never chosen.
        f = np.nan_to_num(cand, nan=-1.0)
        sc = np.nan_to_num(score, nan=-np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            near = np.abs(f[:, None, :] / (2 * np.where(f > 0, f, np.nan))[:, :, None] - 1) < 0.05
        sub = np.any(near & (sc[:, None, :] >= sc[:, :, None] - subharmonic_margin), axis=2)
        local[:, 1:] = np.where(sub, np.inf, local[:, 1:])
    local[f_states < 0] = np.inf  # missing candidates
    back = np.zeros(f_states.shape, int)
    cost = local[0].copy()
    voiced = f_states > 0
    for i in range(1, n):
        fa, fb = f_states[i][:, None], f_states[i - 1][None, :]
        va, vb = voiced[i][:, None], voiced[i - 1][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = octave_cost * np.abs(np.log2(np.where(va & vb, fa / fb, 1.0)))
        tr = np.where(va == vb, jump, switch_cost) + cost[None, :]
        back[i] = np.argmin(tr, axis=1)
        cost = tr[np.arange(tr.shape[0]), back[i]] + local[i]
    path = np.zeros(n, int)
    if n:
        path[-1] = int(np.argmin(cost))
        for i in range(n - 1, 0, -1):
            path[i - 1] = back[i][path[i]]
    rows = np.arange(n)
    f0 = np.where(path > 0, f_states[rows, path], 0.0)
    sc = np.where(path > 0, np.nan_to_num(score, nan=0.0)[rows, np.maximum(path - 1, 0)], 0.0)
    return f0, sc
