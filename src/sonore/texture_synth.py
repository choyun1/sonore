"""Imposing texture statistics on envelopes (McDermott & Simoncelli, 2011).

Milestone 3 of the synthesis: adjust *one* channel's compressed envelope so
its statistics approach the targets, holding the other channels fixed. The
full loop (channel ordering, fine structure, subband rescaling, iterations)
builds on :func:`impose_channel`.

The objective for channel ``k`` is the unweighted sum of squared errors of
the statistics that depend on ``k``:

* its envelope moments, modulation power, and C2;
* its envelope correlation C and C1 with channels already adjusted in this
  pass (``adjusted``), at the model's offsets in both directions.

It is minimized by nonlinear conjugate gradient (SciPy's Polak-Ribiere CG)
with a fixed number of iterations, then the envelope is clipped at 0.

The loss is nearly flat along two directions: a uniform shift of the envelope
and a scaling of its deviations from the mean. Only the mean and variance
depend on them, so a few CG iterations barely correct those two. Since every
other statistic is invariant to shifting and scaling, they are set exactly
after CG (see ``affine`` in :func:`impose_channel`); this is a deviation from
the toolbox, listed in :data:`sonore.texture.DIFFERENCES_FROM_TOOLBOX`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import hilbert, resample

from sonore import texture_grad as tg
from sonore.filterbank import Subbands
from sonore.sound import Sound
from sonore.texture import PAPER_CLASSES, TextureStats
from sonore.utils import as_rng

__all__ = ["ChannelObjective", "impose_channel", "synthesize", "channel_order"]

MOMENTS = ("env_mean", "env_var", "env_skew", "env_kurt")


@dataclass(frozen=True, eq=False)
class ChannelObjective:
    """The squared-error objective for channel ``k`` given the other channels'
    envelopes ``env`` ``(n, B)`` and which of them count as ``adjusted``."""

    ctx: tg.ChannelContext
    target: TextureStats
    k: int
    env: np.ndarray
    adjusted: np.ndarray  # (B,) bool
    classes: tuple[str, ...] = PAPER_CLASSES

    def _pairs(self, offsets, table):
        """Neighbors ``k +/- d`` that are adjusted, with their targets from
        ``table[j, i]`` (the stat for channels ``j`` and ``j + offsets[i]``)."""
        B, k = self.env.shape[1], self.k
        idx, rows = [], []
        for i, d in enumerate(offsets):
            for j, row in ((k + d, k), (k - d, k - d)):  # the stat is stored under the lower channel
                if 0 <= j < B and self.adjusted[j] and j != k:
                    idx.append(j)
                    rows.append(table[row][..., i])
        return idx, rows

    def terms(self):
        """``[(name, f(s) -> (value, vjp), target)]`` for this channel."""
        t, k, ctx = self.target, self.k, self.ctx
        moments = np.array([t.env_mean[k], t.env_var[k], t.env_skew[k], t.env_kurt[k]])
        use = set(self.classes)
        out = []
        if use & set(MOMENTS):
            out.append(("env_moments", lambda s: tg.env_moments(s, ctx), moments))
        if "mod_power" in use:
            out.append(("mod_power", lambda s: tg.mod_power(s, ctx), t.mod_power[k]))
        if "c2" in use:
            out.append(("c2", lambda s: tg.c2(s, ctx), t.c2[k]))
        idx, rows = self._pairs(t.model.corr_offsets, t.env_corr)
        if idx and "env_corr" in use:
            others = self.env[:, idx]
            out.append(("env_corr", lambda s, o=others: tg.env_corr(s, o, ctx), np.array(rows)))
        idx, rows = self._pairs(t.model.c1_offsets, t.c1)
        if idx and "c1" in use:
            others = self.env[:, idx]
            out.append(("c1", lambda s, o=others: tg.c1(s, o, ctx), np.stack(rows, axis=-1)))
        return out

    def __call__(self, s: np.ndarray, per_term: bool = False):
        """Loss ``0.5 * sum |f(s) - target|**2`` and its gradient."""
        loss, grad, parts = 0.0, np.zeros_like(s), {}
        for name, f, tgt in self.terms():
            val, vjp = f(s)
            err = val - tgt
            if name == "env_moments":
                err = err * np.array([m in self.classes for m in MOMENTS])
            e = 0.5 * float(np.sum(np.abs(err) ** 2))
            parts[name] = e
            loss += e
            grad += vjp(err)
        return (loss, grad, parts) if per_term else (loss, grad)


def _match_mean_var(s, target, k, w, classes):
    mu = w @ s
    d = s - mu
    mean = target.env_mean[k] if "env_mean" in classes else mu
    if "env_var" in classes:
        m2 = w @ d**2
        d = d * np.sqrt(target.env_var[k] * mean**2 / max(m2, 1e-300))
    return mean + d


def impose_channel(
    target: TextureStats,
    env: np.ndarray,
    k: int,
    adjusted: np.ndarray | None = None,
    n_iter: int = 5,
    ctx: tg.ChannelContext | None = None,
    classes=PAPER_CLASSES,
    affine: bool = True,
) -> tuple[np.ndarray, dict]:
    """Adjust channel ``k`` of ``env`` ``(n, B)`` toward ``target``.

    ``adjusted`` marks channels whose envelopes are already final in this pass
    (default: none). Runs ``n_iter`` conjugate-gradient iterations and clips
    the result at 0. With ``affine`` (default), CG is followed by an exact
    affine correction ``s -> a + b*(s - mean)`` that sets the envelope mean
    and variance to their targets. Every other statistic is invariant to that
    map (skew, kurtosis and all correlations are scale- and shift-invariant;
    modulation power is normalized by the variance and its filters reject
    DC), so the correction costs nothing elsewhere, apart from clipping. It
    is needed because those two directions are nearly flat in the objective.
    Returns the new envelope ``(n,)`` and a report with the
    objective before and after (total and per term).
    """
    B = env.shape[1]
    adjusted = np.zeros(B, bool) if adjusted is None else np.asarray(adjusted, bool)
    ctx = tg.ChannelContext.build(target.model, env.shape[0]) if ctx is None else ctx
    obj = ChannelObjective(ctx, target, k, env, adjusted, tuple(classes))
    s0 = env[:, k]
    before = obj(s0, per_term=True)
    res = minimize(obj, s0, jac=True, method="CG", options={"maxiter": n_iter, "gtol": 0.0})
    s = res.x
    if affine:
        s = _match_mean_var(s, target, k, ctx.w, classes)
    s = np.maximum(s, 0.0)
    after = obj(s, per_term=True)
    report = {"before": before[0], "after": after[0], "terms_before": before[2], "terms_after": after[2]}
    return s, report


def channel_order(env_mean: np.ndarray) -> list[int]:
    """Start at the channel with the largest target envelope mean and
    alternate outward: ``k, k+1, k-1, k+2, k-2, ...``."""
    B = len(env_mean)
    k0 = int(np.argmax(env_mean))
    order = [k0]
    for d in range(1, B):
        for k in (k0 + d, k0 - d):
            if 0 <= k < B:
                order.append(k)
    return order


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    """Circular (FFT) resampling along axis 0."""
    return resample(x, n, axis=0)


def synthesize(
    target: TextureStats,
    duration: float = 5.0,
    classes=PAPER_CLASSES,
    rng=None,
    max_iter: int = 60,
    stop_db: float = 30.0,
    converged_db: float = 20.0,
    init: Sound | None = None,
    callback=None,
) -> tuple[Sound, dict]:
    """Synthesize a sound whose statistics match ``target``.

    Starts from Gaussian noise (or ``init``) and iterates: decompose into
    subbands; take compressed envelopes (downsampled, plus the high-rate
    residual the downsampling removes) and fine structure; impose the
    statistics channel by channel (:func:`impose_channel`, in
    :func:`channel_order`, each channel's correlations measured against the
    channels already adjusted); restore the full-rate envelopes, decompress,
    recombine with the fine structure; rescale each subband to its target
    variance; resynthesize. All analysis is circular, so the result loops
    seamlessly.

    Stops when every class in ``classes`` is at least ``stop_db`` SNR, or
    after ``max_iter`` iterations. The synthesis counts as converged if the
    average SNR over ``classes`` is at least ``converged_db``.

    Returns the sound (at ``target.model.fs``, RMS ``target.model.rms``) and a
    report: ``snr`` (per-iteration dicts), ``converged``, ``iterations``,
    ``best_iteration``. The returned sound is the iterate with the best
    average SNR.
    """
    model = target.model
    classes = tuple(classes)
    rng = as_rng(rng)
    if init is None:
        x = rng.standard_normal(int(round(duration * model.fs)))
    else:
        x = model.prepare(init)
    x = model.prepare(Sound(x, model.fs))
    fb = model.filterbank
    N = len(x)
    n_env = N // model.decimation
    ctx = tg.ChannelContext.build(model, n_env)
    order = channel_order(target.env_mean)
    history, best = [], (-np.inf, x, 0)
    for it in range(1, max_iter + 1):
        sb = model.subbands(x)
        analytic = hilbert(sb, axis=0)
        fine = np.cos(np.angle(analytic))
        comp = np.abs(analytic) ** model.compression
        env = np.maximum(_resample(comp, n_env), 0.0)
        residual = comp - _resample(env, N)

        adjusted = np.zeros(env.shape[1], bool)
        n_ls = 5 + round(0.2 * (it - 1))
        for k in order:
            env[:, k], _ = impose_channel(target, env, k, adjusted, n_iter=n_ls, ctx=ctx, classes=classes)
            adjusted[k] = True

        full = np.maximum(_resample(env, N) + residual, 0.0) ** (1 / model.compression)
        new_sb = full * fine
        var = np.mean(new_sb**2, axis=0) - np.mean(new_sb, axis=0) ** 2
        new_sb *= np.sqrt(target.subband_var / np.maximum(var, 1e-300))[None, :]
        y = Subbands(new_sb[:, :, None], model.fs, fb).synthesize().data[:, 0]
        x = y * (model.rms / np.sqrt(np.mean(y**2)))

        stats = TextureStats.from_subbands(model.subbands(x), model, window="uniform")
        snr = target.snr(stats, classes)
        history.append(snr)
        avg = float(np.mean(list(snr.values())))
        if avg > best[0]:
            best = (avg, x, it)
        if callback is not None:
            callback(it, Sound(x, model.fs), snr)
        if min(snr.values()) >= stop_db:
            break
    report = {
        "snr": history,
        "iterations": len(history),
        "best_iteration": best[2],
        "average_snr": best[0],
        "converged": best[0] >= converged_db,
    }
    return Sound(best[1], model.fs), report
