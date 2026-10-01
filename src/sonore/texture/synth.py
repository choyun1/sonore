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

import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
from scipy.signal import hilbert, resample

from sonore.analysis.filterbank import Subbands
from sonore.core.fft import threads
from sonore.core.sound import Sound
from sonore.core.utils import as_rng
from sonore.texture import grad as tg
from sonore.texture.stats import PAPER_CLASSES, TextureStats

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

    def reference(self, s: np.ndarray, per_term: bool = False):
        """The objective computed class by class from :meth:`terms` (the
        straightforward, separately tested path; :meth:`__call__` must agree)."""
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

    @cached_property
    def _fixed(self) -> dict:
        """Everything that doesn't depend on ``s``: targets, and the neighbors'
        centered envelopes and octave bands (computed once per channel instead
        of on every objective call)."""
        t, k, ctx, use = self.target, self.k, self.ctx, set(self.classes)
        f = {"mask": np.array([m in use for m in MOMENTS], float)}
        f["moments"] = np.array([t.env_mean[k], t.env_var[k], t.env_skew[k], t.env_kurt[k]])
        idx, rows = self._pairs(t.model.corr_offsets, t.env_corr)
        if idx and "env_corr" in use:
            f["corr"] = (self.env[:, idx], np.array(rows))
        idx, rows = self._pairs(t.model.c1_offsets, t.c1)
        if idx and "c1" in use:
            bands = list(t.model.c1_bands)
            f["c1"] = (bands, ctx.analytic_many(self.env[:, idx], bands).real, np.stack(rows, axis=-1))
        return f

    def __call__(self, s: np.ndarray, per_term: bool = False):
        """Loss ``0.5 * sum |f(s) - target|**2`` and its gradient.

        Fused: one forward transform per filterbank shared by all classes,
        and the band-domain cotangents of C2 and C1 summed before a single
        adjoint transform."""
        t, k, ctx, w, fx = self.target, self.k, self.ctx, self.ctx.w, self._fixed
        use = set(self.classes)
        loss, grad, parts = 0.0, np.zeros_like(s), {}

        def add(name, err, g):
            nonlocal loss, grad
            e = 0.5 * float(np.sum(np.abs(err) ** 2))
            parts[name] = e
            loss += e
            if g is not None:
                grad += g

        d = s - w @ s
        if fx["mask"].any():
            val, vjp = tg.env_moments(s, ctx)
            err = (val - fx["moments"]) * fx["mask"]
            add("env_moments", err, vjp(err))
        if "mod_power" in use:
            val, vjp = tg.mod_power_core(ctx.mod_filter(s), d, w)
            err = val - t.mod_power[k]
            G, direct = vjp(err)
            add("mod_power", err, ctx.mod_adjoint(G) + direct)
        if "c2" in use or "c1" in fx:
            K = ctx.A_oct.shape[1]
            A = ctx.analytic(s, range(K))
            G = np.zeros((len(s), K), complex)  # cotangent of Re A + i * cotangent of Im A
            if "c2" in use:
                val, vjp = tg.c2_core(A, w)
                err = val - t.c2[k]
                g_re, g_im = vjp(err)
                G += g_re + 1j * g_im
                add("c2", err, None)
            if "c1" in fx:
                bands, Ro, tgt = fx["c1"]
                val, vjp = tg.c1_core(A[:, bands].real, Ro, w)
                err = val - tgt
                G[:, bands] += vjp(err)
                add("c1", err, None)
            grad += ctx.analytic_adjoint(G.real, G.imag, range(K))
        if "corr" in fx:
            others, tgt = fx["corr"]
            val, vjp = tg.env_corr(s, others, ctx)
            err = val - tgt
            add("env_corr", err, vjp(err))
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
    with threads():
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
    progress: bool = False,
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

    **Run time.** Cost is linear in ``duration`` and in the number of
    iterations: about 0.4 s per iteration per second of sound on one core
    (2 s per iteration for 5 s), so the default 60 iterations of 5 s take
    about 2 minutes. Most textures are close to their final quality by 20-30
    iterations; use ``max_iter`` to trade quality for time, ``progress=True``
    to print one line per iteration, or ``callback(iteration, sound, snr)``
    to watch or stop from your own code.

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

    def analyze(x):
        sb = model.subbands(x)
        with threads():
            analytic = hilbert(sb, axis=0)
        comp = np.abs(analytic) ** model.compression
        env = np.maximum(_resample(comp, n_env), 0.0)
        return sb, analytic, comp, env

    def score(it, x, sb, env):
        """SNR of iterate ``x``, from the analysis the next iteration needs anyway."""
        nonlocal best
        sub_var = np.mean(sb**2, axis=0) - np.mean(sb, axis=0) ** 2
        snr = target.snr(TextureStats.from_envelopes(env, sub_var, model, window="uniform"), classes)
        history.append(snr)
        avg = float(np.mean(list(snr.values())))
        if avg > best[0]:
            best = (avg, x, it)
        if callback is not None:
            callback(it, Sound(x, model.fs), snr)
        if progress:
            elapsed = time.time() - t_start
            print(f"iteration {it:3d}/{max_iter}  average SNR {avg:5.1f} dB  ({elapsed:6.1f} s)", flush=True)
        return min(snr.values()) >= stop_db

    t_start = time.time()
    sb, analytic, comp, env = analyze(x)
    for it in range(1, max_iter + 1):
        fine = np.cos(np.angle(analytic))
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

        sb, analytic, comp, env = analyze(x)
        if score(it, x, sb, env):
            break
    report = {
        "snr": history,
        "iterations": len(history),
        "best_iteration": best[2],
        "average_snr": best[0],
        "converged": best[0] >= converged_db,
    }
    return Sound(best[1], model.fs), report
