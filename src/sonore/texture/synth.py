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
from scipy.signal import resample

from sonore.core.fft import threads
from sonore.core.sound import Sound
from sonore.core.utils import as_rng, n_samples
from sonore.frames.filterbank import Subbands
from sonore.texture import grad as tg
from sonore.texture.stats import PAPER_CLASSES, TextureStats

__all__ = ["ChannelObjective", "impose_channel", "synthesize", "channel_order"]

MOMENTS = ("env_mean", "env_var", "env_skew", "env_kurt")


@dataclass(frozen=True, eq=False)
class ChannelObjective:
    """The squared-error objective for channel ``k`` given the other channels'
    envelopes ``env`` ``(n, B)`` and which of them count as ``adjusted``."""

    context: tg.ChannelContext
    target: TextureStats
    k: int
    env: np.ndarray
    adjusted: np.ndarray  # (B,) bool
    classes: tuple[str, ...] = PAPER_CLASSES

    def _pairs(self, offsets, table):
        """Neighbors ``k +/- d`` that are adjusted, with their targets from
        ``table[j, i]`` (the stat for channels ``j`` and ``j + offsets[i]``)."""
        n_channels, k = self.env.shape[1], self.k
        neighbors, targets = [], []
        for i, d in enumerate(offsets):
            for j, stored_row in ((k + d, k), (k - d, k - d)):  # the stat is stored under the lower channel
                if 0 <= j < n_channels and self.adjusted[j] and j != k:
                    neighbors.append(j)
                    targets.append(table[stored_row][..., i])
        return neighbors, targets

    def terms(self):
        """``[(name, f(s) -> (value, vjp), target)]`` for this channel."""
        target, k, context = self.target, self.k, self.context
        moments = np.array([target.env_mean[k], target.env_var[k], target.env_skew[k], target.env_kurt[k]])
        active = set(self.classes)
        out = []
        if active & set(MOMENTS):
            out.append(("env_moments", lambda s: tg.env_moments(s, context), moments))
        if "mod_power" in active:
            out.append(("mod_power", lambda s: tg.mod_power(s, context), target.mod_power[k]))
        if "c2" in active:
            out.append(("c2", lambda s: tg.c2(s, context), target.c2[k]))
        neighbors, pair_targets = self._pairs(target.model.corr_offsets, target.env_corr)
        if neighbors and "env_corr" in active:
            others = self.env[:, neighbors]
            out.append(("env_corr", lambda s, o=others: tg.env_corr(s, o, context), np.array(pair_targets)))
        neighbors, pair_targets = self._pairs(target.model.c1_offsets, target.c1)
        if neighbors and "c1" in active:
            others = self.env[:, neighbors]
            out.append(("c1", lambda s, o=others: tg.c1(s, o, context), np.stack(pair_targets, axis=-1)))
        return out

    def reference(self, s: np.ndarray, per_term: bool = False):
        """The objective computed class by class from :meth:`terms` (the
        straightforward, separately tested path; :meth:`__call__` must agree)."""
        loss, grad, parts = 0.0, np.zeros_like(s), {}
        for name, stat_fn, target_value in self.terms():
            value, vjp = stat_fn(s)
            error = value - target_value
            if name == "env_moments":
                error = error * np.array([m in self.classes for m in MOMENTS])
            term_loss = 0.5 * float(np.sum(np.abs(error) ** 2))
            parts[name] = term_loss
            loss += term_loss
            grad += vjp(error)
        return (loss, grad, parts) if per_term else (loss, grad)

    @cached_property
    def _fixed(self) -> dict:
        """Everything that doesn't depend on ``s``: targets, and the neighbors'
        centered envelopes and octave bands (computed once per channel instead
        of on every objective call)."""
        target, k, context, active = self.target, self.k, self.context, set(self.classes)
        fixed = {"mask": np.array([m in active for m in MOMENTS], float)}
        fixed["moments"] = np.array(
            [target.env_mean[k], target.env_var[k], target.env_skew[k], target.env_kurt[k]]
        )
        neighbors, pair_targets = self._pairs(target.model.corr_offsets, target.env_corr)
        if neighbors and "env_corr" in active:
            fixed["corr"] = (self.env[:, neighbors], np.array(pair_targets))
        neighbors, pair_targets = self._pairs(target.model.c1_offsets, target.c1)
        if neighbors and "c1" in active:
            bands = list(target.model.c1_bands)
            fixed["c1"] = (
                bands,
                context.analytic_many(self.env[:, neighbors], bands).real,
                np.stack(pair_targets, axis=-1),
            )
        return fixed

    def __call__(self, s: np.ndarray, per_term: bool = False):
        """Loss ``0.5 * sum |f(s) - target|**2`` and its gradient.

        Fused: one forward transform per filterbank shared by all classes,
        and the band-domain cotangents of C2 and C1 summed before a single
        adjoint transform."""
        target, k, context, w, fixed = self.target, self.k, self.context, self.context.w, self._fixed
        active = set(self.classes)
        loss, grad, parts = 0.0, np.zeros_like(s), {}

        def add(name, error, grad_term):
            nonlocal loss, grad
            term_loss = 0.5 * float(np.sum(np.abs(error) ** 2))
            parts[name] = term_loss
            loss += term_loss
            if grad_term is not None:
                grad += grad_term

        centered = s - w @ s
        if fixed["mask"].any():
            value, vjp = tg.env_moments(s, context)
            error = (value - fixed["moments"]) * fixed["mask"]
            add("env_moments", error, vjp(error))
        if "mod_power" in active:
            value, vjp = tg.mod_power_core(context.mod_filter(s), centered, w)
            error = value - target.mod_power[k]
            mod_cotangent, direct = vjp(error)
            add("mod_power", error, context.mod_adjoint(mod_cotangent) + direct)
        if "c2" in active or "c1" in fixed:
            n_oct = context.A_oct.shape[1]
            oct_bands = context.analytic(s, range(n_oct))
            band_cotangent = np.zeros(
                (len(s), n_oct), complex
            )  # cotangents of oct_bands.real + 1j * of .imag
            if "c2" in active:
                value, vjp = tg.c2_core(oct_bands, w)
                error = value - target.c2[k]
                g_re, g_im = vjp(error)
                band_cotangent += g_re + 1j * g_im
                add("c2", error, None)
            if "c1" in fixed:
                bands, neighbor_bands, target_value = fixed["c1"]
                value, vjp = tg.c1_core(oct_bands[:, bands].real, neighbor_bands, w)
                error = value - target_value
                band_cotangent[:, bands] += vjp(error)
                add("c1", error, None)
            grad += context.analytic_adjoint(band_cotangent.real, band_cotangent.imag, range(n_oct))
        if "corr" in fixed:
            others, target_value = fixed["corr"]
            value, vjp = tg.env_corr(s, others, context)
            error = value - target_value
            add("env_corr", error, vjp(error))
        return (loss, grad, parts) if per_term else (loss, grad)


def _match_mean_var(s, target, k, w, classes):
    current_mean = w @ s
    centered = s - current_mean
    new_mean = target.env_mean[k] if "env_mean" in classes else current_mean
    if "env_var" in classes:
        variance = w @ centered**2
        centered = centered * np.sqrt(target.env_var[k] * new_mean**2 / max(variance, 1e-300))
    return new_mean + centered


def impose_channel(
    target: TextureStats,
    env: np.ndarray,
    k: int,
    adjusted: np.ndarray | None = None,
    n_iter: int = 5,
    context: tg.ChannelContext | None = None,
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
    n_channels = env.shape[1]
    adjusted = np.zeros(n_channels, bool) if adjusted is None else np.asarray(adjusted, bool)
    context = tg.ChannelContext.build(target.model, env.shape[0]) if context is None else context
    objective = ChannelObjective(context, target, k, env, adjusted, tuple(classes))
    s_init = env[:, k]
    before = objective(s_init, per_term=True)
    result = minimize(objective, s_init, jac=True, method="CG", options={"maxiter": n_iter, "gtol": 0.0})
    s = result.x
    if affine:
        s = _match_mean_var(s, target, k, context.w, classes)
    s = np.maximum(s, 0.0)
    after = objective(s, per_term=True)
    report = {"before": before[0], "after": after[0], "terms_before": before[2], "terms_after": after[2]}
    return s, report


def channel_order(env_mean: np.ndarray) -> list[int]:
    """Start at the channel with the largest target envelope mean and
    alternate outward: ``k, k+1, k-1, k+2, k-2, ...``."""
    n_channels = len(env_mean)
    start = int(np.argmax(env_mean))
    order = [start]
    for distance in range(1, n_channels):
        for k in (start + distance, start - distance):
            if 0 <= k < n_channels:
                order.append(k)
    return order


def _resample_circular(x: np.ndarray, n: int) -> np.ndarray:
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
        x = rng.standard_normal(n_samples(duration, model.fs))
    else:
        x = model.prepare(init)
    x = model.prepare(Sound(x, model.fs))
    filterbank = model.filterbank
    signal_length = len(x)
    n_env = signal_length // model.decimation
    context = tg.ChannelContext.build(model, n_env)
    order = channel_order(target.env_mean)
    history, best = [], (-np.inf, x, 0)

    def analyze(x):
        subbands = model.subbands(x)
        return (subbands, *model._envelope_stages(subbands))

    def score(iteration, x, subbands, env):
        """SNR of iterate ``x``, from the analysis the next iteration needs anyway."""
        nonlocal best
        subband_var = np.mean(subbands**2, axis=0) - np.mean(subbands, axis=0) ** 2
        snr = target.snr(TextureStats.from_envelopes(env, subband_var, model, window="uniform"), classes)
        history.append(snr)
        average_snr = float(np.mean(list(snr.values())))
        if average_snr > best[0]:
            best = (average_snr, x, iteration)
        if callback is not None:
            callback(iteration, Sound(x, model.fs), snr)
        if progress:
            elapsed = time.time() - t_start
            print(
                f"iteration {iteration:3d}/{max_iter}  average SNR {average_snr:5.1f} dB  ({elapsed:6.1f} s)",
                flush=True,
            )
        return min(snr.values()) >= stop_db

    t_start = time.time()
    subbands, analytic, compressed, env = analyze(x)
    for iteration in range(1, max_iter + 1):
        fine_structure = np.cos(np.angle(analytic))
        residual = compressed - _resample_circular(env, signal_length)

        adjusted = np.zeros(env.shape[1], bool)
        n_cg_iter = 5 + round(0.2 * (iteration - 1))
        for k in order:
            env[:, k], _ = impose_channel(
                target, env, k, adjusted, n_iter=n_cg_iter, context=context, classes=classes
            )
            adjusted[k] = True

        full_env = np.maximum(_resample_circular(env, signal_length) + residual, 0.0)
        full_env = full_env ** (1 / model.compression)
        new_subbands = full_env * fine_structure
        new_subband_var = np.mean(new_subbands**2, axis=0) - np.mean(new_subbands, axis=0) ** 2
        new_subbands *= np.sqrt(target.subband_var / np.maximum(new_subband_var, 1e-300))[None, :]
        resynth = Subbands(new_subbands[:, :, None], model.fs, filterbank).to_sound().data[:, 0]
        x = resynth * (model.rms / np.sqrt(np.mean(resynth**2)))

        subbands, analytic, compressed, env = analyze(x)
        if score(iteration, x, subbands, env):
            break
    report = {
        "snr": history,
        "iterations": len(history),
        "best_iteration": best[2],
        "average_snr": best[0],
        "converged": best[0] >= converged_db,
    }
    return Sound(best[1], model.fs), report
