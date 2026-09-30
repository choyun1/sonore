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

Known limitation (as in the paper's objective): the loss is nearly flat along a
uniform shift of the envelope, since the mean enters only one term, so a few
CG iterations barely correct the envelope *mean*. In the full loop the mean
should come mainly from rescaling each subband to its target variance (to be
checked in milestone 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from sonore import texture_grad as tg
from sonore.texture import TextureStats

__all__ = ["ChannelObjective", "impose_channel"]


@dataclass(frozen=True, eq=False)
class ChannelObjective:
    """The squared-error objective for channel ``k`` given the other channels'
    envelopes ``env`` ``(n, B)`` and which of them count as ``adjusted``."""

    ctx: tg.ChannelContext
    target: TextureStats
    k: int
    env: np.ndarray
    adjusted: np.ndarray  # (B,) bool

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
        out = [
            ("env_moments", lambda s: tg.env_moments(s, ctx), moments),
            ("mod_power", lambda s: tg.mod_power(s, ctx), t.mod_power[k]),
            ("c2", lambda s: tg.c2(s, ctx), t.c2[k]),
        ]
        idx, rows = self._pairs(t.model.corr_offsets, t.env_corr)
        if idx:
            others = self.env[:, idx]
            out.append(("env_corr", lambda s, o=others: tg.env_corr(s, o, ctx), np.array(rows)))
        idx, rows = self._pairs(t.model.c1_offsets, t.c1)
        if idx:
            others = self.env[:, idx]
            out.append(("c1", lambda s, o=others: tg.c1(s, o, ctx), np.stack(rows, axis=-1)))
        return out

    def __call__(self, s: np.ndarray, per_term: bool = False):
        """Loss ``0.5 * sum |f(s) - target|**2`` and its gradient."""
        loss, grad, parts = 0.0, np.zeros_like(s), {}
        for name, f, tgt in self.terms():
            val, vjp = f(s)
            err = val - tgt
            e = 0.5 * float(np.sum(np.abs(err) ** 2))
            parts[name] = e
            loss += e
            grad += vjp(err)
        return (loss, grad, parts) if per_term else (loss, grad)


def impose_channel(
    target: TextureStats,
    env: np.ndarray,
    k: int,
    adjusted: np.ndarray | None = None,
    n_iter: int = 5,
    ctx: tg.ChannelContext | None = None,
) -> tuple[np.ndarray, dict]:
    """Adjust channel ``k`` of ``env`` ``(n, B)`` toward ``target``.

    ``adjusted`` marks channels whose envelopes are already final in this pass
    (default: none). Runs ``n_iter`` conjugate-gradient iterations and clips
    the result at 0. Returns the new envelope ``(n,)`` and a report with the
    objective before and after (total and per term).
    """
    B = env.shape[1]
    adjusted = np.zeros(B, bool) if adjusted is None else np.asarray(adjusted, bool)
    ctx = tg.ChannelContext.build(target.model, env.shape[0]) if ctx is None else ctx
    obj = ChannelObjective(ctx, target, k, env, adjusted)
    s0 = env[:, k]
    before = obj(s0, per_term=True)
    res = minimize(obj, s0, jac=True, method="CG", options={"maxiter": n_iter, "gtol": 0.0})
    s = np.maximum(res.x, 0.0)
    after = obj(s, per_term=True)
    report = {"before": before[0], "after": after[0], "terms_before": before[2], "terms_after": after[2]}
    return s, report
