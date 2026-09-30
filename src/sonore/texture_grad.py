"""Texture statistics of one envelope channel, with analytic gradients.

Synthesis (McDermott & Simoncelli, 2011) adjusts one cochlear channel's
compressed, downsampled envelope ``s`` at a time, holding the others fixed.
Each function here computes one class of statistics as a function of ``s``
and returns ``(value, vjp)``: the statistics, exactly as
:meth:`sonore.texture.TextureStats.measure` defines them, and a
vector-Jacobian product ``vjp(g) = J(s).T @ g``, so the gradient of
``0.5 * ||value - target||**2`` is ``vjp(value - target)``.

Every gradient is checked against finite differences in the tests.

Notation: ``w`` is the measurement window (sums to 1), ``d = s - w@s``,
``m_p = w @ d**p``. The modulation filters are real, zero-phase and circular,
so they are symmetric operators (their own adjoints). The analytic filter
``A`` (the octave bank with negative frequencies removed) is Hermitian, so the
adjoint of ``s -> Re(A s)`` is ``g -> Re(A g)`` and of ``s -> Im(A s)`` is
``g -> -Im(A g)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonore.texture import TextureModel

__all__ = [
    "ChannelContext",
    "env_moments",
    "mod_power",
    "env_corr",
    "c1",
    "c2",
    "mod_power_core",
    "c1_core",
    "c2_core",
]


@dataclass(frozen=True, eq=False)
class ChannelContext:
    """Everything that depends only on the model and the envelope length:
    the window and the filter responses on the FFT grid."""

    model: TextureModel
    w: np.ndarray  # (n,), sums to 1
    H_mod: np.ndarray  # (n//2 + 1, n_mod), real responses on the rfft grid
    A_oct: np.ndarray  # (n, n_oct), analytic responses on the fft grid

    @classmethod
    def build(cls, model: TextureModel, n: int, w: np.ndarray | None = None) -> ChannelContext:
        w = np.full(n, 1.0 / n) if w is None else np.asarray(w, float) / np.sum(w)
        H_mod = model.mod_bank.response(np.fft.rfftfreq(n, 1 / model.env_fs))
        f = np.fft.fftfreq(n, 1 / model.env_fs)
        gain = np.where(f > 0, 2.0, np.where(f == 0, 1.0, 0.0))
        if n % 2 == 0:
            gain[n // 2] = 1.0
        A_oct = model.oct_bank.response(np.abs(f)) * gain[:, None]
        return cls(model, w, H_mod, A_oct)

    @property
    def n(self) -> int:
        return len(self.w)

    def mod_filter(self, x: np.ndarray) -> np.ndarray:
        """``(n,) -> (n, n_mod)``; symmetric, so also its own adjoint."""
        return np.fft.irfft(np.fft.rfft(x)[:, None] * self.H_mod, n=self.n, axis=0)

    def mod_adjoint(self, G: np.ndarray) -> np.ndarray:
        """Adjoint of :meth:`mod_filter`: ``(n, n_mod) -> (n,)``."""
        return np.fft.irfft((np.fft.rfft(G, axis=0) * self.H_mod).sum(axis=1), n=self.n)

    def analytic(self, x: np.ndarray, bands) -> np.ndarray:
        """``(n,) -> (n, len(bands))`` complex analytic octave bands."""
        return np.fft.ifft(np.fft.fft(x)[:, None] * self.A_oct[:, list(bands)], axis=0)

    def analytic_many(self, X: np.ndarray, bands) -> np.ndarray:
        """``(n, k) -> (n, len(bands), k)``: :meth:`analytic` of each column."""
        return np.fft.ifft(np.fft.fft(X, axis=0)[:, None, :] * self.A_oct[:, list(bands), None], axis=0)

    def analytic_adjoint(self, g_re: np.ndarray, g_im: np.ndarray, bands) -> np.ndarray:
        """Adjoint of ``x -> (Re A x, Im A x)`` for the given bands, summed over bands.

        Since ``A`` is Hermitian, this is ``Re(A g_re) - Im(A g_im)``
        ``= Re(A (g_re + i g_im))``: one complex FFT, and the sum over bands
        is taken before the inverse transform."""
        G = np.fft.fft(g_re + 1j * g_im, axis=0)
        return np.fft.ifft((G * self.A_oct[:, list(bands)]).sum(axis=1)).real


def _central(s, w):
    mu = w @ s
    d = s - mu
    return mu, d


def env_moments(s: np.ndarray, ctx: ChannelContext):
    """``[mean, var/mean**2, skew, kurtosis]`` of the envelope."""
    w = ctx.w
    mu, d = _central(s, w)
    d2 = d * d
    d3 = d2 * d
    m2, m3, m4 = w @ d2, w @ d3, w @ (d2 * d2)
    val = np.array([mu, m2 / mu**2, m3 / m2**1.5, m4 / m2**2])

    def vjp(g):
        # Derivatives of the central moments (the w @ d**(p-1) terms use w @ d = 0):
        #   dm2 = 2 w d,  dm3 = 3 w d^2 - 3 m2 w,  dm4 = 4 w d^3 - 4 m3 w.
        # Collected as coefficients of w, w*d, w*d^2, w*d^3 to avoid n-length temporaries.
        c_var = g[1] / mu**2
        c_skew = g[2] / m2**1.5
        c_kurt = g[3] / m2**2
        c_dm2 = c_var - c_skew * 1.5 * m3 / m2 - c_kurt * 2 * m4 / m2
        c_w = g[0] - g[1] * 2 * m2 / mu**3 - 3 * m2 * c_skew - 4 * m3 * c_kurt
        return w * (c_w + 2 * c_dm2 * d + 3 * c_skew * d2 + 4 * c_kurt * d3)

    return val, vjp


def mod_power(s: np.ndarray, ctx: ChannelContext):
    """Modulation power in each constant-Q band, relative to envelope variance."""
    _, d = _central(s, ctx.w)
    val, core_vjp = mod_power_core(ctx.mod_filter(s), d, ctx.w)

    def vjp(g):
        G, direct = core_vjp(g)
        return ctx.mod_adjoint(G) + direct

    return val, vjp


def mod_power_core(B: np.ndarray, d: np.ndarray, w: np.ndarray):
    """:func:`mod_power` from the filtered envelope ``B`` ``(n, M)`` and the
    centered envelope ``d``. ``vjp(g)`` returns ``(G, direct)``: the cotangent
    of ``B`` (to be passed through the filter adjoint) and the gradient term
    that reaches ``s`` directly through the variance."""
    m2 = w @ d**2
    val = (w @ B**2) / m2

    def vjp(g):
        return 2 * w[:, None] * B * g[None, :] / m2, -(g @ val) / m2 * (2 * w * d)

    return val, vjp


def env_corr(s: np.ndarray, others: np.ndarray, ctx: ChannelContext):
    """Correlation of ``s`` with each column of ``others`` ``(n, k)`` (fixed
    envelopes of other channels)."""
    w = ctx.w
    _, d = _central(s, w)
    U = others - (w @ others)[None, :]
    m2, mu2 = w @ d**2, w @ U**2
    den = np.sqrt(m2 * mu2)
    val = (w @ (d[:, None] * U)) / den

    def vjp(g):
        # w @ U = 0, so the mean-subtraction term in d's derivative vanishes
        return w * (U @ (g / den)) - (g @ val) / m2 * (w * d)

    return val, vjp


def c1(s: np.ndarray, others: np.ndarray, ctx: ChannelContext, bands=None):
    """C1: correlation of ``s``'s octave modulation bands with the same bands
    of each column of ``others`` ``(n, k)``. Shape ``(len(bands), k)``. No
    mean subtraction (paper Eq. 6)."""
    bands = ctx.model.c1_bands if bands is None else bands
    val, core_vjp = c1_core(ctx.analytic(s, bands).real, ctx.analytic_many(others, bands).real, ctx.w)

    def vjp(g):
        gR = core_vjp(g)
        return ctx.analytic_adjoint(gR, np.zeros_like(gR), bands)

    return val, vjp


def c1_core(R: np.ndarray, Ro: np.ndarray, w: np.ndarray):
    """:func:`c1` from the octave bands of ``s`` (``R``, ``(n, K)``) and of the
    neighbors (``Ro``, ``(n, K, k)``). ``vjp(g)`` returns the cotangent of ``R``."""
    ps = w @ R**2  # (K,)
    po = np.einsum("t,tkj->kj", w, Ro**2)
    den = np.sqrt(ps[:, None] * po)
    val = np.einsum("t,tk,tkj->kj", w, R, Ro) / den

    def vjp(g):
        return w[:, None] * (np.einsum("tkj,kj->tk", Ro, g / den) - R * ((g * val).sum(1) / ps)[None, :])

    return val, vjp


def c2(s: np.ndarray, ctx: ChannelContext):
    """C2 within one channel: correlation of each octave band, frequency-doubled,
    with the next band up. Complex ``(n_oct - 1,)``: real part against the
    band's real part, imaginary part against its imaginary (quadrature) part."""
    K = ctx.A_oct.shape[1]
    val, core_vjp = c2_core(ctx.analytic(s, range(K)), ctx.w)

    def vjp(g):
        g_re, g_im = core_vjp(g)
        return ctx.analytic_adjoint(g_re, g_im, range(K))

    return val, vjp


def c2_core(A: np.ndarray, w: np.ndarray):
    """:func:`c2` from all analytic octave bands ``A`` ``(n, K)``. ``vjp(g)``
    returns the cotangents ``(g_re, g_im)`` of ``A.real`` and ``A.imag``."""
    K = A.shape[1]
    x, y = A.real, A.imag
    lo, hx, hy = slice(0, K - 1), x[:, 1:], y[:, 1:]
    r = np.maximum(np.abs(A[:, lo]), 1e-300)
    D = (x[:, lo] ** 2 - y[:, lo] ** 2) / r  # Re(a**2 / |a|)
    sd, sh = np.sqrt(w @ D**2), np.sqrt(w @ hx**2)
    P, Q = w @ (D * hx), w @ (D * hy)
    cre, cim = P / (sd * sh), Q / (sd * sh)

    def vjp(g):
        g = np.asarray(g, complex)
        gr, gi = g.real, g.imag
        ww = w[:, None]
        gD = ww * (hx * gr + hy * gi) / (sd * sh) - ww * D * (gr * cre + gi * cim) / sd**2
        ghx = ww * D * gr / (sd * sh) - ww * hx * (gr * cre + gi * cim) / sh**2
        ghy = ww * D * gi / (sd * sh)
        xl, yl = x[:, lo], y[:, lo]
        gxl = gD * (2 * xl / r - D * xl / r**2)
        gyl = gD * (-2 * yl / r - D * yl / r**2)
        g_re, g_im = np.zeros_like(x), np.zeros_like(y)
        g_re[:, lo] += gxl
        g_im[:, lo] += gyl
        g_re[:, 1:] += ghx
        g_im[:, 1:] += ghy
        return g_re, g_im

    return cre + 1j * cim, vjp
