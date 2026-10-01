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

from sonore.texture.stats import TextureModel

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
        # Gradient w.r.t. s of g @ [mean, var/mean**2, skew, kurtosis].
        # Derivatives of the central moments (the w @ d**(p-1) terms use w @ d = 0):
        #   dm2 = 2 w d,  dm3 = 3 w d^2 - 3 m2 w,  dm4 = 4 w d^3 - 4 m3 w.
        # Collected as coefficients of w, w*d, w*d^2, w*d^3 to avoid n-length temporaries:
        # c_var, c_skew, c_kurt are g scaled by each statistic's normalizer, c_dm2 the
        # coefficient of dm2 (and so of 2 w d), c_w the coefficient of w.
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
        # Gradient w.r.t. s: back through the modulation filters, plus the variance term.
        g_filtered, direct = core_vjp(g)
        return ctx.mod_adjoint(g_filtered) + direct

    return val, vjp


def mod_power_core(B: np.ndarray, d: np.ndarray, w: np.ndarray):
    """:func:`mod_power` from the filtered envelope ``B`` ``(n, M)`` and the
    centered envelope ``d``. ``vjp(g)`` returns ``(G, direct)``: the cotangent
    of ``B`` (to be passed through the filter adjoint) and the gradient term
    that reaches ``s`` directly through the variance."""
    m2 = w @ d**2
    val = (w @ B**2) / m2

    def vjp(g):
        # Gradients w.r.t. B and (through the variance m2) directly w.r.t. s.
        return 2 * w[:, None] * B * g[None, :] / m2, -(g @ val) / m2 * (2 * w * d)

    return val, vjp


def env_corr(s: np.ndarray, others: np.ndarray, ctx: ChannelContext):
    """Correlation of ``s`` with each column of ``others`` ``(n, k)`` (fixed
    envelopes of other channels)."""
    w = ctx.w
    _, d = _central(s, w)
    others_c = others - (w @ others)[None, :]  # centered
    m2, var_others = w @ d**2, w @ others_c**2
    norm = np.sqrt(m2 * var_others)  # product of standard deviations
    val = (w @ (d[:, None] * others_c)) / norm

    def vjp(g):
        # Gradient w.r.t. s. w @ others_c = 0, so the mean-subtraction term in
        # d's derivative vanishes.
        return w * (others_c @ (g / norm)) - (g @ val) / m2 * (w * d)

    return val, vjp


def c1(s: np.ndarray, others: np.ndarray, ctx: ChannelContext, bands=None):
    """C1: correlation of ``s``'s octave modulation bands with the same bands
    of each column of ``others`` ``(n, k)``. Shape ``(len(bands), k)``. No
    mean subtraction (paper Eq. 6)."""
    bands = ctx.model.c1_bands if bands is None else bands
    val, core_vjp = c1_core(ctx.analytic(s, bands).real, ctx.analytic_many(others, bands).real, ctx.w)

    def vjp(g):
        # Gradient w.r.t. s: back through the real part of the analytic bands.
        g_bands = core_vjp(g)
        return ctx.analytic_adjoint(g_bands, np.zeros_like(g_bands), bands)

    return val, vjp


def c1_core(R: np.ndarray, Ro: np.ndarray, w: np.ndarray):
    """:func:`c1` from the octave bands of ``s`` (``R``, ``(n, K)``) and of the
    neighbors (``Ro``, ``(n, K, k)``). ``vjp(g)`` returns the cotangent of ``R``."""
    pow_s = w @ R**2  # (K,) weighted power of each band of s
    pow_o = np.einsum("t,tkj->kj", w, Ro**2)  # (K, k) same for the neighbors
    norm = np.sqrt(pow_s[:, None] * pow_o)
    val = np.einsum("t,tk,tkj->kj", w, R, Ro) / norm

    def vjp(g):
        # Gradient w.r.t. R (the neighbors Ro are fixed).
        return w[:, None] * (np.einsum("tkj,kj->tk", Ro, g / norm) - R * ((g * val).sum(1) / pow_s)[None, :])

    return val, vjp


def c2(s: np.ndarray, ctx: ChannelContext):
    """C2 within one channel: correlation of each octave band, frequency-doubled,
    with the next band up. Complex ``(n_oct - 1,)``: real part against the
    band's real part, imaginary part against its imaginary (quadrature) part."""
    K = ctx.A_oct.shape[1]
    val, core_vjp = c2_core(ctx.analytic(s, range(K)), ctx.w)

    def vjp(g):
        # Gradient w.r.t. s: back through the real and imaginary analytic bands.
        g_re, g_im = core_vjp(g)
        return ctx.analytic_adjoint(g_re, g_im, range(K))

    return val, vjp


def c2_core(A: np.ndarray, w: np.ndarray):
    """:func:`c2` from all analytic octave bands ``A`` ``(n, K)``. ``vjp(g)``
    returns the cotangents ``(g_re, g_im)`` of ``A.real`` and ``A.imag``.

    Each band pair is a lower band (bands ``0..K-2``) and the band above it."""
    K = A.shape[1]
    re, im = A.real, A.imag
    lower = slice(0, K - 1)
    up_re, up_im = re[:, 1:], im[:, 1:]  # the upper band of each pair
    mag = np.maximum(np.abs(A[:, lower]), 1e-300)  # |lower band|, floored
    # Re(a**2 / |a|): the lower band at twice its frequency, same magnitude
    doubled = (re[:, lower] ** 2 - im[:, lower] ** 2) / mag
    rms_doubled, rms_up = np.sqrt(w @ doubled**2), np.sqrt(w @ up_re**2)
    cross_re, cross_im = w @ (doubled * up_re), w @ (doubled * up_im)
    corr_re, corr_im = cross_re / (rms_doubled * rms_up), cross_im / (rms_doubled * rms_up)

    def vjp(g):
        # Gradients w.r.t. A.real and A.imag, via doubled, up_re and up_im.
        g = np.asarray(g, complex)
        g_corr_re, g_corr_im = g.real, g.imag
        w_col = w[:, None]
        g_doubled = (
            w_col * (up_re * g_corr_re + up_im * g_corr_im) / (rms_doubled * rms_up)
            - w_col * doubled * (g_corr_re * corr_re + g_corr_im * corr_im) / rms_doubled**2
        )
        g_up_re = (
            w_col * doubled * g_corr_re / (rms_doubled * rms_up)
            - w_col * up_re * (g_corr_re * corr_re + g_corr_im * corr_im) / rms_up**2
        )
        g_up_im = w_col * doubled * g_corr_im / (rms_doubled * rms_up)
        lo_re, lo_im = re[:, lower], im[:, lower]
        g_lo_re = g_doubled * (2 * lo_re / mag - doubled * lo_re / mag**2)
        g_lo_im = g_doubled * (-2 * lo_im / mag - doubled * lo_im / mag**2)
        g_re, g_im = np.zeros_like(re), np.zeros_like(im)
        g_re[:, lower] += g_lo_re
        g_im[:, lower] += g_lo_im
        g_re[:, 1:] += g_up_re
        g_im[:, 1:] += g_up_im
        return g_re, g_im

    return corr_re + 1j * corr_im, vjp
