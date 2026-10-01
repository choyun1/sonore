"""Modulation filterbanks: bandpass filters applied to envelopes.

Two banks from McDermott & Simoncelli (2011), usable on their own (e.g. for
Dau-style modulation models) and by :mod:`sonore.texture`:

* :class:`ConstantQModulationFilterbank`: half-cycle cosines on a *linear*
  frequency axis, each spanning ``cf +/- cf/Q``, with centers log-spaced. Used
  for the modulation power spectrum.
* :class:`OctaveModulationFilterbank`: half-cycle cosines on a log2 axis, each
  two octaves wide (``cf/2 .. 2*cf``), centers one octave apart. Used for the
  C1/C2 modulation correlations.

Filtering is done in the frequency domain and is therefore *circular*: the
envelope is treated as one period of a periodic signal. That is exact for
texture synthesis (seamless loops); for other uses pad the envelope first.

A third bank, :class:`HannModulationFilterbank`, is defined in time instead:
Hann-windowed complex exponentials of finite length, applied by direct
(zero-padded, not circular) correlation, centred or causal. It is the bank
behind :class:`~sonore.analysis.modspectrogram.ModulationSpectrogram`, and
its finite kernels are what make a causal, block-by-block version possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

__all__ = [
    "ModulationFilterbank",
    "ConstantQModulationFilterbank",
    "OctaveModulationFilterbank",
    "HannModulationFilterbank",
]


@dataclass(frozen=True)
class ModulationFilterbank:
    """Base class. Subclasses define ``n_bands``, :attr:`cfs` and :meth:`response`."""

    @property
    def cfs(self) -> np.ndarray:
        raise NotImplementedError

    def response(self, freqs) -> np.ndarray:
        """Magnitude responses, shape ``(len(freqs), n_bands)`` (``freqs`` >= 0)."""
        raise NotImplementedError

    def filter(self, x, fs: float, axis: int = 0, analytic: bool = False) -> np.ndarray:
        """Filter ``x`` (circularly) along ``axis`` with every band.

        The band index is appended as a new *last* axis. With ``analytic=True``
        the output is the complex analytic signal of each band (negative
        frequencies removed, positive ones doubled), whose real part equals the
        ordinary filtered output.
        """
        x = np.moveaxis(np.asarray(x, float), axis, 0)
        n = x.shape[0]
        extra = (None,) * (x.ndim - 1)
        if not analytic:
            H = self.response(np.fft.rfftfreq(n, 1 / fs))  # (F, M)
            X = np.fft.rfft(x, axis=0)[..., None]
            out = np.fft.irfft(X * H[(slice(None),) + extra], n=n, axis=0)
        else:
            f = np.fft.fftfreq(n, 1 / fs)
            H = self.response(np.abs(f))
            gain = np.where(f > 0, 2.0, np.where(f == 0, 1.0, 0.0))
            if n % 2 == 0:
                gain[n // 2] = 1.0  # Nyquist bin is its own conjugate
            H = H * gain[:, None]
            X = np.fft.fft(x, axis=0)[..., None]
            out = np.fft.ifft(X * H[(slice(None),) + extra], axis=0)
        return np.moveaxis(out, 0, axis) if axis != 0 else out


@dataclass(frozen=True)
class ConstantQModulationFilterbank(ModulationFilterbank):
    """``n_bands`` constant-Q filters with centers log-spaced from ``f_lo`` to
    ``f_hi`` [Hz]. Filter ``k`` is a half-cycle cosine on a linear frequency
    axis, nonzero on ``cf_k*(1 - 1/Q) .. cf_k*(1 + 1/Q)``.

    Responses are scaled so that the summed squared response, averaged over
    the middle of the bank (between the 4th and the 4th-from-last center),
    is 1. That makes band powers sum to about the envelope's variance.
    """

    n_bands: int = 20
    f_lo: float = 0.5
    f_hi: float = 200.0
    Q: float = 2.0

    @property
    def cfs(self) -> np.ndarray:
        return np.geomspace(self.f_lo, self.f_hi, self.n_bands)

    @property
    def scale(self) -> float:
        cfs = self.cfs
        lo, hi = cfs[min(3, len(cfs) - 1)], cfs[max(len(cfs) - 4, 0)]
        f = np.linspace(lo, hi, 4097) if hi > lo else np.array([lo])
        return float(1 / np.sqrt(np.mean(np.sum(self._raw(f) ** 2, axis=1))))

    def _raw(self, freqs) -> np.ndarray:
        f = np.asarray(freqs, float)[:, None]
        cf = self.cfs[None, :]
        u = (f - cf) / (2 * cf / self.Q)  # -1/2 .. 1/2 across the support
        return np.where(np.abs(u) < 0.5, np.cos(np.pi * np.clip(u, -0.5, 0.5)), 0.0)

    def response(self, freqs) -> np.ndarray:
        return self._raw(freqs) * self.scale


@dataclass(frozen=True)
class OctaveModulationFilterbank(ModulationFilterbank):
    """``n_bands`` filters with centers one octave apart, the highest at
    ``f_hi`` [Hz] (default: 1.5625, 3.125, ..., 100 Hz). Filter ``k`` is a
    half-cycle cosine on log2 frequency spanning ``cf_k/2 .. 2*cf_k``. Peak
    gain is 1 (the C1/C2 correlations are scale-invariant, so no further
    normalization is applied)."""

    n_bands: int = 7
    f_hi: float = 100.0

    @property
    def cfs(self) -> np.ndarray:
        return self.f_hi / 2.0 ** np.arange(self.n_bands - 1, -1, -1)

    def response(self, freqs) -> np.ndarray:
        f = np.asarray(freqs, float)[:, None]
        with np.errstate(divide="ignore"):
            u = (np.log2(f) - np.log2(self.cfs)[None, :]) / 2  # octaves / width
        return np.where(np.abs(u) < 0.5, np.cos(np.pi * np.clip(u, -0.5, 0.5)), 0.0)


@dataclass(frozen=True)
class HannModulationFilterbank(ModulationFilterbank):
    """Hann-windowed complex exponentials, defined in time.

    Band ``k`` correlates the envelope with ``h_k[j] = w_k[j] exp(i 2 pi f_k
    (j - (L_k - 1)/2) / fs)``, where ``w_k`` is a Hann window of ``L_k``
    samples normalized to sum 1. Two ways to set the window:

    * ``cycles`` (the default, 3): every window holds that many cycles of its
      own rate, ``L_k = cycles * fs / f_k``, with centres ``per_octave`` to
      the octave from ``f_lo`` to ``f_hi``. A constant-Q bank (Q = cycles /
      1.44, about 2.1 for 3 cycles), long windows for slow rates and short
      ones for fast rates.
    * ``window`` [s]: every band uses the same window T, with centres on the
      linear grid ``k / T`` from the first one at or above ``max(f_lo, 2/T)``
      to ``f_hi``. This is the STFT of the envelope.

    Either way each window holds a whole number (at least 2) of cycles of its
    rate, so the bank ignores the envelope's mean: the Hann window's transform
    is zero at every whole bin from 2 on.

    Unlike the other two banks, :meth:`filter` works in time and is not
    circular: the envelope is taken to be zero outside its extent. The
    analytic output ``2 y`` has magnitude ``A * m`` for an envelope component
    ``A * m * cos(2 pi f_k t)``, and its real part is the real filtered
    output, whose frequency response is :meth:`response`.
    """

    f_lo: float = 0.5
    f_hi: float = 64.0
    per_octave: float = 2
    cycles: int = 3
    window: float | None = None

    def __post_init__(self):
        if self.window is None:
            if self.cycles != int(self.cycles) or self.cycles < 2:
                raise ValueError("cycles must be a whole number of at least 2, so the bank ignores the mean")
            if not 0 < self.f_lo <= self.f_hi:
                raise ValueError("need 0 < f_lo <= f_hi")
        elif self.window <= 0 or self.f_hi < max(self.f_lo, 2 / self.window):
            raise ValueError("need window > 0 and f_hi >= max(f_lo, 2 / window)")

    @property
    def n_bands(self) -> int:
        return len(self.cfs)

    @property
    def cfs(self) -> np.ndarray:
        """Centre rates [Hz]."""
        if self.window is None:
            n = int(np.floor(np.log2(self.f_hi / self.f_lo) * self.per_octave + 1e-9)) + 1
            return self.f_lo * 2.0 ** (np.arange(n) / self.per_octave)
        k_lo = max(2, int(np.ceil(self.f_lo * self.window - 1e-9)))
        k_hi = int(np.floor(self.f_hi * self.window + 1e-9))
        return np.arange(k_lo, k_hi + 1) / self.window

    def durations(self) -> np.ndarray:
        """Window length [s] of each band."""
        if self.window is None:
            return self.cycles / self.cfs
        return np.full(self.n_bands, float(self.window))

    def lengths(self, fs: float) -> np.ndarray:
        """Window length in samples at ``fs`` (rounded, at least 3)."""
        return np.maximum(np.round(self.durations() * fs).astype(int), 3)

    def check_fs(self, fs: float) -> None:
        """The envelope rate must be at least 3 * f_hi: the top band reaches
        about 1.25 * f_hi, and its window must hold enough samples."""
        if fs < 3 * self.cfs.max():
            raise ValueError(
                f"envelope rate {fs:g} Hz is too low for modulation rates up to {self.cfs.max():g} Hz; "
                f"resample the envelopes to at least {3 * self.cfs.max():g} Hz (1000 Hz is a good default)"
            )

    def kernels(self, fs: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(h_k, w_k)`` for every band at ``fs``: the complex kernel and its
        normalized Hann window."""
        out = []
        for f, n in zip(self.cfs, self.lengths(fs), strict=True):
            w = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
            w = w / w.sum()
            tc = (np.arange(n) - (n - 1) / 2) / fs
            out.append((w * np.exp(2j * np.pi * f * tc), w))
        return out

    def response(self, freqs) -> np.ndarray:
        """Magnitude response of the real filter (the real part of
        :meth:`filter`'s analytic output), shape ``(len(freqs), n_bands)``,
        for continuous-time windows. Peak gain is about 1; zero at 0 Hz."""
        f = np.asarray(freqs, float)[:, None]
        T, fc = self.durations()[None, :], self.cfs[None, :]
        return np.abs(_hann_ft((f - fc) * T) + _hann_ft((f + fc) * T))

    def filter(
        self, x, fs: float, axis: int = 0, analytic: bool = False, align: str = "center"
    ) -> np.ndarray:
        """Filter ``x`` along ``axis`` with every band (zero outside ``x``).

        The band index is appended as a new *last* axis. ``align="center"``
        puts each window's middle on the output sample; ``"causal"`` ends it
        there, which is the centred output delayed by ``(L_k - 1) // 2``
        samples and is what a live, block-by-block analysis would produce.
        """
        if align not in ("center", "causal"):
            raise ValueError("align must be 'center' or 'causal'")
        self.check_fs(fs)
        x = np.moveaxis(np.asarray(x, float), axis, 0)
        out = np.empty(x.shape + (self.n_bands,), complex)
        for k, (h, _) in enumerate(self.kernels(fs)):
            out[..., k] = 2 * _correlate(x, h, align)
        out = out if analytic else out.real
        return np.moveaxis(out, 0, axis) if axis != 0 else out


def _hann_ft(u: np.ndarray) -> np.ndarray:
    """Fourier transform of a unit-area Hann window of length 1, at ``u``
    cycles per window length: sinc(u) / (1 - u^2), with its limit at |u| = 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sinc(u) / (1 - u * u)
    return np.where(np.isclose(np.abs(u), 1.0), 0.5, v)


def _correlate(x: np.ndarray, h: np.ndarray, align: str) -> np.ndarray:
    """``y[n] = sum_j x[n + j - c] conj(h[j])`` along axis 0, with ``x`` zero
    outside its extent; ``c = L - 1`` (causal) or ``L - 1 - (L - 1) // 2``
    (centred)."""
    n, L = x.shape[0], len(h)
    g = np.conj(h)[::-1].reshape((L,) + (1,) * (x.ndim - 1))
    full = fftconvolve(x, g, axes=0)  # full[m] = sum_j x[m - L + 1 + j] conj(h[j])
    start = 0 if align == "causal" else (L - 1) // 2
    return full[start : start + n]
