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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ModulationFilterbank", "ConstantQModulationFilterbank", "OctaveModulationFilterbank"]


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
