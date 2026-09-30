"""Shared test constants and helpers."""

from dataclasses import dataclass

import numpy as np

import sonore as so

FS = 44100
# A lower rate for spectral and modulation tests. These check where modulation
# peaks and decay rates land, which doesn't need CD-quality audio; octave
# filterbank analyses at 44.1 kHz dominated the suite's runtime.
FAST = 16000
FAST_HI = 6000  # upper frequency limit comfortably below FAST's Nyquist


def peak_freq(s: so.Sound) -> float:
    X = np.abs(np.fft.rfft(s.data[:, 0]))
    return np.fft.rfftfreq(len(s), 1 / s.fs)[np.argmax(X)]


def dominant_freq(s: so.Sound) -> float:
    """Peak frequency with parabolic interpolation on a zero-padded spectrum."""
    x = s.data[:, 0] * np.hanning(len(s))
    n = 8 * len(x)
    X = np.abs(np.fft.rfft(x, n))
    k = np.argmax(X[1:-1]) + 1
    a, b, c = np.log(X[k - 1 : k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * s.fs / n


def cents(f, ref):
    return 1200 * np.log2(f / ref)


@dataclass(frozen=True)
class GaussianFilterbank(so.Filterbank):
    """Test-only, deliberately non-tight frame: ``n`` Gaussian filters with
    centers linearly spaced from 0 Hz to ``f_hi`` and standard deviation
    ``width`` times the spacing, truncated to zero beyond 3 SD (so a small
    ``width`` leaves true gaps in coverage)."""

    n: int = 8
    f_hi: float = 4000.0
    width: float = 0.7

    @property
    def n_filters(self) -> int:
        return self.n

    @property
    def cfs(self) -> np.ndarray:
        return np.linspace(0.0, self.f_hi, self.n)

    def response(self, freqs):
        sd = self.width * self.f_hi / (self.n - 1)
        u = (np.asarray(freqs, float)[:, None] - self.cfs[None, :]) / sd
        return np.where(np.abs(u) <= 3, np.exp(-0.5 * u**2), 0.0)
