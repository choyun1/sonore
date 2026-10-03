"""Shared test constants and helpers."""

from dataclasses import dataclass

import numpy as np

import sonore as so
from sonore.frames.filterbank import FilterType

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
class GaussianType(FilterType):
    """Test-only filter type: Gaussians with standard deviation ``width``
    times the spacing of the bank's centers, truncated to zero beyond 3 SD
    (so a small ``width`` leaves true gaps in coverage), delayed by ``delay``
    [s]. A delay makes the response complex (conjugate-symmetric) with a
    value at Nyquist that is not real, which exercises the Nyquist rule (see
    ``Filterbank.rfft_response``)."""

    width: float = 0.7
    delay: float = 0.0

    def band_response(self, bank, freqs):
        centers = bank.band_cfs
        u = (np.asarray(freqs, float)[:, None] - centers[None, :]) / (self.width * (centers[1] - centers[0]))
        gaussian = np.where(np.abs(u) <= 3, np.exp(-0.5 * u**2), 0.0)
        if self.delay:
            return gaussian * np.exp(-2j * np.pi * np.asarray(freqs, float) * self.delay)[:, None]
        return gaussian


def GaussianFilterbank(n: int = 8, f_hi: float = 4000.0, width: float = 0.7, delay: float = 0.0):
    """Test-only, deliberately non-tight frame: ``n`` Gaussian filters
    (:class:`GaussianType`) with centers linearly spaced from 0 Hz to
    ``f_hi``, and no edge filters."""
    step = f_hi / (n - 1)
    knots = np.linspace(-step, f_hi + step, n + 2)
    return so.Filterbank("linear", knots, GaussianType(width, delay), edges=False)


def DelayedGaussianFilterbank(f_hi: float = 4000.0, delay: float = 0.3e-3):
    """:func:`GaussianFilterbank` with a fractional-sample delay ``delay`` [s]."""
    return GaussianFilterbank(f_hi=f_hi, delay=delay)


# ------------------------------------------------ dense-matrix frame oracle
# docs/design/frames/frames.md: small dense matrices, built from the fast path itself
# by analyzing unit impulses, so the oracle tests what the code actually does.


def coef_matrix(coefs) -> np.ndarray:
    """Coefficients as a matrix, one column per channel: ``Subbands`` in
    (time, band) order including padding, ``STFT`` and ``TVSTFT`` in
    (freq, time window) order."""
    if isinstance(coefs, so.Subbands):
        return coefs._full.reshape(-1, coefs._full.shape[2])
    return np.moveaxis(coefs.data, 0, -1).reshape(-1, coefs.data.shape[0])


def coef_weights(frame, coefs) -> np.ndarray:
    """The coefficient-norm weight (``bin_weights``) of each row of :func:`coef_matrix`."""
    if isinstance(frame, (so.GaborFrame, so.TVGaborFrame)):
        return np.repeat(frame.bin_weights(coefs.fs), coefs.data.shape[2])
    return np.ones(coef_matrix(coefs).shape[0])


def dense_operator(frame, n, fs, pad=None):
    """Analysis matrix ``T`` (coefficients x ``n``) and coefficient-norm weights, from
    ``frame.analyze`` of the ``n`` unit impulses (``pad`` for filterbanks)."""
    kw = {} if pad is None else {"pad": pad}
    eye, batch = np.eye(n), 64  # Sound caps the channel count (it suspects channels-first data)
    parts = [frame.analyze(so.Sound(eye[:, i : i + batch], fs), **kw) for i in range(0, n, batch)]
    return np.hstack([coef_matrix(c) for c in parts]), coef_weights(frame, parts[0])


def weighted_frame_operator(T, w) -> np.ndarray:
    """``S = Re(T^H W T)``, the frame operator on real signals."""
    return np.real(T.conj().T @ (w[:, None] * T))


def canonical_lstsq(T, w, c) -> np.ndarray:
    """The real ``x`` minimizing ``sum w |T x - c|^2``: canonical-dual synthesis."""
    r = np.sqrt(w)[:, None] * T
    b = np.sqrt(w) * c
    A = np.vstack([r.real, r.imag]) if np.iscomplexobj(r) else r
    y = np.concatenate([b.real, b.imag]) if np.iscomplexobj(b) else b
    return np.linalg.lstsq(A, y, rcond=None)[0]


def toy_hrirs(fs=48000, taps=256):
    """Spherical-head-ish toy set: ITD from Woodworth, ILD from azimuth."""
    az = np.arange(0, 360, 10)
    el = np.arange(-40, 91, 20)
    hcc = np.array([(100, e, a) for e in el for a in az])
    pos = np.column_stack(so.hcc_to_rect(*hcc.T))
    theta = np.radians(hcc[:, 2])
    lat = np.arcsin(np.sin(theta) * np.cos(np.radians(hcc[:, 1])))
    itd = 0.0875 / 343 * (lat + np.sin(lat))
    irs = np.zeros((len(hcc), 2, taps))
    for i, d in enumerate(itd):
        base = 20
        irs[i, 0, base + int(round(max(d, 0) * fs))] = 10 ** (-np.sin(lat[i]) * 5 / 20)
        irs[i, 1, base + int(round(max(-d, 0) * fs))] = 10 ** (np.sin(lat[i]) * 5 / 20)
    return so.HRIRSet(irs, pos, fs), itd
