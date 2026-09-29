"""Binaural manipulation and analysis.

Sign convention throughout: **positive ITD = right ear leads** and **positive
ILD = right ear louder**, i.e. positive values point to the right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert
from scipy.signal.windows import hann

from sigtools.filterbank import ERBFilterbank
from sigtools.generators import gaussian_noise
from sigtools.processing import pad
from sigtools.sound import Sound
from sigtools.utils import time_axis

__all__ = [
    "apply_itd_ild",
    "simple_bir",
    "InterauralCues",
    "interaural_cues",
    "oscor",
    "phasewarp",
]


def apply_itd_ild(sound: Sound, itd: float = 0.0, ild: float = 0.0) -> Sound:
    """Impose an ITD [s] and ILD [dB] on a mono (or diotic stereo) sound.

    The lagging ear is delayed by ``|itd|`` (fractional delays are exact up
    to band-limiting); the ILD is split symmetrically, ``±ild/2`` per ear.
    """
    left, right = (sound.to_stereo().channel(i) for i in (0, 1))
    if itd > 0:
        left = left.delay(itd)
    elif itd < 0:
        right = right.delay(-itd)
    left, right = pad([left.gain_db(-ild / 2), right.gain_db(ild / 2)])
    return Sound.from_channels(left, right)


def _fractional_impulse(delay: float, n: int, half_width: int = 32) -> np.ndarray:
    """Hann-windowed sinc centered at ``delay`` samples (exact impulse if integer)."""
    h = np.zeros(n)
    if abs(delay - round(delay)) < 1e-9:
        h[int(round(delay))] = 1.0
        return h
    k = np.arange(n)
    u = k - delay
    win = np.where(np.abs(u) < half_width, 0.5 * (1 + np.cos(np.pi * u / half_width)), 0.0)
    return np.sinc(u) * win


def simple_bir(fs: float, itd: float = 0.0, ild: float = 0.0, half_width: int = 32) -> Sound:
    """A binaural impulse response with only an ITD [s] and ILD [dB].

    Each ear is a (possibly fractionally delayed) unit impulse scaled by
    ``±ild/2`` dB. Fractional delays use a windowed sinc, so the IR starts
    ``half_width`` samples late whenever the ITD isn't a whole number of samples.
    """
    d = abs(itd) * fs
    frac = abs(d - round(d)) > 1e-9
    offset = half_width if frac else 0
    n = int(np.ceil(d)) + 2 * offset + 1
    lead, lag = _fractional_impulse(offset, n, half_width), _fractional_impulse(offset + d, n, half_width)
    left, right = (lag, lead) if itd > 0 else (lead, lag)
    g = 10 ** (ild / 40)
    return Sound(np.column_stack([left / g, right * g]), fs)


@dataclass(frozen=True)
class InterauralCues:
    """Short-time interaural cues. Arrays are ``(n_frames,)`` for broadband
    analysis or ``(n_frames, n_bands)`` per band. Silent frames are NaN."""

    t: np.ndarray
    itd: np.ndarray
    ild: np.ndarray
    iac: np.ndarray
    cfs: np.ndarray | None = None

    def plot(self, ax=None, **kwargs):
        from sigtools.plotting import plot_interaural_cues

        return plot_interaural_cues(self, ax=ax, **kwargs)


def interaural_cues(
    sound: Sound,
    win_dur: float = 20e-3,
    hop_dur: float | None = None,
    max_itd: float = 1e-3,
    filterbank: ERBFilterbank | None = None,
    silence_db: float = -40.0,
) -> InterauralCues:
    """Windowed ITD, ILD and interaural coherence.

    ITD is the lag (within ``±max_itd``) of the peak of the normalized
    cross-correlation, refined by parabolic interpolation. IAC is that peak's
    height. Frames more than ``silence_db`` below the loudest frame are NaN.
    If a ``filterbank`` is given, cues are computed per band.
    """
    if sound.n_channels != 2:
        raise ValueError("interaural_cues needs a 2-channel sound")
    fs = sound.fs
    nwin = int(round(win_dur * fs))
    hop = max(1, int(round((hop_dur if hop_dur is not None else win_dur / 2) * fs)))
    max_lag = int(np.ceil(max_itd * fs))

    if filterbank is None:
        x = sound.data[:, None, :]  # (n, 1, 2)
        cfs = None
    else:
        x = filterbank.analyze(sound).data  # (n, B, 2)
        cfs = filterbank.cfs

    w = hann(nwin, sym=False)
    frames = sliding_window_view(x, nwin, axis=0)[::hop]  # (F, B, 2, nwin)
    frames = frames * w
    L, R = frames[:, :, 0, :], frames[:, :, 1, :]
    nfft = 1 << int(np.ceil(np.log2(2 * nwin)))
    cc = np.fft.irfft(np.fft.rfft(L, nfft) * np.conj(np.fft.rfft(R, nfft)), nfft)
    lags = np.r_[0 : max_lag + 1, -max_lag:0]
    cc = cc[..., lags]  # c[tau] = sum_n L[n+tau] R[n]; peak at tau>0 means L lags
    eL, eR = np.sum(L**2, -1), np.sum(R**2, -1)
    norm = np.sqrt(eL * eR)
    # undo the taper of the windowed CCF: E[cc(tau)] = R_LR(tau) * (w*w)(tau)
    ww = np.correlate(w, w, "full")[nwin - 1 :]
    taper = ww[np.abs(lags)] / ww[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = cc / norm[..., None] / taper
        ild = 10 * np.log10(eR / eL)

    k = np.argmax(ncc, axis=-1)
    iac = np.take_along_axis(ncc, k[..., None], -1)[..., 0]
    # parabolic interpolation around the peak (skip at the lag-range edges)
    km, kp = (k - 1) % len(lags), (k + 1) % len(lags)
    y0 = np.take_along_axis(ncc, km[..., None], -1)[..., 0]
    y2 = np.take_along_axis(ncc, kp[..., None], -1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = 0.5 * (y0 - y2) / (y0 - 2 * iac + y2)
    lag = lags[k].astype(float)
    ok = (np.abs(lags[k]) < max_lag) & np.isfinite(delta)
    delta = np.where(ok, np.clip(delta, -0.5, 0.5), 0.0)
    itd = (lag + delta) / fs
    iac = np.minimum(iac - 0.25 * (y0 - y2) * delta, 1.0)

    energy = eL + eR
    with np.errstate(divide="ignore"):
        e_db = 10 * np.log10(energy / np.max(energy, axis=0, keepdims=True))
    silent = e_db < silence_db
    itd, ild, iac = (np.where(silent, np.nan, a) for a in (itd, ild, iac))

    t = (np.arange(frames.shape[0]) * hop + nwin / 2) / fs
    if filterbank is None:
        itd, ild, iac = itd[:, 0], ild[:, 0], iac[:, 0]
    return InterauralCues(t, itd, ild, iac, cfs)


def oscor(duration: float, fs: float, f_mod: float, rng=None, **noise_kwargs) -> Sound:
    """Oscillating-correlation noise (Siveke et al., 2008): the interaural
    correlation follows ``sin(2*pi*f_mod*t)``."""
    n = gaussian_noise(duration, fs, n_channels=2, rng=rng, **noise_kwargs).data
    t = time_axis(len(n), fs)
    right = np.sin(2 * np.pi * f_mod * t) * n[:, 0] + np.cos(2 * np.pi * f_mod * t) * n[:, 1]
    return Sound(np.column_stack([n[:, 0], right]), fs).normalize()


def phasewarp(duration: float, fs: float, f_mod: float, rng=None, **noise_kwargs) -> Sound:
    """Phasewarp noise (Siveke et al., 2008): every component's interaural
    phase difference rotates at ``f_mod`` Hz, so the IPD sweeps through 360°
    and the zero-lag interaural correlation follows ``cos(2*pi*f_mod*t)``.

    Implemented as a single-sideband frequency shift of the left-ear noise.
    """
    left = gaussian_noise(duration, fs, rng=rng, **noise_kwargs).data[:, 0]
    t = time_axis(len(left), fs)
    right = np.real(hilbert(left) * np.exp(2j * np.pi * f_mod * t))
    return Sound(np.column_stack([left, right]), fs).normalize()
