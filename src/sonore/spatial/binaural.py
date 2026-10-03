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

from sonore.core.processing import pad
from sonore.core.sound import Sound
from sonore.core.units import dB
from sonore.core.utils import n_samples, time_axis
from sonore.frames.filterbank import ERBFilterbank
from sonore.sources.waveforms import gaussian_noise
from sonore.views.view import View

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
    left, right = pad([left - ild / 2 * dB, right + ild / 2 * dB])
    return Sound.from_channels(left, right)


def _fractional_impulse(delay: float, n: int, half_width: int = 32) -> np.ndarray:
    """Hann-windowed sinc centered at ``delay`` samples (exact impulse if integer)."""
    impulse = np.zeros(n)
    if abs(delay - round(delay)) < 1e-9:
        impulse[int(round(delay))] = 1.0
        return impulse
    samples = np.arange(n)
    offset = samples - delay
    window = np.where(np.abs(offset) < half_width, 0.5 * (1 + np.cos(np.pi * offset / half_width)), 0.0)
    return np.sinc(offset) * window


def simple_bir(fs: float, itd: float = 0.0, ild: float = 0.0, half_width: int = 32) -> Sound:
    """A binaural impulse response with only an ITD [s] and ILD [dB].

    Each ear is a (possibly fractionally delayed) unit impulse scaled by
    ``±ild/2`` dB. Fractional delays use a windowed sinc, so the IR starts
    ``half_width`` samples late whenever the ITD isn't a whole number of samples.
    """
    delay_samples = abs(itd) * fs
    fractional = abs(delay_samples - round(delay_samples)) > 1e-9
    offset = half_width if fractional else 0
    length = int(np.ceil(delay_samples)) + 2 * offset + 1
    lead, lag = (
        _fractional_impulse(offset, length, half_width),
        _fractional_impulse(offset + delay_samples, length, half_width),
    )
    left, right = (lag, lead) if itd > 0 else (lead, lag)
    ear_gain = 10 ** (ild / 40)
    return Sound(np.column_stack([left / ear_gain, right * ear_gain]), fs)


@dataclass(frozen=True)
class InterauralCues(View):
    """Short-time interaural cues. Arrays are ``(n_windows,)`` for broadband
    analysis or ``(n_windows, n_bands)`` per band. Silent time windows are NaN.
    A view: it keeps a few numbers per time window and drops the sounds."""

    discards = (
        "InterauralCues keeps only the time and level differences and the coherence between the ears in "
        "each time window: it discards the sounds themselves."
    )

    t: np.ndarray
    itd: np.ndarray
    ild: np.ndarray
    iac: np.ndarray
    corr0: np.ndarray
    cfs: np.ndarray | None = None

    def plot(self, ax=None, **kwargs):
        from sonore.plotting import plot_interaural_cues

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
    cross-correlation, refined by parabolic interpolation. ``iac`` is that
    peak's height (interaural coherence); ``corr0`` is the zero-lag
    correlation, which can be negative (use it for Oscor/Phasewarp).
    Time windows more than ``silence_db`` below the loudest one are NaN. If a
    ``filterbank`` is given, cues are computed per band.
    """
    if sound.n_channels != 2:
        raise ValueError("interaural_cues needs a 2-channel sound")
    fs = sound.fs
    n_win = n_samples(win_dur, fs)
    hop = max(1, n_samples(hop_dur if hop_dur is not None else win_dur / 2, fs))
    max_lag = int(np.ceil(max_itd * fs))

    if filterbank is None:
        bands = sound.data[:, None, :]  # (n, 1, 2)
        cfs = None
    else:
        bands = filterbank.analyze(sound).data  # (n, B, 2)
        cfs = filterbank.cfs

    window = hann(n_win, sym=False)
    segments = sliding_window_view(bands, n_win, axis=0)[::hop]  # (F, B, 2, n_win)
    segments = segments * window
    left, right = segments[:, :, 0, :], segments[:, :, 1, :]
    n_fft = 1 << int(np.ceil(np.log2(2 * n_win)))
    xcorr = np.fft.irfft(np.fft.rfft(left, n_fft) * np.conj(np.fft.rfft(right, n_fft)), n_fft)
    lags = np.r_[0 : max_lag + 1, -max_lag:0]
    xcorr = xcorr[..., lags]  # xcorr[tau] = sum_n left[n+tau] right[n]; peak at tau>0 means left lags
    left_energy, right_energy = np.sum(left**2, -1), np.sum(right**2, -1)
    norm = np.sqrt(left_energy * right_energy)
    # undo the taper of the windowed CCF: E[xcorr(tau)] = R_LR(tau) * (window*window)(tau)
    window_acf = np.correlate(window, window, "full")[n_win - 1 :]
    taper = window_acf[np.abs(lags)] / window_acf[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_xcorr = xcorr / norm[..., None] / taper
        ild = 10 * np.log10(right_energy / left_energy)

    peak_idx = np.argmax(norm_xcorr, axis=-1)
    iac = np.take_along_axis(norm_xcorr, peak_idx[..., None], -1)[..., 0]
    # parabolic interpolation around the peak (skip at the lag-range edges)
    before_idx, after_idx = (peak_idx - 1) % len(lags), (peak_idx + 1) % len(lags)
    corr_before = np.take_along_axis(norm_xcorr, before_idx[..., None], -1)[..., 0]
    corr_after = np.take_along_axis(norm_xcorr, after_idx[..., None], -1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = 0.5 * (corr_before - corr_after) / (corr_before - 2 * iac + corr_after)
    peak_lag = lags[peak_idx].astype(float)
    refinable = (np.abs(lags[peak_idx]) < max_lag) & np.isfinite(delta)
    delta = np.where(refinable, np.clip(delta, -0.5, 0.5), 0.0)
    itd = (peak_lag + delta) / fs
    iac = np.minimum(iac - 0.25 * (corr_before - corr_after) * delta, 1.0)

    corr0 = norm_xcorr[..., 0]
    energy = left_energy + right_energy
    with np.errstate(divide="ignore"):
        level_db = 10 * np.log10(energy / np.max(energy, axis=0, keepdims=True))
    silent = level_db < silence_db
    itd, ild, iac, corr0 = (np.where(silent, np.nan, cue) for cue in (itd, ild, iac, corr0))

    t = (np.arange(segments.shape[0]) * hop + n_win / 2) / fs
    if filterbank is None:
        itd, ild, iac, corr0 = itd[:, 0], ild[:, 0], iac[:, 0], corr0[:, 0]
    return InterauralCues(t, itd, ild, iac, corr0, cfs)


def oscor(duration: float, fs: float, f_mod: float, rng=None, **noise_kwargs) -> Sound:
    """Oscillating-correlation noise (Siveke et al., 2008): the interaural
    correlation follows ``sin(2*pi*f_mod*t)``."""
    noise = gaussian_noise(duration, fs, n_channels=2, rng=rng, **noise_kwargs).data
    t = time_axis(len(noise), fs)
    right = np.sin(2 * np.pi * f_mod * t) * noise[:, 0] + np.cos(2 * np.pi * f_mod * t) * noise[:, 1]
    return Sound(np.column_stack([noise[:, 0], right]), fs).normalize()


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
