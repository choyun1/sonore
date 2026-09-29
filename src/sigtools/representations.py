"""Spectral representations: magnitude spectra, STFTs, T-F masks, and
modulation spectra. All levels are in dB of *power* (``20*log10|X|``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import ShortTimeFFT, welch
from scipy.signal.windows import hann

from sigtools.sound import Sound
from sigtools.utils import amp_to_db, as_rng

__all__ = [
    "Spectrum",
    "long_term_spectrum",
    "STFT",
    "Mask",
    "ideal_binary_mask",
    "ideal_ratio_mask",
    "ModulationSpectrum",
]

_FLOOR_DB = -200.0


# ---------------------------------------------------------------- spectrum
@dataclass(frozen=True)
class Spectrum:
    """A magnitude spectrum: frequencies [Hz] and levels [dB]."""

    f: np.ndarray
    level: np.ndarray

    @classmethod
    def from_sound(cls, sound: Sound) -> Spectrum:
        """Exact FFT magnitude of the (mono mixdown of the) sound. Noisy for
        noise-like signals; see :func:`long_term_spectrum` for a smooth estimate."""
        x = sound.mono().data[:, 0]
        return cls(np.fft.rfftfreq(len(x), 1 / sound.fs), amp_to_db(np.fft.rfft(x), floor_db=_FLOOR_DB))

    def level_at(self, freqs: np.ndarray) -> np.ndarray:
        """Levels [dB] linearly interpolated at ``freqs``."""
        return np.interp(freqs, self.f, self.level)

    def relative(self) -> Spectrum:
        """Levels re the maximum (0 dB peak)."""
        return Spectrum(self.f, self.level - self.level.max())

    def smooth(self, fraction: float = 1 / 3) -> Spectrum:
        """Fractional-octave smoothing (power average over ``fraction`` octave)."""
        p = 10 ** (self.level / 10)
        c = np.concatenate([[0], np.cumsum(p)])
        half = 2 ** (fraction / 2)
        lo = np.searchsorted(self.f, self.f / half, side="left")
        hi = np.searchsorted(self.f, self.f * half, side="right")
        hi = np.maximum(hi, lo + 1)
        avg = (c[hi] - c[lo]) / (hi - lo)
        return Spectrum(self.f, 10 * np.log10(np.maximum(avg, 10 ** (_FLOOR_DB / 10))))

    def to_noise(self, duration: float, fs: float, rng=None, **kwargs) -> Sound:
        """Gaussian noise with this spectral shape."""
        from sigtools.generators import gaussian_noise

        return gaussian_noise(duration, fs, spectrum=self, rng=rng, **kwargs)

    def plot(self, ax=None, **kwargs):
        from sigtools.plotting import plot_spectrum

        return plot_spectrum(self, ax=ax, **kwargs)


def long_term_spectrum(sounds: Sound | Sequence[Sound], nperseg: int = 4096) -> Spectrum:
    """Long-term average spectrum via Welch's method, averaged over sounds
    (weighted by duration). This is the right input for speech-shaped noise."""
    if isinstance(sounds, Sound):
        sounds = [sounds]
    fs = sounds[0].fs
    total, weight = 0.0, 0
    for s in sounds:
        if s.fs != fs:
            s = s.resample(fs)
        f, p = welch(s.mono().data[:, 0], fs, nperseg=min(nperseg, len(s)))
        total = total + p * len(s)
        weight += len(s)
    psd = total / weight
    return Spectrum(f, 10 * np.log10(np.maximum(psd, 10 ** (_FLOOR_DB / 10))))


# -------------------------------------------------------------------- STFT
class STFT:
    """Short-time Fourier transform (wraps :class:`scipy.signal.ShortTimeFFT`).

    ``data`` has shape ``(n_channels, n_freqs, n_frames)``. Resynthesis with
    :meth:`to_sound` is exact for an unmodified STFT.

    Parameters
    ----------
    win_dur
        Window length [s] (periodic Hann).
    hop_dur
        Hop [s]; defaults to a quarter window (75% overlap).
    """

    __array_ufunc__ = None

    def __init__(
        self,
        sound: Sound,
        win_dur: float = 20e-3,
        hop_dur: float | None = None,
        *,
        _data=None,
        _sft=None,
    ):
        self.fs = sound.fs if sound is not None else _sft.fs
        if _sft is None:
            nwin = int(round(win_dur * self.fs))
            hop = max(1, int(round((hop_dur if hop_dur is not None else win_dur / 4) * self.fs)))
            _sft = ShortTimeFFT(hann(nwin, sym=False), hop=hop, fs=self.fs, fft_mode="onesided")
        self.sft = _sft
        if _data is None:
            self.n_samples = len(sound)
            _data = _sft.stft(sound.data.T, axis=-1)
        self.data = _data

    @classmethod
    def _from(cls, template: STFT, data: np.ndarray) -> STFT:
        new = cls(None, _data=data, _sft=template.sft)
        new.n_samples = template.n_samples
        return new

    def __repr__(self) -> str:
        c, f, t = self.data.shape
        win, hop = self.sft.m_num / self.fs * 1e3, self.sft.hop / self.fs * 1e3
        return f"STFT({f} freqs x {t} frames, {c} ch, win {win:.1f} ms, hop {hop:.1f} ms)"

    @property
    def f(self) -> np.ndarray:
        return self.sft.f

    @property
    def t(self) -> np.ndarray:
        """Frame center times [s]."""
        return self.sft.t(self.n_samples)

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self.data, floor_db=_FLOOR_DB)

    def __mul__(self, other):
        m = other.values if isinstance(other, Mask) else other
        return STFT._from(self, self.data * m)

    __rmul__ = __mul__

    def to_sound(self) -> Sound:
        """Inverse STFT (least-squares overlap-add)."""
        x = self.sft.istft(self.data, k1=self.n_samples)
        return Sound(np.real(x).T, self.fs)

    def griffin_lim(self, n_iter: int = 100, momentum: float = 0.99, rng=None) -> Sound:
        """Reconstruct a signal from the magnitude only (fast Griffin-Lim,
        Perraudin et al., 2013). ``momentum=0`` gives classic Griffin-Lim."""
        mag = self.magnitude
        rng = as_rng(rng)
        c = mag * np.exp(2j * np.pi * rng.random(mag.shape))
        t_prev = c
        for _ in range(n_iter):
            x = self.sft.istft(c, k1=self.n_samples)
            t = self.sft.stft(np.real(x), axis=-1)
            c = t + momentum * (t - t_prev)
            t_prev = t
            c = mag * np.exp(1j * np.angle(c))
        return STFT._from(self, c).to_sound()

    def plot(self, ax=None, channel: int = 0, **kwargs):
        from sigtools.plotting import plot_stft

        return plot_stft(self, ax=ax, channel=channel, **kwargs)


# ------------------------------------------------------------------- masks
@dataclass(frozen=True)
class Mask:
    """A time-frequency mask (binary or soft) aligned with an STFT."""

    values: np.ndarray
    t: np.ndarray
    f: np.ndarray

    def __mul__(self, other):
        if isinstance(other, STFT):
            return other * self
        return NotImplemented

    def plot(self, ax=None, channel: int = 0, **kwargs):
        from sigtools.plotting import plot_mask

        return plot_mask(self, ax=ax, channel=channel, **kwargs)


def ideal_binary_mask(target: STFT, masker: STFT, lc_db: float = 0.0) -> Mask:
    """1 where the target exceeds the masker by more than ``lc_db`` (local SNR criterion)."""
    return Mask((target.db - masker.db > lc_db).astype(float), target.t, target.f)


def ideal_ratio_mask(target: STFT, masker: STFT, beta: float = 0.5) -> Mask:
    """``(|T|^2 / (|T|^2 + |M|^2))**beta``."""
    pt, pm = target.magnitude**2, masker.magnitude**2
    with np.errstate(invalid="ignore", divide="ignore"):
        irm = np.nan_to_num((pt / (pt + pm)) ** beta)
    return Mask(irm, target.t, target.f)


# ------------------------------------------------------ modulation spectrum
class ModulationSpectrum:
    """2-D Fourier transform of a dB spectrogram (Singh & Theunissen, 2003).

    Temporal modulation [Hz] on one axis, spectral modulation [cycles/kHz] on
    the other. Spectral modulations are w.r.t. *linear* frequency here.
    """

    def __init__(self, stft: STFT, channel: int = 0):
        d = stft.db[channel]
        d = d - d.mean()
        F = np.fft.fft2(d)
        df = stft.f[1] - stft.f[0]
        dt = stft.sft.hop / stft.fs
        w_f = np.fft.fftfreq(d.shape[0], d=df / 1000)
        w_t = np.fft.fftfreq(d.shape[1], d=dt)
        keep = w_f >= 0
        self.w_f = w_f[keep]
        self.w_t = np.fft.fftshift(w_t)
        self.level = amp_to_db(np.fft.fftshift(F[keep], axes=1), floor_db=_FLOOR_DB)

    def plot(self, ax=None, **kwargs):
        from sigtools.plotting import plot_modulation_spectrum

        return plot_modulation_spectrum(self, ax=ax, **kwargs)
