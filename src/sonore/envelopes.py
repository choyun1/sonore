"""Envelopes: the slowly varying amplitudes that shape sounds.

An envelope is not a sound. It is non-negative, it usually varies slowly
enough to live at a low sampling rate, and you don't listen to it: you apply it
to a sound. The types reflect that:

- :class:`Envelope` is one envelope. ``Envelope * Sound`` modulates the sound.
- :class:`Envelopes` is one envelope per frequency band of a filterbank, i.e. a
  spectrotemporal envelope. This is conceptually the same thing as a
  **cochleagram** (the envelope of each cochlear-filter output over time);
  here the "cochlea" is whichever :class:`~sonore.filterbank.CosineFilterbank`
  produced it. ``Envelopes * Subbands`` modulates each band.

The Hilbert decomposition of a band is then literal::

    band == band.envelope() * (band / band.envelope())      # envelope x fine structure
    sb == sb.envelopes() * sb.tfs()                          # for every band at once

Envelopes resampled to a lower rate are upsampled automatically (band-limited
polyphase interpolation, clipped at zero) when applied to a sound.
"""

from __future__ import annotations

import numbers
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from sonore.utils import amp_to_db

if TYPE_CHECKING:
    from sonore.filterbank import CosineFilterbank
    from sonore.representations import ModulationSpectrum
    from sonore.sound import Sound

__all__ = ["Envelope", "Envelopes"]


def _nonnegative(arr: np.ndarray) -> np.ndarray:
    """Clip round-off negatives; reject genuinely negative envelopes."""
    if arr.size and np.min(arr) < 0:
        scale = np.max(np.abs(arr)) or 1.0
        if np.min(arr) < -1e-9 * scale:
            raise ValueError("envelopes must be non-negative")
        arr = np.maximum(arr, 0.0)
    return arr


def _lowpass(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    sos = butter(order, cutoff, fs=fs, output="sos")
    return np.maximum(sosfiltfilt(sos, data, axis=0), 0.0)


def _resample(data: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    ratio = Fraction(fs_new / fs_old).limit_denominator(10000)
    # padtype="line": extend the envelope at its ends instead of padding with
    # zeros, which would drag the first and last few ms toward zero
    return np.maximum(resample_poly(data, ratio.numerator, ratio.denominator, axis=0, padtype="line"), 0.0)


def _upsample_to(data: np.ndarray, fs: float, n: int, fs_new: float) -> np.ndarray:
    """Envelope (axis 0) onto ``n`` samples at ``fs_new``.

    Band-limited polyphase resampling (an envelope that was lowpassed and
    downsampled is band-limited, so this is the accurate choice), clipped at
    zero, then trimmed or edge-padded to exactly ``n`` samples.
    """
    ratio = Fraction(fs_new / fs).limit_denominator(10000)
    if abs(float(ratio) - fs_new / fs) > 1e-9 * fs_new / fs:  # irrational ratio: fall back
        t_old, t_new = np.arange(data.shape[0]) / fs, np.arange(n) / fs_new
        flat = data.reshape(data.shape[0], -1)
        out = np.column_stack([np.interp(t_new, t_old, col) for col in flat.T])
        return out.reshape((n,) + data.shape[1:])
    out = np.maximum(resample_poly(data, ratio.numerator, ratio.denominator, axis=0, padtype="line"), 0.0)
    if out.shape[0] >= n:
        return out[:n]
    pad = [(0, n - out.shape[0])] + [(0, 0)] * (data.ndim - 1)
    return np.pad(out, pad, mode="edge")


def _check_duration(n_env: int, fs_env: float, n: int, fs: float) -> None:
    if abs(n_env / fs_env - n / fs) > 1.5 / fs_env:
        raise ValueError(f"durations differ ({n_env / fs_env:.4f} s envelope vs {n / fs:.4f} s signal)")


class Envelope:
    """A single (possibly multichannel) envelope, shape ``(n_samples, n_channels)``.

    Arithmetic: ``*``, ``/`` and ``+`` with numbers and other Envelopes (so
    ``1 + 0.5 * env`` works), and ``env * snd`` / ``snd * env`` modulate a
    :class:`~sonore.Sound`. ``snd / env`` divides the envelope out of a sound.
    """

    __array_ufunc__ = None

    def __init__(self, data, fs: float):
        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"envelope data must be 1-D or 2-D, got shape {arr.shape}")
        arr = _nonnegative(arr)
        arr.flags.writeable = False
        self._data, self.fs = arr, fs

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def duration(self) -> float:
        return len(self) / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self)) / self.fs

    @property
    def db(self) -> np.ndarray:
        """Envelope in dB (``20*log10``)."""
        return amp_to_db(self._data)

    def __repr__(self) -> str:
        return f"Envelope({self.duration:.3f} s, {self.fs:g} Hz, {self._data.shape[1]} ch)"

    def lowpass(self, cutoff: float, order: int = 4) -> Envelope:
        """Zero-phase lowpass (result clipped at 0)."""
        return Envelope(_lowpass(self._data, cutoff, self.fs, order), self.fs)

    def resample(self, fs: float) -> Envelope:
        return self if fs == self.fs else Envelope(_resample(self._data, self.fs, fs), fs)

    def _values_for(self, sound: Sound) -> np.ndarray:
        _check_duration(len(self), self.fs, len(sound), sound.fs)
        if self.fs == sound.fs and len(self) == len(sound):
            return self._data
        return _upsample_to(self._data, self.fs, len(sound), sound.fs)

    def _other(self, other):
        if isinstance(other, Envelope):
            if other.fs != self.fs or len(other) != len(self):
                raise ValueError("envelopes must share fs and length")
            return other._data
        if isinstance(other, numbers.Real) and not isinstance(other, bool):
            return float(other)
        return NotImplemented

    def __mul__(self, other):
        from sonore.sound import Sound

        if isinstance(other, Sound):
            return Sound(other.data * self._values_for(other), other.fs)
        o = self._other(other)
        return NotImplemented if o is NotImplemented else Envelope(self._data * o, self.fs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._other(other)
        return NotImplemented if o is NotImplemented else Envelope(self._data / o, self.fs)

    def __rtruediv__(self, other):
        from sonore.sound import Sound

        if isinstance(other, Sound):
            env = self._values_for(other)
            floor = 1e-12 * (np.max(env) or 1.0)
            return Sound(other.data / np.maximum(env, floor), other.fs)
        return NotImplemented

    def __add__(self, other):
        o = self._other(other)
        return NotImplemented if o is NotImplemented else Envelope(self._data + o, self.fs)

    __radd__ = __add__

    def plot(self, ax=None, **kwargs):
        from sonore.plotting import plot_envelope

        return plot_envelope(self, ax=ax, **kwargs)


class Envelopes:
    """One envelope per band of a filterbank: a spectrotemporal envelope.

    Conceptually this is a **cochleagram**: the envelope of each filter's
    output over time. Shape ``(n_samples, n_bands, n_channels)``; the bands
    include the filterbank's lowpass and highpass edge filters, as in
    :class:`~sonore.filterbank.Subbands`.

    ``env[i]`` is an :class:`Envelope`; ``env * subbands`` modulates each band;
    ``env.modulation_spectrum()`` gives its 2-D modulation spectrum on the
    filterbank's frequency scale (cycles/octave or cycles/ERB).
    """

    __array_ufunc__ = None

    def __init__(self, data, fs: float, filterbank: CosineFilterbank):
        arr = np.array(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3 or arr.shape[1] != filterbank.n_bands + 2:
            raise ValueError(
                f"expected shape (n_samples, {filterbank.n_bands + 2}, n_channels), got {arr.shape}"
            )
        arr = _nonnegative(arr)
        arr.flags.writeable = False
        self._data, self.fs, self.filterbank = arr, fs, filterbank

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def cfs(self) -> np.ndarray:
        return self.filterbank.cfs

    def __len__(self) -> int:
        """Number of bands (including the two edge filters)."""
        return self._data.shape[1]

    def __getitem__(self, i: int) -> Envelope:
        return Envelope(self._data[:, i, :], self.fs)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def n_samples(self) -> int:
        return self._data.shape[0]

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.fs

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self._data)

    def __repr__(self) -> str:
        n, b, c = self._data.shape
        return f"Envelopes({b} bands, {n / self.fs:.3f} s, {self.fs:g} Hz, {c} ch)"

    def _new(self, data, fs=None) -> Envelopes:
        return Envelopes(data, self.fs if fs is None else fs, self.filterbank)

    def lowpass(self, cutoff: float, order: int = 4) -> Envelopes:
        return self._new(_lowpass(self._data, cutoff, self.fs, order))

    def resample(self, fs: float) -> Envelopes:
        return self if fs == self.fs else self._new(_resample(self._data, self.fs, fs), fs)

    def without_edges(self) -> Envelopes:
        """Zero the lowpass and highpass edge bands (which lie outside
        ``f_lo..f_hi``), keeping the band count unchanged."""
        data = self._data.copy()
        data[:, [0, -1], :] = 0.0
        return self._new(data)

    def __mul__(self, other):
        from sonore.filterbank import Subbands

        if isinstance(other, Subbands):
            if len(other) != len(self):
                raise ValueError(f"band counts differ ({len(self)} vs {len(other)})")
            _check_duration(self.n_samples, self.fs, other.data.shape[0], other.fs)
            env = self._data
            if self.fs != other.fs or self.n_samples != other.data.shape[0]:
                env = _upsample_to(env, self.fs, other.data.shape[0], other.fs)
            return Subbands(other.data * env, other.fs, other.filterbank)
        if isinstance(other, Envelopes):
            if other._data.shape != self._data.shape or other.fs != self.fs:
                raise ValueError("Envelopes must share shape and fs")
            return self._new(self._data * other._data)
        if isinstance(other, Envelope):
            if other.fs != self.fs or len(other) != self.n_samples:
                raise ValueError("Envelope must share fs and length")
            return self._new(self._data * other.data[:, None, :])
        if isinstance(other, numbers.Real) and not isinstance(other, bool):
            return self._new(self._data * float(other))
        return NotImplemented

    __rmul__ = __mul__

    def modulation_spectrum(self, scale: str = "linear", drop_edges: bool = True) -> ModulationSpectrum:
        """2-D modulation spectrum (temporal Hz x spectral cycles per scale unit).

        ``scale="db"`` transforms log envelopes. The edge bands are dropped by
        default, since they aren't evenly spaced with the others.
        """
        from sonore.representations import ModulationSpectrum

        env = self._data.mean(axis=2)  # (n, B), channels averaged
        if drop_edges:
            env = env[:, 1:-1]
        if scale == "db":
            env = amp_to_db(env + 1e-12 * (env.max() or 1.0))
        elif scale != "linear":
            raise ValueError("scale must be 'linear' or 'db'")
        return ModulationSpectrum.from_array(
            env.T, dt=1 / self.fs, dx=self.filterbank.spacing, spectral_unit=f"cyc/{self.filterbank.unit}"
        )

    def plot(self, ax=None, **kwargs):
        from sonore.plotting import plot_envelopes

        return plot_envelopes(self, ax=ax, **kwargs)
