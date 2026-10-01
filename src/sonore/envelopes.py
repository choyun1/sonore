"""Envelopes: the slowly varying amplitudes that shape sounds.

An envelope is not a sound. It is non-negative, it usually varies slowly
enough to live at a low sampling rate, and you don't listen to it: you apply it
to a sound. The types reflect that:

- :class:`Envelope` is one envelope. ``Envelope * Sound`` modulates the sound.
- :class:`Envelopes` is one envelope per frequency band of a filterbank, i.e. a
  spectrotemporal envelope. This is conceptually the same thing as a
  **cochleagram** (the envelope of each cochlear-filter output over time);
  here the "cochlea" is whichever :class:`~sonore.frames.Filterbank` (by
  default a :class:`~sonore.filterbank.CosineFilterbank`) produced it.
  ``Envelopes * Subbands`` modulates each band.

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

from sonore.core.utils import amp_to_db

if TYPE_CHECKING:
    from sonore.core.sound import Sound
    from sonore.frames import Filterbank
    from sonore.representations import ModulationSpectrum

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
        from sonore.core.sound import Sound

        if isinstance(other, Sound):
            return Sound(other.data * self._values_for(other), other.fs)
        o = self._other(other)
        return NotImplemented if o is NotImplemented else Envelope(self._data * o, self.fs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._other(other)
        return NotImplemented if o is NotImplemented else Envelope(self._data / o, self.fs)

    def __rtruediv__(self, other):
        from sonore.core.sound import Sound

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

    Envelopes derived from padded Subbands carry the same zero-padding
    (``pad`` samples at each end, hidden from :attr:`data`), so that
    modulating and re-synthesizing bands can't wrap around. Envelopes made
    from scratch (e.g. a rendered ripple pattern) have ``pad=0`` and are
    zero outside their own extent when combined with padded bands.
    """

    __array_ufunc__ = None

    def __init__(self, data, fs: float, filterbank: Filterbank, pad: int = 0):
        arr = np.array(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3 or arr.shape[1] != filterbank.n_filters:
            raise ValueError(
                f"expected shape (n_samples, {filterbank.n_filters}, n_channels), got {arr.shape}"
            )
        if 2 * pad >= arr.shape[0]:
            raise ValueError("padding is longer than the data")
        arr = _nonnegative(arr)
        arr.flags.writeable = False
        self._full, self.fs, self.filterbank, self.pad = arr, fs, filterbank, int(pad)

    @property
    def data(self) -> np.ndarray:
        """Envelopes without the padding, shape ``(n_samples, n_bands, n_channels)``."""
        return self._full[self.pad : self._full.shape[0] - self.pad]

    @property
    def cfs(self) -> np.ndarray:
        return self.filterbank.cfs

    def __len__(self) -> int:
        """Number of bands (including the two edge filters)."""
        return self._full.shape[1]

    def __getitem__(self, i: int) -> Envelope:
        return Envelope(self.data[:, i, :], self.fs)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def n_samples(self) -> int:
        return self._full.shape[0] - 2 * self.pad

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.fs

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self.data)

    def __repr__(self) -> str:
        n, b, c = self.data.shape
        return f"Envelopes({b} bands, {n / self.fs:.3f} s, {self.fs:g} Hz, {c} ch)"

    def _new(self, full, fs=None, pad=None) -> Envelopes:
        return Envelopes(
            full, self.fs if fs is None else fs, self.filterbank, self.pad if pad is None else pad
        )

    def lowpass(self, cutoff: float, order: int = 4) -> Envelopes:
        return self._new(_lowpass(self._full, cutoff, self.fs, order))

    def resample(self, fs: float) -> Envelopes:
        if fs == self.fs:
            return self
        ratio = Fraction(fs / self.fs).limit_denominator(10000)
        # extend the front padding so it maps to a whole number of samples at
        # the new rate: the inner signal then starts exactly on a sample
        extra = (-self.pad) % ratio.denominator
        full = np.pad(self._full, ((extra, 0), (0, 0), (0, 0))) if extra else self._full
        full = _resample(full, self.fs, fs)
        pad = (self.pad + extra) * ratio.numerator // ratio.denominator
        total = pad + int(round(self.n_samples * float(ratio))) + pad
        if full.shape[0] < total:
            full = np.pad(full, [(0, total - full.shape[0])] + [(0, 0)] * 2, mode="edge")
        return self._new(full[:total], fs, pad)

    def without_edges(self) -> Envelopes:
        """Zero the lowpass and highpass edge bands (which lie outside
        ``f_lo..f_hi``), keeping the band count unchanged. A bank built with
        ``edges=False`` has none, and its envelopes are returned as they are."""
        if getattr(self.filterbank, "edges", True) is False:
            return self
        full = self._full.copy()
        full[:, [0, -1], :] = 0.0
        return self._new(full)

    def _on_grid(self, fs: float, n_inner: int, pad: int) -> np.ndarray:
        """These envelopes on another time grid (``n_inner`` samples at ``fs``
        with ``pad`` samples each side), zero outside their own extent."""
        full, own_pad = self._full, self.pad
        if fs != self.fs:
            n_up = int(round(full.shape[0] * fs / self.fs))
            full = _upsample_to(full, self.fs, n_up, fs)
            own_pad = int(round(self.pad * fs / self.fs))
        out = np.zeros((n_inner + 2 * pad,) + full.shape[1:])
        shift = pad - own_pad  # where our sample 0 lands on the target grid
        lo, hi = max(0, shift), min(out.shape[0], shift + full.shape[0])
        out[lo:hi] = full[lo - shift : hi - shift]
        return out

    def __mul__(self, other):
        from sonore.filterbank import Subbands

        if isinstance(other, Subbands):
            if len(other) != len(self):
                raise ValueError(f"band counts differ ({len(self)} vs {len(other)})")
            _check_duration(self.n_samples, self.fs, other.n_samples, other.fs)
            pad = max(other.pad, int(round(self.pad * other.fs / self.fs)))
            extra = pad - other.pad
            bands = np.pad(other._full, ((extra, extra), (0, 0), (0, 0))) if extra else other._full
            env = self._on_grid(other.fs, other.n_samples, pad)
            return Subbands(bands * env, other.fs, other.filterbank, pad=pad)
        if isinstance(other, Envelopes):
            if other.fs != self.fs or other.n_samples != self.n_samples or len(other) != len(self):
                raise ValueError("Envelopes must share fs, length and band count")
            pad = max(self.pad, other.pad)
            a = self._on_grid(self.fs, self.n_samples, pad)
            b = other._on_grid(self.fs, self.n_samples, pad)
            return self._new(a * b, pad=pad)
        if isinstance(other, Envelope):
            if other.fs != self.fs or len(other) != self.n_samples:
                raise ValueError("Envelope must share fs and length")
            gain = np.zeros((self._full.shape[0], 1, other.data.shape[1]))
            gain[self.pad : self.pad + self.n_samples, 0, :] = other.data
            return self._new(self._full * gain)
        if isinstance(other, numbers.Real) and not isinstance(other, bool):
            return self._new(self._full * float(other))
        return NotImplemented

    __rmul__ = __mul__

    def modulation_spectrum(self, scale: str = "linear", drop_edges: bool = True) -> ModulationSpectrum:
        """2-D modulation spectrum (temporal Hz x spectral cycles per scale unit).

        ``scale="db"`` transforms log envelopes. The edge bands are dropped by
        default, since they aren't evenly spaced with the others.
        """
        from sonore.representations import ModulationSpectrum

        fb = self.filterbank
        if getattr(fb, "spacing", None) is None or getattr(fb, "unit", None) is None:
            raise TypeError(
                "modulation_spectrum needs filters evenly spaced on a frequency scale (a filterbank "
                "with 'spacing' and 'unit', such as ERBFilterbank or OctaveFilterbank); "
                f"{type(fb).__name__} has none"
            )
        env = self.data.mean(axis=2)  # (n, B), channels averaged
        if drop_edges:
            env = env[:, 1:-1]
        if scale == "db":
            env = amp_to_db(env + 1e-12 * (env.max() or 1.0))
        elif scale != "linear":
            raise ValueError("scale must be 'linear' or 'db'")
        return ModulationSpectrum.from_array(
            env.T, dt=1 / self.fs, dx=fb.spacing, spectral_unit=f"cyc/{fb.unit}"
        )

    def plot(self, ax=None, **kwargs):
        from sonore.plotting import plot_envelopes

        return plot_envelopes(self, ax=ax, **kwargs)
