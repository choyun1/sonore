"""The :class:`Sound` container.

A ``Sound`` is an immutable ``(n_samples, n_channels)`` float array plus a
sampling rate. Every operation returns a new ``Sound``; nothing is modified in
place.
"""

from __future__ import annotations

import numbers
import warnings
from fractions import Fraction
from os import PathLike

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve, hilbert, resample_poly

from sonore.core.units import Decibels
from sonore.core.utils import amp_to_db, db_to_amp, rms, time_axis

__all__ = ["Sound", "load"]

_MAX_CHANNELS = 64


class Sound:
    """A sampled sound.

    Parameters
    ----------
    data
        Samples, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    fs
        Sampling rate [Hz].

    Notes
    -----
    Arithmetic works the way you'd expect for signals: ``a + b`` mixes,
    ``a * b`` multiplies sample-by-sample (e.g. an envelope), ``2 * a`` scales,
    and ``a + 6*dB`` / ``a - 3*dB`` change the level (``from sonore import dB``).
    Mono sounds broadcast against multichannel ones. Use :meth:`pad` /
    :func:`sonore.pad` to match lengths.

    Indexing with a slice selects by *time in seconds*: ``snd[0.1:0.5]``.
    """

    # Make numpy defer to our operators, so ``np.float64(2) * snd`` calls
    # ``Sound.__rmul__`` instead of trying to iterate over the Sound.
    __array_ufunc__ = None

    def __init__(self, data: ArrayLike, fs: float):
        arr = np.array(data, dtype=float)  # always copy
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"data must be 1-D or 2-D, got shape {arr.shape}")
        if arr.shape[1] > _MAX_CHANNELS:
            raise ValueError(
                f"data has shape {arr.shape}; expected (n_samples, n_channels). "
                "Did you pass channels-first data? Transpose it, or use "
                "Sound.from_channels(left, right, fs=...)."
            )
        if fs <= 0:
            raise ValueError("fs must be positive")
        arr.flags.writeable = False
        self._data = arr
        self.fs = fs

    # ------------------------------------------------------------------ basics
    @classmethod
    def from_channels(cls, *channels: ArrayLike | Sound, fs: float | None = None) -> Sound:
        """Stack 1-D arrays or mono Sounds into a multichannel Sound."""
        arrays = []
        for ch in channels:
            if isinstance(ch, Sound):
                fs = ch.fs if fs is None else fs
                if ch.fs != fs:
                    raise ValueError("all channels must share a sampling rate")
                ch = ch.mono().data[:, 0]
            arrays.append(np.asarray(ch, dtype=float))
        if fs is None:
            raise ValueError("fs is required when channels are plain arrays")
        n = max(len(a) for a in arrays)
        out = np.zeros((n, len(arrays)))
        for i, a in enumerate(arrays):
            out[: len(a), i] = a
        return cls(out, fs)

    @classmethod
    def load(cls, path: str | PathLike, **kwargs) -> Sound:
        """Read an audio file (anything libsndfile supports)."""
        import soundfile as sf

        data, fs = sf.read(path, always_2d=True, **kwargs)
        return cls(data, fs)

    @property
    def data(self) -> np.ndarray:
        """Read-only ``(n_samples, n_channels)`` array."""
        return self._data

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def n_samples(self) -> int:
        return self._data.shape[0]

    @property
    def n_channels(self) -> int:
        return self._data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def t(self) -> np.ndarray:
        """Sample times [s]."""
        return time_axis(self.n_samples, self.fs)

    @property
    def rms(self) -> float:
        """RMS over all samples and channels."""
        return float(rms(self._data))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self._data))) if self.n_samples else 0.0

    def channel(self, i: int) -> Sound:
        return Sound(self._data[:, i], self.fs)

    @property
    def left(self) -> Sound:
        return self.channel(0)

    @property
    def right(self) -> Sound:
        return self.channel(1)

    def mono(self) -> Sound:
        """Average of all channels."""
        return Sound(self._data.mean(axis=1), self.fs)

    def to_channels(self, n: int = 2) -> Sound:
        """Copy a mono sound into ``n`` identical channels (diotic)."""
        if self.n_channels == n:
            return self
        if self.n_channels != 1:
            raise ValueError(f"cannot upmix {self.n_channels} channels to {n}")
        return Sound(np.repeat(self._data, n, axis=1), self.fs)

    def to_stereo(self) -> Sound:
        return self.to_channels(2)

    def __repr__(self) -> str:
        level = amp_to_db(self.rms)
        return f"Sound({self.duration:.3f} s, {self.fs:g} Hz, {self.n_channels} ch, rms {level:.1f} dB)"

    def _repr_html_(self) -> str | None:
        """Notebook display: an audio player."""
        try:
            from IPython.display import Audio
        except ImportError:
            return None
        if self.n_samples == 0 or self.peak == 0:
            return f"<code>{self!r}</code>"
        player = Audio(self._data.T, rate=int(self.fs), normalize=True)
        return f"<code>{self!r}</code><br>{player._repr_html_()}"

    # -------------------------------------------------------------- arithmetic
    def _coerce(self, other) -> np.ndarray:
        if isinstance(other, Sound):
            if other.fs != self.fs:
                raise ValueError(
                    f"sampling rates differ ({self.fs} vs {other.fs}); use .resample() or sonore.match_fs()"
                )
            if len(other) != len(self):
                raise ValueError(
                    f"lengths differ ({len(self)} vs {len(other)} samples); "
                    "use sonore.pad() or sonore.truncate() first"
                )
            return other._data
        if isinstance(other, numbers.Real):
            return np.asarray(float(other))
        if isinstance(other, np.ndarray):
            arr = other.astype(float)
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr
        return NotImplemented

    def _binary(self, other, op, reflected=False):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return Sound(op(o, self._data) if reflected else op(self._data, o), self.fs)

    def __add__(self, other):
        if isinstance(other, Decibels):
            return self * other.gain
        if isinstance(other, numbers.Real) and not isinstance(other, bool):
            if other == 0:  # lets built-in sum() work
                return self
            raise TypeError(
                f"adding a bare number to a Sound is ambiguous; write snd + {other!r}*dB "
                "for a level change (from sonore import dB), or add an array for a DC offset"
            )
        return self._binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Decibels | numbers.Real) and not isinstance(other, bool):
            return self + (-other)
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        if isinstance(other, Decibels):
            raise TypeError("dB - Sound is undefined; did you mean snd - x*dB?")
        return (-self) + other

    def __neg__(self):
        return Sound(-self._data, self.fs)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __rtruediv__(self, other):
        return self._binary(other, np.divide, reflected=True)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step is not None:
            raise TypeError(
                "Sound indexing takes a time slice in seconds, e.g. snd[0.1:0.5]; index snd.data for samples"
            )
        start = None if key.start is None else int(round(key.start * self.fs))
        stop = None if key.stop is None else int(round(key.stop * self.fs))
        return Sound(self._data[start:stop], self.fs)

    # ------------------------------------------------------------- operations
    def gain_db(self, db: float | Decibels) -> Sound:
        """Change level by ``db`` decibels. Same as ``snd + db*dB``."""
        return self * float(db_to_amp(float(db)))

    def normalize(self, rms: float | None = 1.0, peak: float | None = None) -> Sound:
        """Scale to a target RMS (default 1) or, if ``peak`` is given, a target peak."""
        if peak is not None:
            return self * (peak / self.peak)
        return self * (rms / self.rms)

    def zero_mean(self) -> Sound:
        return Sound(self._data - self._data.mean(axis=0), self.fs)

    def ramp(self, duration: float, shape: str = "cosine") -> Sound:
        """Apply onset and offset ramps of ``duration`` seconds.

        ``shape`` is ``"cosine"`` (raised cosine) or ``"linear"``.
        """
        n = int(round(duration * self.fs))
        if n == 0:
            return self
        if 2 * n > self.n_samples:
            raise ValueError("ramps are longer than the sound")
        u = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
        if shape == "cosine":
            r = (1 - np.cos(np.pi * u)) / 2
        elif shape == "linear":
            r = u
        else:
            raise ValueError("shape must be 'cosine' or 'linear'")
        env = np.ones(self.n_samples)
        env[:n] = r
        env[-n:] = r[::-1]
        return self * env

    def pad(self, before: float = 0.0, after: float = 0.0) -> Sound:
        """Zero-pad by ``before``/``after`` seconds."""
        nb, na = int(round(before * self.fs)), int(round(after * self.fs))
        return Sound(np.pad(self._data, ((nb, na), (0, 0))), self.fs)

    def pad_to(self, n: int, align: str = "start") -> Sound:
        """Zero-pad to ``n`` samples. ``align`` is ``start``, ``center``, or ``end``."""
        extra = n - self.n_samples
        if extra < 0:
            raise ValueError("sound is already longer than n")
        before = {"start": 0, "center": extra // 2, "end": extra}[align]
        return Sound(np.pad(self._data, ((before, extra - before), (0, 0))), self.fs)

    def delay(self, seconds: float) -> Sound:
        """Delay by ``seconds`` (may be fractional samples); the result is longer.

        Integer-sample delays are exact. Fractional delays use an FFT phase ramp
        (band-limited interpolation); the output is ``ceil(delay)`` samples
        longer, so sinc ringing past the end is cut off. That's inaudible for
        ramped stimuli; for impulse responses use :func:`sonore.simple_bir`.
        """
        if seconds < 0:
            raise ValueError("delay must be non-negative")
        d = seconds * self.fs
        d_int = int(np.floor(d))
        frac = d - d_int
        data = np.pad(self._data, ((d_int, 0), (0, 0)))
        if frac > 1e-9:
            guard = data.shape[0]  # sinc tails decay slowly; keep them from wrapping around
            n = data.shape[0] + 1 + guard
            spec = np.fft.rfft(data, n=n, axis=0)
            f = np.fft.rfftfreq(n)
            spec *= np.exp(-2j * np.pi * f * frac)[:, None]
            data = np.fft.irfft(spec, n=n, axis=0)[: data.shape[0] + 1]
        return Sound(data, self.fs)

    def resample(self, fs: float) -> Sound:
        """Polyphase resampling to a new rate."""
        if fs == self.fs:
            return self
        ratio = Fraction(fs / self.fs).limit_denominator(10000)
        data = resample_poly(self._data, ratio.numerator, ratio.denominator, axis=0)
        return Sound(data, fs)

    def convolve(self, ir: Sound | ArrayLike) -> Sound:
        """Convolve with an impulse response (full length).

        A mono IR is applied to every channel; a multichannel IR convolves a
        mono sound into that many channels, or channel-by-channel otherwise.
        """
        h = ir.data if isinstance(ir, Sound) else np.atleast_1d(np.asarray(ir, float))
        if isinstance(ir, Sound) and ir.fs != self.fs:
            raise ValueError("impulse response has a different sampling rate")
        if h.ndim == 1:
            h = h[:, None]
        return Sound(fftconvolve(self._data, h, axes=0), self.fs)

    def envelope(self, pad: float | str = "auto"):
        """Hilbert envelope of each channel, as an :class:`~sonore.envelopes.Envelope`
        (not a Sound: you apply an envelope to a sound rather than listen to it).
        ``snd / snd.envelope()`` is the fine structure.

        The Hilbert transform is FFT-based. By default the sound is zero-padded
        by its own length on each side first, so a loud start can't leak into
        the end of the envelope (or vice versa). ``pad=0`` is circular;
        a number pads by that many seconds.
        """
        from sonore.envelopes import Envelope

        p = self.n_samples if pad == "auto" else int(round(float(pad) * self.fs))
        x = np.pad(self._data, ((p, p), (0, 0))) if p else self._data
        return Envelope(np.abs(hilbert(x, axis=0))[p : p + self.n_samples], self.fs)

    # ---------------------------------------------------------------- output
    def play(self, blocking: bool = False, **kwargs) -> None:
        """Play through the default device (requires the ``sounddevice`` extra)."""
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            raise RuntimeError(
                "playback needs sounddevice and PortAudio: pip install 'sonore[play]'. "
                "In a notebook, just display the Sound instead."
            ) from e
        sd.play(self._data, self.fs, blocking=blocking, **kwargs)

    def save(self, path: str | PathLike, **kwargs) -> None:
        """Write to an audio file (format from the extension; mp3 needs libsndfile>=1.1)."""
        import soundfile as sf

        if self.peak > 1:
            warnings.warn(
                f"peak is {self.peak:.2f}; integer formats will clip. Consider .normalize(peak=0.9) first.",
                stacklevel=2,
            )
        sf.write(path, self._data, int(self.fs), **kwargs)

    def plot(self, ax=None, **kwargs):
        """Waveform plot; returns the matplotlib Axes."""
        from sonore.plotting import plot_waveform

        return plot_waveform(self, ax=ax, **kwargs)


def load(path: str | PathLike, **kwargs) -> Sound:
    """Read an audio file. Alias for :meth:`Sound.load`."""
    return Sound.load(path, **kwargs)
