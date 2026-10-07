"""Envelopes: the slowly varying amplitudes that shape sounds.

An envelope is not a sound. It is non-negative, it usually varies slowly
enough to live at a low sampling rate, and you don't listen to it: you apply it
to a sound. The types reflect that:

- :class:`Envelope` is one envelope. ``Envelope * Sound`` modulates the sound.
- :class:`Envelopes` is one envelope per frequency band of a filterbank, i.e. a
  spectrotemporal envelope. This is conceptually the same thing as a
  **cochleagram** (the envelope of each cochlear-filter output over time);
  here the "cochlea" is whichever :class:`~sonore.frames.filterbank.Filterbank` (by
  default an ERB :func:`~sonore.frames.filterbank.cosine_filterbank`) produced it.
  ``Envelopes * Subbands`` modulates each band.

The Hilbert decomposition of a band is then literal::

    band == band.envelope() * (band / band.envelope())      # envelope x fine structure
    sb == sb.envelopes() * sb.tfs()                          # for every band at once

Envelopes resampled to a lower rate are upsampled automatically (band-limited:
polyphase, or the FFT for unusual rate ratios; clipped at zero) when applied
to a sound.

``to_sound(carrier)`` is the route back: the envelopes imposed on a carrier.
:func:`noise_vocode` is the noise vocoder after Shannon et al. (1995): a
sound's own band envelopes imposed on bands of noise, as in simulations of
cochlear-implant hearing.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, resample, resample_poly, sosfiltfilt

from sonore.core.utils import _fit_length, amp_to_db, as_rng, time_axis
from sonore.frames.filterbank import Subbands, _PaddedBands, cosine_filterbank
from sonore.views.view import View

if TYPE_CHECKING:
    from sonore.core.sound import Sound
    from sonore.frames.filterbank import Filterbank
    from sonore.views.modulation import ModulationSpectrum

__all__ = ["Envelope", "Envelopes", "noise_vocode"]


@dataclass(frozen=True)
class _EnvelopeAnalysis:
    """How a modulation spectrum's envelopes were made, so that envelopes
    can be rebuilt on the same grid: the filterbank, the envelope rate and
    length, linear or dB envelopes, and whether the edge bands were dropped."""

    filterbank: Filterbank
    fs: float
    n_samples: int
    scale: str
    drop_edges: bool


def _nonnegative(values: np.ndarray) -> np.ndarray:
    """Clip round-off negatives (within 1e-9 of the largest magnitude); reject
    genuinely negative or non-finite envelopes."""
    if not np.all(np.isfinite(values)):
        raise ValueError("envelopes must be finite (a division by zero?)")
    if values.size and np.min(values) < 0:
        scale = np.max(np.abs(values)) or 1.0
        if np.min(values) < -1e-9 * scale:
            raise ValueError("envelopes must be non-negative")
        values = np.maximum(values, 0.0)
    return values


def _lowpass(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    sos = butter(order, cutoff, fs=fs, output="sos")
    return np.maximum(sosfiltfilt(sos, data, axis=0), 0.0)


# A polyphase filter grows with the up and down factors of the rate ratio (a
# 1 s envelope at 314.16 Hz brought to 44.1 kHz, 464219/3307, took 3.6 s
# against 3 ms at 1000 Hz), so ratios that need larger factors go through the
# FFT instead, which is band-limited too.
_MAX_FACTOR = 1000


def _simple_ratio(fs_old: float, fs_new: float) -> Fraction | None:
    """``fs_new / fs_old`` as a fraction with both terms at most _MAX_FACTOR,
    or None when no such fraction is exact to 1e-9."""
    ratio = Fraction(fs_new / fs_old).limit_denominator(_MAX_FACTOR)
    exact = abs(float(ratio) - fs_new / fs_old) <= 1e-9 * fs_new / fs_old
    return ratio if exact and ratio.numerator <= _MAX_FACTOR else None


def _resample(data: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """Band-limited resampling along axis 0, clipped at zero. The ends are
    extended along a line rather than padded with zeros, which would drag the
    first and last few ms toward zero."""
    ratio = _simple_ratio(fs_old, fs_new)
    if ratio is not None:
        resampled = resample_poly(data, ratio.numerator, ratio.denominator, axis=0, padtype="line")
        return np.maximum(resampled, 0.0)
    # FFT resampling treats the signal as periodic: take out the line through
    # the end samples first, so the wrap from last to first sample is smooth
    n_old = data.shape[0]
    n_new = int(round(n_old * fs_new / fs_old))
    line_shape = (-1,) + (1,) * (data.ndim - 1)
    first, rise = data[:1], data[-1:] - data[:1]
    old_position = np.arange(n_old).reshape(line_shape) / max(n_old - 1, 1)
    new_position = (np.arange(n_new) * n_old / n_new).reshape(line_shape) / max(n_old - 1, 1)
    resampled = resample(data - first - rise * old_position, n_new, axis=0)
    return np.maximum(resampled + first + rise * new_position, 0.0)


def _upsample_to(data: np.ndarray, fs: float, n: int, fs_new: float) -> np.ndarray:
    """Envelope (axis 0) onto ``n`` samples at ``fs_new``: band-limited
    resampling (an envelope that was lowpassed and downsampled is
    band-limited, so this is the accurate choice), clipped at zero, then
    trimmed or edge-padded to exactly ``n`` samples."""
    return _fit_length(_resample(data, fs, fs_new), n, mode="edge")


def _check_duration(n_env: int, fs_env: float, n: int, fs: float) -> None:
    if abs(n_env / fs_env - n / fs) > 1.5 / fs_env:
        raise ValueError(f"durations differ ({n_env / fs_env:.4f} s envelope vs {n / fs:.4f} s signal)")


class Envelope(View):
    """A single (possibly multichannel) envelope, shape ``(n_samples, n_channels)``.
    A view: it keeps a magnitude over time and drops the fine structure.

    Arithmetic: ``*``, ``/`` and ``+`` with numbers and other Envelopes (so
    ``1 + 0.5 * env`` works), and ``env * snd`` / ``snd * env`` modulate a
    :class:`~sonore.Sound`. ``snd / env`` divides the envelope out of a sound.
    A one-channel envelope applies to every channel of a sound; otherwise the
    channel counts must match.
    :meth:`to_sound` puts it on another sound's fine structure.
    """

    discards = "Envelope discards the fine structure: only a magnitude over time is kept."
    back_to_sound = (
        "Envelope.to_sound puts it on the fine structure of a carrier sound, which is not the analysed "
        "sound's fine structure."
    )

    __array_ufunc__ = None

    def __init__(self, data, fs: float):
        values = np.array(data, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"envelope data must be 1-D or 2-D, got shape {values.shape}")
        values = _nonnegative(values)
        values.flags.writeable = False
        self._data, self.fs = values, fs

    @property
    def data(self) -> np.ndarray:
        """The envelope values, shape ``(n_samples, n_channels)``, read-only."""
        return self._data

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return len(self) / self.fs

    @property
    def t(self) -> np.ndarray:
        """Sample times in seconds, starting at 0."""
        return np.arange(len(self)) / self.fs

    @property
    def db(self) -> np.ndarray:
        """Envelope in dB (``20*log10``)."""
        return amp_to_db(self._data)

    def __repr__(self) -> str:
        return f"Envelope({self.duration:.3f} s, {self.fs:g} Hz, {self._data.shape[1]} ch)"

    def lowpass(self, cutoff: float, order: int = 4) -> Envelope:
        """Zero-phase Butterworth lowpass at ``cutoff`` Hz (result clipped at 0)."""
        return Envelope(_lowpass(self._data, cutoff, self.fs, order), self.fs)

    def resample(self, fs: float) -> Envelope:
        """The envelope at rate ``fs``: band-limited, clipped at zero, with the
        ends extended along a line rather than dragged toward zero."""
        return self if fs == self.fs else Envelope(_resample(self._data, self.fs, fs), fs)

    def _values_for(self, sound: Sound) -> np.ndarray:
        _check_duration(len(self), self.fs, len(sound), sound.fs)
        if self.fs == sound.fs and len(self) == len(sound):
            return self._data
        return _upsample_to(self._data, self.fs, len(sound), sound.fs)

    def _other(self, other):
        match other:
            case Envelope():
                if other.fs != self.fs or len(other) != len(self):
                    raise ValueError("envelopes must share fs and length")
                return other._data
            case bool():
                return NotImplemented
            case numbers.Real():
                return float(other)
            case _:
                return NotImplemented

    def __mul__(self, other):
        from sonore.core.sound import Sound

        match other:
            case Sound():
                return Sound(other.data * self._values_for(other), other.fs)
            case _:
                other_values = self._other(other)
                if other_values is NotImplemented:
                    return NotImplemented
                return Envelope(self._data * other_values, self.fs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other_values = self._other(other)
        if other_values is NotImplemented:
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore"):  # a zero divisor is refused below
            quotient = self._data / other_values
        return Envelope(quotient, self.fs)

    def __rtruediv__(self, other):
        from sonore.core.sound import Sound

        match other:
            case Sound():
                env_values = self._values_for(other)
                floor = 1e-12 * (np.max(env_values) or 1.0)
                return Sound(other.data / np.maximum(env_values, floor), other.fs)
            case _:
                return NotImplemented

    def __add__(self, other):
        other_values = self._other(other)
        if other_values is NotImplemented:
            return NotImplemented
        return Envelope(self._data + other_values, self.fs)

    __radd__ = __add__

    def to_sound(self, carrier: Sound) -> Sound:
        """This envelope on the fine structure of ``carrier``: the carrier is
        divided by its own Hilbert envelope (``carrier / carrier.envelope()``)
        and multiplied by this one. ``carrier`` must last as long as the
        envelope; it sets the sampling rate. Plain ``envelope * sound`` keeps
        the sound's own envelope as well (amplitude modulation)."""
        from sonore.core.sound import Sound

        if not isinstance(carrier, Sound):
            raise TypeError("carrier must be a Sound")
        return self * (carrier / carrier.envelope())

    def plot(self, ax=None, **kwargs):
        """The envelope against time (see :func:`~sonore.plotting.plot_envelope`)."""
        from sonore.plotting import plot_envelope

        return plot_envelope(self, ax=ax, **kwargs)


class Envelopes(_PaddedBands, View):
    """One envelope per band of a filterbank: a spectrotemporal envelope.

    Conceptually this is a **cochleagram**: the envelope of each filter's
    output over time. Shape ``(n_samples, n_bands, n_channels)``; the bands
    include the filterbank's lowpass and highpass edge filters, as in
    :class:`~sonore.frames.filterbank.Subbands`.

    A view: it keeps each band's Hilbert magnitude and drops the fine
    structure (and, with ``lowpass``, the envelope's own fast detail).

    ``env[i]`` is an :class:`Envelope`; ``env * subbands`` modulates each band
    (imposing the envelopes on a carrier, not an inverse);
    ``env.modulation_spectrum()`` gives its 2-D modulation spectrum on the
    filterbank's frequency scale (cycles/octave or cycles/ERB).

    Envelopes derived from padded Subbands carry the same zero-padding
    (``pad`` samples at each end, hidden from :attr:`data`), so that
    modulating and re-synthesizing bands can't wrap around. Envelopes made
    from scratch (e.g. a rendered ripple pattern) have ``pad=0`` and are
    zero outside their own extent when combined with padded bands.
    """

    discards = (
        "Envelopes discard the fine structure: only the Hilbert magnitude of each band is kept, "
        "smoothed further when a lowpass was asked for."
    )
    back_to_sound = (
        "Envelopes.to_sound puts them on the fine structure of a carrier (a new noise, tones at the band "
        "centers, or another sound) and synthesizes the bands, as the noise vocoder does."
    )

    __array_ufunc__ = None

    def __init__(self, data, fs: float, filterbank: Filterbank, pad: int = 0):
        data = np.asarray(data, dtype=float)
        super().__init__(data[:, :, None] if data.ndim == 2 else data, fs, filterbank, pad)

    def _checked(self, bands: np.ndarray) -> np.ndarray:
        return _nonnegative(bands)

    def __getitem__(self, i: int) -> Envelope:
        return Envelope(self.data[:, i, :], self.fs)

    @property
    def duration(self) -> float:
        """Length in seconds (without the padding)."""
        return self.n_samples / self.fs

    @property
    def t(self) -> np.ndarray:
        """Sample times in seconds, starting at 0."""
        return time_axis(self.n_samples, self.fs)

    @property
    def db(self) -> np.ndarray:
        """Envelopes in dB (``20*log10``), shape ``(n_samples, n_bands, n_channels)``."""
        return amp_to_db(self.data)

    def _new(self, full, fs=None, pad=None) -> Envelopes:
        return Envelopes(
            full, self.fs if fs is None else fs, self.filterbank, self.pad if pad is None else pad
        )

    def lowpass(self, cutoff: float, order: int = 4) -> Envelopes:
        """Zero-phase Butterworth lowpass of every band at ``cutoff`` Hz,
        padding included (result clipped at 0)."""
        return self._new(_lowpass(self._full, cutoff, self.fs, order))

    def resample(self, fs: float) -> Envelopes:
        """The envelopes at rate ``fs``, padding included, as
        :meth:`Envelope.resample`. When the rate ratio is a small fraction the
        inner signal starts exactly on a sample; otherwise within half of
        one."""
        if fs == self.fs:
            return self
        ratio = _simple_ratio(self.fs, fs)
        if ratio is None:
            # no small fraction: the inner signal starts within half a sample
            pad = int(round(self.pad * fs / self.fs))
            full = _resample(self._full, self.fs, fs)
        else:
            # extend the front padding so it maps to a whole number of samples at
            # the new rate: the inner signal then starts exactly on a sample
            extra = (-self.pad) % ratio.denominator
            full = np.pad(self._full, ((extra, 0), (0, 0), (0, 0))) if extra else self._full
            full = _resample(full, self.fs, fs)
            pad = (self.pad + extra) * ratio.numerator // ratio.denominator
        n_total = pad + int(round(self.n_samples * fs / self.fs)) + pad
        return self._new(_fit_length(full, n_total, mode="edge"), fs, pad)

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
        on_grid = np.zeros((n_inner + 2 * pad,) + full.shape[1:])
        shift = pad - own_pad  # where our sample 0 lands on the target grid
        dst_start, dst_stop = max(0, shift), min(on_grid.shape[0], shift + full.shape[0])
        on_grid[dst_start:dst_stop] = full[dst_start - shift : dst_stop - shift]
        return on_grid

    def __mul__(self, other):
        match other:
            case Subbands():
                if len(other) != len(self):
                    raise ValueError(f"band counts differ ({len(self)} vs {len(other)})")
                _check_duration(self.n_samples, self.fs, other.n_samples, other.fs)
                pad = max(other.pad, int(round(self.pad * other.fs / self.fs)))
                extra = pad - other.pad
                bands = np.pad(other._full, ((extra, extra), (0, 0), (0, 0))) if extra else other._full
                env_on_grid = self._on_grid(other.fs, other.n_samples, pad)
                return Subbands(bands * env_on_grid, other.fs, other.filterbank, pad=pad)
            case Envelopes():
                if other.fs != self.fs or other.n_samples != self.n_samples or len(other) != len(self):
                    raise ValueError("Envelopes must share fs, length and band count")
                pad = max(self.pad, other.pad)
                own_values = self._on_grid(self.fs, self.n_samples, pad)
                other_values = other._on_grid(self.fs, self.n_samples, pad)
                return self._new(own_values * other_values, pad=pad)
            case Envelope():
                if other.fs != self.fs or len(other) != self.n_samples:
                    raise ValueError("Envelope must share fs and length")
                gain = np.zeros((self._full.shape[0], 1, other.data.shape[1]))
                gain[self.pad : self.pad + self.n_samples, 0, :] = other.data
                return self._new(self._full * gain)
            case bool():
                return NotImplemented
            case numbers.Real():
                return self._new(self._full * float(other))
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def modulation_spectrum(self, scale: str = "linear", drop_edges: bool = True) -> ModulationSpectrum:
        """2-D modulation spectrum (temporal Hz x spectral cycles per scale unit).

        ``scale="db"`` transforms log envelopes. The edge bands are dropped by
        default, since they aren't evenly spaced with the others; a bank built
        with ``edges=False`` has none to drop, and all its bands are kept.
        """
        from sonore.views.modulation import ModulationSpectrum

        filterbank = self.filterbank
        if filterbank.spacing is None:
            raise TypeError(
                "modulation_spectrum needs filters evenly spaced on a frequency scale; "
                "this filterbank's centers are not"
            )
        band_env = self.data.mean(axis=2)  # (n, B), channels averaged
        drop_edges = drop_edges and getattr(filterbank, "edges", True) is not False
        if drop_edges:
            band_env = band_env[:, 1:-1]
        if scale == "db":
            band_env = amp_to_db(band_env + 1e-12 * (band_env.max() or 1.0))
        elif scale != "linear":
            raise ValueError("scale must be 'linear' or 'db'")
        spectrum = ModulationSpectrum.from_array(
            band_env.T, dt=1 / self.fs, dx=filterbank.spacing, spectral_unit=f"cyc/{filterbank.unit}"
        )
        spectrum._analysis = _EnvelopeAnalysis(filterbank, self.fs, self.n_samples, scale, drop_edges)
        return spectrum

    def to_sound(self, carrier: Sound | str = "noise", fs: float | None = None, rng=None) -> Sound:
        """These envelopes imposed on a carrier, band by band, and synthesized
        with their filterbank.

        - ``"noise"``: each band of a Gaussian noise (from ``rng``), scaled
          to a mean-square envelope of 1. The bands keep their own random
          envelope fluctuations, as in the classic noise vocoder.
        - ``"tone"``: a cosine at each band's center.
        - a Sound lasting as long as the envelopes, analyzed with the same
          filterbank. Only its fine structure is used, so this is
          :meth:`Envelope.to_sound` in every band.

        ``fs`` is the output rate for ``"noise"`` and ``"tone"`` (by default
        the envelopes' own); a Sound carrier sets its own. The edge bands are
        kept (see :meth:`without_edges`) and the level is the envelopes' own.
        """
        from sonore.core.sound import Sound

        match carrier:
            case Sound():
                if fs is not None or rng is not None:
                    raise TypeError("fs and rng apply only to the 'noise' and 'tone' carriers")
                fine = self.filterbank.analyze(carrier).tfs()
            case str("noise"):
                from sonore.sources.waveforms import gaussian_noise

                fs = self.fs if fs is None else fs
                noise = gaussian_noise(self.duration, fs, n_channels=self.data.shape[2], rng=as_rng(rng))
                # generated noise is periodic, so circular analysis is exact
                bands = self.filterbank.analyze(noise, pad=0)
                envelope_power = np.mean(np.abs(bands._analytic()) ** 2, axis=0, keepdims=True)
                fine = bands._new(bands._full / np.sqrt(np.maximum(envelope_power, 1e-300)))
            case str("tone"):
                if rng is not None:
                    raise TypeError("rng applies only to the 'noise' carrier")
                fs = self.fs if fs is None else fs
                n_out = int(round(self.duration * fs))
                t = np.arange(n_out) / fs
                tones = np.cos(2 * np.pi * self.filterbank.cfs[None, :, None] * t[:, None, None])
                fine = Subbands(
                    np.broadcast_to(tones, (n_out, len(self.filterbank.cfs), self.data.shape[2])).copy(),
                    fs,
                    self.filterbank,
                )
            case _:
                raise ValueError("carrier must be 'noise', 'tone', or a Sound")
        return (self * fine).to_sound()

    def plot(self, ax=None, **kwargs):
        """The envelopes as a cochleagram, time by band in dB (see
        :func:`~sonore.plotting.plot_envelopes`)."""
        from sonore.plotting import plot_envelopes

        return plot_envelopes(self, ax=ax, **kwargs)


def noise_vocode(
    sound: Sound,
    n_bands: int = 16,
    f_lo: float = 80.0,
    f_hi: float = 8000.0,
    carrier: Sound | str = "noise",
    env_lowpass: float | None = 50.0,
    rng=None,
) -> Sound:
    """A noise vocoder, after Shannon et al. (1995), with Hilbert envelopes.

    The sound is split by a ``cosine_filterbank`` of ``n_bands`` bands from
    ``f_lo`` to ``f_hi`` (capped at Nyquist), each band's Hilbert envelope is
    lowpassed at ``env_lowpass`` Hz, and the envelopes are imposed on the
    ``carrier`` with :meth:`Envelopes.to_sound`: ``"noise"`` (bands of
    noise), ``"tone"`` (sinusoids at the band centers), or a Sound at the
    same rate and at least as long, whose fine structure carries them. The
    edge bands (outside ``f_lo..f_hi``) are silenced. The output matches the
    input's RMS.
    """
    from sonore.core.sound import Sound

    filterbank = cosine_filterbank(n_bands, f_lo, min(f_hi, sound.fs / 2))
    envelopes = filterbank.analyze(sound).envelopes(lowpass=env_lowpass).without_edges()
    match carrier:
        case Sound():
            if carrier.fs != sound.fs:
                raise ValueError(f"sample rates differ ({carrier.fs:g} Hz carrier vs {sound.fs:g} Hz sound)")
            if len(carrier) < len(sound):
                raise ValueError("carrier is shorter than the sound")
            vocoded = envelopes.to_sound(Sound(carrier.data[: len(sound)], carrier.fs))
        case str("noise"):
            vocoded = envelopes.to_sound("noise", rng=rng)
        case str("tone"):
            vocoded = envelopes.to_sound("tone")
        case _:
            raise ValueError("carrier must be 'noise', 'tone', or a Sound")
    return vocoded.normalize(sound.rms)
