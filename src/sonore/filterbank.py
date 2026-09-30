"""Cosine filterbanks and the subband representation.

The filters follow McDermott & Simoncelli (2011): half-cycle cosines on a
perceptual frequency scale, each spanning two band spacings, plus a lowpass and
a highpass at the edges. Their squared responses sum to exactly 1, so filtering
on analysis *and* synthesis reconstructs the input perfectly.

Two scales are provided: :class:`ERBFilterbank` (ERB-number, the auditory
default) and :class:`OctaveFilterbank` (log2 frequency, the axis on which
spectral modulation is measured in cycles/octave).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from sonore.frames import Filterbank
from sonore.sound import Sound
from sonore.utils import as_rng, erb_to_freq, freq_to_erb

__all__ = ["CosineFilterbank", "ERBFilterbank", "OctaveFilterbank", "Subbands", "subbands", "noise_vocode"]


@dataclass(frozen=True)
class CosineFilterbank(Filterbank):
    """``n_bands`` bandpass filters equally spaced on some frequency scale
    between ``f_lo`` and ``f_hi`` [Hz], plus a lowpass below ``f_lo`` and a
    highpass above ``f_hi`` (``n_bands + 2`` filters in total). Subclasses
    define the scale.

    A tight :class:`~sonore.frames.Filterbank`: the squared responses sum to
    exactly 1, so the frame bounds are ``(1, 1)`` and synthesis re-filters
    with the analysis filters."""

    n_bands: int = 30
    f_lo: float = 50.0
    f_hi: float = 8000.0
    unit = "unit"  # name of one step on the scale, e.g. "oct" or "ERB"
    tight = True

    @property
    def n_filters(self) -> int:
        return self.n_bands + 2

    @staticmethod
    def to_scale(freq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def from_scale(value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def _knots(self) -> np.ndarray:
        return np.linspace(self.to_scale(self.f_lo), self.to_scale(self.f_hi), self.n_bands + 2)

    @property
    def spacing(self) -> float:
        """Distance between adjacent filter centers, in scale units."""
        k = self._knots
        return float(k[1] - k[0])

    @property
    def cfs(self) -> np.ndarray:
        """Center frequencies [Hz] of all filters (edges at ``f_lo``/``f_hi``)."""
        return self.from_scale(self._knots)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Magnitude responses, shape ``(len(freqs), n_bands + 2)``."""
        with np.errstate(divide="ignore"):
            e = self.to_scale(np.asarray(freqs, float))[:, None]
        k = self._knots
        u = (e - k[None, :]) / self.spacing  # distance from each center in band spacings
        H = np.where(np.abs(u) < 1, np.cos(np.pi / 2 * np.clip(u, -1, 1)), 0.0)
        H[:, 0] = np.where(e[:, 0] <= k[0], 1.0, H[:, 0])
        H[:, -1] = np.where(e[:, 0] >= k[-1], 1.0, H[:, -1])
        return H


@dataclass(frozen=True)
class ERBFilterbank(CosineFilterbank):
    """Filters equally spaced on the ERB-number scale (Glasberg & Moore, 1990)."""

    unit = "ERB"

    @staticmethod
    def to_scale(freq):
        return freq_to_erb(freq)

    @staticmethod
    def from_scale(value):
        return erb_to_freq(value)


@dataclass(frozen=True)
class OctaveFilterbank(CosineFilterbank):
    """Filters equally spaced in log2 frequency (octaves)."""

    f_lo: float = 125.0
    unit = "oct"

    @staticmethod
    def to_scale(freq):
        return np.log2(freq)

    @staticmethod
    def from_scale(value):
        return np.exp2(value)

    @classmethod
    def per_octave(cls, bands_per_octave: float, f_lo: float, f_hi: float) -> OctaveFilterbank:
        """Choose ``n_bands`` so filter centers are about ``1/bands_per_octave``
        octaves apart between ``f_lo`` and ``f_hi``."""
        n = max(1, int(round(bands_per_octave * np.log2(f_hi / f_lo))) - 1)
        return cls(n, f_lo, f_hi)


def subbands(sound: Sound, n_bands: int = 30, f_lo: float = 50.0, f_hi: float | None = None) -> Subbands:
    """Convenience: build an :class:`ERBFilterbank` and analyze ``sound``.
    ``f_hi`` defaults to just under Nyquist."""
    f_hi = 0.95 * sound.fs / 2 if f_hi is None else min(f_hi, sound.fs / 2)
    return ERBFilterbank(n_bands, f_lo, f_hi).analyze(sound)


class Subbands:
    """The output of a :class:`~sonore.frames.Filterbank`: one band-limited :class:`~sonore.Sound` per filter.

    ``sb[i]`` is a Sound, iterating yields Sounds, and ``sb.cfs`` labels them.
    The first and last bands are the filterbank's lowpass and highpass edges.
    Internally the bands are stored as one ``(n_samples, n_bands, n_channels)``
    array (:attr:`data`).

    The Hilbert decomposition of every band at once::

        sb.envelopes()      # Envelopes: one Envelope per band (a cochleagram)
        sb.tfs()            # Subbands: the fine structure, itself audible
        sb.envelopes() * sb.tfs()   # == sb

    so a vocoder is ``(speech.envelopes() * carrier.tfs()).synthesize()``.
    """

    def __init__(self, data: np.ndarray, fs: float, filterbank: Filterbank, pad: int = 0):
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 3 or arr.shape[1] != filterbank.n_filters:
            raise ValueError(f"expected shape (n_samples, {filterbank.n_filters}, n_channels)")
        if 2 * pad >= arr.shape[0]:
            raise ValueError("padding is longer than the data")
        arr.flags.writeable = False
        self._full, self.fs, self.filterbank, self.pad = arr, fs, filterbank, int(pad)

    @property
    def data(self) -> np.ndarray:
        """Band signals without the padding, shape ``(n_samples, n_bands, n_channels)``."""
        return self._full[self.pad : self._full.shape[0] - self.pad]

    @property
    def n_samples(self) -> int:
        return self._full.shape[0] - 2 * self.pad

    @property
    def cfs(self) -> np.ndarray:
        return self.filterbank.cfs

    def __len__(self) -> int:
        return self._full.shape[1]

    def __getitem__(self, i: int) -> Sound:
        return Sound(self.data[:, i, :], self.fs)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        n, b, c = self.data.shape
        return f"Subbands({b} bands, {n / self.fs:.3f} s, {self.fs:g} Hz, {c} ch)"

    def _new(self, full: np.ndarray, pad: int | None = None) -> Subbands:
        return Subbands(full, self.fs, self.filterbank, self.pad if pad is None else pad)

    def _analytic(self) -> np.ndarray:
        return hilbert(self._full, axis=0)

    def envelopes(self, lowpass: float | None = None, fs: float | None = None):
        """Hilbert envelope of every band, as :class:`~sonore.envelopes.Envelopes`.
        Optionally lowpass-filtered [Hz] and then resampled to ``fs``."""
        from sonore.envelopes import Envelopes

        env = Envelopes(np.abs(self._analytic()), self.fs, self.filterbank, pad=self.pad)
        if lowpass is not None:
            env = env.lowpass(lowpass)
        if fs is not None:
            env = env.resample(fs)
        return env

    def tfs(self) -> Subbands:
        """Temporal fine structure of every band: ``cos`` of the instantaneous
        phase (unit amplitude)."""
        return self._new(np.cos(np.angle(self._analytic())))

    def synthesize(self) -> Sound:
        """Back to a Sound with the filterbank's canonical dual
        (:meth:`~sonore.frames.Filterbank.synthesize`): the exact inverse of
        :meth:`~sonore.frames.Filterbank.analyze`, and the least-squares
        signal after the bands are modified. For the cosine banks this is
        re-filtering each band and summing; with the default padding,
        re-filtering can't wrap around either."""
        return self.filterbank.synthesize(self)

    def sum(self) -> Sound:
        """Add the bands without re-filtering. :meth:`synthesize` is almost
        always what you want; ``sum`` is for bands that were already shaped to
        add up correctly."""
        return Sound(self.data.sum(axis=1), self.fs)

    def plot(self, axes=None, channel: int = 0, **kwargs):
        """Stacked band waveforms (see :func:`sonore.plotting.plot_subbands`).
        For an image, plot the envelopes: ``sb.envelopes().plot()``."""
        from sonore.plotting import plot_subbands

        return plot_subbands(self, axes=axes, channel=channel, **kwargs)


def noise_vocode(
    sound: Sound,
    n_bands: int = 16,
    f_lo: float = 80.0,
    f_hi: float = 8000.0,
    carrier: Sound | str = "noise",
    env_lowpass: float | None = 50.0,
    rng=None,
) -> Sound:
    """Channel vocoder (Shannon et al., 1995).

    Band envelopes of ``sound`` (lowpassed at ``env_lowpass`` Hz) modulate the
    fine structure of the ``carrier``: ``"noise"``, ``"tone"`` (sinusoids at the
    band centers), or any Sound at least as long. The edge bands (outside
    ``f_lo..f_hi``) are silenced. The output matches the input's RMS.
    """
    fb = ERBFilterbank(n_bands, f_lo, min(f_hi, 0.95 * sound.fs / 2))
    envelopes = fb.analyze(sound).envelopes(lowpass=env_lowpass).without_edges()
    if isinstance(carrier, Sound):
        if len(carrier) < len(sound):
            raise ValueError("carrier is shorter than the sound")
        fine = fb.analyze(Sound(carrier.data[: len(sound)], carrier.fs)).tfs()
    elif carrier == "noise":
        from sonore.generators import gaussian_noise

        noise = gaussian_noise(sound.duration, sound.fs, n_channels=sound.n_channels, rng=as_rng(rng))
        fine = fb.analyze(noise, pad=0).tfs()  # generated noise is periodic: circular is exact
    elif carrier == "tone":
        tones = np.cos(2 * np.pi * fb.cfs[None, :, None] * sound.t[:, None, None])
        fine = Subbands(
            np.broadcast_to(tones, (len(sound), len(fb.cfs), sound.n_channels)).copy(), sound.fs, fb
        )
    else:
        raise ValueError("carrier must be 'noise', 'tone', or a Sound")
    return (envelopes * fine).synthesize().normalize(sound.rms)
