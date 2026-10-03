"""Filterbanks: frames whose filters are given by their responses on the DFT
grid, and the subband representation they produce.

:class:`Filterbank` is the frequency-domain, undecimated family. Its frame
operator is diagonal in frequency, ``s(f) = sum_k |H_k(f)|**2``, and the
canonical dual filters are ``H_k / s`` (Balazs et al., 2011). Banks whose ``s``
is identically 1 set ``tight = True`` and skip the division, as
:class:`CosineFilterbank` does.

The cosine filters follow McDermott & Simoncelli (2011): half-cycle cosines on
a perceptual frequency scale, each spanning two band spacings, plus a lowpass
and a highpass at the edges. Their squared responses sum to exactly 1, so
filtering on analysis *and* synthesis reconstructs the input perfectly.

Two scales are provided: :class:`ERBFilterbank` (ERB-number, the auditory
default) and :class:`OctaveFilterbank` (log2 frequency, the axis on which
spectral modulation is measured in cycles/octave).

Two non-tight shapes (docs/design/frames.md, step 2) share
:class:`BandpassFilterbank`'s edge filters: :class:`GammatoneFilterbank`
(4th-order gammatones on the ERB-number scale) and :class:`MorletFilterbank`
(Morlet wavelets, log spaced). Their synthesis is the canonical dual.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft as sp_fft

from sonore.core.fft import fast_padding, threads
from sonore.core.sound import Sound
from sonore.frames.frame import Frame, _check_frame

if TYPE_CHECKING:
    pass
from dataclasses import KW_ONLY

from scipy.signal import hilbert

from sonore.core.utils import erb_bandwidth, erb_to_freq, freq_to_erb

__all__ = [
    "Filterbank",
    "CosineFilterbank",
    "ERBFilterbank",
    "OctaveFilterbank",
    "BandpassFilterbank",
    "GammatoneFilterbank",
    "MorletFilterbank",
    "subbands",
    "Subbands",
]


class Filterbank(Frame):
    """Undecimated filters applied by multiplication on the DFT grid.

    Subclasses define :meth:`response`, :attr:`n_filters` and :attr:`cfs`, and
    must be hashable (a frozen dataclass is the easy way) so the ringing time
    can be cached. Responses are zero-phase or, more generally, satisfy
    ``H(-f) = conj(H(f))``; only ``f >= 0`` is ever evaluated, and the
    subbands are real.

    Set ``tight = True`` only if the squared responses sum to exactly 1 at
    every frequency; synthesis then skips the division by ``s``.
    """

    tight: bool = False

    @property
    @abstractmethod
    def n_filters(self) -> int:
        """Number of filters (the band axis of :class:`~sonore.frames.filterbank.Subbands`)."""

    @property
    @abstractmethod
    def cfs(self) -> np.ndarray:
        """Center frequency [Hz] of each filter, shape ``(n_filters,)``."""

    @abstractmethod
    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Responses at ``freqs`` [Hz] (all >= 0), shape ``(len(freqs), n_filters)``."""

    def rfft_response(self, n: int, fs: float) -> np.ndarray:
        """Responses on the ``rfft`` grid of an ``n``-sample signal.

        On an even grid, a complex response is replaced by its real part at
        Nyquist: ``irfft`` keeps only the real part of that bin, so this is
        the filter actually applied, and analysis, :meth:`frame_power` and
        :meth:`synthesize` must all see the same one. Without this, the dual
        divides by ``|H(fs/2)|**2`` while analysis applied ``Re H(fs/2)``, and
        reconstruction is off by up to ~1e-5. Real responses are returned
        unchanged."""
        transfer = self.response(np.fft.rfftfreq(n, 1 / fs))
        if n % 2 == 0 and np.iscomplexobj(transfer):
            transfer = np.concatenate([transfer[:-1], transfer[-1:].real.astype(transfer.dtype)])
        return transfer

    def frame_power(self, n: int, fs: float) -> np.ndarray:
        """``s(f) = sum_k |H_k(f)|**2`` on the ``rfft`` grid of ``n`` samples:
        the eigenvalues of the (circular) frame operator."""
        return np.sum(np.abs(self.rfft_response(n, fs)) ** 2, axis=1)

    def ringing(self, fs: float, level_db: float = -60.0) -> int:
        """How long [samples] the filters ring: the longest time, over all
        filters, before the zero-phase impulse response stays below
        ``level_db`` re its peak (capped at 2 s). The minimum padding of
        ``pad="auto"``."""
        try:
            return _ringing_samples(self, float(fs), float(level_db))
        except TypeError:  # unhashable subclass: compute without the cache
            return _ringing_samples.__wrapped__(self, float(fs), float(level_db))

    def _pad_samples(self, pad: float | str, fs: float, n_samples: int) -> int:
        if pad == "auto":
            # at least the ringing, rounded up to an FFT length without large prime factors
            return fast_padding(int(n_samples), self.ringing(fs))
        return int(round(float(pad) * fs))

    def analyze(self, sound: Sound, pad: float | str = "auto") -> Subbands:
        """Split ``sound`` into subbands (zero-phase, via FFT).

        By default the sound is zero-padded by at least the filters' ringing
        time (:meth:`ringing`), so filter ringing near one end can't wrap
        around to the other. The padding is rounded up (typically by well
        under 1% of the length) so the FFT length has no large prime factor,
        which makes the transforms several times faster. The padding travels
        with the Subbands (and any Envelopes derived from them) and is
        removed on output, so ``.data`` and :meth:`Subbands.synthesize` have
        the sound's own length, and analysis followed by synthesis is exact.

        ``pad=0`` makes the analysis circular, which is what you want for
        periodic signals and for texture synthesis (seamless loops). A number
        pads by that many seconds.
        """

        n_pad = self._pad_samples(pad, sound.fs, sound.data.shape[0])
        padded = np.pad(sound.data, ((n_pad, n_pad), (0, 0))) if n_pad else sound.data
        n_padded = padded.shape[0]
        transfer = self.rfft_response(n_padded, sound.fs)  # (F, B)
        with threads():
            spectrum = sp_fft.rfft(padded, axis=0)  # (F, C)
            bands = sp_fft.irfft(spectrum[:, None, :] * transfer[:, :, None], n=n_padded, axis=0)  # (n, B, C)
        return Subbands(bands, sound.fs, self, pad=n_pad)

    def synthesize(self, coefs: Subbands) -> Sound:
        """Canonical dual synthesis: filter each band with ``conj(H) / s`` and
        sum (with ``tight = True``, ``s = 1`` and this is re-filtering with
        ``H``), then remove the padding.

        The frame is the circular operator on the (padded) grid the
        coefficients live on. Unmodified coefficients reconstruct exactly.
        For modified ones the result is the least-squares fit on that grid,
        cropped; with ``pad=0`` this is the canonical least squares on the
        signal itself. With padding on a non-tight bank it differs slightly
        (about 1% on masked coefficients) from the canonical least squares on
        the unpadded signal, because masked energy that lands in the padding is
        discarded rather than refit; this is a deliberate choice, exact and
        non-iterative. Raises if the bank leaves a frequency
        uncovered (A = 0).
        """
        padded_bands, fs, n_pad = coefs._full, coefs.fs, coefs.pad
        n_padded = padded_bands.shape[0]
        transfer = self.rfft_response(n_padded, fs)
        if self.tight:
            with threads():
                weighted_spectra = sp_fft.rfft(padded_bands, axis=0) * transfer[:, :, None]
                signal = sp_fft.irfft(weighted_spectra.sum(axis=1), n=n_padded, axis=0)
        else:
            power_sum = np.sum(np.abs(transfer) ** 2, axis=1)
            _check_frame(float(power_sum.min()), float(power_sum.max()), type(self).__name__)
            with threads():
                weighted_spectra = sp_fft.rfft(padded_bands, axis=0) * np.conj(transfer)[:, :, None]
                signal = sp_fft.irfft(weighted_spectra.sum(axis=1) / power_sum[:, None], n=n_padded, axis=0)
        return Sound(signal[n_pad : n_padded - n_pad], fs)

    def frame_bounds(self, n_samples: int, fs: float, pad: float | str = "auto") -> tuple[float, float]:
        """``(min s, max s)`` on the DFT grid that :meth:`analyze` would use
        for a signal of ``n_samples`` at ``fs``, including its padding: the
        bounds of the grid the coefficients actually live on, which depends on
        the signal length and rate (hence a method, not a property). Tight banks return ``(1.0, 1.0)``."""
        if self.tight:
            return (1.0, 1.0)
        power_sum = self.frame_power(int(n_samples) + 2 * self._pad_samples(pad, fs, n_samples), fs)
        return (float(power_sum.min()), float(power_sum.max()))

    def energy(self, coefs: Subbands) -> np.ndarray:
        """Sum of squares of the subbands, padding included, per channel.
        Subbands are real, so every sample has weight 1."""
        return np.sum(coefs._full**2, axis=(0, 1))

    def adjoint(self, coefs: Subbands) -> Sound:
        """Filter each band with ``conj(H)`` and sum, with no division by
        ``s``, then remove the padding (the adjoint of zero-padding is
        cropping). For a tight bank this equals :meth:`synthesize`."""
        padded_bands, fs, n_pad = coefs._full, coefs.fs, coefs.pad
        n_padded = padded_bands.shape[0]
        with threads():
            weighted_spectra = (
                sp_fft.rfft(padded_bands, axis=0) * np.conj(self.rfft_response(n_padded, fs))[:, :, None]
            )
            signal = sp_fft.irfft(weighted_spectra.sum(axis=1), n=n_padded, axis=0)
        return Sound(signal[n_pad : n_padded - n_pad], fs)


@lru_cache(maxsize=64)
def _ringing_samples(filterbank: Filterbank, fs: float, level_db: float) -> int:
    n_grid = 1 << int(np.ceil(np.log2(4 * fs)))  # a 4 s grid: impulse responses up to 2 s each side
    impulse_mag = np.abs(np.fft.irfft(filterbank.rfft_response(n_grid, fs), n=n_grid, axis=0)[: n_grid // 2])
    above = impulse_mag > impulse_mag.max(axis=0, keepdims=True) * 10 ** (level_db / 20)
    last_above = np.array(
        [np.flatnonzero(band_above).max() if band_above.any() else 0 for band_above in above.T]
    )
    return int(last_above.max()) + 1


@dataclass(frozen=True)
class CosineFilterbank(Filterbank):
    """``n_bands`` bandpass filters equally spaced on some frequency scale
    between ``f_lo`` and ``f_hi`` [Hz], plus a lowpass below ``f_lo`` and a
    highpass above ``f_hi`` (``n_bands + 2`` filters in total). Subclasses
    define the scale.

    A tight :class:`~sonore.frames.filterbank.Filterbank`: the squared responses sum to
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
        knots = self._knots
        return float(knots[1] - knots[0])

    @property
    def cfs(self) -> np.ndarray:
        """Center frequencies [Hz] of all filters (edges at ``f_lo``/``f_hi``)."""
        return self.from_scale(self._knots)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Magnitude responses, shape ``(len(freqs), n_bands + 2)``."""
        with np.errstate(divide="ignore"):
            scale_pos = self.to_scale(np.asarray(freqs, float))[:, None]
        knots = self._knots
        distance = (scale_pos - knots[None, :]) / self.spacing  # distance from each center in band spacings
        transfer = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
        transfer[:, 0] = np.where(scale_pos[:, 0] <= knots[0], 1.0, transfer[:, 0])
        transfer[:, -1] = np.where(scale_pos[:, 0] >= knots[-1], 1.0, transfer[:, -1])
        return transfer


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
        n_bands = max(1, int(round(bands_per_octave * np.log2(f_hi / f_lo))) - 1)
        return cls(n_bands, f_lo, f_hi)


@dataclass(frozen=True)
class BandpassFilterbank(Filterbank):
    """``n_bands`` bandpass filters with centers equally spaced on a scale from
    ``f_lo`` to ``f_hi`` [Hz] (both included), plus, with ``edges=True``, a
    zero-phase lowpass and highpass that make the bank a well-conditioned
    frame on the whole band, as the cosine banks' two extra filters do.
    Subclasses define the scale and :meth:`band_response`.

    The edge filters have magnitude ``sqrt(s_floor)``, where ``s_floor`` is
    the smallest ``s = sum_k |H_k|**2`` of the bandpass filters between the
    lowest and highest center, and a raised-cosine transition
    ``edge_width`` filter spacings wide (on the bank's scale): the lowpass
    falls to 0 at the lowest center, the highpass rises from 0 at the highest.
    One spacing trades frame bounds (A/B about 0.5-0.6 for typical banks)
    against ringing (2-3 times the bank's own, which sets the padding);
    filling the gap exactly gives better bounds but rings for about half a
    second. They depend only on the bank, not on ``fs`` or the signal
    length. Their nominal centers in :attr:`cfs` are where the transitions
    start, ``edge_width`` spacings outside the band (0 Hz at the lowest).

    ``edges=False`` gives the bare bank (for cochleagrams). It is badly
    conditioned or not a frame at all: ``analyze`` still works, but
    ``synthesize`` may refuse.
    """

    n_bands: int = 30
    f_lo: float = 50.0
    f_hi: float = 8000.0
    _: KW_ONLY
    edges: bool = True
    edge_width: float = 1.0
    unit = "unit"

    def __post_init__(self):
        if self.n_bands < 2:
            raise ValueError("n_bands must be at least 2")
        if not 0 < self.f_lo < self.f_hi:
            raise ValueError("need 0 < f_lo < f_hi")
        if not self.edge_width > 0:
            raise ValueError("edge_width must be positive")

    @staticmethod
    def to_scale(freq: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def from_scale(value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def band_response(self, freqs: np.ndarray) -> np.ndarray:
        """Responses of the bandpass filters at ``freqs`` [Hz] (all >= 0),
        shape ``(len(freqs), n_bands)``."""
        raise NotImplementedError

    @property
    def _knots(self) -> np.ndarray:
        return np.linspace(self.to_scale(self.f_lo), self.to_scale(self.f_hi), self.n_bands)

    @property
    def spacing(self) -> float:
        """Distance between adjacent bandpass centers, in scale units."""
        knots = self._knots
        return float(knots[1] - knots[0])

    @property
    def band_cfs(self) -> np.ndarray:
        """Center frequencies [Hz] of the bandpass filters."""
        return self.from_scale(self._knots)

    @property
    def n_filters(self) -> int:
        return self.n_bands + 2 if self.edges else self.n_bands

    @property
    def _edge_corners(self) -> tuple[float, float]:
        """Where the lowpass starts to fall and the highpass stops rising [Hz]."""
        knots, edge_offset = self._knots, self.edge_width * self.spacing
        return max(float(self.from_scale(knots[0] - edge_offset)), 0.0), float(
            self.from_scale(knots[-1] + edge_offset)
        )

    @property
    def cfs(self) -> np.ndarray:
        """Center frequencies [Hz]: the edge filters' nominal centers (see the
        class docstring) around the bandpass centers."""
        if not self.edges:
            return self.band_cfs
        lo, hi = self._edge_corners
        return np.concatenate([[lo], self.band_cfs, [hi]])

    @property
    def s_floor(self) -> float:
        """The smallest ``s`` of the bandpass filters between the lowest and
        highest center, on a grid of 64 points per spacing; the edge filters'
        squared magnitude."""
        return _s_floor(self)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Responses at ``freqs`` [Hz], shape ``(len(freqs), n_filters)``,
        edges first and last."""
        freq_array = np.asarray(freqs, float)
        transfer = self.band_response(freq_array)
        if not self.edges:
            return transfer
        lo, hi = self._edge_corners
        cf_lo, cf_hi = self.band_cfs[[0, -1]]
        edge_gain = np.sqrt(self.s_floor)
        low = edge_gain * _taper((freq_array - lo) / (cf_lo - lo))
        high = edge_gain * (1 - _taper((freq_array - cf_hi) / (hi - cf_hi)))
        return np.concatenate([low[:, None], transfer, high[:, None]], axis=1)


def _taper(u: np.ndarray) -> np.ndarray:
    """Raised cosine: 1 for ``u <= 0``, 0 for ``u >= 1``."""
    return 0.5 * (1 + np.cos(np.pi * np.clip(u, 0, 1)))


@lru_cache(maxsize=64)
def _s_floor(filterbank: BandpassFilterbank) -> float:
    knots = filterbank._knots
    grid = filterbank.from_scale(np.linspace(knots[0], knots[-1], 64 * (filterbank.n_bands - 1) + 1))
    return float(np.min(np.sum(np.abs(filterbank.band_response(grid)) ** 2, axis=1)))


@dataclass(frozen=True)
class GammatoneFilterbank(BandpassFilterbank):
    """4th-order gammatone filters (Patterson et al., 1992) equally spaced on
    the ERB-number scale, with bandwidth parameter ``b = 1.019 ERB(cf)``, plus
    edge filters (see :class:`BandpassFilterbank`). ERB is Glasberg & Moore's
    (1990); the factor 1.019 is the usual gammatone convention, commonly
    credited to Patterson et al. and Slaney (1993), a source not checked here.

    The responses are the exact Fourier transform of the impulse response
    ``t**3 exp(-2 pi b t) cos(2 pi cf t)``, ``t >= 0``, not an IIR
    approximation, normalized to unit gain at ``cf``. ``phase="causal"`` (the
    default) is that filter, with its CF-dependent delay (group delay
    ``4 / (2 pi b)`` at ``cf``); ``phase="zero"`` keeps the magnitude and
    aligns onsets across bands. Synthesis is exact either way: the dual
    filters with ``conj(H)``, which undoes the delay.
    """

    _: KW_ONLY
    phase: str = "causal"
    unit = "ERB"
    order = 4
    bandwidth_factor = 1.019

    def __post_init__(self):
        super().__post_init__()
        if self.phase not in ("causal", "zero"):
            raise ValueError("phase must be 'causal' or 'zero'")

    @staticmethod
    def to_scale(freq):
        return freq_to_erb(freq)

    @staticmethod
    def from_scale(value):
        return erb_to_freq(value)

    @property
    def b(self) -> np.ndarray:
        """Bandwidth parameter ``b`` [Hz] of each bandpass filter."""
        return self.bandwidth_factor * erb_bandwidth(self.band_cfs)

    @property
    def envelope_peak_delay(self) -> np.ndarray:
        """Time [s] from an impulse to the peak of each filter's envelope, one
        per filter (as :attr:`cfs`): ``(order - 1) / (2 pi b)`` for causal
        bandpass filters, 0 for zero-phase ones and for the (zero-phase) edge
        filters. Drawing each band this much earlier shows a click as a
        vertical line (see ``plot_envelopes(align="peak")``). This is the
        envelope's peak, not the group delay ``order / (2 pi b)``, which is
        the envelope's centroid and leaves a visible sweep.
        """
        delays = (self.order - 1) / (2 * np.pi * self.b) if self.phase == "causal" else np.zeros(self.n_bands)
        return np.concatenate([[0.0], delays, [0.0]]) if self.edges else delays

    def band_response(self, freqs):
        freq_col = np.asarray(freqs, float)[:, None]
        fc, b, n = self.band_cfs[None, :], self.b[None, :], self.order

        def gammatone(freq):
            return (b + 1j * (freq - fc)) ** -n + (b + 1j * (freq + fc)) ** -n

        normalized = gammatone(freq_col) / np.abs(gammatone(fc))  # the constant (n-1)!/(2 pi)^n/2 cancels
        return normalized if self.phase == "causal" else np.abs(normalized)


@dataclass(frozen=True)
class MorletFilterbank(BandpassFilterbank):
    """Morlet wavelets equally spaced in log2 frequency, plus edge filters
    (see :class:`BandpassFilterbank`).

    Each filter is a Gaussian in frequency with standard deviation
    ``cf / cycles``, minus the standard DC correction so that the response
    is exactly 0 at 0 Hz, normalized to unit gain at ``cf``; zero-phase.
    Because of the correction, the bare bank (``edges=False``) is not a frame:
    every response is 0 at DC, so A = 0.
    """

    f_lo: float = 50.0
    _: KW_ONLY
    cycles: float = 6.0
    unit = "oct"

    def __post_init__(self):
        super().__post_init__()
        if not self.cycles > 0:
            raise ValueError("cycles must be positive")

    @staticmethod
    def to_scale(freq):
        return np.log2(freq)

    @staticmethod
    def from_scale(value):
        return np.exp2(value)

    def band_response(self, freqs):
        freq_col = np.abs(np.asarray(freqs, float))[:, None]
        fc = self.band_cfs[None, :]
        sigma = fc / self.cycles
        gaussian = np.exp(-((freq_col - fc) ** 2) / (2 * sigma**2)) - np.exp(
            -(freq_col**2 + fc**2) / (2 * sigma**2)
        )
        return gaussian / (1 - np.exp(-(fc**2) / sigma**2))


def subbands(sound: Sound, n_bands: int = 30, f_lo: float = 50.0, f_hi: float | None = None) -> Subbands:
    """Convenience: build an :class:`ERBFilterbank` and analyze ``sound``.
    ``f_hi`` defaults to just under Nyquist."""
    f_hi = 0.95 * sound.fs / 2 if f_hi is None else min(f_hi, sound.fs / 2)
    return ERBFilterbank(n_bands, f_lo, f_hi).analyze(sound)


class _PaddedBands:
    """What Subbands and Envelopes share: one signal per filter of a filterbank,
    stored as ``(n_samples, n_bands, n_channels)`` with ``pad`` zero samples at
    each end that :attr:`data` hides. The array is a copy, read-only, so the
    caller's array stays writeable and the object can't change under its users."""

    def __init__(self, data, fs: float, filterbank: Filterbank, pad: int = 0):
        bands = np.array(data, dtype=float)
        if bands.ndim != 3 or bands.shape[1] != filterbank.n_filters:
            raise ValueError(
                f"expected shape (n_samples, {filterbank.n_filters}, n_channels), got {bands.shape}"
            )
        if 2 * pad >= bands.shape[0]:
            raise ValueError("padding is longer than the data")
        bands = self._checked(bands)
        bands.flags.writeable = False
        self._full, self.fs, self.filterbank, self.pad = bands, fs, filterbank, int(pad)

    def _checked(self, bands: np.ndarray) -> np.ndarray:
        """A subclass's own check on the band values; Envelopes clips round-off negatives."""
        return bands

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
        """Number of bands (including the two edge filters)."""
        return self._full.shape[1]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        n_samples, n_bands, n_channels = self.data.shape
        name = type(self).__name__
        return f"{name}({n_bands} bands, {n_samples / self.fs:.3f} s, {self.fs:g} Hz, {n_channels} ch)"


class Subbands(_PaddedBands):
    """The output of a :class:`~sonore.frames.filterbank.Filterbank`: one band-limited
    :class:`~sonore.Sound` per filter.

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

    def __getitem__(self, i: int) -> Sound:
        return Sound(self.data[:, i, :], self.fs)

    def _new(self, full: np.ndarray, pad: int | None = None) -> Subbands:
        return Subbands(full, self.fs, self.filterbank, self.pad if pad is None else pad)

    def _analytic(self) -> np.ndarray:
        with threads():
            return hilbert(self._full, axis=0)

    def envelopes(self, lowpass: float | None = None, fs: float | None = None):
        """Hilbert envelope of every band, as :class:`~sonore.views.envelopes.Envelopes`.
        Optionally lowpass-filtered [Hz] and then resampled to ``fs``."""
        from sonore.views.envelopes import Envelopes

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
        (:meth:`~sonore.frames.filterbank.Filterbank.synthesize`): the exact inverse of
        :meth:`~sonore.frames.filterbank.Filterbank.analyze`, and the least-squares
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
