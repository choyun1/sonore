"""Filterbanks: frames whose filters are given by their responses on the DFT
grid, and the subband representation they produce.

One class, :class:`Filterbank`, holds any undecimated bank: a frequency
:class:`Scale`, the filter centers as positions on that scale, and a
:class:`FilterType` that gives every filter's response. Functions return
the banks in common use, as :func:`~sonore.pure_tone` returns a Sound::

    so.cosine_filterbank(30, 50, 8000)                    # ERB-spaced cosines
    so.cosine_filterbank(f_lo=125, f_hi=8000, spacing=1/12, scale="octave")
    so.gammatone_filterbank(30, 50, 8000)
    so.morlet_filterbank(30, 50, 8000, cycles=6)

The frame operator of such a bank is diagonal in frequency,
``s(f) = sum_k |H_k(f)|**2``, and its canonical dual filters are ``H_k / s``
(Balazs et al., 2011). Whether ``s`` is constant (the bank is tight) is
measured on the grid the coefficients live on, never declared
(docs/design/frames/filterbanks.md).

``n_bands`` bandpass filters sit at equally spaced points strictly between
``f_lo`` and ``f_hi`` on the scale; with ``edges=True`` (the default) a
lowpass and a highpass take over below ``f_lo`` and above ``f_hi``, which
makes the bank a frame on the whole band.

The cosine filter type is a half-cycle cosine on the scale, spanning two band
spacings. Its squared responses sum to 1 (they are power complementary), so
filtering on analysis *and* synthesis reconstructs the input. McDermott &
Simoncelli (2011) used these filters on the ERB scale for sound texture,
with a lowpass and a highpass at the ends so that "the summed squared
frequency response of the filter bank was constant across frequency". The
construction comes from image processing: a squared response summing to one
is the "flat system response" of the steerable pyramid (Simoncelli &
Freeman, 1995), and its radial filters in Portilla & Simoncelli (2000) are
these cosines on a log2 scale. The gammatone and Morlet types are not tight;
their synthesis is the canonical dual.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache

import numpy as np
import scipy.fft as sp_fft
from scipy.signal import hilbert

from sonore.core.fft import fast_padding, threads
from sonore.core.sound import Sound
from sonore.core.utils import (
    _below_nyquist,
    _parabola_vertex,
    db_to_amp,
    erb_bandwidth,
    erb_to_freq,
    freq_to_erb,
    freq_to_mel,
    mel_to_freq,
)
from sonore.frames.frame import Frame, _check_frame

__all__ = [
    "Scale",
    "SCALES",
    "FilterType",
    "Cosine",
    "Gammatone",
    "Morlet",
    "Filterbank",
    "cosine_filterbank",
    "gammatone_filterbank",
    "morlet_filterbank",
    "subbands",
    "Subbands",
]

# A bank is tight when s varies by at most this fraction of its largest value.
# Cosine banks reach s = 1 only to about 1e-14 (rounding), and taking the tight
# path on a bank whose s ripples loses exactly that ripple, so this bounds the
# loss to 1e-12 of the signal (docs/design/frames/filterbanks.md, C1 and C9).
TIGHT_TOLERANCE = 1e-12


# ------------------------------------------------------------------ scales
def _identity(value):
    return value


@dataclass(frozen=True)
class Scale:
    """A frequency axis that filter centers are equally spaced on: its
    ``name``, the ``unit`` of one step on it, and the conversions to and
    from Hz."""

    name: str
    unit: str
    to_scale: Callable[[np.ndarray], np.ndarray]
    from_scale: Callable[[np.ndarray], np.ndarray]


#: The scales a bank can be built on by name: the ERB-number scale (Glasberg &
#: Moore, 1990), octaves (log2 frequency), the HTK mel scale, and Hz.
SCALES = {
    "erb": Scale("erb", "ERB", freq_to_erb, erb_to_freq),
    "octave": Scale("octave", "oct", np.log2, np.exp2),
    "mel": Scale("mel", "mel", freq_to_mel, mel_to_freq),
    "linear": Scale("linear", "Hz", _identity, _identity),
}


def _as_scale(scale: str | Scale) -> Scale:
    if isinstance(scale, Scale):
        return scale
    try:
        return SCALES[scale.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"scale must be one of {sorted(SCALES)} or a Scale, not {scale!r}") from None


# ------------------------------------------------------------- filter types
class FilterType:
    """The kind of filter a bank is made of: it gives every filter's
    response. A filter type defines
    :meth:`band_response`, the bandpass filters at the bank's centers; the
    edge filters default to the raised-cosine lowpass and highpass described
    under :meth:`responses`. Shapes must be hashable (a frozen dataclass)."""

    edge_width = 1.0

    def band_response(self, bank: Filterbank, freqs: np.ndarray) -> np.ndarray:
        """Responses of the bandpass filters at ``freqs`` [Hz] (all >= 0),
        shape ``(len(freqs), bank.n_bands)``."""
        raise NotImplementedError

    def edge_cfs(self, bank: Filterbank) -> tuple[float, float]:
        """Nominal centers [Hz] of the lowpass and highpass: where their
        transitions start, ``edge_width`` gaps outside the outermost bandpass
        centers (0 Hz at the lowest)."""
        knots = bank._knots
        lowest, highest = knots[1], knots[-2]
        low = bank.scale.from_scale(lowest - self.edge_width * (lowest - knots[0]))
        high = bank.scale.from_scale(highest + self.edge_width * (knots[-1] - highest))
        return max(float(low), 0.0), float(high)

    def responses(self, bank: Filterbank, freqs: np.ndarray) -> np.ndarray:
        """All of the bank's responses at ``freqs`` [Hz], shape
        ``(len(freqs), bank.n_filters)``, edges first and last.

        The default edge filters have magnitude ``sqrt(s_floor)``, where
        ``s_floor`` is the smallest ``s`` of the bandpass filters between the
        lowest and highest center, and a raised-cosine transition from the
        nominal center (:meth:`edge_cfs`) to the outermost bandpass center.
        One gap trades frame bounds (A/B about 0.5-0.6 for typical banks)
        against ringing (2-3 times the bank's own); filling the gap exactly
        gives better bounds but rings for about half a second
        (docs/design/frames/frames.md, step 2)."""
        transfer = self.band_response(bank, freqs)
        if not bank.edges:
            return transfer
        lo, hi = self.edge_cfs(bank)
        cf_lo, cf_hi = bank.band_cfs[[0, -1]]
        edge_gain = np.sqrt(_s_floor(bank))
        low = edge_gain * _taper((freqs - lo) / (cf_lo - lo))
        high = edge_gain * (1 - _taper((freqs - cf_hi) / (hi - cf_hi)))
        return np.concatenate([low[:, None], transfer, high[:, None]], axis=1)


def _taper(u: np.ndarray) -> np.ndarray:
    """Raised cosine: 1 for ``u <= 0``, 0 for ``u >= 1``."""
    return 0.5 * (1 + np.cos(np.pi * np.clip(u, 0, 1)))


@lru_cache(maxsize=64)
def _s_floor(bank: Filterbank) -> float:
    band_knots = bank._knots[1:-1]
    n_grid = 64 * (len(band_knots) - 1) + 1
    grid = bank.scale.from_scale(np.linspace(band_knots[0], band_knots[-1], n_grid))
    return float(np.min(np.sum(np.abs(bank.filter_type.band_response(bank, grid)) ** 2, axis=1)))


@dataclass(frozen=True)
class Cosine(FilterType):
    """Half-cycle cosines on the bank's scale, each ``width`` gaps to either
    side of its center (``width=1``: zero at the neighbouring centers). The
    lowpass and highpass are the cosines beyond the edges summed in power, so
    they are flat outside ``f_lo..f_hi``.

    With ``width=1`` the squared responses sum to 1 on any increasing
    centers. Wider cosines on equally spaced centers sum to ``width`` when
    ``2 * width`` is a whole number, and ripple otherwise; the bank measures
    which (:meth:`Filterbank.is_tight`)."""

    width: float = 1.0

    def __post_init__(self):
        if not self.width > 0:
            raise ValueError("width must be positive")

    def edge_cfs(self, bank: Filterbank) -> tuple[float, float]:
        """The outermost knots, ``f_lo`` and ``f_hi``, where the flat parts begin."""
        return float(bank.scale.from_scale(bank._knots[0])), float(bank.scale.from_scale(bank._knots[-1]))

    def band_response(self, bank: Filterbank, freqs: np.ndarray) -> np.ndarray:
        return self.responses(bank, freqs, edges=False)

    def responses(self, bank: Filterbank, freqs: np.ndarray, edges: bool | None = None) -> np.ndarray:
        with np.errstate(divide="ignore"):
            scale_pos = bank.scale.to_scale(freqs)[:, None]
        knots = bank._knots
        spacing = bank.spacing
        if spacing is None:
            if self.width != 1:
                raise ValueError("cosines wider than one gap need equally spaced centers")
            transfer = _cosine_gaps(scale_pos[:, 0], knots)
        elif self.width == 1:
            distance = (scale_pos - knots[None, :]) / spacing  # distance from each center in band spacings
            transfer = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
            transfer[:, 0] = np.where(scale_pos[:, 0] <= knots[0], 1.0, transfer[:, 0])
            transfer[:, -1] = np.where(scale_pos[:, 0] >= knots[-1], 1.0, transfer[:, -1])
        else:
            transfer = self._wide(scale_pos[:, 0], knots, spacing)
        keep_edges = bank.edges if edges is None else edges
        return transfer if keep_edges else transfer[:, 1:-1]

    def _wide(self, scale_pos: np.ndarray, knots: np.ndarray, spacing: float) -> np.ndarray:
        width = self.width

        def cosine(distance):
            return np.where(
                np.abs(distance) < width, np.cos(np.pi / 2 * np.clip(distance / width, -1, 1)), 0.0
            )

        position = (scale_pos - knots[0]) / spacing  # in gaps above the lowest knot
        n_gaps = len(knots) - 1
        transfer = cosine(position[:, None] - np.arange(n_gaps + 1)[None, :])
        # Each edge filter is every cosine at or beyond its knot, summed in
        # power, held constant where the bandpass filters no longer reach.
        beyond = np.arange(int(np.ceil(2 * width)) + 1)
        below = np.maximum(position, 1 - width)
        transfer[:, 0] = np.sqrt(np.sum(cosine(below[:, None] + beyond[None, :]) ** 2, axis=1))
        above = np.minimum(position - n_gaps, width - 1)
        transfer[:, -1] = np.sqrt(np.sum(cosine(above[:, None] - beyond[None, :]) ** 2, axis=1))
        return transfer


def _cosine_gaps(scale_pos: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Cosine bank on any increasing knots: between knots ``k`` and ``k + 1``,
    with ``u`` the fractional position in that gap, filter ``k`` is
    ``cos(pi u / 2)`` and filter ``k + 1`` is ``sin(pi u / 2)``; flat
    lowpass and highpass outside."""
    transfer = np.zeros((len(scale_pos), len(knots)))
    gap = np.clip(np.searchsorted(knots, scale_pos, side="right") - 1, 0, len(knots) - 2)
    fraction = np.clip((scale_pos - knots[gap]) / (knots[gap + 1] - knots[gap]), 0, 1)
    rows = np.arange(len(scale_pos))
    transfer[rows, gap] = np.cos(np.pi / 2 * fraction)
    transfer[rows, gap + 1] = np.cos(np.pi / 2 * (1 - fraction))
    below, above = scale_pos <= knots[0], scale_pos >= knots[-1]
    transfer[below | above] = 0.0
    transfer[below, 0] = 1.0
    transfer[above, -1] = 1.0
    return transfer


@dataclass(frozen=True)
class Gammatone(FilterType):
    """Gammatone filters of the given ``order`` with bandwidth parameter
    ``b = bandwidth_factor * ERB(cf)`` (Patterson et al., 1992). ERB is
    Glasberg & Moore's (1990); the factor 1.019 for order 4 is the usual
    convention, commonly credited to Patterson et al. and Slaney (1993), a
    source not checked here.

    The responses are the exact Fourier transform of the impulse response
    ``t**(order - 1) exp(-2 pi b t) cos(2 pi cf t)``, ``t >= 0``, not an IIR
    approximation, normalized to unit gain at ``cf``. ``phase="causal"`` is
    that filter, with its CF-dependent delay (group delay ``order / (2 pi b)``
    at ``cf``); ``phase="zero"`` keeps the magnitude and aligns onsets across
    bands. Synthesis is exact either way: the dual filters with ``conj(H)``,
    which undoes the delay."""

    order: int = 4
    bandwidth_factor: float = 1.019
    phase: str = "causal"
    edge_width: float = 1.0

    def __post_init__(self):
        if self.phase not in ("causal", "zero"):
            raise ValueError("phase must be 'causal' or 'zero'")
        if not (self.order >= 1 and self.bandwidth_factor > 0 and self.edge_width > 0):
            raise ValueError("order, bandwidth_factor and edge_width must be positive")

    def band_response(self, bank: Filterbank, freqs: np.ndarray) -> np.ndarray:
        freq_col = np.asarray(freqs, float)[:, None]
        fc = bank.band_cfs[None, :]
        b = self.bandwidth_factor * erb_bandwidth(bank.band_cfs)[None, :]

        def gammatone(freq):
            return (b + 1j * (freq - fc)) ** -self.order + (b + 1j * (freq + fc)) ** -self.order

        normalized = gammatone(freq_col) / np.abs(gammatone(fc))  # the constant (n-1)!/(2 pi)^n/2 cancels
        return normalized if self.phase == "causal" else np.abs(normalized)


@dataclass(frozen=True)
class Morlet(FilterType):
    """Morlet wavelets: a Gaussian in frequency with standard deviation
    ``cf / cycles``, minus the standard DC correction so that the response is
    exactly 0 at 0 Hz, normalized to unit gain at ``cf``; zero-phase. Because
    of the correction a bank without edge filters is not a frame: every
    response is 0 at DC, so A = 0."""

    cycles: float = 6.0
    edge_width: float = 1.0

    def __post_init__(self):
        if not (self.cycles > 0 and self.edge_width > 0):
            raise ValueError("cycles and edge_width must be positive")

    def band_response(self, bank: Filterbank, freqs: np.ndarray) -> np.ndarray:
        freq_col = np.abs(np.asarray(freqs, float))[:, None]
        fc = bank.band_cfs[None, :]
        sigma = fc / self.cycles
        gaussian = np.exp(-((freq_col - fc) ** 2) / (2 * sigma**2)) - np.exp(
            -(freq_col**2 + fc**2) / (2 * sigma**2)
        )
        return gaussian / (1 - np.exp(-(fc**2) / sigma**2))


# --------------------------------------------------------------- the bank
@dataclass(frozen=True)
class Filterbank(Frame):
    """Undecimated filters applied by multiplication on the DFT grid.

    ``knots`` are positions on ``scale``: the bandpass centers, with one more
    point at each end (``f_lo`` and ``f_hi``) marking where the edge filters
    take over. ``filter_type`` gives the responses; ``edges`` says whether the bank
    has its lowpass and highpass. Build one with :func:`cosine_filterbank`,
    :func:`gammatone_filterbank` or :func:`morlet_filterbank`, or directly
    from any :class:`FilterType`.

    Responses are zero-phase or, more generally, satisfy
    ``H(-f) = conj(H(f))``; only ``f >= 0`` is ever evaluated, and the
    subbands are real.
    """

    scale: Scale
    knots: tuple[float, ...]
    filter_type: FilterType
    _: KW_ONLY
    edges: bool = True

    def __post_init__(self):
        object.__setattr__(self, "scale", _as_scale(self.scale))
        knots = tuple(float(knot) for knot in self.knots)
        object.__setattr__(self, "knots", knots)
        if len(knots) < 3:
            raise ValueError("need at least one bandpass filter (three knots)")
        if not (np.all(np.isfinite(knots)) and np.all(np.diff(knots) > 0)):
            raise ValueError("knots must be finite and increasing")

    @property
    def _knots(self) -> np.ndarray:
        return np.asarray(self.knots)

    @property
    def n_bands(self) -> int:
        """Number of bandpass filters."""
        return len(self.knots) - 2

    @property
    def n_filters(self) -> int:
        """Number of filters (the band axis of :class:`~sonore.frames.filterbank.Subbands`)."""
        return self.n_bands + 2 if self.edges else self.n_bands

    @property
    def f_lo(self) -> float:
        """Lowest knot [Hz]: below it the lowpass takes over."""
        return float(self.scale.from_scale(self.knots[0]))

    @property
    def f_hi(self) -> float:
        """Highest knot [Hz]: above it the highpass takes over."""
        return float(self.scale.from_scale(self.knots[-1]))

    @property
    def unit(self) -> str:
        """The scale's unit, e.g. ``"ERB"`` or ``"oct"``."""
        return self.scale.unit

    @property
    def spacing(self) -> float | None:
        """Distance between adjacent knots in scale units, or None if they are
        not equally spaced."""
        knots = self._knots
        if not np.array_equal(knots, np.linspace(knots[0], knots[-1], len(knots))):
            return None
        return float(knots[1] - knots[0])

    @property
    def band_cfs(self) -> np.ndarray:
        """Center frequencies [Hz] of the bandpass filters."""
        return self.scale.from_scale(self._knots[1:-1])

    @property
    def cfs(self) -> np.ndarray:
        """Center frequency [Hz] of each filter, shape ``(n_filters,)``; the
        edge filters' nominal centers first and last."""
        if not self.edges:
            return self.band_cfs
        lo, hi = self.filter_type.edge_cfs(self)
        return np.concatenate([[lo], self.band_cfs, [hi]])

    @property
    def s_floor(self) -> float:
        """The smallest ``s`` of the bandpass filters between the lowest and
        highest center, on a grid of 64 points per gap."""
        return _s_floor(self)

    @property
    def envelope_peak_delay(self) -> np.ndarray:
        """Time [s] from an impulse to the peak of each filter's Hilbert
        envelope, one per filter (as :attr:`cfs`), measured from the impulse
        responses and refined between samples; 0 for zero-phase filters.
        Drawing each band this much earlier shows a click as a vertical line
        (see ``plot_envelopes(align="peak")``). For a causal gammatone it is
        ``(order - 1) / (2 pi b)``, the envelope's peak, not the group delay
        ``order / (2 pi b)``."""
        return _envelope_peak_delays(self)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Responses at ``freqs`` [Hz] (all >= 0), shape ``(len(freqs), n_filters)``."""
        return self.filter_type.responses(self, np.asarray(freqs, float))

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

    def _tight_gain(self, n: int, fs: float) -> float | None:
        """The constant value of ``s`` on this grid, or None if ``s`` varies
        by more than :data:`TIGHT_TOLERANCE` of its largest value. Exactly 1.0
        when every ``s`` is within the tolerance of 1."""
        return _tight_gain(self, int(n), float(fs))

    def is_tight(self, n_samples: int, fs: float, pad: float | str = "auto") -> bool:
        """Whether ``s`` is constant (to :data:`TIGHT_TOLERANCE`) on the grid
        :meth:`analyze` would use for ``n_samples`` at ``fs``, padding
        included. Synthesis is then re-filtering, divided by that constant."""
        return self._tight_gain(int(n_samples) + 2 * self._pad_samples(pad, fs, n_samples), fs) is not None

    def ringing(self, fs: float, level_db: float = -60.0) -> int:
        """How long [samples] the filters ring: the longest time, over all
        filters, before the zero-phase impulse response stays below
        ``level_db`` re its peak (capped at 2 s). The minimum padding of
        ``pad="auto"``."""
        return _ringing_samples(self, float(fs), float(level_db))

    def _pad_samples(self, pad: float | str, fs: float, n_samples: int) -> int:
        if pad == "auto":
            # at least the ringing, rounded up to an FFT length without large prime factors
            return fast_padding(int(n_samples), self.ringing(fs))
        return int(round(float(pad) * fs))

    def analyze(self, sound: Sound, pad: float | str = "auto") -> Subbands:
        """Split ``sound`` into subbands (via FFT).

        By default the sound is zero-padded by at least the filters' ringing
        time (:meth:`ringing`), so filter ringing near one end can't wrap
        around to the other. The padding is rounded up (by a median 0.3% of
        the length, at most about 4%, for a 30-band ERB bank on 1 to 10 s)
        so the FFT length has no large prime factor, which makes the
        transforms about four times faster than at a prime length (measured
        by tools/measure_docstring_numbers.py). The padding travels
        with the Subbands (and any Envelopes derived from them) and is
        removed on output, so ``.data`` and :meth:`Subbands.to_sound` have
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
        sum, then remove the padding. When ``s`` is constant on the grid
        (:meth:`is_tight`) this is re-filtering with ``H`` and dividing by
        that constant.

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
        gain = self._tight_gain(n_padded, fs)
        if gain is not None:
            with threads():
                weighted_spectra = sp_fft.rfft(padded_bands, axis=0) * transfer[:, :, None]
                summed = weighted_spectra.sum(axis=1)
                if gain != 1.0:
                    summed = summed / gain
                signal = sp_fft.irfft(summed, n=n_padded, axis=0)
        else:
            power_sum = np.sum(np.abs(transfer) ** 2, axis=1)
            _check_frame(float(power_sum.min()), float(power_sum.max()), "this filterbank")
            with threads():
                weighted_spectra = sp_fft.rfft(padded_bands, axis=0) * np.conj(transfer)[:, :, None]
                signal = sp_fft.irfft(weighted_spectra.sum(axis=1) / power_sum[:, None], n=n_padded, axis=0)
        return Sound(signal[n_pad : n_padded - n_pad], fs)

    def frame_bounds(self, n_samples: int, fs: float, pad: float | str = "auto") -> tuple[float, float]:
        """``(min s, max s)`` on the DFT grid that :meth:`analyze` would use
        for a signal of ``n_samples`` at ``fs``, including its padding: the
        bounds of the grid the coefficients actually live on, which depends on
        the signal length and rate (hence a method, not a property). A tight
        bank returns its constant twice, ``(1.0, 1.0)`` for the cosine banks."""
        n_grid = int(n_samples) + 2 * self._pad_samples(pad, fs, n_samples)
        gain = self._tight_gain(n_grid, fs)
        if gain is not None:
            return (gain, gain)
        power_sum = self.frame_power(n_grid, fs)
        return (float(power_sum.min()), float(power_sum.max()))

    def energy(self, coefs: Subbands) -> np.ndarray:
        """Sum of squares of the subbands, padding included, per channel.
        Subbands are real, so every sample has weight 1."""
        return np.sum(coefs._full**2, axis=(0, 1))

    def adjoint(self, coefs: Subbands) -> Sound:
        """Filter each band with ``conj(H)`` and sum, with no division by
        ``s``, then remove the padding (the adjoint of zero-padding is
        cropping). For a bank with ``s = 1`` this equals :meth:`synthesize`."""
        padded_bands, fs, n_pad = coefs._full, coefs.fs, coefs.pad
        n_padded = padded_bands.shape[0]
        with threads():
            weighted_spectra = (
                sp_fft.rfft(padded_bands, axis=0) * np.conj(self.rfft_response(n_padded, fs))[:, :, None]
            )
            signal = sp_fft.irfft(weighted_spectra.sum(axis=1), n=n_padded, axis=0)
        return Sound(signal[n_pad : n_padded - n_pad], fs)


@lru_cache(maxsize=64)
def _tight_gain(bank: Filterbank, n: int, fs: float) -> float | None:
    power_sum = bank.frame_power(n, fs)
    largest = float(power_sum.max())
    if np.all(np.abs(power_sum - 1.0) <= TIGHT_TOLERANCE):
        return 1.0
    if largest > 0 and largest - float(power_sum.min()) <= TIGHT_TOLERANCE * largest:
        return float(np.median(power_sum))
    return None


@lru_cache(maxsize=64)
def _ringing_samples(bank: Filterbank, fs: float, level_db: float) -> int:
    n_grid = 1 << int(np.ceil(np.log2(4 * fs)))  # a 4 s grid: impulse responses up to 2 s each side
    impulse_mag = np.abs(np.fft.irfft(bank.rfft_response(n_grid, fs), n=n_grid, axis=0)[: n_grid // 2])
    above = impulse_mag > impulse_mag.max(axis=0, keepdims=True) * db_to_amp(level_db)
    last_above = np.array(
        [np.flatnonzero(band_above).max() if band_above.any() else 0 for band_above in above.T]
    )
    return int(last_above.max()) + 1


@lru_cache(maxsize=64)
def _envelope_peak_delays(bank: Filterbank) -> np.ndarray:
    # A rate well above every filter, and a 2 s grid so even slow filters'
    # envelopes peak long before the impulse response wraps around.
    fs = max(64000.0, 4.0 * float(np.max(bank.cfs)))
    n_grid = 1 << int(np.ceil(np.log2(2 * fs)))
    transfer = bank.rfft_response(n_grid, fs)
    delays = np.zeros(bank.n_filters)
    if not np.iscomplexobj(transfer):
        return delays  # every filter is zero-phase: its envelope peaks at t = 0
    with threads():
        envelopes = np.abs(hilbert(np.fft.irfft(transfer, n=n_grid, axis=0), axis=0))
    for k in range(bank.n_filters):
        if not np.any(transfer[:, k].imag):
            continue  # zero-phase
        envelope = envelopes[:, k]
        peak = int(np.argmax(envelope))
        offset = _parabola_vertex(envelope[peak - 1], envelope[peak], envelope[(peak + 1) % n_grid])
        position = peak + float(offset)
        delays[k] = (position - n_grid if position > n_grid / 2 else position) / fs
    return delays


# ------------------------------------------------------------- factories
def _knots(
    scale: Scale,
    n_bands: int | None,
    f_lo: float,
    f_hi: float,
    spacing: float | None,
    centers,
) -> tuple[float, ...]:
    if centers is not None:
        if n_bands is not None or spacing is not None:
            raise ValueError("give centers, or n_bands or spacing, not both")
        with np.errstate(divide="ignore", invalid="ignore"):
            knots = scale.to_scale(np.asarray(centers, float))
        if not np.all(np.isfinite(knots)):
            raise ValueError(
                f"every center must lie on the {scale.name} scale (0 Hz has no place in octaves)"
            )
        return tuple(knots)
    if not f_lo < f_hi:
        raise ValueError("need f_lo < f_hi")
    with np.errstate(divide="ignore", invalid="ignore"):
        lowest, highest = scale.to_scale(f_lo), scale.to_scale(f_hi)
    if not (np.isfinite(lowest) and np.isfinite(highest)):
        raise ValueError(
            f"f_lo = {f_lo:g} Hz is not on the {scale.name} scale (0 Hz has no place in octaves)"
        )
    if spacing is not None:
        if n_bands is not None:
            raise ValueError("give n_bands or spacing, not both")
        if not spacing > 0:
            raise ValueError("spacing must be positive")
        n_bands = max(1, int(round((highest - lowest) / spacing)) - 1)
    n_bands = 30 if n_bands is None else int(n_bands)
    if n_bands < 1:
        raise ValueError("n_bands must be at least 1")
    return tuple(np.linspace(lowest, highest, n_bands + 2))


def cosine_filterbank(
    n_bands: int | None = None,
    f_lo: float = 50.0,
    f_hi: float = 8000.0,
    *,
    scale: str | Scale = "erb",
    spacing: float | None = None,
    centers=None,
    width: float = 1.0,
    edges: bool = True,
) -> Filterbank:
    """Half-cycle cosine filters (:class:`Cosine`) on ``scale``.

    ``n_bands`` bandpass filters (30 if neither it nor ``spacing`` is given)
    sit at equally spaced points strictly between ``f_lo`` and ``f_hi``
    [Hz]; ``spacing`` (in scale units, e.g. ``1/12`` octave) chooses
    ``n_bands`` instead. ``centers`` [Hz] gives every knot directly,
    increasing, the first and last being where the lowpass and highpass take
    over; the cosines then follow the gaps and stay tight. With
    ``edges=True`` the bank has ``n_bands + 2`` filters and its squared
    responses sum to 1 (``width=1``)."""
    scale = _as_scale(scale)
    return Filterbank(scale, _knots(scale, n_bands, f_lo, f_hi, spacing, centers), Cosine(width), edges=edges)


def gammatone_filterbank(
    n_bands: int | None = None,
    f_lo: float = 50.0,
    f_hi: float = 8000.0,
    *,
    scale: str | Scale = "erb",
    spacing: float | None = None,
    centers=None,
    phase: str = "causal",
    order: int = 4,
    bandwidth_factor: float = 1.019,
    edges: bool = True,
    edge_width: float = 1.0,
) -> Filterbank:
    """Gammatone filters (:class:`Gammatone`), equally spaced on ``scale``
    (the ERB-number scale by default), with the raised-cosine edge filters
    of :meth:`FilterType.responses`. ``n_bands``, ``f_lo``, ``f_hi``,
    ``spacing`` and ``centers`` place the filters as in
    :func:`cosine_filterbank`. ``edges=False`` gives the bare bank (for
    cochleagrams); it may be badly conditioned or not a frame at all, in
    which case ``analyze`` still works but ``synthesize`` refuses."""
    scale = _as_scale(scale)
    filter_type = Gammatone(order, bandwidth_factor, phase, edge_width)
    return Filterbank(scale, _knots(scale, n_bands, f_lo, f_hi, spacing, centers), filter_type, edges=edges)


def morlet_filterbank(
    n_bands: int | None = None,
    f_lo: float = 50.0,
    f_hi: float = 8000.0,
    *,
    scale: str | Scale = "octave",
    spacing: float | None = None,
    centers=None,
    cycles: float = 6.0,
    edges: bool = True,
    edge_width: float = 1.0,
) -> Filterbank:
    """Morlet wavelets (:class:`Morlet`), equally spaced on ``scale`` (log2
    frequency by default), with the raised-cosine edge filters of
    :meth:`FilterType.responses`. ``n_bands``, ``f_lo``, ``f_hi``,
    ``spacing`` and ``centers`` place the filters as in
    :func:`cosine_filterbank`. Without edges the bank is not a frame (A = 0
    at DC)."""
    scale = _as_scale(scale)
    filter_type = Morlet(cycles, edge_width)
    return Filterbank(scale, _knots(scale, n_bands, f_lo, f_hi, spacing, centers), filter_type, edges=edges)


def subbands(sound: Sound, n_bands: int = 30, f_lo: float = 50.0, f_hi: float | None = None) -> Subbands:
    """Convenience: build an ERB :func:`cosine_filterbank` and analyze ``sound``.
    ``f_hi`` defaults to just under Nyquist."""
    f_hi = _below_nyquist(sound.fs) if f_hi is None else min(f_hi, sound.fs / 2)
    return cosine_filterbank(n_bands, f_lo, f_hi).analyze(sound)


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

    so a vocoder is ``(speech.envelopes() * carrier.tfs()).to_sound()``.
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

    def __mul__(self, other):
        """``subbands * mask`` (a :class:`~sonore.views.mask.Mask` made for
        these subbands' grid): the masked subbands."""
        from sonore.views.mask import Mask  # views.mask imports Subbands from here

        if isinstance(other, Mask):
            return other.apply(self)
        return NotImplemented

    __rmul__ = __mul__

    def to_sound(self) -> Sound:
        """Back to a Sound with the filterbank's canonical dual
        (:meth:`~sonore.frames.filterbank.Filterbank.synthesize`): the exact inverse of
        :meth:`~sonore.frames.filterbank.Filterbank.analyze`, and the least-squares
        signal after the bands are modified. For the cosine banks this is
        re-filtering each band and summing; with the default padding,
        re-filtering can't wrap around either."""
        return self.filterbank.synthesize(self)

    def sum(self) -> Sound:
        """Add the bands without re-filtering. :meth:`to_sound` is almost
        always what you want; ``sum`` is for bands that were already shaped to
        add up correctly."""
        return Sound(self.data.sum(axis=1), self.fs)

    def plot(self, axes=None, channel: int = 0, **kwargs):
        """Stacked band waveforms (see :func:`sonore.plotting.plot_subbands`).
        For an image, plot the envelopes: ``sb.envelopes().plot()``."""
        from sonore.plotting import plot_subbands

        return plot_subbands(self, axes=axes, channel=channel, **kwargs)
