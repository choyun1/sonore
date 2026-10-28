"""Spectral envelopes, and moving them along frequency.

A spectral envelope is the smooth curve over a time window's spectrum with the
source taken out: the harmonics, their phase and everything finer than the
smoothing. (A *temporal* envelope, :mod:`sonore.views.temporal_envelope`, is the other
"envelope": smoothed along time, with the fine structure taken out.) Every
estimator in sonore returns the one class :class:`SpectralEnvelope`, power on a
grid of times and frequencies with the estimator's name in ``method`` and its
parameters in ``settings``: :func:`pitch_adaptive_envelope` here,
:meth:`Cepstrum.envelope_view <sonore.views.cepstrum.Cepstrum.envelope_view>`,
:meth:`MFCC.envelope_view <sonore.views.mfcc.MFCC.envelope_view>` and
:meth:`LPC.envelope_view <sonore.views.lpc.LPC.envelope_view>`.

:func:`pitch_adaptive_envelope` is CheapTrick (Morise, 2015), written from the paper:
a window three periods long, its power spectrum smoothed over 2 F0 / 3, and a
cepstral lifter that removes the harmonic ripple and restores the level the
smoothing took from the harmonic peaks. Where the paper leaves a choice open
(the window's gain, what happens below 0 Hz, unvoiced time windows, the FFT
size), the choice is stated in the function's docstring. It is not WORLD's
code: with the same ``q1`` its envelopes are within a few hundredths of a dB of
WORLD's (median over voiced time windows: 0.00 to 0.01 dB on two recorded
sentences, 0.03 dB on a breathy synthetic vowel), not equal.

An envelope is read as ``env(t, f)``, giving power at times ``t`` and
frequencies ``f`` with shape ``(n_channels, len(f), len(t))``: a
:class:`SpectralEnvelope`, anything else held on a grid
(:class:`~sonore.views.view.TimeFrequencyGrid`), or a plain function.
:class:`~sonore.views.aperiodicity.Aperiodicity` is read the same way.
:func:`warp_frequency` moves an envelope (or an aperiodicity) along frequency
without looking at how it was made, so any envelope and any synthesizer
combine: :func:`~sonore.sources.source_filter.source_filter_synthesize` and
:func:`~sonore.sources.waveforms.harmonic_complex` both take any envelope.
"""

from __future__ import annotations

import copy
import numbers
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from sonore.core.sound import Sound
from sonore.sources.waveforms import _contour_per_channel
from sonore.views.view import TimeFrequencyGrid, View

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from sonore.sources.waveforms import ContourLike

__all__ = ["SpectralEnvelope", "pitch_adaptive_envelope", "warp_frequency"]


_FLOOR_F0 = 71.0  # the lowest F0 analyzed; it sets the FFT size


_UNVOICED_F0 = 500.0  # unvoiced time windows are analyzed as if F0 were this


# Added to a power spectrum before a logarithm or a division. The windows have unit
# energy, so white noise of variance s**2 has power s**2 in every bin; 16-bit quantization
# noise on a +-1 scale is about (2**-15)**2 / 12 = 8e-11, so 1e-12 lies about 19 dB below
# the quietest noise a 16-bit recording holds and only changes bins that are numerically 0
# (digital silence, or the zeros beyond the ends of the sound).
_POWER_FLOOR = 1e-12


def _power_of_two_above(value: float) -> int:
    size = 1
    while size <= value:
        size *= 2
    return size


def _envelope_fft_size(fs: float, f0_floor: float) -> int:
    """The smallest power of two above ``3 fs / f0_floor``, so that a window three
    periods of ``f0_floor`` long fits."""
    return _power_of_two_above(3.0 * fs / f0_floor)


def _fft_frequencies(fs: float, n_fft: int) -> np.ndarray:
    """The ``n_fft // 2 + 1`` bin frequencies [Hz] from 0 to ``fs / 2``."""
    return np.arange(n_fft // 2 + 1) * fs / n_fft


def _windowed_segment(samples, center, window_function, half_length):
    """The samples around a (possibly fractional) ``center`` under a window.

    ``window_function(offset)`` gives the window at an offset in samples from the
    center. Every whole sample with ``|offset| < half_length`` is used, and samples
    beyond either end of the sound are zeros. Returns the offsets, the window and
    the windowed segment."""
    first = int(np.floor(center - half_length)) + 1
    last = int(np.ceil(center + half_length)) - 1
    indices = np.arange(first, last + 1)
    offsets = indices - center
    window = window_function(offsets)
    segment = np.zeros(len(indices))
    inside = (indices >= 0) & (indices < len(samples))
    segment[inside] = samples[indices[inside]]
    return offsets, window, window * segment


def _hann(length):
    """A Hann window ``length`` samples long (may be fractional), as a function of the
    offset from its center."""
    return lambda offset: 0.5 + 0.5 * np.cos(2.0 * np.pi * offset / length)


def _unit_energy_power(samples, center, window_function, half_length, n_fft):
    """The power spectrum of the samples under a window scaled to unit energy, so that
    its level does not change with the window's length."""
    _, window, segment = _windowed_segment(samples, center, window_function, half_length)
    return np.abs(np.fft.rfft(segment / np.sqrt(np.sum(window**2)), n_fft)) ** 2


def _smooth_rectangular(values, half_width_bins):
    """A moving average over ``+-half_width_bins`` of a one-sided spectrum.

    The average is the exact integral of the linearly interpolated spectrum over the
    width, from a cumulative sum. Values below 0 Hz and above Nyquist come from
    mirroring the spectrum about both, which for a real signal is its periodic
    two-sided spectrum."""
    n_half = len(values)
    n_fft = 2 * (n_half - 1)
    two_sided = np.concatenate([values, values[-2:0:-1]])
    margin = int(np.ceil(half_width_bins)) + 2
    extended = two_sided[np.mod(np.arange(-margin, n_half + margin), n_fft)]
    # the integral from the first extended bin up to each bin, by trapezoids
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (extended[1:] + extended[:-1]))])

    def integral_up_to(position):
        shifted = position + margin
        left = np.floor(shifted).astype(int)
        fraction = shifted - left
        slope = extended[left + 1] - extended[left]
        return cumulative[left] + extended[left] * fraction + 0.5 * slope * fraction**2

    centers = np.arange(n_half, dtype=float)
    average = (integral_up_to(centers + half_width_bins) - integral_up_to(centers - half_width_bins)) / (
        2.0 * half_width_bins
    )
    # An average of powers is never negative, but the difference of two large cumulative sums can
    # come out just below zero where the spectrum is empty, and its logarithm would be NaN.
    return np.maximum(average, 0.0)


# ------------------------------------------------------------ reading a grid
def _positions(query: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query value, the grid index below it, the one above, and the
    weight of the upper one; held at the ends of the grid."""
    position = np.interp(query, grid, np.arange(len(grid)))
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(grid) - 1)
    return lower, upper, position - lower


def _pointwise(log_values: np.ndarray, grid_t: np.ndarray, grid_f: np.ndarray, t, f) -> np.ndarray:
    """``log_values`` (one channel, ``(len(grid_f), len(grid_t))``) read at
    the points ``(t[i], f[i])``: linear in time and frequency, held beyond
    the ends of the grid."""
    t, f = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(f, dtype=float))
    t_lower, t_upper, t_weight = _positions(t.ravel(), grid_t)
    f_lower, f_upper, f_weight = _positions(f.ravel(), grid_f)
    below = log_values[f_lower, t_lower] * (1 - t_weight) + log_values[f_lower, t_upper] * t_weight
    above = log_values[f_upper, t_lower] * (1 - t_weight) + log_values[f_upper, t_upper] * t_weight
    return (below * (1 - f_weight) + above * f_weight).reshape(t.shape)


def _power_on_grid(
    view: TimeFrequencyGrid | Callable, times: np.ndarray, freqs: np.ndarray, shape
) -> np.ndarray:
    """``view``'s values at these times and frequencies, shape ``(n_channels,
    len(freqs), len(times))`` with ``shape`` naming the channel count: its own
    data when it is held on that grid already, otherwise read there as
    ``view(t, f)``."""
    held = getattr(view, "data", None)
    if (
        held is not None
        and np.shape(held) == shape
        and hasattr(view, "t")
        and hasattr(view, "f")
        and np.array_equal(view.t, times)
        and np.array_equal(view.f, freqs)
    ):
        return np.asarray(held, dtype=float)
    if not callable(view):
        raise TypeError(
            "the envelope must be a SpectralEnvelope (or any TimeFrequencyGrid) or a function envelope(t, f)"
        )
    values = np.asarray(view(times, freqs), dtype=float)
    n_channels = shape[0]
    if values.ndim != 3 or values.shape[1:] != shape[1:] or values.shape[0] not in (1, n_channels):
        raise ValueError(
            f"view(t, f) gave shape {values.shape}; expected (n_channels, {shape[1]}, {shape[2]})"
        )
    return np.repeat(values, n_channels, axis=0) if values.shape[0] != n_channels else values


# ------------------------------------------------------------ the views
class _GridView(View):
    """What the spectral envelope and the aperiodicity share: non-negative
    values on a grid of times and frequencies, ``data`` of shape
    ``(n_channels, len(f), len(t))``, read at any point by ``__call__``."""

    def __init__(self, data: ArrayLike, t: ArrayLike, f: ArrayLike):
        self.data = np.asarray(data, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.f = np.asarray(f, dtype=float)
        if self.data.ndim == 2:
            self.data = self.data[None]
        if self.data.shape[1:] != (len(self.f), len(self.t)):
            raise ValueError(
                f"data has shape {self.data.shape[1:]} per channel, but there are"
                f" {len(self.f)} frequencies and {len(self.t)} times"
            )
        if not np.all(np.isfinite(self.data)):
            raise ValueError(f"{type(self).__name__} values must be finite")
        if np.any(self.data < 0):
            raise ValueError(f"{type(self).__name__} values cannot be negative")

    @staticmethod
    def _power_db(power: np.ndarray) -> np.ndarray:
        """``10 log10`` of a power, floored at the smallest positive float so that a zero
        reads as a very low, finite level."""
        return 10 * np.log10(np.maximum(power, np.finfo(float).tiny))

    def _log(self) -> np.ndarray:
        """The natural logarithm of the data, floored at the smallest positive float."""
        return np.log(np.maximum(self.data, np.finfo(float).tiny))

    def __call__(self, t: ArrayLike, f: ArrayLike) -> np.ndarray:
        """The values at times ``t`` [s] and frequencies ``f`` [Hz], shape
        ``(n_channels, len(f), len(t))``: linear in time between time
        windows, linear in dB between frequencies, held beyond the ends of
        the grid."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        freqs = np.atleast_1d(np.asarray(f, dtype=float))
        t_lower, t_upper, t_weight = _positions(times, self.t)
        f_lower, f_upper, f_weight = _positions(freqs, self.f)
        log_values = self._log()
        f_weight = f_weight[:, None]
        at_freqs = log_values[:, f_lower] * (1 - f_weight) + log_values[:, f_upper] * f_weight
        return np.exp(at_freqs[:, :, t_lower] * (1 - t_weight) + at_freqs[:, :, t_upper] * t_weight)


class SpectralEnvelope(_GridView):
    """A spectral envelope: power on a grid of times :attr:`t` and frequencies
    :attr:`f`, ``data`` of shape ``(n_channels, len(f), len(t))``. A view: it
    keeps the envelope and drops the harmonics and the phase.

    Every estimator returns one, with its name in :attr:`method`
    (``"CheapTrick"``, ``"cepstrum"``, ``"MFCC"``, ``"LPC"``) and its
    parameters in :attr:`settings` (for CheapTrick ``q1``, ``f0_floor`` and
    ``unvoiced_f0``); one built by hand has neither. :attr:`fs` is the
    sampling rate of the analyzed sound, or None for an envelope made by
    hand. Calling it, ``env(t, f)``, reads the power at any times and
    frequencies, linear in time and in dB over frequency, held beyond the
    ends of the grid, so it goes wherever an envelope is taken:
    :func:`warp_frequency`,
    :func:`~sonore.sources.source_filter.source_filter_synthesize`,
    :func:`~sonore.sources.waveforms.harmonic_complex`.
    """

    discards = (
        "SpectralEnvelope discards the harmonics, the phase, and everything finer than the envelope's "
        "smoothing."
    )
    back_to_sound = (
        "so.source_filter_synthesize rebuilds an approximation of the voice from it together with an F0 "
        "track and an aperiodicity."
    )

    def __init__(
        self,
        data: ArrayLike,
        t: ArrayLike,
        f: ArrayLike,
        *,
        fs: float | None = None,
        method: str | None = None,
        settings: Mapping[str, float] | None = None,
    ):
        super().__init__(data, t, f)
        self.fs = None if fs is None else float(fs)
        self.method = method
        self.settings: dict[str, float] = {} if settings is None else dict(settings)

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        parts = [f"{n_freqs} freqs x {n_windows} time windows", f"{n_channels} ch"]
        if self.method is not None:
            parts.insert(0, self.method)
        parts += [f"{name} {value:g}" for name, value in self.settings.items()]
        return f"SpectralEnvelope({', '.join(parts)})"

    @property
    def n_fft(self) -> int | None:
        """The FFT size whose bins the frequencies are, ``2 (len(f) - 1)``,
        when :attr:`fs` is known and :attr:`f` is that FFT's grid from 0 to
        ``fs / 2``; None for an envelope on any other grid."""
        if self.fs is None:
            return None
        n_fft = 2 * (len(self.f) - 1)
        on_grid = n_fft > 0 and np.allclose(self.f, _fft_frequencies(self.fs, n_fft), rtol=1e-9, atol=1e-6)
        return n_fft if on_grid else None

    @property
    def db(self) -> np.ndarray:
        """``10 log10`` of the power."""
        return self._power_db(self.data)

    def amplitude(self, t: ArrayLike, f: ArrayLike, *, channel: int = 0) -> np.ndarray:
        """The amplitude (square root of the power) at the points ``(t[i],
        f[i])``, ``t`` and ``f`` of the same shape: the form
        :func:`~sonore.sources.waveforms.harmonic_complex` reads, one value
        per sample and harmonic. Linear in time and in dB over frequency,
        held beyond the first and last time windows."""
        return np.exp(0.5 * _pointwise(self._log()[channel], self.t, self.f, t, f))

    def plot(
        self, ax: Axes | None = None, *, channel: int = 0, db_range: float = 70.0, fmax: float | None = None
    ) -> Axes:
        """The envelope in dB as a time-frequency image."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(
            self.db[channel],
            self.t,
            self.f,
            ax=ax,
            db_range=db_range,
            fmax=fmax,
            title="Spectral envelope" if self.method is None else f"Spectral envelope ({self.method})",
        )


# ------------------------------------------------------------ CheapTrick
def pitch_adaptive_envelope(
    sound: Sound,
    f0: ContourLike,
    *,
    q1: float = -0.09,
    f0_floor: float = _FLOOR_F0,
    unvoiced_f0: float = _UNVOICED_F0,
) -> SpectralEnvelope:
    """The CheapTrick spectral envelope (Morise, 2015), written from the paper.

    At each time window of the F0 track:

    1. The sound under a Hann window three periods long, centered on the sample
       nearest the window's time and scaled to unit energy (so the level does not
       change with the window's length); its power spectrum.
    2. That power smoothed by a moving average 2 F0 / 3 wide (the paper's Eq. 4),
       which fills the dips between harmonics so the logarithm can be taken.
    3. The cepstrum of its logarithm, multiplied by ``sinc(F0 q)``, which removes
       the ripple at the harmonic spacing (Eq. 7), and by the recovery lifter
       ``(1 - 2 q1) + 2 q1 cos(2 pi F0 q)``, which gives back the level the
       smoothing took from the harmonic peaks (Eq. 8); then back to a spectrum.

    The result follows the harmonic peaks and barely changes with the window's
    position within a period. It is a :class:`SpectralEnvelope` on the FFT grid
    (``n_fft // 2 + 1`` frequencies from 0 to ``fs / 2``) at the F0 track's
    times, with ``method="CheapTrick"`` and the three parameters in ``settings``.

    Choices the paper leaves open: where the smoothing reaches below 0 Hz or above
    Nyquist the spectrum is mirrored there; samples beyond the ends of the sound are
    zeros; ``1e-12`` is added to the power before the logarithm; unvoiced time
    windows are analyzed at ``unvoiced_f0`` (a short window and wide smoothing,
    which suits noise); F0 below ``f0_floor`` is raised to it; the FFT size is the
    smallest power of two above ``3 fs / f0_floor``.

    Parameters
    ----------
    sound
        The sound, at any sampling rate.
    f0
        An :class:`~sonore.views.f0.F0Track` (one row per channel) or a
        ``(times, f0)`` pair; 0 marks unvoiced time windows. The envelope is
        computed at these times.
    q1
        The recovery lifter's parameter. ``-0.09`` is the paper's value
        (WORLD's code later used ``-0.15``, which sharpens the peaks a little
        more); ``0`` is no recovery.
    f0_floor
        The lowest F0 analyzed; it sets the FFT size.
    unvoiced_f0
        The F0 at which unvoiced time windows are analyzed.
    """
    fs = float(sound.fs)
    times, f0_values = _contour_per_channel(f0, sound.n_channels, "the sound")
    n_fft = _envelope_fft_size(fs, f0_floor)
    data = np.empty((sound.n_channels, n_fft // 2 + 1, len(times)))
    for channel in range(sound.n_channels):
        samples = sound.data[:, channel].astype(float)
        for window_index, (time, window_f0) in enumerate(zip(times, f0_values[channel], strict=True)):
            analyzed_f0 = max(unvoiced_f0 if window_f0 == 0 else window_f0, f0_floor)
            data[channel, :, window_index] = _cheaptrick_window(
                samples, fs, int(np.round(time * fs)), analyzed_f0, n_fft, q1
            )
    settings = {"q1": q1, "f0_floor": f0_floor, "unvoiced_f0": unvoiced_f0}
    freqs = _fft_frequencies(fs, n_fft)
    return SpectralEnvelope(data, times, freqs, fs=fs, method="CheapTrick", settings=settings)


def _cheaptrick_window(samples, fs, center, f0, n_fft, q1):
    """Steps 1 to 3 of :func:`pitch_adaptive_envelope` at one time window."""
    window_length = 3.0 * fs / f0
    power = _unit_energy_power(samples, center, _hann(window_length), window_length / 2.0, n_fft)
    # the moving average over 2 F0 / 3 (Eq. 4)
    smoothed = _smooth_rectangular(power, (f0 / 3.0) / (fs / n_fft))
    # the cepstrum of the log power (Eq. 9), at quefrencies tau in seconds
    cepstrum = np.fft.irfft(np.log(smoothed + _POWER_FLOOR), n_fft)
    bins = np.arange(n_fft)
    tau = np.minimum(bins, n_fft - bins) / fs
    smoothing_lifter = np.sinc(f0 * tau)  # Eq. 7
    recovery_lifter = (1.0 - 2.0 * q1) + 2.0 * q1 * np.cos(2.0 * np.pi * tau * f0)  # Eq. 8
    return np.exp(np.fft.rfft(smoothing_lifter * recovery_lifter * cepstrum, n_fft).real)  # Eq. 6


# ------------------------------------------------------------ the formants
#: Anything read as ``view(t, f)``: a spectral envelope, an aperiodicity, or a function.
EnvelopeFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def warp_frequency(
    view: TimeFrequencyGrid | EnvelopeFunction,
    ratio: float | tuple[ArrayLike, ArrayLike] | Callable[[np.ndarray], ArrayLike],
) -> TimeFrequencyGrid | EnvelopeFunction:
    """A spectral envelope (or aperiodicity) moved along frequency:
    ``new(t, f) = view(t, f / ratio)``. A ratio above 1 moves every formant
    up by that factor, as a shorter vocal tract would; below 1, down. The
    F0 is not touched, so the pitch stays.

    ``view`` is anything read as ``view(t, f)``: a :class:`SpectralEnvelope`,
    an :class:`~sonore.views.aperiodicity.Aperiodicity`, any other
    :class:`~sonore.views.view.TimeFrequencyGrid`, or a function. A view held
    on a grid comes back as the same type on the same grid, with its
    ``method`` and ``settings`` (so ``source_filter_synthesize`` takes it as
    before); a function comes back as a function.

    ``ratio`` is:

    - a number: one ratio for the whole sound;
    - a ``(times, ratios)`` pair: a ratio that changes over time, read at
      each time window (linearly between the given times, held beyond
      them); only for a view held on a grid;
    - a function ``f -> source frequency``: any frequency map, giving for
      each new frequency the one it is read from (a piecewise or bilinear
      warp, for example).

    Values above the view's top frequency (when lowering) hold the top
    value. The level is not adjusted: a warp stretches or squeezes the
    envelope's area, which changes the output level by a little (about 1.5
    dB for a ratio of 0.8 on the speech example), which ``normalize``
    removes. A ratio of exactly 1 returns the view itself, so a voice
    resynthesized with it is the same, sample for sample.
    """
    match ratio:
        case numbers.Real():
            if not float(ratio) > 0:
                raise ValueError(f"ratio must be positive, not {ratio}")
            if ratio == 1:
                return view

            def source_freqs(freqs, time=None):
                return freqs / ratio

            varies_in_time = False
        case _ if callable(ratio):

            def source_freqs(freqs, time=None):
                return np.asarray(ratio(freqs), dtype=float)

            varies_in_time = False
        case (times, ratios):
            times = np.asarray(times, dtype=float)
            ratios = np.asarray(ratios, dtype=float)
            if times.shape != ratios.shape or times.ndim != 1:
                raise ValueError("a ratio contour is a (times, ratios) pair of equal-length 1-d arrays")
            if np.any(ratios <= 0):
                raise ValueError("ratios must be positive")

            def source_freqs(freqs, time=None):
                return freqs / np.interp(time, times, ratios)

            varies_in_time = True
        case _:
            raise TypeError("ratio must be a number, a (times, ratios) pair, or a function of frequency")

    if isinstance(view, TimeFrequencyGrid):
        freqs = np.asarray(view.f, dtype=float)
        if varies_in_time:
            data = np.stack([view(time, source_freqs(freqs, time))[:, :, 0] for time in view.t], axis=2)
        else:
            data = view(view.t, source_freqs(freqs))
        warped = copy.copy(view)
        warped.data = data
        return warped
    if varies_in_time:
        raise TypeError("a ratio that changes over time needs a view held on a grid (with .t and .f)")
    if not callable(view):
        raise TypeError("view must be read as view(t, f)")
    return _WarpedFunction(view, source_freqs)


class _WarpedFunction:
    """A function view read at moved frequencies; it keeps whatever shape
    convention the function has."""

    def __init__(self, view: Callable, source_freqs: Callable):
        self.view = view
        self.source_freqs = source_freqs

    def __call__(self, t: ArrayLike, f: ArrayLike) -> np.ndarray:
        return self.view(t, self.source_freqs(np.asarray(f, dtype=float)))
