"""Spectral envelopes, and moving them along frequency.

:func:`cheaptrick` is WORLD's CheapTrick (Morise, 2015), ported exactly: every
step reproduces WORLD's C++ code (github.com/mmorise/World, commit d625e76) to
floating-point precision, including the tiny noise WORLD adds to keep
logarithms and divisions finite, drawn from WORLD's own generator in WORLD's
order (:mod:`sonore.signals.world`, where :data:`DIFFERENCES_FROM_WORLD
<sonore.signals.world.DIFFERENCES_FROM_WORLD>` lists every option that departs
from WORLD).

An envelope here is anything read as ``env(t, f)``, giving power at times
``t`` and frequencies ``f`` with shape ``(n_channels, len(f), len(t))``:
a :class:`SpectralEnvelope` (CheapTrick), a :class:`GridEnvelope` (from
:meth:`Cepstrum.envelope_view <sonore.views.cepstrum.Cepstrum.envelope_view>`
or :meth:`MFCC.envelope_view <sonore.views.mfcc.MFCC.envelope_view>`), or a
function. :class:`~sonore.views.aperiodicity.Aperiodicity` is read the same
way. :func:`warp_frequency` moves an envelope (or an aperiodicity) along
frequency without looking at how it was made, so any envelope and any
synthesizer combine: :func:`~sonore.signals.world.world_synthesize` and
:func:`~sonore.signals.generators.harmonic_complex` both take any envelope.
"""

from __future__ import annotations

import copy
import numbers
from collections.abc import Callable

import numpy as np

from sonore.core.sound import Sound
from sonore.signals.world import (
    _DEFAULT_F0,
    _EPS,
    _FLOOR_F0,
    _SAFEGUARD,
    _integer_fs,
    _matlab_round,
    _Stream,
    _time_windows,
    world_fft_size,
)
from sonore.views.view import View

__all__ = ["SpectralEnvelope", "cheaptrick", "GridEnvelope", "warp_frequency"]


def _interp1q(x_start: float, x_step: float, values: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """WORLD's interp1Q: ``values`` sampled at ``x_start + k x_step``, read
    linearly at ``x_query`` (index truncated toward zero, last step flat)."""
    position = (np.asarray(x_query, float) - x_start) / x_step
    index = position.astype(int)
    steps = np.append(np.diff(values), 0.0)
    return values[index] + steps[index] * (position - index)


def _linear_smoothing(spectrum: np.ndarray, width: float, fs: int, n_fft: int) -> np.ndarray:
    """WORLD's LinearSmoothing: the mean over ``width`` Hz of a half
    spectrum, piecewise constant per bin, mirrored at 0 Hz and Nyquist."""
    n_half = n_fft // 2
    bin_width = fs / n_fft
    boundary = int(width * n_fft / fs) + 1
    mirrored = np.concatenate(
        [spectrum[boundary:0:-1], spectrum[:n_half], spectrum[n_half::-1][: boundary + 1]]
    )
    integral = np.cumsum(mirrored * bin_width)
    lower_edges = np.arange(n_half + 1) * bin_width - width / 2
    origin = -(boundary - 0.5) * bin_width
    upper = _interp1q(origin, bin_width, integral, lower_edges + width)
    lower = _interp1q(origin, bin_width, integral, lower_edges)
    return (upper - lower) / width


def _dc_correction(spectrum: np.ndarray, f0: float, fs: int, n_fft: int) -> np.ndarray:
    """WORLD's DCCorrection: the spectrum below F0 folded back about F0 / 2."""
    bin_width = fs / n_fft
    upper_index = 2 + int(f0 * n_fft / fs)
    below_f0 = np.arange(upper_index - 1) * bin_width
    corrected = spectrum.copy()
    corrected[: upper_index - 1] += _interp1q(f0, -bin_width, spectrum[: upper_index + 1], below_f0)
    return corrected


def _windowed_waveform(samples, fs, f0, time, window, periods, noise, noise_scale):
    """An F0-adaptive Hann or Blackman window ``periods`` long, centred at
    ``time`` (WORLD's rounding), samples beyond the ends repeated; plus
    WORLD's safety noise; less the window times the weighted mean."""
    half_length = _matlab_round(periods / 2 * fs / f0)
    offsets = np.arange(-half_length, half_length + 1)
    if window == "cheaptrick":  # Hann, scaled to unit energy
        shape = 0.5 * np.cos(np.pi * (offsets / 1.5 / fs) * f0) + 0.5
        shape = shape / np.sqrt(np.sum(shape * shape))
    else:
        position = 2.0 * offsets / periods / fs
        if window == "hann":
            shape = 0.5 * np.cos(np.pi * position * f0) + 0.5
        else:  # Blackman
            shape = 0.42 + 0.5 * np.cos(np.pi * position * f0) + 0.08 * np.cos(2 * np.pi * position * f0)
    origin = _matlab_round(time * fs + 0.001)
    indices = np.clip(origin + offsets, 0, len(samples) - 1)
    segment = samples[indices] * shape + noise.draw(len(offsets)) * noise_scale
    return segment - shape * (segment.sum() / shape.sum())


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


# ------------------------------------------------------------ the views
class _FrequencyView(View):
    """Shared by the envelope and the aperiodicity: data of shape
    ``(n_channels, n_freqs, n_windows)`` on WORLD's grid."""

    data: np.ndarray
    t: np.ndarray
    fs: float
    n_fft: int

    @property
    def f(self) -> np.ndarray:
        """Frequencies [Hz]: ``n_fft // 2 + 1`` bins from 0 to ``fs / 2``."""
        return np.arange(self.n_fft // 2 + 1) * self.fs / self.n_fft

    def to_world(self, channel: int = 0) -> np.ndarray:
        """One channel as WORLD and pyworld hold it: shape ``(n_windows,
        n_freqs)``, C-contiguous, ready for ``pyworld.synthesize``."""
        return np.ascontiguousarray(self.data[channel].T)

    def _interpolate(self, log_values: np.ndarray, t, f) -> np.ndarray:
        """``log_values`` (same shape as data) read linearly at times ``t``
        and frequencies ``f``: shape ``(n_channels, len(f), len(t))``."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        freqs = np.atleast_1d(np.asarray(f, dtype=float))
        window_position = np.interp(times, self.t, np.arange(len(self.t)))
        lower = np.floor(window_position).astype(int)
        upper = np.minimum(lower + 1, len(self.t) - 1)
        weight = window_position - lower
        bin_pos = np.clip(freqs * self.n_fft / self.fs, 0, self.n_fft // 2)
        bin_lower = np.floor(bin_pos).astype(int)
        bin_upper = np.minimum(bin_lower + 1, self.n_fft // 2)
        bin_weight = (bin_pos - bin_lower)[:, None]
        at_freqs = log_values[:, bin_lower] * (1 - bin_weight) + log_values[:, bin_upper] * bin_weight
        return at_freqs[:, :, lower] * (1 - weight) + at_freqs[:, :, upper] * weight


class SpectralEnvelope(_FrequencyView):
    """A spectral envelope from :func:`cheaptrick`: power on ``n_fft // 2 + 1``
    frequencies :attr:`f` from 0 to ``fs / 2``, at the window times :attr:`t`
    of the F0 track. ``data`` has shape ``(n_channels, n_freqs, n_windows)``;
    :meth:`to_world` gives one channel in WORLD's layout.

    It is a view (it keeps the envelope, not the sound): calling it,
    ``env(t, f)``, reads the power at any times and frequencies, linearly
    in time and in dB over frequency.
    """

    discards = (
        "SpectralEnvelope discards the harmonics, the phase, and everything finer than the envelope's "
        "smoothing."
    )
    back_to_sound = (
        "so.world_synthesize rebuilds a voice from it together with an F0 track and an aperiodicity."
    )

    def __init__(self, data: np.ndarray, t: np.ndarray, fs: float, q1: float):
        self.data = data
        self.t = t
        self.fs = fs
        self.n_fft = 2 * (data.shape[1] - 1)
        self.q1 = q1

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        return (
            f"SpectralEnvelope({n_freqs} freqs x {n_windows} time windows, {n_channels} ch, q1 {self.q1:g})"
        )

    @property
    def db(self) -> np.ndarray:
        """``10 log10`` of the power."""
        return 10 * np.log10(self.data)

    def __call__(self, t, f) -> np.ndarray:
        """Power at times ``t`` [s] and frequencies ``f`` [Hz], shape
        ``(n_channels, len(f), len(t))``: linear in time between time windows,
        linear in dB between bins."""
        return np.exp(self._interpolate(np.log(self.data), t, f))

    def amplitude(self, t, f, channel: int = 0) -> np.ndarray:
        """The amplitude (square root of the power) at the points ``(t[i],
        f[i])``, ``t`` and ``f`` of the same shape: the form
        :func:`~sonore.signals.generators.harmonic_complex` reads, one value
        per sample and harmonic. Linear in time and in dB over frequency,
        held beyond the first and last time windows."""
        return np.exp(0.5 * _pointwise(np.log(self.data[channel]), self.t, self.f, t, f))

    def plot(self, ax=None, channel: int = 0, db_range: float = 70.0, fmax: float | None = None):
        """The envelope in dB as a time-frequency image."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(
            self.db[channel], self.t, self.f, ax=ax, db_range=db_range, fmax=fmax, title="Spectral envelope"
        )


# ------------------------------------------------------------ CheapTrick
def cheaptrick(sound: Sound, f0, *, q1: float = -0.15, f0_floor: float = _FLOOR_F0) -> SpectralEnvelope:
    """WORLD's CheapTrick spectral envelope (Morise, 2015), ported exactly.

    At each time window of the F0 track: the sound under a Hann window three
    periods long, scaled to unit energy, less its weighted mean; its power
    spectrum on :func:`world_fft_size` bins; the power below F0 folded back
    about F0 / 2; a moving average over 2 F0 / 3; then the log, liftered by
    ``sinc(F0 q)`` (smoothing over F0) times the recovery lifter
    ``(1 - 2 q1) + 2 q1 cos(2 pi F0 q)``, and back. The result follows the
    shape of the harmonic peaks a few dB below them and, because of the
    smoothing over 2 F0 / 3, barely changes with the time window's position
    within a period. Time windows with F0 at or below the floor
    ``3 fs / (n_fft - 3)`` (unvoiced ones included) are analysed at 500 Hz.

    Parameters
    ----------
    sound
        The sound. Its sampling rate must be a whole number of Hz.
    f0
        An :class:`~sonore.views.f0.F0Track` (one row per channel) or a
        ``(times, f0)`` pair; 0 marks unvoiced time windows. The envelope is
        computed at these times.
    q1
        The recovery lifter's parameter. ``-0.15`` is WORLD's code; the
        paper gives ``-0.09``; ``0`` is no recovery.
    f0_floor
        The lowest F0 the FFT size must hold (WORLD's 71 Hz).
    """
    fs = _integer_fs(sound)
    times, f0_values = _time_windows(sound, f0)
    n_fft = world_fft_size(fs, f0_floor)
    analysis_floor = 3.0 * fs / (n_fft - 3.0)
    quefrencies = np.arange(n_fft // 2 + 1) / fs
    data = np.empty((sound.n_channels, n_fft // 2 + 1, len(times)))
    for channel in range(sound.n_channels):
        samples = sound.data[:, channel]
        noise = _Stream()
        for window_index, (time, window_f0) in enumerate(zip(times, f0_values[channel], strict=True)):
            current_f0 = _DEFAULT_F0 if window_f0 <= analysis_floor else window_f0
            segment = _windowed_waveform(samples, fs, current_f0, time, "cheaptrick", 3.0, noise, _SAFEGUARD)
            power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
            power = _dc_correction(power, current_f0, fs, n_fft)
            power = _linear_smoothing(power, current_f0 * 2.0 / 3.0, fs, n_fft)
            power = power + np.abs(noise.draw(n_fft // 2 + 1)) * _EPS
            data[channel, :, window_index] = _smooth_with_recovery(power, current_f0, q1, quefrencies, n_fft)
    return SpectralEnvelope(data, times, fs, q1)


def _smooth_with_recovery(power, f0, q1, quefrencies, n_fft):
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothing = np.sin(np.pi * f0 * quefrencies) / (np.pi * f0 * quefrencies)
    smoothing[0] = 1.0
    recovery = (1.0 - 2.0 * q1) + 2.0 * q1 * np.cos(2.0 * np.pi * quefrencies * f0)
    log_power = np.log(power)
    cepstrum = np.fft.rfft(np.concatenate([log_power, log_power[-2:0:-1]])).real
    return np.exp(np.fft.irfft(cepstrum * smoothing * recovery, n_fft)[: n_fft // 2 + 1])


class GridEnvelope(View):
    """A spectral envelope held as power on any grid of times and
    frequencies: ``data`` has shape ``(n_channels, len(f), len(t))``.

    Any envelope estimate can be put in this form, so that it reads like a
    :class:`~sonore.views.spectral_envelope.SpectralEnvelope` and goes wherever an
    envelope is taken: :func:`warp_frequency`,
    :func:`~sonore.signals.world.world_synthesize`,
    :func:`~sonore.signals.generators.harmonic_complex`.
    :meth:`Cepstrum.envelope_view <sonore.views.cepstrum.Cepstrum.envelope_view>`
    and :meth:`MFCC.envelope_view <sonore.views.mfcc.MFCC.envelope_view>`
    return one.

    Calling it, ``env(t, f)``, reads the power at any times and frequencies,
    linear in time and in dB over frequency, as ``SpectralEnvelope`` does;
    beyond the ends of the grid the end values are held.
    """

    discards = (
        "GridEnvelope discards the harmonics, the phase, and everything finer than the envelope's smoothing."
    )
    back_to_sound = (
        "so.world_synthesize rebuilds a voice from it together with an F0 track and an aperiodicity."
    )

    def __init__(self, data: np.ndarray, t, f):
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
        if np.any(self.data < 0):
            raise ValueError("an envelope is power, so it cannot be negative")

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        return f"GridEnvelope({n_freqs} freqs x {n_windows} time windows, {n_channels} ch)"

    @property
    def db(self) -> np.ndarray:
        """``10 log10`` of the power."""
        return 10 * np.log10(np.maximum(self.data, np.finfo(float).tiny))

    def _log(self) -> np.ndarray:
        return np.log(np.maximum(self.data, np.finfo(float).tiny))

    def __call__(self, t, f) -> np.ndarray:
        """Power at times ``t`` [s] and frequencies ``f`` [Hz], shape
        ``(n_channels, len(f), len(t))``."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        freqs = np.atleast_1d(np.asarray(f, dtype=float))
        t_lower, t_upper, t_weight = _positions(times, self.t)
        f_lower, f_upper, f_weight = _positions(freqs, self.f)
        log_values = self._log()
        f_weight = f_weight[:, None]
        at_freqs = log_values[:, f_lower] * (1 - f_weight) + log_values[:, f_upper] * f_weight
        return np.exp(at_freqs[:, :, t_lower] * (1 - t_weight) + at_freqs[:, :, t_upper] * t_weight)

    def amplitude(self, t, f, channel: int = 0) -> np.ndarray:
        """The amplitude (square root of the power) at the points ``(t[i],
        f[i])``, ``t`` and ``f`` of the same shape: the form
        :func:`~sonore.signals.generators.harmonic_complex` reads, one value
        per sample and harmonic."""
        return np.exp(0.5 * _pointwise(self._log()[channel], self.t, self.f, t, f))

    def plot(self, ax=None, channel: int = 0, db_range: float = 70.0, fmax: float | None = None):
        """The envelope in dB as a time-frequency image."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(
            self.db[channel], self.t, self.f, ax=ax, db_range=db_range, fmax=fmax, title="Spectral envelope"
        )


# ------------------------------------------------------------ the formants
def _is_grid_view(view) -> bool:
    return isinstance(view, (_FrequencyView, GridEnvelope))


def warp_frequency(view, ratio):
    """A spectral envelope (or aperiodicity) moved along frequency:
    ``new(t, f) = view(t, f / ratio)``. A ratio above 1 moves every formant
    up by that factor, as a shorter vocal tract would; below 1, down. The
    F0 is not touched, so the pitch stays.

    ``view`` is anything read as ``view(t, f)``: a
    :class:`~sonore.views.spectral_envelope.SpectralEnvelope`, an
    :class:`~sonore.views.aperiodicity.Aperiodicity`, a :class:`GridEnvelope`,
    or a function. A view held on a grid comes back as the same type on the
    same grid (so ``world_synthesize`` takes it as before); a function comes
    back as a function.

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
            if not ratio > 0:
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

    if _is_grid_view(view):
        freqs = view.f
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

    def __call__(self, t, f):
        return self.view(t, self.source_freqs(np.asarray(f, dtype=float)))
