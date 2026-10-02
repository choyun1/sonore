"""Changing a voice's description: its pitch and its formants, whatever
measured it.

A voice is described here by two things that every method shares:

- an **F0 contour**: anything with window times ``.t`` and values ``.f0``
  (0 where unvoiced), such as an :class:`~sonore.analysis.f0.F0Track`, or a
  ``(times, f0)`` pair, such as Harvest's output or the first two outputs of
  :meth:`~sonore.analysis.cepstrum.Cepstrum.f0`;
- an **envelope**: anything read as ``env(t, f)``, giving power at times
  ``t`` and frequencies ``f`` with shape ``(n_channels, len(f), len(t))``,
  such as :class:`~sonore.analysis.vocoder.SpectralEnvelope` (CheapTrick),
  :class:`GridEnvelope` (from :meth:`Cepstrum.envelope_view
  <sonore.analysis.cepstrum.Cepstrum.envelope_view>` or
  :meth:`MFCC.envelope_view <sonore.analysis.mfcc.MFCC.envelope_view>`),
  or a function. :class:`~sonore.analysis.vocoder.Aperiodicity` is read
  the same way.

:func:`scale_f0` changes the pitch and :func:`warp_frequency` moves an
envelope (or an aperiodicity) along frequency. Neither looks at how its
input was made, so any tracker, any envelope and any synthesizer combine:
:func:`~sonore.stimuli.vocoder.world_synthesize` and
:func:`~sonore.signals.generators.harmonic_complex` both take any envelope.
"""

from __future__ import annotations

import copy
import numbers
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from sonore.analysis.f0 import F0Track
from sonore.analysis.vocoder import _FrequencyView, _pointwise, _positions

__all__ = ["GridEnvelope", "scale_f0", "warp_frequency"]


class GridEnvelope:
    """A spectral envelope held as power on any grid of times and
    frequencies: ``data`` has shape ``(n_channels, len(f), len(t))``.

    Any envelope estimate can be put in this form, so that it reads like a
    :class:`~sonore.analysis.vocoder.SpectralEnvelope` and goes wherever an
    envelope is taken: :func:`warp_frequency`,
    :func:`~sonore.stimuli.vocoder.world_synthesize`,
    :func:`~sonore.signals.generators.harmonic_complex`.
    :meth:`Cepstrum.envelope_view <sonore.analysis.cepstrum.Cepstrum.envelope_view>`
    and :meth:`MFCC.envelope_view <sonore.analysis.mfcc.MFCC.envelope_view>`
    return one.

    Calling it, ``env(t, f)``, reads the power at any times and frequencies,
    linear in time and in dB over frequency, as ``SpectralEnvelope`` does;
    beyond the ends of the grid the end values are held.
    """

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


# ------------------------------------------------------------ the pitch
def _scaled(values: np.ndarray, ratio: float, spread: float) -> np.ndarray:
    """Voiced values (> 0, one row per channel) times ratio, spread around
    each row's median on a log scale; 0 and NaN stay as they are."""
    values = np.asarray(values, dtype=float)
    if spread == 1:
        return values * ratio
    rows = np.atleast_2d(values)
    out = rows.copy()
    for row_in, row_out in zip(rows.reshape(len(rows), -1), out.reshape(len(out), -1), strict=True):
        voiced = row_in > 0
        if voiced.any():
            median = np.median(row_in[voiced])
            row_out[voiced] = median * ratio * (row_in[voiced] / median) ** spread
    return out.reshape(values.shape)


def scale_f0(contour, ratio: float, *, range: float = 1.0):
    """An F0 contour with its pitch changed: every voiced value multiplied
    by ``ratio`` and, if ``range`` is not 1, spread around its median on a
    log scale, ``median * ratio * (f0 / median) ** range`` (``range=0`` is a
    monotone at ``ratio`` times the median, 2 doubles every interval from
    it). Unvoiced time windows (0) stay unvoiced.

    ``contour`` is an :class:`~sonore.analysis.f0.F0Track`, which comes back
    as an ``F0Track`` (its candidates changed the same way), a ``(times,
    f0)`` pair, which comes back as a pair, or anything with ``.t`` and
    ``.f0``, which comes back as a ``(times, f0)`` pair. A ratio in
    semitones ``st`` is ``2 ** (st / 12)``. With ``ratio=1`` and
    ``range=1`` the contour itself is returned.

    The envelope is not touched, so a voice resynthesized on the new contour
    keeps its formants (unlike :func:`~sonore.stimuli.phasevocoder.pitch_shift`,
    which moves them with the pitch).
    """
    if not ratio > 0:
        raise ValueError(f"ratio must be positive, not {ratio}")
    if not range >= 0:
        raise ValueError(f"range must be 0 or more, not {range}")
    if ratio == 1 and range == 1:
        return contour
    match contour:
        case F0Track():
            return replace(
                contour,
                f0=_scaled(contour.f0, ratio, range),
                candidates=_scaled_candidates(contour, ratio, range),
            )
        case (times, values):
            return times, _scaled(values, ratio, range)
        case _ if hasattr(contour, "t") and hasattr(contour, "f0"):
            return contour.t, _scaled(contour.f0, ratio, range)
        case _:
            raise TypeError("contour must be an F0Track, a (times, f0) pair, or have .t and .f0")


def _scaled_candidates(track: F0Track, ratio: float, spread: float) -> np.ndarray:
    """The candidates moved by the same map as the chosen F0 (each channel's
    median of voiced F0 as the centre)."""
    if spread == 1:
        return track.candidates * ratio
    out = track.candidates.copy()
    for channel, f0_row in enumerate(track.f0):
        voiced = f0_row > 0
        if voiced.any():
            median = np.median(f0_row[voiced])
            out[channel] = median * ratio * (track.candidates[channel] / median) ** spread
    return out


# ------------------------------------------------------------ the formants
def _is_grid_view(view) -> bool:
    return isinstance(view, (_FrequencyView, GridEnvelope))


def warp_frequency(view, ratio):
    """A spectral envelope (or aperiodicity) moved along frequency:
    ``new(t, f) = view(t, f / ratio)``. A ratio above 1 moves every formant
    up by that factor, as a shorter vocal tract would; below 1, down. The
    F0 is not touched, so the pitch stays.

    ``view`` is anything read as ``view(t, f)``: a
    :class:`~sonore.analysis.vocoder.SpectralEnvelope`, an
    :class:`~sonore.analysis.vocoder.Aperiodicity`, a :class:`GridEnvelope`,
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


# ------------------------------------------------------------ a contour on a grid
def _contour_on_grid(times, f0_values, grid) -> np.ndarray:
    """An F0 contour (one row per channel) read at other window times: voiced
    where the nearest given time window is voiced, there linear between
    voiced values; 0 elsewhere."""
    times = np.asarray(times, dtype=float)
    grid = np.asarray(grid, dtype=float)
    rows = np.atleast_2d(np.asarray(f0_values, dtype=float))
    upper = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    lower = np.clip(upper - 1, 0, len(times) - 1)
    nearest = np.where(np.abs(times[lower] - grid) <= np.abs(times[upper] - grid), lower, upper)
    out = np.zeros((len(rows), len(grid)))
    for row_in, row_out in zip(rows, out, strict=True):
        voiced = row_in > 0
        if voiced.any():
            row_out[:] = np.where(voiced[nearest], np.interp(grid, times[voiced], row_in[voiced]), 0.0)
    return out
