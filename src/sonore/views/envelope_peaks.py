"""Envelope peaks: how big a sound's pops are, as the exponent of a power law.

Pops (fire crackling, a typewriter, tapping) make a few moments far louder
than the typical ones. When the sizes of those moments follow a power law
with a small exponent, the envelope's variance and kurtosis have no stable
value: measured on a recording, they keep growing with its length. The
exponent itself does not. The design and its numbers are in
docs/design/views/envelope-peaks.md.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt

from sonore.core.sound import Sound
from sonore.views.view import View

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["EnvelopePeaks", "envelope_peaks"]


SMOOTHING_HZ = 500.0  # low-pass on the Hilbert envelope


MIN_FITTED = 10  # the Hill fit uses at least this many of the largest values


WARN_BELOW = 50  # fewer fitted values than this leave the exponent uncertain by more than about 14%


@dataclass(frozen=True)
class EnvelopePeaks(View):
    """The largest envelope value in each bin of a sound high-passed at each
    cutoff, and the power-law exponent of their upper tail. A view: it keeps
    only those maxima.

    ``maxima`` has shape ``(n_bins, n_cutoffs)``. ``exponent`` and
    ``uncertainty`` have one value per cutoff: ``exponent`` is alpha in
    ``P(maximum > a) ~ a**-alpha``, fitted by the Hill estimator on the
    largest ``fraction`` of the maxima, and ``uncertainty`` is its standard
    error, ``exponent / sqrt(n_fitted)``. ``threshold`` is, per cutoff, the
    largest maximum left out of the fit. Below about 2 the sound is made of
    pops, with no smooth background under them; Gaussian noise gives 20 to
    40, more at a higher sample rate, which puts more samples in each bin.
    """

    discards = (
        "EnvelopePeaks keeps only the largest envelope value in each bin above each cutoff, "
        "and drops when the bins occurred, the spectrum and everything below each maximum."
    )
    back_to_sound = (
        "There is no inverse; so.event_train with level_db drawn from an exponential distribution "
        "of scale 20 / (exponent * ln 10) makes pops with a given exponent, to hear what it means."
    )

    maxima: np.ndarray
    exponent: np.ndarray
    uncertainty: np.ndarray
    threshold: np.ndarray
    cutoffs_hz: tuple[float, ...]
    bin_seconds: float
    fraction: float
    n_fitted: int

    def __repr__(self) -> str:
        fits = ", ".join(
            f"{cutoff:g} Hz: {exponent:.2f} ± {uncertainty:.2f}"
            for cutoff, exponent, uncertainty in zip(
                self.cutoffs_hz, self.exponent, self.uncertainty, strict=True
            )
        )
        return f"EnvelopePeaks({len(self.maxima)} bins, exponent {fits})"

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        """The survival of the maxima on log-log axes, with the fitted line
        (see :func:`~sonore.plotting.plot_envelope_peaks`)."""
        from sonore.plotting import plot_envelope_peaks

        return plot_envelope_peaks(self, ax=ax, **kwargs)


def envelope_peaks(
    sound: Sound,
    *,
    cutoffs_hz: Sequence[float] = (300.0, 2000.0),
    bin_seconds: float = 0.005,
    fraction: float = 0.05,
) -> EnvelopePeaks:
    """Measure how heavy the tail of a sound's envelope peaks is.

    For each cutoff, separately:

    1. High-pass the sound at the cutoff (4th-order Butterworth, forward
       and backward). Pops are broadband, and a high cutoff keeps low
       rumble out.
    2. Take the Hilbert envelope and smooth it with a 500 Hz low-pass.
    3. Take the envelope's maximum in each bin ``bin_seconds`` long. The
       maximum, not the mean, because the mean smooths short pops away.
    4. Fit the exponent of the upper tail of the maxima by the Hill
       estimator (Hill, 1975) on the largest ``fraction`` of them, at least
       10: with ``x_1 >= x_2 >= ...`` the sorted maxima and ``k`` the number
       fitted, ``exponent = k / sum(log(x_i / x_(k+1)))`` over ``i <= k``.

    The uncertainty is ``exponent / sqrt(k)``. With the defaults, 2 s of
    sound fits 20 values (about 25%) and 7 s fits 70 (about 12%), and a
    short excerpt can miss the big events altogether; fewer than 50 fitted
    values raise a warning. Mono only.
    """
    if sound.n_channels != 1:
        raise ValueError(
            f"envelope_peaks needs a mono sound, not {sound.n_channels} channels; try sound.mono()"
        )
    fs = sound.fs
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    if any(not 0 < cutoff < fs / 2 for cutoff in cutoffs_hz) or not cutoffs_hz:
        raise ValueError(f"each cutoff must be between 0 and Nyquist ({fs / 2:g} Hz)")
    bin_length = int(round(bin_seconds * fs))
    if bin_length < 1:
        raise ValueError("bin_seconds is shorter than one sample")
    n_bins = len(sound) // bin_length
    n_fitted = max(MIN_FITTED, int(fraction * n_bins))
    if n_bins <= n_fitted:
        raise ValueError(f"the sound has {n_bins} bins, too few to fit the largest {n_fitted}")

    samples = sound.data[:, 0]
    smoothing = butter(2, min(SMOOTHING_HZ, 0.99 * fs / 2), "lowpass", fs=fs, output="sos")
    maxima = np.empty((n_bins, len(cutoffs_hz)))
    for column, cutoff in enumerate(cutoffs_hz):
        high = sosfiltfilt(butter(4, cutoff, "highpass", fs=fs, output="sos"), samples)
        analytic = hilbert(high, N=next_fast_len(len(high)))[: len(high)]
        envelope = sosfiltfilt(smoothing, np.abs(analytic))
        maxima[:, column] = envelope[: n_bins * bin_length].reshape(n_bins, bin_length).max(axis=1)

    ordered = -np.sort(-maxima, axis=0)
    threshold = ordered[n_fitted]
    if np.any(threshold <= 0):
        raise ValueError("the sound is silent above a cutoff, so its peaks have no tail to fit")
    exponent = n_fitted / np.sum(np.log(ordered[:n_fitted] / threshold), axis=0)
    uncertainty = exponent / math.sqrt(n_fitted)
    if n_fitted < WARN_BELOW:
        warnings.warn(
            f"envelope_peaks fitted only {n_fitted} values, so each exponent is uncertain by about "
            f"{1 / math.sqrt(n_fitted):.0%}; a longer sound narrows it",
            stacklevel=2,
        )
    return EnvelopePeaks(
        maxima=maxima,
        exponent=exponent,
        uncertainty=uncertainty,
        threshold=threshold,
        cutoffs_hz=tuple(float(cutoff) for cutoff in cutoffs_hz),
        bin_seconds=bin_length / fs,
        fraction=fraction,
        n_fitted=n_fitted,
    )
