"""Time-frequency masks: gains on a frame's coefficients.

A mask keeps one gain per coefficient and nothing of any sound, so it is a
:class:`~sonore.views.view.View`. The way back to a sound goes through the
coefficients it multiplies: ``(stft * mask).to_sound()`` is the frame's
least-squares inverse of the masked coefficients. A mask works with
:class:`~sonore.frames.gabor.STFT`, :class:`~sonore.frames.gabor.TVSTFT` and
:class:`~sonore.frames.filterbank.Subbands`.

The ideal binary mask is the goal Wang (2005) proposed for computational
auditory scene analysis: keep the coefficients where the target is stronger
than the masker by more than a local criterion ``lc_db``, the parameter
whose effect on listeners Brungart, Chang, Simpson & Wang (2006) measured.
Ratio masks, soft gains from the target's share of the power, go back to
Srinivasan, Roman & Wang (2006); the ideal ratio mask
``(T**2 / (T**2 + M**2))**beta`` with ``beta = 0.5`` is the form Wang,
Narayanan & Wang (2014) found best, close to a square-root Wiener filter.

For complex coefficients the power of a cell is ``|X|**2``. Subbands are
real and oscillate, so their power is the Hilbert envelope squared: the mask
then follows each band's level, not its waveform.
"""

from __future__ import annotations

import numpy as np

from sonore.core.utils import power_to_db, time_axis
from sonore.frames.filterbank import Subbands
from sonore.frames.gabor import _FLOOR_DB, STFT, TVSTFT
from sonore.views.view import View

__all__ = ["Mask", "ideal_binary_mask", "ideal_ratio_mask"]

Coefficients = STFT | TVSTFT | Subbands


class Mask(View):
    """Gains, usually between 0 and 1, one per coefficient of the
    coefficients ``like`` it was made for (same frame, same grid).

    ``mask * coefs`` and ``coefs * mask`` give the masked coefficients;
    ``.to_sound()`` on those is the way back to a sound. :attr:`values`
    has the layout of the coefficients' own array: ``(n_channels, n_freqs,
    n_windows)`` for an STFT, ``(n_samples, n_bands, n_channels)`` for
    Subbands (their padding included)."""

    discards = (
        "Mask holds gains for a frame's coefficients, not the coefficients themselves: "
        "it keeps no level or phase of any sound."
    )
    back_to_sound = "Multiply coefficients by it and go back from those: (stft * mask).to_sound()."

    def __init__(self, values, like: Coefficients):
        gains = np.array(values, dtype=float)
        if gains.shape != _array(like).shape:
            raise ValueError(f"mask values have shape {gains.shape}, the coefficients {_array(like).shape}")
        gains.flags.writeable = False
        self.values, self._like = gains, like

    @property
    def t(self) -> np.ndarray:
        """Times [s] of the coefficients' columns (time windows, or samples
        for subbands)."""
        if isinstance(self._like, Subbands):
            return time_axis(self._like.n_samples, self._like.fs)
        return self._like.t

    @property
    def f(self) -> np.ndarray:
        """Frequencies [Hz] of the coefficients' rows: FFT bins, or the
        filterbank's center frequencies for subbands."""
        return self._like.cfs if isinstance(self._like, Subbands) else self._like.f

    def apply(self, coefs: Coefficients) -> Coefficients:
        """The coefficients multiplied by this mask (what ``coefs * mask`` does)."""
        if type(coefs) is not type(self._like) or _array(coefs).shape != self.values.shape:
            raise ValueError("this mask was made for coefficients on a different grid")
        if isinstance(coefs, Subbands):
            return coefs._new(coefs._full * self.values)
        return type(coefs)._from(coefs, coefs.data * self.values)

    def __repr__(self) -> str:
        return f"Mask({type(self._like).__name__}, mean gain {self.values.mean():.3g})"

    def plot(self, ax=None, channel: int = 0, **kwargs):
        """Draw one channel's gains as an image on the coefficients' time
        and frequency axes; see :func:`sonore.plotting.plot_mask`."""
        from sonore.plotting import plot_mask

        return plot_mask(self, ax=ax, channel=channel, **kwargs)


def _array(coefs: Coefficients) -> np.ndarray:
    return coefs._full if isinstance(coefs, Subbands) else coefs.data


def _level_db(coefs: Coefficients) -> np.ndarray:
    """Level of every coefficient [dB]: of ``|X|`` for complex coefficients,
    of the Hilbert envelope for subbands."""
    if isinstance(coefs, Subbands):
        return power_to_db(np.abs(coefs._analytic()) ** 2, floor_db=_FLOOR_DB)
    return coefs.db


def _power(coefs: Coefficients) -> np.ndarray:
    if isinstance(coefs, Subbands):
        return np.abs(coefs._analytic()) ** 2
    return coefs.magnitude**2


def _check_pair(target: Coefficients, masker: Coefficients) -> None:
    if not isinstance(target, Coefficients) or type(target) is not type(masker):
        raise TypeError("target and masker must be coefficients of the same kind (STFT, TVSTFT or Subbands)")
    if _array(target).shape != _array(masker).shape:
        raise ValueError("target and masker must be analyzed by the same frame, at the same length")


def ideal_binary_mask(target: Coefficients, masker: Coefficients, lc_db: float = 0.0) -> Mask:
    """1 where the target's level exceeds the masker's by more than
    ``lc_db``, the local SNR criterion, else 0 (Wang, 2005; Brungart et al.,
    2006)."""
    _check_pair(target, masker)
    return Mask((_level_db(target) - _level_db(masker) > lc_db).astype(float), target)


def ideal_ratio_mask(target: Coefficients, masker: Coefficients, beta: float = 0.5) -> Mask:
    """``(T**2 / (T**2 + M**2))**beta`` with ``T**2`` and ``M**2`` the powers
    of each coefficient (Wang, Narayanan & Wang, 2014; 0 where both are
    silent)."""
    _check_pair(target, masker)
    target_power, masker_power = _power(target), _power(masker)
    with np.errstate(invalid="ignore", divide="ignore"):
        irm = np.nan_to_num((target_power / (target_power + masker_power)) ** beta)
    return Mask(irm, target)
