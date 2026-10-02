"""Time-frequency masks: a change to STFT coefficients whose resynthesis is
the frame's least-squares inverse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonore.frames.gabor import STFT

__all__ = ["Mask", "ideal_binary_mask", "ideal_ratio_mask"]


# ------------------------------------------------------------------- masks
@dataclass(frozen=True)
class Mask:
    """A time-frequency mask (binary or soft) aligned with an STFT."""

    values: np.ndarray
    t: np.ndarray
    f: np.ndarray

    def __mul__(self, other):
        if isinstance(other, STFT):
            return other * self
        return NotImplemented

    def plot(self, ax=None, channel: int = 0, **kwargs):
        from sonore.plotting import plot_mask

        return plot_mask(self, ax=ax, channel=channel, **kwargs)


def ideal_binary_mask(target: STFT, masker: STFT, lc_db: float = 0.0) -> Mask:
    """1 where the target exceeds the masker by more than ``lc_db`` (local SNR criterion)."""
    return Mask((target.db - masker.db > lc_db).astype(float), target.t, target.f)


def ideal_ratio_mask(target: STFT, masker: STFT, beta: float = 0.5) -> Mask:
    """``(|T|^2 / (|T|^2 + |M|^2))**beta``."""
    target_power, masker_power = target.magnitude**2, masker.magnitude**2
    with np.errstate(invalid="ignore", divide="ignore"):
        irm = np.nan_to_num((target_power / (target_power + masker_power)) ** beta)
    return Mask(irm, target.t, target.f)
