"""Frames: invertible analyses with a common contract.

A :class:`Frame` turns a :class:`~sonore.Sound` into coefficients and back.
The contract (docs/design/frames/frames.md) is:

- ``synthesize(analyze(x)) == x`` to floating-point precision.
- For modified coefficients, ``synthesize`` returns the least-squares signal:
  the one whose coefficients are nearest to the given ones, in the norm of
  :meth:`Frame.energy`.
- ``frame_bounds(n_samples, fs) -> (A, B)`` are the extreme eigenvalues of the
  frame operator. ``A == B`` means tight. ``A == 0`` means not a frame, and
  ``synthesize`` raises (``analyze`` still works).

Every frame also has :meth:`Frame.adjoint`, the adjoint of ``analyze`` for
the coefficient inner product that ``energy`` uses: synthesis without the
division by ``s``. It is the gradient of a coefficient-domain loss with
respect to the signal.

The frames themselves are in :mod:`sonore.frames.filterbank` (filters on the
DFT grid) and :mod:`sonore.frames.gabor` (the STFT and its time-varying
version). The code is written as pure array functions (no in-place mutation)
so that a JAX port is mechanical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from sonore.core.sound import Sound

if TYPE_CHECKING:
    pass

__all__ = ["Frame"]


# analyze() always works, even on a non-frame (a bank with coverage gaps still
# gives a usable cochleagram), but synthesize() refuses when A <= NOT_A_FRAME * B,
# because no stable inverse exists.
NOT_A_FRAME = 1e-12


def _check_frame(lower: float, upper: float, what: str, note: str = " analyze() still works.") -> None:
    if not lower > NOT_A_FRAME * upper:
        raise ValueError(
            f"{what} is not a frame (bounds A={lower:.3g}, B={upper:.3g}): some part of the signal "
            f"is not covered, so it cannot be synthesized.{note}"
        )


class Frame(ABC):
    """An invertible analysis. See the module docstring for the contract."""

    @abstractmethod
    def analyze(self, sound: Sound, **kwargs):
        """Coefficients of ``sound``."""

    @abstractmethod
    def synthesize(self, coefs) -> Sound:
        """The least-squares signal for ``coefs`` (exact for unmodified ones)."""

    @abstractmethod
    def frame_bounds(self, n_samples: int, fs: float) -> tuple[float, float]:
        """Extreme eigenvalues ``(A, B)`` of the frame operator for signals of
        ``n_samples`` samples at ``fs`` Hz."""

    @abstractmethod
    def energy(self, coefs) -> np.ndarray:
        """Coefficient energy per channel, in the norm the bounds and the least
        squares are defined in, so that bounds, SNRs and tests all agree."""

    @abstractmethod
    def adjoint(self, coefs) -> Sound:
        """The adjoint of :meth:`analyze`: the signal ``y`` with
        ``<analyze(x), coefs> == <x, y>`` for every ``x``, the coefficient
        inner product being the one :meth:`energy` uses. It is
        :meth:`synthesize` without the division by the frame operator, and
        the gradient of a loss on the coefficients with respect to the
        signal."""
