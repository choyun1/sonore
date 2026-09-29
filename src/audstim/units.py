"""A decibel unit, so level changes read like arithmetic: ``snd + 6*dB``.

``6*dB`` is a :class:`Decibels` value, not a number. Adding it to a
:class:`~audstim.Sound` changes the level; adding a Sound mixes; adding a bare
number is an error. That keeps ``snd + 0.5`` from silently meaning either
"+0.5 dB" or "add a DC offset of 0.5".
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

__all__ = ["Decibels", "dB"]


@dataclass(frozen=True)
class Decibels:
    """A level change in decibels (amplitude: ``gain = 10**(value/20)``)."""

    value: float

    # Let numpy scalars defer to us, so ``np.float64(6) * dB`` works.
    __array_ufunc__ = None

    @property
    def gain(self) -> float:
        """Linear amplitude factor."""
        return 10 ** (self.value / 20)

    def __repr__(self) -> str:
        return f"{self.value:g} dB"

    def __float__(self) -> float:
        return float(self.value)

    # scaling: 6*dB, dB*6, -3*dB, (6*dB)/2
    def __mul__(self, other):
        if isinstance(other, numbers.Real) and not isinstance(other, bool):
            return Decibels(self.value * float(other))
        from audstim.sound import Sound

        if isinstance(other, Sound):
            raise TypeError(
                "multiplying a Sound by dB is ambiguous; write snd + 6*dB for a gain "
                "(note that snd * 6*dB parses as (snd * 6) * dB)"
            )
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, numbers.Real) and not isinstance(other, bool):
            return Decibels(self.value / float(other))
        return NotImplemented

    def __neg__(self):
        return Decibels(-self.value)

    def __pos__(self):
        return self

    # combining levels: 6*dB + 3*dB == 9*dB
    def __add__(self, other):
        if isinstance(other, Decibels):
            return Decibels(self.value + other.value)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Decibels):
            return Decibels(self.value - other.value)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Decibels):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Decibels):
            return self.value <= other.value
        return NotImplemented


dB = Decibels(1.0)
"""One decibel. Use as ``snd + 6*dB``, ``snd - 3*dB``, or ``masker + snr*dB``."""
