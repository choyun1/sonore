"""A decibel unit, so level changes read like arithmetic: ``snd + 6*dB``.

``6*dB`` is a :class:`Decibels` value, not a number. Adding it to a
:class:`~sonore.Sound` changes the level; adding a Sound mixes; adding a bare
number is an error. That keeps ``snd + 0.5`` from silently meaning either
"+0.5 dB" or "add a DC offset of 0.5".
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

__all__ = ["Decibels", "dB"]


@dataclass(frozen=True)
class Decibels:
    """A level change in decibels (amplitude: ``gain = 10**(value/20)``).

    Write the number first (``6*dB``, ``2*(6*dB)``); ``dB*6`` is refused. A
    level is not a plain number either: ``float(6*dB)`` raises, and
    ``(6*dB).value`` is 6."""

    value: float

    # Let numpy scalars defer to us, so ``np.float64(6) * dB`` works.
    __array_ufunc__ = None

    @property
    def gain(self) -> float:
        """Linear amplitude factor."""
        return 10 ** (self.value / 20)

    def __repr__(self) -> str:
        return f"{self.value:g} dB"

    def __float__(self):
        # A level never passes silently as a bare number; read .value on purpose.
        raise TypeError(f"a level is not a plain number; use ({self!r}).value".replace(" dB)", "*dB)"))

    # scaling: 6*dB, -3*dB, 2*(6*dB), (6*dB)/2. The number comes first, as it is
    # written and read; dB*6 is refused rather than quietly meaning the same.
    def __rmul__(self, other):
        match other:
            case numbers.Real() if not isinstance(other, bool):
                return Decibels(self.value * float(other))
            case _:
                return self._refuse(other)

    def __mul__(self, other):
        match other:
            case numbers.Real() if not isinstance(other, bool):
                level = "dB" if self.value == 1 else f"({self!r})".replace(" dB", "*dB")
                raise TypeError(f"write the number first: did you mean {other:g}*{level}?")
            case _:
                return self._refuse(other)

    def _refuse(self, other):
        from sonore.core.sound import Sound

        match other:
            case Sound():
                raise TypeError(
                    "multiplying a Sound by dB is ambiguous; write snd + 6*dB for a gain "
                    "(note that snd * 6*dB parses as (snd * 6) * dB)"
                )
            case _:
                return NotImplemented

    def __truediv__(self, other):
        match other:
            case numbers.Real() if not isinstance(other, bool):
                return Decibels(self.value / float(other))
            case _:
                return NotImplemented

    def __neg__(self):
        return Decibels(-self.value)

    def __pos__(self):
        return self

    # combining levels: 6*dB + 3*dB == 9*dB
    def __add__(self, other):
        match other:
            case Decibels():
                return Decibels(self.value + other.value)
            case _:
                return NotImplemented

    def __sub__(self, other):
        match other:
            case Decibels():
                return Decibels(self.value - other.value)
            case _:
                return NotImplemented

    def __lt__(self, other):
        match other:
            case Decibels():
                return self.value < other.value
            case _:
                return NotImplemented

    def __le__(self, other):
        match other:
            case Decibels():
                return self.value <= other.value
            case _:
                return NotImplemented


dB = Decibels(1.0)
"""One decibel. Use as ``snd + 6*dB``, ``snd - 3*dB``, or ``masker + snr*dB``."""
