"""Views: one-way analyses that say what they drop.

A :class:`~sonore.frames.frame.Frame` turns a sound into coefficients and
back. A view keeps only part of a sound, so many sounds give the same view
and none can be read back from it alone. Every view therefore has a
:meth:`View.synthesize` that refuses, raising :class:`NotInvertibleError`
with the mathematical reason (:attr:`View.discards`) and the route that does
lead back to a sound (:attr:`View.back_to_sound`), if sonore has one.

That route is ``to_sound``, on the views that have a canonical one, and its
arguments are what the view discarded: ``Spectrum.to_sound`` takes a carrier
for the phase, ``Cepstrum.to_sound`` a phase, ``PVAnalysis.to_sound`` a time
scale and a frequency map. On any other view, :meth:`View.to_sound` refuses
as ``synthesize`` does.
"""

from __future__ import annotations

__all__ = ["NotInvertibleError", "View"]

_NO_ROUTE = "sonore has no canonical route from it back to a sound yet."


class NotInvertibleError(NotImplementedError):
    """Raised by :meth:`View.synthesize`, and by :meth:`View.to_sound` on a
    view with no route back: the view keeps too little of a sound for any
    sound to be recovered from it alone."""


class View:
    """Base class of every view.

    Each subclass sets two class attributes: :attr:`discards`, one sentence
    saying what the view drops and so why it cannot be inverted, and
    :attr:`back_to_sound`, one sentence naming the route to a sound that
    exists, or an empty string if there is none.
    """

    discards = ""
    back_to_sound = ""

    def _refuse(self):
        raise NotInvertibleError(f"{self.discards} {self.back_to_sound or _NO_ROUTE}")

    def synthesize(self, *args, **kwargs):
        """Refuse: raise :class:`NotInvertibleError` with :attr:`discards`
        and :attr:`back_to_sound` as the message."""
        self._refuse()

    def to_sound(self, *args, **kwargs):
        """Refuse, as :meth:`synthesize` does, on a view with no canonical
        route back to a sound; views that have one override it."""
        self._refuse()
