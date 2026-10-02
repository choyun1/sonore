"""Views: one-way analyses that say what they drop.

A :class:`~sonore.frames.frame.Frame` turns a sound into coefficients and
back. A view keeps only part of a sound, so many sounds give the same view
and none can be read back from it. Every view therefore has a
:meth:`View.synthesize` that refuses, raising :class:`NotInvertibleError`
with the mathematical reason (:attr:`View.discards`) and, where sonore has
one, the route that does lead back to a sound (:attr:`View.back_to_sound`).
Routes that make a sound from a view keep names that say what they assume,
such as ``Cepstrum.to_sound`` or ``Spectrum.to_noise``, and are never called
``synthesize``.
"""

from __future__ import annotations

__all__ = ["NotInvertibleError", "View"]

_NO_ROUTE = "sonore has no route from it back to a sound."


class NotInvertibleError(NotImplementedError):
    """Raised by :meth:`View.synthesize`: the view keeps too little of a
    sound for any sound to be recovered from it alone."""


class View:
    """Base class of every view.

    Each subclass sets two class attributes: :attr:`discards`, one sentence
    saying what the view drops and so why it cannot be inverted, and
    :attr:`back_to_sound`, one sentence naming the route to a sound that
    exists, or an empty string if there is none.
    """

    discards = ""
    back_to_sound = ""

    def synthesize(self, *args, **kwargs):
        """Refuse: raise :class:`NotInvertibleError` with :attr:`discards`
        and :attr:`back_to_sound` as the message."""
        raise NotInvertibleError(f"{self.discards} {self.back_to_sound or _NO_ROUTE}")
