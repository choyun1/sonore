"""Frames: invertible analyses with a common contract.

A :class:`Frame` turns a :class:`~sonore.Sound` into coefficients and back.
The contract (docs/design/frames.md) is:

- ``synthesize(analyze(x)) == x`` to floating-point precision.
- For modified coefficients, ``synthesize`` returns the least-squares signal:
  the one whose coefficients are nearest to the given ones, in the norm of
  :meth:`Frame.energy`.
- ``frame_bounds(n_samples, fs) -> (A, B)`` are the extreme eigenvalues of the
  frame operator. ``A == B`` means tight. ``A == 0`` means not a frame, and
  ``synthesize`` raises (``analyze`` still works).

:class:`Filterbank` is the frequency-domain, undecimated family: filters given
by their responses on the DFT grid. Its frame operator is diagonal in frequency,
``s(f) = sum_k |H_k(f)|**2``, and the canonical dual filters are ``H_k / s``
(claim C4). Banks whose ``s`` is identically 1 set ``tight = True`` and skip
the division, as :class:`~sonore.filterbank.CosineFilterbank` does.

The code is written as pure array functions (no in-place mutation) so that a
JAX port is mechanical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from sonore.sound import Sound

if TYPE_CHECKING:
    from sonore.filterbank import Subbands

__all__ = ["Frame", "Filterbank"]

# synthesize() refuses when A <= NOT_A_FRAME * B (decision D4).
NOT_A_FRAME = 1e-12


def _check_frame(lower: float, upper: float, what: str) -> None:
    if not lower > NOT_A_FRAME * upper:
        raise ValueError(
            f"{what} is not a frame (bounds A={lower:.3g}, B={upper:.3g}): some part of the signal "
            "is not covered, so it cannot be synthesized. analyze() still works."
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
        squares are defined in (decision D2)."""


class Filterbank(Frame):
    """Undecimated filters applied by multiplication on the DFT grid.

    Subclasses define :meth:`response`, :attr:`n_filters` and :attr:`cfs`, and
    must be hashable (a frozen dataclass is the easy way) so the ringing time
    can be cached. Responses are zero-phase or, more generally, satisfy
    ``H(-f) = conj(H(f))``; only ``f >= 0`` is ever evaluated, and the
    subbands are real.

    Set ``tight = True`` only if the squared responses sum to exactly 1 at
    every frequency; synthesis then skips the division by ``s``.
    """

    tight: bool = False

    @property
    @abstractmethod
    def n_filters(self) -> int:
        """Number of filters (the band axis of :class:`~sonore.filterbank.Subbands`)."""

    @property
    @abstractmethod
    def cfs(self) -> np.ndarray:
        """Center frequency [Hz] of each filter, shape ``(n_filters,)``."""

    @abstractmethod
    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Responses at ``freqs`` [Hz] (all >= 0), shape ``(len(freqs), n_filters)``."""

    def rfft_response(self, n: int, fs: float) -> np.ndarray:
        """Responses on the ``rfft`` grid of an ``n``-sample signal."""
        return self.response(np.fft.rfftfreq(n, 1 / fs))

    def frame_power(self, n: int, fs: float) -> np.ndarray:
        """``s(f) = sum_k |H_k(f)|**2`` on the ``rfft`` grid of ``n`` samples:
        the eigenvalues of the (circular) frame operator."""
        return np.sum(np.abs(self.rfft_response(n, fs)) ** 2, axis=1)

    def ringing(self, fs: float, level_db: float = -60.0) -> int:
        """How long [samples] the filters ring: the longest time, over all
        filters, before the zero-phase impulse response stays below
        ``level_db`` re its peak (capped at 2 s). Used to size the padding."""
        try:
            return _ringing_samples(self, float(fs), float(level_db))
        except TypeError:  # unhashable subclass: compute without the cache
            return _ringing_samples.__wrapped__(self, float(fs), float(level_db))

    def _pad_samples(self, pad: float | str, fs: float) -> int:
        if pad == "auto":
            return self.ringing(fs)
        return int(round(float(pad) * fs))

    def analyze(self, sound: Sound, pad: float | str = "auto") -> Subbands:
        """Split ``sound`` into subbands (zero-phase, via FFT).

        By default the sound is zero-padded by the filters' ringing time
        (:meth:`ringing`), so filter ringing near one end can't wrap around to
        the other. The padding travels with the Subbands (and any Envelopes
        derived from them) and is removed on output, so ``.data`` and
        :meth:`Subbands.synthesize` have the sound's own length, and
        analysis followed by synthesis is exact.

        ``pad=0`` makes the analysis circular, which is what you want for
        periodic signals and for texture synthesis (seamless loops). A number
        pads by that many seconds.
        """
        from sonore.filterbank import Subbands

        p = self._pad_samples(pad, sound.fs)
        x = np.pad(sound.data, ((p, p), (0, 0))) if p else sound.data
        n = x.shape[0]
        H = self.rfft_response(n, sound.fs)  # (F, B)
        X = np.fft.rfft(x, axis=0)  # (F, C)
        bands = np.fft.irfft(X[:, None, :] * H[:, :, None], n=n, axis=0)  # (n, B, C)
        return Subbands(bands, sound.fs, self, pad=p)

    def synthesize(self, coefs: Subbands) -> Sound:
        """Canonical dual synthesis: filter each band with ``conj(H) / s`` and
        sum (with ``tight = True``, ``s = 1`` and this is re-filtering with
        ``H``), then remove the padding.

        The frame is the circular operator on the (padded) grid the
        coefficients live on (decision D1). Unmodified coefficients
        reconstruct exactly. For modified ones the result is the least-squares
        fit on that grid, cropped; with ``pad=0`` this is the canonical least
        squares on the signal itself. Raises if the bank leaves a frequency
        uncovered (A = 0).
        """
        full, fs, p = coefs._full, coefs.fs, coefs.pad
        n = full.shape[0]
        H = self.rfft_response(n, fs)
        if self.tight:
            X = np.fft.rfft(full, axis=0) * H[:, :, None]
            out = np.fft.irfft(X.sum(axis=1), n=n, axis=0)
        else:
            s = np.sum(np.abs(H) ** 2, axis=1)
            _check_frame(float(s.min()), float(s.max()), type(self).__name__)
            X = np.fft.rfft(full, axis=0) * np.conj(H)[:, :, None]
            out = np.fft.irfft(X.sum(axis=1) / s[:, None], n=n, axis=0)
        return Sound(out[p : n - p], fs)

    def frame_bounds(self, n_samples: int, fs: float, pad: float | str = "auto") -> tuple[float, float]:
        """``(min s, max s)`` on the DFT grid that :meth:`analyze` would use
        for a signal of ``n_samples`` at ``fs``, including its padding
        (decisions D1, D3). Tight banks return ``(1.0, 1.0)``."""
        if self.tight:
            return (1.0, 1.0)
        s = self.frame_power(int(n_samples) + 2 * self._pad_samples(pad, fs), fs)
        return (float(s.min()), float(s.max()))

    def energy(self, coefs: Subbands) -> np.ndarray:
        """Sum of squares of the subbands, padding included, per channel.
        Subbands are real, so every sample has weight 1 (claim C3)."""
        return np.sum(coefs._full**2, axis=(0, 1))


@lru_cache(maxsize=64)
def _ringing_samples(fb: Filterbank, fs: float, level_db: float) -> int:
    n = 1 << int(np.ceil(np.log2(4 * fs)))  # a 4 s grid: impulse responses up to 2 s each side
    h = np.abs(np.fft.irfft(fb.rfft_response(n, fs), n=n, axis=0)[: n // 2])
    above = h > h.max(axis=0, keepdims=True) * 10 ** (level_db / 20)
    last = np.array([np.flatnonzero(col).max() if col.any() else 0 for col in above.T])
    return int(last.max()) + 1
