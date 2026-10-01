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

:class:`GaborFrame` is the one-sided STFT (wrapping
:class:`scipy.signal.ShortTimeFFT`). Its frame operator is diagonal in time,
``s(t) = K sum_q |w(t - q hop)|**2`` with ``K = n_fft`` (claim C5), and its
coefficient norm weights each stored bin by 2 except DC and, for even ``K``,
Nyquist (claim C3).

Every frame also has :meth:`Frame.adjoint`, the adjoint of ``analyze`` for
the coefficient inner product that ``energy`` uses (decision D10 of
docs/design/frames-step2.md): synthesis without the division by ``s``.

The code is written as pure array functions (no in-place mutation) so that a
JAX port is mechanical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import ShortTimeFFT, get_window

from sonore.sound import Sound

if TYPE_CHECKING:
    from sonore.filterbank import Subbands
    from sonore.representations import STFT

__all__ = ["Frame", "Filterbank", "GaborFrame"]

# synthesize() refuses when A <= NOT_A_FRAME * B (decision D4).
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
        squares are defined in (decision D2)."""

    @abstractmethod
    def adjoint(self, coefs) -> Sound:
        """The adjoint of :meth:`analyze`: the signal ``y`` with
        ``<analyze(x), coefs> == <x, y>`` for every ``x``, the coefficient
        inner product being the one :meth:`energy` uses (step 2, decision
        D10). It is :meth:`synthesize` without the division by the frame
        operator, and the gradient of a loss on the coefficients with respect
        to the signal."""


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
        """Responses on the ``rfft`` grid of an ``n``-sample signal.

        On an even grid, a complex response is replaced by its real part at
        Nyquist: ``irfft`` keeps only the real part of that bin, so this is
        the filter actually applied, and analysis, :meth:`frame_power` and
        :meth:`synthesize` must all see the same one (step 2, claim C14 and
        decision D8). Real responses are returned unchanged."""
        H = self.response(np.fft.rfftfreq(n, 1 / fs))
        if n % 2 == 0 and np.iscomplexobj(H):
            H = np.concatenate([H[:-1], H[-1:].real.astype(H.dtype)])
        return H

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

    def adjoint(self, coefs: Subbands) -> Sound:
        """Filter each band with ``conj(H)`` and sum, with no division by
        ``s``, then remove the padding (the adjoint of zero-padding is
        cropping). For a tight bank this equals :meth:`synthesize`."""
        full, fs, p = coefs._full, coefs.fs, coefs.pad
        n = full.shape[0]
        X = np.fft.rfft(full, axis=0) * np.conj(self.rfft_response(n, fs))[:, :, None]
        out = np.fft.irfft(X.sum(axis=1), n=n, axis=0)
        return Sound(out[p : n - p], fs)


@lru_cache(maxsize=64)
def _ringing_samples(fb: Filterbank, fs: float, level_db: float) -> int:
    n = 1 << int(np.ceil(np.log2(4 * fs)))  # a 4 s grid: impulse responses up to 2 s each side
    h = np.abs(np.fft.irfft(fb.rfft_response(n, fs), n=n, axis=0)[: n // 2])
    above = h > h.max(axis=0, keepdims=True) * 10 ** (level_db / 20)
    last = np.array([np.flatnonzero(col).max() if col.any() else 0 for col in above.T])
    return int(last.max()) + 1


@dataclass(frozen=True)
class GaborFrame(Frame):
    """The one-sided short-time Fourier transform as a frame.

    Parameters
    ----------
    win_dur
        Window length [s].
    hop_dur
        Hop [s]; defaults to a quarter window (75% overlap).
    window
        A :func:`scipy.signal.get_window` spec (name or tuple), sampled
        periodically, or a callable ``n -> array`` of length ``n``.
    n_fft
        FFT length K (``>=`` the window length); defaults to the window length.

    Durations are rounded to samples per sampling rate, and the
    :class:`~scipy.signal.ShortTimeFFT` is built once per rate
    (:meth:`sft`). ``analyze`` returns an :class:`~sonore.STFT`.

    A window/hop pair that leaves gaps (A = 0) is refused when the frame is
    first used at a sampling rate, because SciPy builds the dual window
    eagerly; the error gives the frame bounds (decision D4). A hop longer than
    the window is refused at construction.
    """

    win_dur: float
    hop_dur: float | None = None
    window: str | tuple | Callable[[int], np.ndarray] = "hann"
    n_fft: int | None = None

    def __post_init__(self):
        if not self.win_dur > 0:
            raise ValueError("win_dur must be positive")
        if self.hop_dur is not None and not 0 < self.hop_dur <= self.win_dur:
            raise ValueError(
                f"hop_dur must be in (0, win_dur]: a hop of {self.hop_dur:g} s with a {self.win_dur:g} s "
                "window leaves gaps, so it is not a frame"
            )

    def lengths(self, fs: float) -> tuple[int, int, int]:
        """``(window, hop, n_fft)`` in samples at ``fs``."""
        n_win = int(round(self.win_dur * fs))
        hop = max(1, int(round((self.hop_dur if self.hop_dur is not None else self.win_dur / 4) * fs)))
        n_fft = n_win if self.n_fft is None else int(self.n_fft)
        if n_win < 1:
            raise ValueError(f"win_dur {self.win_dur:g} s is shorter than one sample at {fs:g} Hz")
        if n_fft < n_win:
            raise ValueError(f"n_fft ({n_fft}) must be at least the window length ({n_win})")
        return n_win, hop, n_fft

    def window_samples(self, fs: float) -> np.ndarray:
        """The analysis window at ``fs``."""
        n_win = self.lengths(fs)[0]
        if callable(self.window):
            w = np.asarray(self.window(n_win), dtype=float)
            if w.shape != (n_win,):
                raise ValueError(f"window callable returned shape {w.shape}, expected ({n_win},)")
            return w
        return get_window(self.window, n_win, fftbins=True)

    def sft(self, fs: float) -> ShortTimeFFT:
        """The :class:`~scipy.signal.ShortTimeFFT` at ``fs`` (cached)."""
        return _gabor_sft(self, float(fs))

    def frame_power(self, n_samples: int, fs: float) -> np.ndarray:
        """``s(t) = K sum_q |w(t - q hop)|**2`` for ``t`` in ``0..n_samples-1``:
        the diagonal of the frame operator (claim C5). Sums over every frame
        that overlaps the signal; frames SciPy leaves out overlap it only
        where the window is zero, so they add nothing."""
        n = int(n_samples)
        w2 = np.abs(self.window_samples(fs)) ** 2
        n_win, hop, n_fft = self.lengths(fs)
        mid = n_win // 2
        q = np.arange((mid - n_win) // hop + 1, -(-(n + mid) // hop))
        t = q[:, None] * hop - mid + np.arange(n_win)[None, :]
        inside = (t >= 0) & (t < n)
        weights = np.broadcast_to(w2, t.shape)[inside]
        return n_fft * np.bincount(t[inside], weights=weights, minlength=n)

    def frame_bounds(self, n_samples: int, fs: float) -> tuple[float, float]:
        """``(min s, max s)`` over a signal of ``n_samples`` at ``fs``, in the
        weighted coefficient norm of :meth:`energy` (claims C3, C5)."""
        s = self.frame_power(n_samples, fs)
        return (float(s.min()), float(s.max()))

    def bin_weights(self, fs: float) -> np.ndarray:
        """Weight of each stored frequency bin (claim C3): 1 for DC and, for
        even ``n_fft``, Nyquist; 2 for the rest."""
        n_fft = self.lengths(fs)[2]
        w = np.full(n_fft // 2 + 1, 2.0)
        w[0] = 1.0
        if n_fft % 2 == 0:
            w[-1] = 1.0
        return w

    def analyze(self, sound: Sound) -> STFT:
        """The STFT of ``sound``, data shape ``(n_channels, n_freqs, n_frames)``."""
        from sonore.representations import STFT

        return STFT(sound, frame=self)

    def synthesize(self, coefs: STFT) -> Sound:
        """SciPy's ``istft``: the weighted real least-squares signal for
        ``coefs`` (claims C3, C6), exact for unmodified coefficients."""
        sft = self.sft(coefs.fs)
        if coefs.data.shape[-2] != sft.f_pts:
            raise ValueError(f"expected {sft.f_pts} frequency bins, got {coefs.data.shape[-2]}")
        x = sft.istft(coefs.data, k1=coefs.n_samples)
        return Sound(np.real(x).T, coefs.fs)

    def energy(self, coefs: STFT) -> np.ndarray:
        """Weighted coefficient energy per channel (claim C3): the energy of
        the full two-sided STFT."""
        w = self.bin_weights(coefs.fs)
        return np.einsum("f,cft->c", w, np.abs(coefs.data) ** 2)

    def adjoint(self, coefs: STFT) -> Sound:
        """``n_fft`` times SciPy's ``istft`` with ``dual_win = win``: the
        adjoint for the C3-weighted inner product (step 2, claim C13)."""
        sft = _gabor_adjoint_sft(self, float(coefs.fs))
        if coefs.data.shape[-2] != sft.f_pts:
            raise ValueError(f"expected {sft.f_pts} frequency bins, got {coefs.data.shape[-2]}")
        x = sft.mfft * sft.istft(coefs.data, k1=coefs.n_samples)
        return Sound(np.real(x).T, coefs.fs)


@lru_cache(maxsize=64)
def _gabor_sft(frame: GaborFrame, fs: float) -> ShortTimeFFT:
    win = frame.window_samples(fs)
    n_win, hop, n_fft = frame.lengths(fs)
    lower, upper = frame.frame_bounds(2 * (n_win + hop), fs)  # s is hop-periodic: this covers a period
    try:
        sft = ShortTimeFFT(win, hop=hop, fs=fs, mfft=n_fft, fft_mode="onesided")
    except ValueError as err:
        raise ValueError(
            f"{frame} at {fs:g} Hz is not a frame (bounds A={lower:.3g}, B={upper:.3g}); SciPy: {err}"
        ) from err
    _check_frame(lower, upper, f"{frame} at {fs:g} Hz", note="")
    return sft


@lru_cache(maxsize=64)
def _gabor_adjoint_sft(frame: GaborFrame, fs: float) -> ShortTimeFFT:
    win = frame.window_samples(fs)
    _, hop, n_fft = frame.lengths(fs)
    return ShortTimeFFT(win, hop=hop, fs=fs, mfft=n_fft, dual_win=win, fft_mode="onesided")
