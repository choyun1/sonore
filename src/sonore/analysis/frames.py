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
(Balazs et al., 2011). Banks whose ``s`` is identically 1 set ``tight = True`` and skip
the division, as :class:`~sonore.analysis.filterbank.CosineFilterbank` does.

:class:`GaborFrame` is the one-sided STFT (wrapping
:class:`scipy.signal.ShortTimeFFT`). Its frame operator is diagonal in time,
``s(t) = K sum_q |w(t - q hop)|**2`` with ``K = n_fft`` (this holds because
SciPy requires the window to fit in ``n_fft``). Only the non-negative
frequencies are stored, so the coefficient norm weights each stored bin by 2,
except DC and, for even ``K``, Nyquist, which have no mirror image: this
recovers the energy of the full two-sided STFT, and it is the norm in which
SciPy's ``istft`` is the least-squares inverse.

Every frame also has :meth:`Frame.adjoint`, the adjoint of ``analyze`` for
the coefficient inner product that ``energy`` uses: synthesis without the
division by ``s``. It is the gradient of a coefficient-domain loss with
respect to the signal.

:class:`TVGaborFrame` is the Gabor frame with one window per position
(nonstationary Gabor, painless case): every window fits in one FFT length
``M``, so the frame operator is again diagonal in time,
``s(t) = M sum_q |w_q(t - a_q)|**2``.

The code is written as pure array functions (no in-place mutation) so that a
JAX port is mechanical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft as sp_fft
from scipy.signal import ShortTimeFFT, get_window

from sonore.core.fft import fast_padding, threads
from sonore.core.sound import Sound

if TYPE_CHECKING:
    from sonore.analysis.filterbank import Subbands
    from sonore.analysis.representations import STFT, TVSTFT

__all__ = ["Frame", "Filterbank", "GaborFrame", "TVGaborFrame"]

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
        """Number of filters (the band axis of :class:`~sonore.analysis.filterbank.Subbands`)."""

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
        :meth:`synthesize` must all see the same one. Without this, the dual
        divides by ``|H(fs/2)|**2`` while analysis applied ``Re H(fs/2)``, and
        reconstruction is off by up to ~1e-5. Real responses are returned
        unchanged."""
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
        ``level_db`` re its peak (capped at 2 s). The minimum padding of
        ``pad="auto"``."""
        try:
            return _ringing_samples(self, float(fs), float(level_db))
        except TypeError:  # unhashable subclass: compute without the cache
            return _ringing_samples.__wrapped__(self, float(fs), float(level_db))

    def _pad_samples(self, pad: float | str, fs: float, n_samples: int) -> int:
        if pad == "auto":
            # at least the ringing, rounded up to an FFT length without large prime factors
            return fast_padding(int(n_samples), self.ringing(fs))
        return int(round(float(pad) * fs))

    def analyze(self, sound: Sound, pad: float | str = "auto") -> Subbands:
        """Split ``sound`` into subbands (zero-phase, via FFT).

        By default the sound is zero-padded by at least the filters' ringing
        time (:meth:`ringing`), so filter ringing near one end can't wrap
        around to the other. The padding is rounded up (typically by well
        under 1% of the length) so the FFT length has no large prime factor,
        which makes the transforms several times faster. The padding travels
        with the Subbands (and any Envelopes derived from them) and is
        removed on output, so ``.data`` and :meth:`Subbands.synthesize` have
        the sound's own length, and analysis followed by synthesis is exact.

        ``pad=0`` makes the analysis circular, which is what you want for
        periodic signals and for texture synthesis (seamless loops). A number
        pads by that many seconds.
        """
        from sonore.analysis.filterbank import Subbands

        p = self._pad_samples(pad, sound.fs, sound.data.shape[0])
        x = np.pad(sound.data, ((p, p), (0, 0))) if p else sound.data
        n = x.shape[0]
        H = self.rfft_response(n, sound.fs)  # (F, B)
        with threads():
            X = sp_fft.rfft(x, axis=0)  # (F, C)
            bands = sp_fft.irfft(X[:, None, :] * H[:, :, None], n=n, axis=0)  # (n, B, C)
        return Subbands(bands, sound.fs, self, pad=p)

    def synthesize(self, coefs: Subbands) -> Sound:
        """Canonical dual synthesis: filter each band with ``conj(H) / s`` and
        sum (with ``tight = True``, ``s = 1`` and this is re-filtering with
        ``H``), then remove the padding.

        The frame is the circular operator on the (padded) grid the
        coefficients live on. Unmodified coefficients reconstruct exactly.
        For modified ones the result is the least-squares fit on that grid,
        cropped; with ``pad=0`` this is the canonical least squares on the
        signal itself. With padding on a non-tight bank it differs slightly
        (about 1% on masked coefficients) from the canonical least squares on
        the unpadded signal, because masked energy that lands in the padding is
        discarded rather than refit; this is a deliberate choice, exact and
        non-iterative. Raises if the bank leaves a frequency
        uncovered (A = 0).
        """
        full, fs, p = coefs._full, coefs.fs, coefs.pad
        n = full.shape[0]
        H = self.rfft_response(n, fs)
        if self.tight:
            with threads():
                X = sp_fft.rfft(full, axis=0) * H[:, :, None]
                out = sp_fft.irfft(X.sum(axis=1), n=n, axis=0)
        else:
            s = np.sum(np.abs(H) ** 2, axis=1)
            _check_frame(float(s.min()), float(s.max()), type(self).__name__)
            with threads():
                X = sp_fft.rfft(full, axis=0) * np.conj(H)[:, :, None]
                out = sp_fft.irfft(X.sum(axis=1) / s[:, None], n=n, axis=0)
        return Sound(out[p : n - p], fs)

    def frame_bounds(self, n_samples: int, fs: float, pad: float | str = "auto") -> tuple[float, float]:
        """``(min s, max s)`` on the DFT grid that :meth:`analyze` would use
        for a signal of ``n_samples`` at ``fs``, including its padding: the
        bounds of the grid the coefficients actually live on, which depends on
        the signal length and rate (hence a method, not a property). Tight banks return ``(1.0, 1.0)``."""
        if self.tight:
            return (1.0, 1.0)
        s = self.frame_power(int(n_samples) + 2 * self._pad_samples(pad, fs, n_samples), fs)
        return (float(s.min()), float(s.max()))

    def energy(self, coefs: Subbands) -> np.ndarray:
        """Sum of squares of the subbands, padding included, per channel.
        Subbands are real, so every sample has weight 1."""
        return np.sum(coefs._full**2, axis=(0, 1))

    def adjoint(self, coefs: Subbands) -> Sound:
        """Filter each band with ``conj(H)`` and sum, with no division by
        ``s``, then remove the padding (the adjoint of zero-padding is
        cropping). For a tight bank this equals :meth:`synthesize`."""
        full, fs, p = coefs._full, coefs.fs, coefs.pad
        n = full.shape[0]
        with threads():
            X = sp_fft.rfft(full, axis=0) * np.conj(self.rfft_response(n, fs))[:, :, None]
            out = sp_fft.irfft(X.sum(axis=1), n=n, axis=0)
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
    eagerly; the error gives the frame bounds. A hop longer than
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
        the diagonal of the frame operator. Sums over every frame
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
        weighted coefficient norm of :meth:`energy`."""
        s = self.frame_power(n_samples, fs)
        return (float(s.min()), float(s.max()))

    def bin_weights(self, fs: float) -> np.ndarray:
        """Weight of each stored frequency bin: 1 for DC and, for even
        ``n_fft``, Nyquist; 2 for the rest, which stand for their negative-
        frequency mirror images too."""
        n_fft = self.lengths(fs)[2]
        w = np.full(n_fft // 2 + 1, 2.0)
        w[0] = 1.0
        if n_fft % 2 == 0:
            w[-1] = 1.0
        return w

    def analyze(self, sound: Sound) -> STFT:
        """The STFT of ``sound``, data shape ``(n_channels, n_freqs, n_frames)``."""
        from sonore.analysis.representations import STFT

        return STFT(sound, frame=self)

    def synthesize(self, coefs: STFT) -> Sound:
        """SciPy's ``istft``: the weighted real least-squares signal for
        ``coefs``, exact for unmodified coefficients. Padding does not change
        this, since the frame operator is diagonal in time."""
        sft = self.sft(coefs.fs)
        if coefs.data.shape[-2] != sft.f_pts:
            raise ValueError(f"expected {sft.f_pts} frequency bins, got {coefs.data.shape[-2]}")
        x = sft.istft(coefs.data, k1=coefs.n_samples)
        return Sound(np.real(x).T, coefs.fs)

    def energy(self, coefs: STFT) -> np.ndarray:
        """Weighted coefficient energy per channel: the energy of the full
        two-sided STFT."""
        w = self.bin_weights(coefs.fs)
        return np.einsum("f,cft->c", w, np.abs(coefs.data) ** 2)

    def adjoint(self, coefs: STFT) -> Sound:
        """``n_fft`` times SciPy's ``istft`` with ``dual_win = win``: the
        adjoint for the bin-weighted inner product of :meth:`energy` (verified
        against a dense matrix, including odd and zero-padded FFTs)."""
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


def _window(spec, n: int) -> np.ndarray:
    """A periodic window of ``n`` samples from a get_window spec or a callable."""
    if callable(spec):
        w = np.asarray(spec(n), dtype=float)
        if w.shape != (n,):
            raise ValueError(f"window callable returned shape {w.shape}, expected ({n},)")
        return w
    return get_window(spec, n, fftbins=True)


@dataclass(frozen=True)
class TVGaborFrame(Frame):
    """A Gabor frame with a time-varying window.

    Window ``q`` is ``win_durs[q]`` long and centered at ``times[q]`` [s];
    each windowed segment is zero-padded to one FFT length ``n_fft`` and
    transformed, with the phase referenced to the window's center, as
    :class:`GaborFrame` (SciPy) does. Every window must fit in ``n_fft``
    samples, the painless condition that keeps the frame operator diagonal
    (Balazs et al., 2011); ``n_fft`` defaults to the longest window. Durations and
    times are rounded to samples at each sampling rate, and a window longer
    than ``n_fft`` is refused when the frame is first used at a rate.

    Parameters
    ----------
    times
        Window centers [s], strictly increasing.
    win_durs
        Window lengths [s], one per center.
    n_fft
        FFT length [samples], at least the longest window.
    window
        A :func:`scipy.signal.get_window` spec, sampled periodically at each
        window's length, or a callable ``n -> array``.

    ``analyze`` returns a :class:`~sonore.analysis.representations.TVSTFT`, data shape
    ``(n_channels, n_fft // 2 + 1, len(times))``. The signal is zero outside
    its own extent, as in :class:`GaborFrame`. A constant schedule over the
    frames SciPy uses reproduces :class:`GaborFrame` exactly.
    """

    times: Sequence[float]
    win_durs: Sequence[float]
    n_fft: int | None = None
    window: str | tuple | Callable[[int], np.ndarray] = "hann"

    def __post_init__(self):
        times = tuple(float(t) for t in np.atleast_1d(np.asarray(self.times, float)))
        durs = tuple(float(d) for d in np.atleast_1d(np.asarray(self.win_durs, float)))
        if len(times) != len(durs) or not times:
            raise ValueError("times and win_durs must be non-empty and of equal length")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite and strictly increasing")
        if not min(durs) > 0:
            raise ValueError("win_durs must be positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "win_durs", durs)

    @classmethod
    def from_function(
        cls,
        win_dur_of_t: Callable[[float], float],
        t_end: float,
        overlap: float = 4,
        t_start: float = 0.0,
        **kwargs,
    ) -> TVGaborFrame:
        """A schedule that steps from ``t_start`` to ``t_end`` [s] with hop
        ``win_dur_of_t(t) / overlap`` at each center ``t``. Use ``t_end`` at
        least the signal's duration so the windows cover it. Other arguments
        go to the constructor."""
        if not overlap >= 1:
            raise ValueError("overlap must be at least 1, or the windows leave gaps")
        times, durs, t = [], [], float(t_start)
        while t <= t_end:
            d = float(win_dur_of_t(t))
            if not d > 0:
                raise ValueError(f"win_dur_of_t({t:g}) = {d:g} is not positive")
            times.append(t)
            durs.append(d)
            t += d / overlap
        return cls(tuple(times), tuple(durs), **kwargs)

    @classmethod
    def pitch_adaptive(
        cls,
        f0_times: Sequence[float],
        f0: Sequence[float],
        t_end: float,
        periods: float = 3.0,
        overlap: float = 4,
        t_start: float = 0.0,
        **kwargs,
    ) -> TVGaborFrame:
        """Windows ``periods`` fundamental periods long, following an F0 track.

        ``f0`` [Hz] at ``f0_times`` [s] is 0 or NaN where unvoiced. F0 is
        carried through unvoiced stretches, log-linearly between the voiced
        neighbors and held constant before the first and after the last voiced
        point, so the window length never jumps: jumps are what cost frame
        bounds, not the adaptation itself. The hop is ``window / overlap``, as
        in :meth:`from_function`, from ``t_start`` to ``t_end``; to cover a
        sound evenly up to its last sample, ``t_end`` should be its duration
        plus half the longest window.

        Three periods is the shortest whole number of periods at which
        neighboring harmonics separate clearly (a peak-to-dip ratio of about
        12 dB with a Hann window, the same at every F0) while the flicker at
        the period rate cancels; it is also the window length of WORLD's
        CheapTrick (Morise, 2015). Other arguments go to the constructor.
        """
        f0_of_t = _bridged_f0(f0_times, f0)
        return cls.from_function(
            lambda t: periods / f0_of_t(t), t_end, overlap=overlap, t_start=t_start, **kwargs
        )

    def layout(self, fs: float) -> _TVLayout:
        """Window lengths, centers and FFT length in samples at ``fs`` (cached)."""
        return _tv_layout(self, float(fs))

    def frame_power(self, n_samples: int, fs: float) -> np.ndarray:
        """``s(t) = M sum_q |w_q(t - a_q)|**2`` for ``t`` in ``0..n_samples-1``:
        the diagonal of the frame operator."""
        lay, n = self.layout(fs), int(n_samples)
        t, valid = lay.positions(n)
        return lay.n_fft * np.bincount(t[valid], weights=np.abs(lay.windows[valid]) ** 2, minlength=n)

    def frame_bounds(self, n_samples: int, fs: float) -> tuple[float, float]:
        """``(min s, max s)`` over a signal of ``n_samples`` at ``fs``, in the
        weighted coefficient norm of :meth:`energy`."""
        s = self.frame_power(n_samples, fs)
        return (float(s.min()), float(s.max()))

    def bin_weights(self, fs: float) -> np.ndarray:
        """Weight of each stored frequency bin, as for :class:`GaborFrame`."""
        n_fft = self.layout(fs).n_fft
        w = np.full(n_fft // 2 + 1, 2.0)
        w[0] = 1.0
        if n_fft % 2 == 0:
            w[-1] = 1.0
        return w

    def analyze(self, sound: Sound) -> TVSTFT:
        """The time-varying STFT of ``sound``, data shape ``(n_channels, n_freqs, n_frames)``."""
        from sonore.analysis.representations import TVSTFT

        lay, n = self.layout(sound.fs), len(sound)
        t, valid = lay.positions(n)
        segs = np.where(valid, lay.windows, 0.0) * sound.data[np.clip(t, 0, n - 1)].transpose(2, 0, 1)
        data = np.fft.rfft(segs, axis=-1).transpose(0, 2, 1)
        return TVSTFT(data, sound.fs, n, self)

    def _check(self, coefs: TVSTFT) -> _TVLayout:
        lay = self.layout(coefs.fs)
        expected = (lay.n_fft // 2 + 1, len(self.times))
        if coefs.data.shape[1:] != expected:
            raise ValueError(f"expected (n_freqs, n_frames) = {expected}, got {coefs.data.shape[1:]}")
        return lay

    def _overlap_add(self, coefs: TVSTFT, lay: _TVLayout) -> np.ndarray:
        n = coefs.n_samples
        t, valid = lay.positions(n)
        y = lay.n_fft * np.fft.irfft(coefs.data, n=lay.n_fft, axis=1).transpose(0, 2, 1)  # (C, Q, M)
        contrib = y * np.conj(lay.windows)[None]
        out = [np.bincount(t[valid], weights=c[valid], minlength=n) for c in contrib]
        return np.stack(out, axis=1)

    def adjoint(self, coefs: TVSTFT) -> Sound:
        """Conjugate-window overlap-add of ``n_fft * irfft`` of each frame,
        without dividing by ``s``."""
        return Sound(self._overlap_add(coefs, self._check(coefs)), coefs.fs)

    def synthesize(self, coefs: TVSTFT) -> Sound:
        """The adjoint divided by ``s(t)``: the canonical dual, so the
        weighted least-squares signal for ``coefs``, exact for unmodified ones.
        Raises if the windows leave part of the signal uncovered."""
        lay = self._check(coefs)
        s = self.frame_power(coefs.n_samples, coefs.fs)
        _check_frame(float(s.min()), float(s.max()), f"{type(self).__name__} at {coefs.fs:g} Hz")
        return Sound(self._overlap_add(coefs, lay) / s[:, None], coefs.fs)

    def energy(self, coefs: TVSTFT) -> np.ndarray:
        """Weighted coefficient energy per channel, as for :class:`GaborFrame`."""
        w = self.bin_weights(coefs.fs)
        return np.einsum("f,cft->c", w, np.abs(coefs.data) ** 2)


def _bridged_f0(f0_times: Sequence[float], f0: Sequence[float]) -> Callable[[float], float]:
    """F0 [Hz] as a function of time, bridged log-linearly across unvoiced
    points (0 or NaN) and held constant beyond the voiced extent."""
    t = np.asarray(f0_times, dtype=float)
    f = np.asarray(f0, dtype=float)
    if t.shape != f.shape or t.ndim != 1:
        raise ValueError("f0_times and f0 must be 1-D and of equal length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("f0_times must be strictly increasing")
    voiced = np.isfinite(f) & (f > 0)
    if not voiced.any():
        raise ValueError("the F0 track has no voiced points")
    tv, log_f = t[voiced], np.log(f[voiced])
    return lambda u: float(np.exp(np.interp(u, tv, log_f)))


@dataclass(frozen=True)
class _TVLayout:
    """A :class:`TVGaborFrame` in samples at one rate. Row ``q`` of
    ``offsets``/``windows`` is FFT buffer index ``j`` of frame ``q``: the
    signal sample ``centers[q] + offsets[q, j]`` and its window weight (0 in
    the zero padding). Index 0 is the window's middle sample, as in SciPy."""

    lengths: np.ndarray
    centers: np.ndarray
    n_fft: int
    offsets: np.ndarray
    windows: np.ndarray

    def positions(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Signal indices ``(Q, n_fft)`` and where they hold real signal samples."""
        t = self.centers[:, None] + self.offsets
        return t, (t >= 0) & (t < n) & (self.windows != 0)


@lru_cache(maxsize=64)
def _tv_layout(frame: TVGaborFrame, fs: float) -> _TVLayout:
    lengths = np.array([int(round(d * fs)) for d in frame.win_durs])
    if lengths.min() < 1:
        raise ValueError(f"a window in win_durs is shorter than one sample at {fs:g} Hz")
    n_fft = int(lengths.max()) if frame.n_fft is None else int(frame.n_fft)
    if n_fft < lengths.max():
        raise ValueError(
            f"n_fft ({n_fft}) must be at least the longest window ({lengths.max()} samples at {fs:g} Hz): "
            "a longer window would make the frame operator non-diagonal"
        )
    centers = np.round(np.asarray(frame.times) * fs).astype(int)
    j = np.arange(n_fft)
    m = (j[None, :] + lengths[:, None] // 2) % n_fft  # window sample at buffer index j
    inside = m < lengths[:, None]
    windows = np.zeros((len(lengths), n_fft))
    for q, L in enumerate(lengths):
        w = _window(frame.window, int(L))
        windows[q, inside[q]] = w[m[q, inside[q]]]
    offsets = np.where(inside, m - lengths[:, None] // 2, 0)
    return _TVLayout(lengths, centers, n_fft, offsets, windows)
