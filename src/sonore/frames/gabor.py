"""Gabor frames: the STFT as a frame, and its coefficients.

:class:`GaborFrame` is the one-sided STFT (wrapping
:class:`scipy.signal.ShortTimeFFT`). Its frame operator is diagonal in time,
``s(t) = K sum_q |w(t - q hop)|**2`` with ``K = n_fft`` (this holds because
SciPy requires the window to fit in ``n_fft``). Only the non-negative
frequencies are stored, so the coefficient norm weights each stored bin by 2,
except DC and, for even ``K``, Nyquist, which have no mirror image: this
recovers the energy of the full two-sided STFT, and it is the norm in which
SciPy's ``istft`` is the least-squares inverse.

:class:`TVGaborFrame` is the Gabor frame with one window per position
(nonstationary Gabor, painless case): every window fits in one FFT length
``M``, so the frame operator is again diagonal in time,
``s(t) = M sum_q |w_q(t - a_q)|**2``.

:class:`STFT` and :class:`TVSTFT` are their coefficients, which synthesize
back exactly. Levels are in dB of *power* (``20*log10|X|``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import ShortTimeFFT, get_window

from sonore.core.sound import Sound
from sonore.frames.frame import Frame, _check_frame

if TYPE_CHECKING:
    pass
from sonore.core.utils import amp_to_db, as_rng

__all__ = ["GaborFrame", "TVGaborFrame", "STFT", "TVSTFT"]


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

    Analysis always works, even for a window and hop that leave gaps
    (A = 0, including a hop longer than the window): the STFT is still a
    picture of the sound. :meth:`synthesize` (and Griffin-Lim) refuse such a
    pair, and the error gives the frame bounds.
    """

    win_dur: float
    hop_dur: float | None = None
    window: str | tuple | Callable[[int], np.ndarray] = "hann"
    n_fft: int | None = None

    def __post_init__(self):
        if not self.win_dur > 0:
            raise ValueError("win_dur must be positive")
        if self.hop_dur is not None and not self.hop_dur > 0:
            raise ValueError("hop_dur must be positive")

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
        """The :class:`~scipy.signal.ShortTimeFFT` at ``fs`` (cached), with
        the canonical dual window for synthesis. Raises, with the frame
        bounds, when the window and hop leave gaps."""
        return _gabor_sft(self, float(fs))

    def frame_power(self, n_samples: int, fs: float) -> np.ndarray:
        """``s(t) = K sum_q |w(t - q hop)|**2`` for ``t`` in ``0..n_samples-1``:
        the diagonal of the frame operator. Sums over every time window
        that overlaps the signal; time windows SciPy leaves out overlap it only
        where the window is zero, so they add nothing."""
        n = int(n_samples)
        window_power = np.abs(self.window_samples(fs)) ** 2
        n_win, hop, n_fft = self.lengths(fs)
        centre_index = n_win // 2
        window_indices = np.arange((centre_index - n_win) // hop + 1, -(-(n + centre_index) // hop))
        positions = window_indices[:, None] * hop - centre_index + np.arange(n_win)[None, :]
        inside = (positions >= 0) & (positions < n)
        weights = np.broadcast_to(window_power, positions.shape)[inside]
        return n_fft * np.bincount(positions[inside], weights=weights, minlength=n)

    def frame_bounds(self, n_samples: int, fs: float) -> tuple[float, float]:
        """``(min s, max s)`` over a signal of ``n_samples`` at ``fs``, in the
        weighted coefficient norm of :meth:`energy`."""
        power_sum = self.frame_power(n_samples, fs)
        return (float(power_sum.min()), float(power_sum.max()))

    def bin_weights(self, fs: float) -> np.ndarray:
        """Weight of each stored frequency bin: 1 for DC and, for even
        ``n_fft``, Nyquist; 2 for the rest, which stand for their negative-
        frequency mirror images too."""
        n_fft = self.lengths(fs)[2]
        weights = np.full(n_fft // 2 + 1, 2.0)
        weights[0] = 1.0
        if n_fft % 2 == 0:
            weights[-1] = 1.0
        return weights

    def analyze(self, sound: Sound) -> STFT:
        """The STFT of ``sound``, data shape ``(n_channels, n_freqs, n_windows)``."""

        return STFT(sound, frame=self)

    def synthesize(self, coefs: STFT) -> Sound:
        """SciPy's ``istft``: the weighted real least-squares signal for
        ``coefs``, exact for unmodified coefficients. Padding does not change
        this, since the frame operator is diagonal in time."""
        short_time_fft = self.sft(coefs.fs)
        if coefs.data.shape[-2] != short_time_fft.f_pts:
            raise ValueError(f"expected {short_time_fft.f_pts} frequency bins, got {coefs.data.shape[-2]}")
        signal = short_time_fft.istft(coefs.data, k1=coefs.n_samples)
        return Sound(np.real(signal).T, coefs.fs)

    def energy(self, coefs: STFT) -> np.ndarray:
        """Weighted coefficient energy per channel: the energy of the full
        two-sided STFT."""
        weights = self.bin_weights(coefs.fs)
        return np.einsum("f,cft->c", weights, np.abs(coefs.data) ** 2)

    def adjoint(self, coefs: STFT) -> Sound:
        """``n_fft`` times SciPy's ``istft`` with ``dual_win = win``: the
        adjoint for the bin-weighted inner product of :meth:`energy` (verified
        against a dense matrix, including odd and zero-padded FFTs)."""
        short_time_fft = _gabor_adjoint_sft(self, float(coefs.fs))
        if coefs.data.shape[-2] != short_time_fft.f_pts:
            raise ValueError(f"expected {short_time_fft.f_pts} frequency bins, got {coefs.data.shape[-2]}")
        signal = short_time_fft.mfft * short_time_fft.istft(coefs.data, k1=coefs.n_samples)
        return Sound(np.real(signal).T, coefs.fs)


@lru_cache(maxsize=64)
def _gabor_sft(frame: GaborFrame, fs: float) -> ShortTimeFFT:
    window = frame.window_samples(fs)
    n_win, hop, n_fft = frame.lengths(fs)
    lower, upper = frame.frame_bounds(2 * (n_win + hop), fs)  # s is hop-periodic: this covers a period
    try:
        short_time_fft = ShortTimeFFT(window, hop=hop, fs=fs, mfft=n_fft, fft_mode="onesided")
    except ValueError as err:
        raise ValueError(
            f"{frame} at {fs:g} Hz is not a frame (bounds A={lower:.3g}, B={upper:.3g}); SciPy: {err}"
        ) from err
    _check_frame(lower, upper, f"{frame} at {fs:g} Hz", note="")
    return short_time_fft


def _analysis_sft(frame: GaborFrame, fs: float) -> ShortTimeFFT:
    """The frame's own ShortTimeFFT, or, when the window and hop leave gaps,
    one with the window as its own dual: analysis does not use the dual, so
    the coefficients are the same, and only synthesis needs to refuse."""
    try:
        return frame.sft(fs)
    except ValueError:
        return _gabor_adjoint_sft(frame, fs)


@lru_cache(maxsize=64)
def _gabor_adjoint_sft(frame: GaborFrame, fs: float) -> ShortTimeFFT:
    window = frame.window_samples(fs)
    _, hop, n_fft = frame.lengths(fs)
    return ShortTimeFFT(window, hop=hop, fs=fs, mfft=n_fft, dual_win=window, fft_mode="onesided")


def _window(spec, n: int) -> np.ndarray:
    """A periodic window of ``n`` samples from a get_window spec or a callable."""
    if callable(spec):
        samples = np.asarray(spec(n), dtype=float)
        if samples.shape != (n,):
            raise ValueError(f"window callable returned shape {samples.shape}, expected ({n},)")
        return samples
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

    ``analyze`` returns a :class:`~sonore.frames.gabor.TVSTFT`, data shape
    ``(n_channels, n_fft // 2 + 1, len(times))``. The signal is zero outside
    its own extent, as in :class:`GaborFrame`. A constant schedule over the
    time windows SciPy uses reproduces :class:`GaborFrame` exactly.
    """

    times: Sequence[float]
    win_durs: Sequence[float]
    n_fft: int | None = None
    window: str | tuple | Callable[[int], np.ndarray] = "hann"

    def __post_init__(self):
        times = tuple(float(t) for t in np.atleast_1d(np.asarray(self.times, float)))
        durs = tuple(float(dur) for dur in np.atleast_1d(np.asarray(self.win_durs, float)))
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
            win_dur = float(win_dur_of_t(t))
            if not win_dur > 0:
                raise ValueError(f"win_dur_of_t({t:g}) = {win_dur:g} is not positive")
            times.append(t)
            durs.append(win_dur)
            t += win_dur / overlap
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
        frame_layout, n = self.layout(fs), int(n_samples)
        signal_index, valid = frame_layout.positions(n)
        return frame_layout.n_fft * np.bincount(
            signal_index[valid], weights=np.abs(frame_layout.windows[valid]) ** 2, minlength=n
        )

    def frame_bounds(self, n_samples: int, fs: float) -> tuple[float, float]:
        """``(min s, max s)`` over a signal of ``n_samples`` at ``fs``, in the
        weighted coefficient norm of :meth:`energy`."""
        power_sum = self.frame_power(n_samples, fs)
        return (float(power_sum.min()), float(power_sum.max()))

    def bin_weights(self, fs: float) -> np.ndarray:
        """Weight of each stored frequency bin, as for :class:`GaborFrame`."""
        n_fft = self.layout(fs).n_fft
        weights = np.full(n_fft // 2 + 1, 2.0)
        weights[0] = 1.0
        if n_fft % 2 == 0:
            weights[-1] = 1.0
        return weights

    def analyze(self, sound: Sound) -> TVSTFT:
        """The time-varying STFT of ``sound``, data shape ``(n_channels, n_freqs, n_windows)``."""

        frame_layout, n = self.layout(sound.fs), len(sound)
        signal_index, valid = frame_layout.positions(n)
        segments = np.where(valid, frame_layout.windows, 0.0) * sound.data[
            np.clip(signal_index, 0, n - 1)
        ].transpose(2, 0, 1)
        data = np.fft.rfft(segments, axis=-1).transpose(0, 2, 1)
        return TVSTFT(data, sound.fs, n, self)

    def _check(self, coefs: TVSTFT) -> _TVLayout:
        frame_layout = self.layout(coefs.fs)
        expected = (frame_layout.n_fft // 2 + 1, len(self.times))
        if coefs.data.shape[1:] != expected:
            raise ValueError(f"expected (n_freqs, n_windows) = {expected}, got {coefs.data.shape[1:]}")
        return frame_layout

    def _overlap_add(self, coefs: TVSTFT, frame_layout: _TVLayout) -> np.ndarray:
        n = coefs.n_samples
        signal_index, valid = frame_layout.positions(n)
        segments = frame_layout.n_fft * np.fft.irfft(coefs.data, n=frame_layout.n_fft, axis=1).transpose(
            0, 2, 1
        )  # (C, Q, M)
        weighted_segments = segments * np.conj(frame_layout.windows)[None]
        channel_signals = [
            np.bincount(signal_index[valid], weights=channel_segments[valid], minlength=n)
            for channel_segments in weighted_segments
        ]
        return np.stack(channel_signals, axis=1)

    def adjoint(self, coefs: TVSTFT) -> Sound:
        """Conjugate-window overlap-add of ``n_fft * irfft`` of each time window,
        without dividing by ``s``."""
        return Sound(self._overlap_add(coefs, self._check(coefs)), coefs.fs)

    def synthesize(self, coefs: TVSTFT) -> Sound:
        """The adjoint divided by ``s(t)``: the canonical dual, so the
        weighted least-squares signal for ``coefs``, exact for unmodified ones.
        Raises if the windows leave part of the signal uncovered."""
        frame_layout = self._check(coefs)
        power_sum = self.frame_power(coefs.n_samples, coefs.fs)
        _check_frame(
            float(power_sum.min()), float(power_sum.max()), f"{type(self).__name__} at {coefs.fs:g} Hz"
        )
        return Sound(self._overlap_add(coefs, frame_layout) / power_sum[:, None], coefs.fs)

    def energy(self, coefs: TVSTFT) -> np.ndarray:
        """Weighted coefficient energy per channel, as for :class:`GaborFrame`."""
        weights = self.bin_weights(coefs.fs)
        return np.einsum("f,cft->c", weights, np.abs(coefs.data) ** 2)


def _bridged_f0(f0_times: Sequence[float], f0: Sequence[float]) -> Callable[[float], float]:
    """F0 [Hz] as a function of time, bridged log-linearly across unvoiced
    points (0 or NaN) and held constant beyond the voiced extent."""
    times = np.asarray(f0_times, dtype=float)
    freqs = np.asarray(f0, dtype=float)
    if times.shape != freqs.shape or times.ndim != 1:
        raise ValueError("f0_times and f0 must be 1-D and of equal length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("f0_times must be strictly increasing")
    voiced = np.isfinite(freqs) & (freqs > 0)
    if not voiced.any():
        raise ValueError("the F0 track has no voiced points")
    voiced_times, log_f = times[voiced], np.log(freqs[voiced])
    return lambda time: float(np.exp(np.interp(time, voiced_times, log_f)))


@dataclass(frozen=True)
class _TVLayout:
    """A :class:`TVGaborFrame` in samples at one rate. Row ``q`` of
    ``offsets``/``windows`` is FFT buffer index ``j`` of time window ``q``: the
    signal sample ``centers[q] + offsets[q, j]`` and its window weight (0 in
    the zero padding). Index 0 is the window's middle sample, as in SciPy."""

    lengths: np.ndarray
    centers: np.ndarray
    n_fft: int
    offsets: np.ndarray
    windows: np.ndarray

    def positions(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Signal indices ``(Q, n_fft)`` and where they hold real signal samples."""
        signal_index = self.centers[:, None] + self.offsets
        return signal_index, (signal_index >= 0) & (signal_index < n) & (self.windows != 0)


@lru_cache(maxsize=64)
def _tv_layout(frame: TVGaborFrame, fs: float) -> _TVLayout:
    lengths = np.array([int(round(dur * fs)) for dur in frame.win_durs])
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
    window_index = (j[None, :] + lengths[:, None] // 2) % n_fft  # window sample at buffer index j
    inside = window_index < lengths[:, None]
    windows = np.zeros((len(lengths), n_fft))
    for q, length in enumerate(lengths):
        window_values = _window(frame.window, int(length))
        windows[q, inside[q]] = window_values[window_index[q, inside[q]]]
    offsets = np.where(inside, window_index - lengths[:, None] // 2, 0)
    return _TVLayout(lengths, centers, n_fft, offsets, windows)


_FLOOR_DB = -200.0


# -------------------------------------------------------------------- STFT
class STFT:
    """Short-time Fourier transform (wraps :class:`scipy.signal.ShortTimeFFT`).

    ``data`` has shape ``(n_channels, n_freqs, n_windows)``. Resynthesis with
    :meth:`to_sound` is exact for an unmodified STFT, and the least-squares
    signal for a modified one.

    ``STFT(sound, win_dur, hop_dur)`` is
    ``GaborFrame(win_dur, hop_dur).analyze(sound)``; the frame is kept as
    :attr:`frame` and SciPy's transform as :attr:`sft`.

    Parameters
    ----------
    win_dur
        Window length [s] (periodic Hann).
    hop_dur
        Hop [s]; defaults to a quarter window (75% overlap).
    frame
        A :class:`~sonore.frames.gabor.GaborFrame` to use instead (other windows,
        zero-padded FFTs); ``win_dur`` and ``hop_dur`` are then ignored.
    """

    __array_ufunc__ = None

    def __init__(
        self,
        sound: Sound,
        win_dur: float = 20e-3,
        hop_dur: float | None = None,
        *,
        frame: GaborFrame | None = None,
    ):
        self.frame = GaborFrame(win_dur, hop_dur) if frame is None else frame
        self.fs = sound.fs
        self.sft = _analysis_sft(self.frame, float(sound.fs))
        self.n_samples = len(sound)
        self.data = self.sft.stft(sound.data.T, axis=-1)

    @classmethod
    def _from(cls, template: STFT, data: np.ndarray) -> STFT:
        new = cls.__new__(cls)
        new.frame, new.fs, new.sft = template.frame, template.fs, template.sft
        new.n_samples, new.data = template.n_samples, data
        return new

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        win_ms, hop_ms = self.sft.m_num / self.fs * 1e3, self.sft.hop / self.fs * 1e3
        return (
            f"STFT({n_freqs} freqs x {n_windows} time windows, {n_channels} ch, "
            f"win {win_ms:.1f} ms, hop {hop_ms:.1f} ms)"
        )

    @property
    def f(self) -> np.ndarray:
        return self.sft.f

    @property
    def t(self) -> np.ndarray:
        """Window center times [s]."""
        return self.sft.t(self.n_samples)

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self.data, floor_db=_FLOOR_DB)

    def __mul__(self, other):
        from sonore.frames.mask import Mask  # mask.py imports STFT from here

        gains = other.values if isinstance(other, Mask) else other
        return STFT._from(self, self.data * gains)

    __rmul__ = __mul__

    def to_sound(self) -> Sound:
        """Inverse STFT (least-squares overlap-add); see
        :meth:`~sonore.frames.gabor.GaborFrame.synthesize`."""
        return self.frame.synthesize(self)

    def griffin_lim(self, n_iter: int = 100, momentum: float = 0.99, rng=None) -> Sound:
        """Reconstruct a signal from the magnitude only (fast Griffin-Lim,
        Perraudin et al., 2013). ``momentum=0`` gives classic Griffin-Lim."""
        target_mag = self.magnitude
        rng = as_rng(rng)
        coefs = target_mag * np.exp(2j * np.pi * rng.random(target_mag.shape))
        prev_projected = coefs
        short_time_fft = self.frame.sft(self.fs)  # the canonical dual; refuses a non-frame
        for _ in range(n_iter):
            signal = short_time_fft.istft(coefs, k1=self.n_samples)
            projected = short_time_fft.stft(np.real(signal), axis=-1)
            coefs = projected + momentum * (projected - prev_projected)
            prev_projected = projected
            coefs = target_mag * np.exp(1j * np.angle(coefs))
        return STFT._from(self, coefs).to_sound()

    def plot(self, ax=None, channel: int = 0, **kwargs):
        from sonore.plotting import plot_stft

        return plot_stft(self, ax=ax, channel=channel, **kwargs)


class TVSTFT:
    """Coefficients of a :class:`~sonore.frames.gabor.TVGaborFrame`: a short-time
    Fourier transform whose window changes over time.

    ``data`` has shape ``(n_channels, n_freqs, n_windows)`` like
    :class:`STFT`, on one frequency grid :attr:`f` (every window is
    zero-padded to the same FFT length) and at non-uniform window centers
    :attr:`t`. :meth:`to_sound` is exact for unmodified coefficients and the
    least-squares signal for modified ones. Multiply by an array to mask.
    """

    __array_ufunc__ = None

    def __init__(self, data: np.ndarray, fs: float, n_samples: int, frame: TVGaborFrame):
        self.data, self.fs, self.n_samples, self.frame = data, fs, int(n_samples), frame

    @classmethod
    def _from(cls, template: TVSTFT, data: np.ndarray) -> TVSTFT:
        return cls(data, template.fs, template.n_samples, template.frame)

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        win_ms = self.frame.layout(self.fs).lengths / self.fs * 1e3
        return (
            f"TVSTFT({n_freqs} freqs x {n_windows} time windows, {n_channels} ch, "
            f"win {win_ms.min():.1f}-{win_ms.max():.1f} ms, n_fft {self.frame.layout(self.fs).n_fft})"
        )

    @property
    def f(self) -> np.ndarray:
        return np.fft.rfftfreq(self.frame.layout(self.fs).n_fft, 1 / self.fs)

    @property
    def t(self) -> np.ndarray:
        """Window center times [s], rounded to samples."""
        return self.frame.layout(self.fs).centers / self.fs

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self.data, floor_db=_FLOOR_DB)

    def __mul__(self, other):
        return TVSTFT._from(self, self.data * other)

    __rmul__ = __mul__

    def to_sound(self) -> Sound:
        """Least-squares resynthesis; see
        :meth:`~sonore.frames.gabor.TVGaborFrame.synthesize`."""
        return self.frame.synthesize(self)

    def plot(self, ax=None, channel: int = 0, **kwargs):
        """Spectrogram in dB at the window centers (see :func:`~sonore.plotting.plot_tf_db`)."""
        from sonore.plotting import plot_tf_db

        kwargs.setdefault("title", "Time-varying spectrogram")
        return plot_tf_db(self.db[channel], self.t, self.f, ax=ax, **kwargs)
