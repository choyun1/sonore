"""The real cepstrum of a short-time Fourier transform: liftering,
resynthesis with the original or minimum phase, and cepstral F0."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sonore.analysis.representations import STFT, TVSTFT
from sonore.core.sound import Sound

__all__ = ["Cepstrum"]


class Cepstrum:
    """The real cepstrum of each frame of an :class:`~sonore.analysis.representations.STFT`
    or :class:`~sonore.analysis.representations.TVSTFT`.

    For a frame with spectrum ``X[k]`` on ``n_fft`` bins, the cepstrum is
    ``c[n] = IDFT(ln |X[k]|)`` at quefrency ``n / fs`` seconds. It is real and
    even in ``n`` because the sound is real, so only ``n = 0 .. n_fft // 2``
    is stored: ``data`` has shape ``(n_channels, n_fft // 2 + 1, n_frames)``
    on quefrencies :attr:`q` [s] and frame times :attr:`t` [s].

    The natural log is used, so ``exp`` undoes it exactly: without liftering,
    :meth:`to_sound` gives back the analyzed sound. Scaling the sound changes
    only ``c[0]``. Before the log, magnitudes are floored at ``floor_db``
    below the channel's largest magnitude over all frames, so a frame of
    digital silence has a flat log spectrum rather than ``-inf``; real
    recordings stay far above the default. Only the real cepstrum is
    provided: the complex cepstrum needs phase unwrapping, and the minimum
    phase comes from the real one.

    Parameters
    ----------
    coefs
        The coefficients to take the cepstrum of. They are kept (as
        :attr:`source`) for their phase, frame and times.
    floor_db
        The floor, in dB below each channel's maximum.
    """

    def __init__(self, coefs: STFT | TVSTFT, floor_db: float = -200.0):
        if not isinstance(coefs, (STFT, TVSTFT)):
            raise TypeError(f"expected an STFT or TVSTFT, not {type(coefs).__name__}")
        self.source = coefs
        self.fs = coefs.fs
        self.n_fft = _n_fft(coefs)
        mag = np.abs(coefs.data)
        peak = mag.max(axis=(1, 2), keepdims=True)
        floor = np.where(peak > 0, peak * 10 ** (floor_db / 20), np.finfo(float).tiny)
        log_mag = np.log(np.maximum(mag, floor))
        self.data = np.fft.irfft(log_mag, n=self.n_fft, axis=1)[:, : self.n_fft // 2 + 1]

    @classmethod
    def _from(cls, template: Cepstrum, data: np.ndarray) -> Cepstrum:
        new = cls.__new__(cls)
        new.source, new.fs, new.n_fft, new.data = template.source, template.fs, template.n_fft, data
        return new

    def __repr__(self) -> str:
        n_channels, n_quefrencies, n_frames = self.data.shape
        return (
            f"Cepstrum({n_quefrencies} quefrencies x {n_frames} frames, {n_channels} ch, "
            f"up to {self.q[-1] * 1e3:.1f} ms)"
        )

    @property
    def q(self) -> np.ndarray:
        """Quefrencies [s]."""
        return np.arange(self.data.shape[1]) / self.fs

    @property
    def t(self) -> np.ndarray:
        """Frame center times [s], those of :attr:`source`."""
        return self.source.t

    def _full(self) -> np.ndarray:
        """All ``n_fft`` quefrencies, mirrored from the stored half."""
        n_half = self.data.shape[1]
        mirror = self.data[:, 1 : self.n_fft - n_half + 1][:, ::-1]
        return np.concatenate([self.data, mirror], axis=1)

    def lifter(self, cutoff: float | Sequence[float], keep: str = "low") -> Cepstrum:
        """A rectangular lifter. ``keep="low"`` keeps the quefrencies below
        ``cutoff`` [s], the smooth spectral envelope; ``"high"`` keeps the
        rest, the fine structure such as the harmonics. ``cutoff`` is one
        value or one per frame, so it can follow an F0 track (half a period
        separates the envelope from the harmonics)."""
        cutoffs = np.asarray(cutoff, dtype=float)
        n_frames = self.data.shape[2]
        if cutoffs.ndim > 1 or (cutoffs.ndim == 1 and len(cutoffs) != n_frames):
            raise ValueError(f"cutoff must be a scalar or have one value per frame ({n_frames})")
        below = self.q[:, None] < np.broadcast_to(cutoffs, (n_frames,))[None, :]
        if keep == "low":
            mask = below
        elif keep == "high":
            mask = ~below
        else:
            raise ValueError(f"keep must be 'low' or 'high', not {keep!r}")
        return Cepstrum._from(self, self.data * mask)

    def envelope(self) -> np.ndarray:
        """``exp(DFT(c))``: the magnitude spectrum this cepstrum stands for,
        shape ``(n_channels, n_freqs, n_frames)`` on the source's frequencies.
        After a low lifter it is the cepstral spectral envelope, which follows
        the shape of the true envelope but sits a few dB below the harmonic
        peaks, because the lifter averages the peaks with the dips between them."""
        return np.exp(np.fft.rfft(self._full(), axis=1).real)

    def to_stft(self, phase: str = "original") -> STFT | TVSTFT:
        """Coefficients of the source's type and frame with :meth:`envelope`
        as magnitude.

        ``phase="original"`` takes the phase from :attr:`source`, so an
        unliftered cepstrum returns the source's coefficients. ``"minimum"``
        uses the minimum phase for that magnitude, from the folded cepstrum
        (``c[0]`` kept, ``2 c[n]`` up to ``n_fft / 2``, zero beyond). Each
        frame's response then starts at the frame's phase reference, the
        middle of its window. The fold is exact up to the time aliasing of
        the cepstrum on ``n_fft`` bins, which is negligible once ``n_fft``
        is several times the response's length.
        """
        if phase == "original":
            data = self.envelope() * np.exp(1j * np.angle(self.source.data))
        elif phase == "minimum":
            n_mid = (self.n_fft + 1) // 2
            folded = np.zeros((self.data.shape[0], self.n_fft, self.data.shape[2]))
            folded[:, 0] = self.data[:, 0]
            folded[:, 1:n_mid] = 2 * self.data[:, 1:n_mid]
            if self.n_fft % 2 == 0:
                folded[:, n_mid] = self.data[:, n_mid]
            data = np.exp(np.fft.rfft(folded, axis=1))
        else:
            raise ValueError(f"phase must be 'original' or 'minimum', not {phase!r}")
        return type(self.source)._from(self.source, data)

    def to_sound(self, phase: str = "original") -> Sound:
        """:meth:`to_stft` synthesized by the source's frame: exact for an
        unliftered cepstrum with the original phase, and otherwise the
        least-squares signal for those coefficients."""
        return self.to_stft(phase).to_sound()

    def f0(
        self, f_lo: float = 75.0, f_hi: float = 400.0, threshold: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classic cepstral F0 (Noll, 1967): the largest cepstral peak between
        quefrencies ``1/f_hi`` and ``1/f_lo``, refined by a parabola through it
        and its neighbors.

        Returns ``(t, f0, peak)``: frame times [s], and F0 [Hz] and the peak's
        height for each channel and frame, shape ``(n_channels, n_frames)``.
        F0 is 0 where the peak is below ``threshold``, a crude voicing rule.

        A periodic sound puts ripples in the log spectrum, one per harmonic,
        and they are resolved only if the window holds about three periods: at
        two periods, many frames come out an octave off. So every window must
        be at least ``3 / f_lo`` long, or this raises. On a male spoken
        sentence (CMU ARCTIC ``bdl``, 40 ms Hann frames), the result agrees
        with WORLD's Harvest within 5% on about 79% of the frames Harvest
        calls voiced, and on 95% of those whose peak also exceeds 0.1. It is a
        baseline, not an F0 tracker: each frame is judged alone.
        """
        if not 0 < f_lo < f_hi < self.fs / 2:
            raise ValueError(f"need 0 < f_lo < f_hi < fs/2, got f_lo={f_lo:g}, f_hi={f_hi:g}")
        shortest = _shortest_window(self.source) / self.fs
        if shortest < 3 / f_lo:
            raise ValueError(
                f"the shortest window is {shortest * 1e3:.1f} ms; cepstral F0 down to f_lo={f_lo:g} Hz "
                f"needs windows of at least three periods, {3 / f_lo * 1e3:.1f} ms"
            )
        q_lo, q_hi = max(int(np.floor(self.fs / f_hi)), 1), int(np.ceil(self.fs / f_lo))
        q_hi = min(q_hi, self.data.shape[1] - 2)
        peak_index = q_lo + np.argmax(self.data[:, q_lo : q_hi + 1], axis=1)[:, None, :]
        left, peak, right = (
            np.take_along_axis(self.data, peak_index + offset, axis=1)[:, 0] for offset in (-1, 0, 1)
        )
        curvature = left - 2 * peak + right
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(curvature != 0, 0.5 * (left - right) / curvature, 0.0)
        f0 = self.fs / (peak_index[:, 0] + shift)
        return self.t, np.where(peak >= threshold, f0, 0.0), peak

    def plot(self, ax=None, channel: int = 0, **kwargs):
        """Cepstrum against time and quefrency in ms (see
        :func:`~sonore.plotting.plot_cepstrum`)."""
        from sonore.plotting import plot_cepstrum

        return plot_cepstrum(self, ax=ax, channel=channel, **kwargs)


def _n_fft(coefs: STFT | TVSTFT) -> int:
    if isinstance(coefs, STFT):
        return int(coefs.sft.mfft)
    return int(coefs.frame.layout(coefs.fs).n_fft)


def _shortest_window(coefs: STFT | TVSTFT) -> int:
    """Shortest window [samples]."""
    if isinstance(coefs, STFT):
        return int(coefs.sft.m_num)
    return int(coefs.frame.layout(coefs.fs).lengths.min())
