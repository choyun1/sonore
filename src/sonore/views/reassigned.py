"""The reassigned spectrogram: STFT power moved to where each coefficient's
energy is centred in time and frequency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import ShortTimeFFT

from sonore.core.sound import Sound
from sonore.core.utils import db_to_power
from sonore.frames.gabor import GaborFrame
from sonore.views.spectrum import TFPower
from sonore.views.view import View

__all__ = ["ReassignedSpectrogram", "reassigned_spectrogram"]


# ---------------------------------------------------------------- reassignment
@dataclass(frozen=True)
class ReassignedSpectrogram(View):
    """Spectrogram cells moved to their reassigned times and frequencies.

    ``t_hat``, ``f_hat`` and ``power`` have shape ``(n_channels, n_freqs,
    n_windows)``, one entry per STFT cell; ``keep`` marks the cells within
    the threshold of the maximum. :meth:`binned` sums the kept power onto a
    grid for display. There is no synthesis: reassignment is not linear.
    """

    discards = (
        "ReassignedSpectrogram discards the phase and moves each cell's power to a new time and frequency, "
        "many cells to one point, so different sounds give the same picture."
    )
    back_to_sound = ""
    no_plot = (
        "ReassignedSpectrogram has no plot: its points lie off any grid until one is chosen, so plot "
        "binned(t_edges, f_edges)."
    )

    t_hat: np.ndarray
    f_hat: np.ndarray
    power: np.ndarray
    keep: np.ndarray

    def binned(self, t_edges: np.ndarray, f_edges: np.ndarray) -> TFPower:
        """Kept power summed into the cells of ``t_edges`` x ``f_edges``
        [s, Hz]; points outside the edges are dropped."""
        t_edges, f_edges = np.asarray(t_edges, float), np.asarray(f_edges, float)
        binned_power = np.stack(
            [
                np.histogram2d(freqs[kept], times[kept], bins=(f_edges, t_edges), weights=powers[kept])[0]
                for times, freqs, powers, kept in zip(
                    self.t_hat, self.f_hat, self.power, self.keep, strict=True
                )
            ]
        )
        return TFPower(binned_power, (t_edges[:-1] + t_edges[1:]) / 2, (f_edges[:-1] + f_edges[1:]) / 2)


def reassigned_spectrogram(
    sound: Sound, frame: GaborFrame, threshold_db: float = -60.0
) -> ReassignedSpectrogram:
    """The reassigned spectrogram (Kodera et al., 1978; Auger & Flandrin, 1995).

    Each cell of ``frame``'s spectrogram is moved from its window time ``t``
    and bin frequency ``f`` to

    - ``t_hat = t + Re(X_tw conj X) / |X|**2``,
    - ``f_hat = f - Im(X_dw conj X) / (2 pi |X|**2)``,

    where ``X`` uses the window ``w``, ``X_tw`` the time-weighted window
    ``tau w(tau)`` and ``X_dw`` its derivative ``w'(tau)``, with ``tau`` in
    seconds from the window's middle sample, which is where SciPy (and so
    ``GaborFrame``) references each time window's phase. A tone off the bin grid,
    an impulse and a linear chirp land on their true frequency, time and
    instantaneous-frequency line.

    The derivative is taken from the window's formula, so only ``"hann"`` and
    ``("gaussian", std)`` windows are accepted. Cells more than
    ``threshold_db`` below the maximum (per channel) are marked not kept:
    their positions are mostly noise.
    """
    n_win, hop, n_fft = frame.lengths(sound.fs)
    window = frame.window_samples(sound.fs)
    tau, window_deriv = _window_tau_and_derivative(frame.window, n_win, sound.fs)
    if not np.allclose(window, _window_from_formula(frame.window, n_win)):
        raise ValueError(f"window {frame.window!r} does not match its formula")

    def stft(window_values):
        short_time_fft = ShortTimeFFT(
            window_values, hop=hop, fs=sound.fs, mfft=n_fft, fft_mode="onesided", dual_win=window
        )
        return short_time_fft, short_time_fft.stft(sound.data.T)

    short_time_fft, X = stft(window)
    X_tw = stft(tau * window)[1]
    X_dw = stft(window_deriv)[1]
    power = np.abs(X) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hat = short_time_fft.t(len(sound))[None, None, :] + np.real(X_tw * X.conj()) / power
        f_hat = short_time_fft.f[None, :, None] - np.imag(X_dw * X.conj()) / power / (2 * np.pi)
    peak = power.max(axis=(1, 2), keepdims=True)
    keep = (power > peak * db_to_power(threshold_db)) & np.isfinite(t_hat) & np.isfinite(f_hat)
    return ReassignedSpectrogram(t_hat, f_hat, power, keep)


def _window_from_formula(spec, n: int) -> np.ndarray:
    k = np.arange(n)
    if spec == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "gaussian":
        return np.exp(-0.5 * ((k - n / 2) / spec[1]) ** 2)
    raise ValueError(f"reassignment needs a 'hann' or ('gaussian', std) window, not {spec!r}")


def _window_tau_and_derivative(spec, n: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Time from the middle sample [s] and the window's derivative [1/s],
    both at the periodic window's samples."""
    k = np.arange(n)
    tau = (k - n // 2) / fs
    window = _window_from_formula(spec, n)
    if spec == "hann":
        derivative = 0.5 * (2 * np.pi / n) * np.sin(2 * np.pi * k / n) * fs
    else:
        derivative = -(k - n / 2) / spec[1] ** 2 * window * fs
    return tau, derivative
