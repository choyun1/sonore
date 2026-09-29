"""ERB-spaced cosine filterbank and subband representation.

The filters follow McDermott & Simoncelli (2011): half-cycle cosines on the
ERB-number scale, each spanning two band spacings, plus a lowpass and highpass
at the edges. Their squared responses sum to exactly 1, so filtering on
analysis *and* synthesis reconstructs the input perfectly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from audstim.sound import Sound
from audstim.utils import as_rng, erb_to_freq, freq_to_erb

__all__ = ["ERBFilterbank", "Subbands", "subbands", "noise_vocode"]


@dataclass(frozen=True)
class ERBFilterbank:
    """``n_bands`` bandpass filters between ``f_lo`` and ``f_hi`` [Hz], plus a
    lowpass below ``f_lo`` and a highpass above ``f_hi`` (``n_bands + 2`` total)."""

    n_bands: int = 30
    f_lo: float = 50.0
    f_hi: float = 8000.0

    @property
    def _knots(self) -> np.ndarray:
        return np.linspace(freq_to_erb(self.f_lo), freq_to_erb(self.f_hi), self.n_bands + 2)

    @property
    def cfs(self) -> np.ndarray:
        """Center frequencies [Hz] of all filters (edges at ``f_lo``/``f_hi``)."""
        return erb_to_freq(self._knots)

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """Magnitude responses, shape ``(len(freqs), n_bands + 2)``."""
        e = freq_to_erb(freqs)[:, None]
        k = self._knots
        step = k[1] - k[0]
        u = (e - k[None, :]) / step  # distance from each center in band spacings
        H = np.where(np.abs(u) < 1, np.cos(np.pi / 2 * u), 0.0)
        H[:, 0] = np.where(e[:, 0] <= k[0], 1.0, H[:, 0])
        H[:, -1] = np.where(e[:, 0] >= k[-1], 1.0, H[:, -1])
        return H

    def rfft_response(self, n: int, fs: float) -> np.ndarray:
        return self.response(np.fft.rfftfreq(n, 1 / fs))

    def analyze(self, sound: Sound) -> Subbands:
        """Split ``sound`` into subbands (zero-phase, via FFT)."""
        n = len(sound)
        H = self.rfft_response(n, sound.fs)  # (F, B)
        X = np.fft.rfft(sound.data, axis=0)  # (F, C)
        bands = np.fft.irfft(X[:, None, :] * H[:, :, None], n=n, axis=0)  # (n, B, C)
        return Subbands(bands, sound.fs, self)


def subbands(sound: Sound, n_bands: int = 30, f_lo: float = 50.0, f_hi: float | None = None) -> Subbands:
    """Convenience: build an :class:`ERBFilterbank` and analyze ``sound``.
    ``f_hi`` defaults to just under Nyquist."""
    f_hi = 0.95 * sound.fs / 2 if f_hi is None else min(f_hi, sound.fs / 2)
    return ERBFilterbank(n_bands, f_lo, f_hi).analyze(sound)


class Subbands:
    """Subband signals, shape ``(n_samples, n_filters, n_channels)``.

    Supports ``*`` and ``/`` with other Subbands or arrays, so e.g. a vocoder
    is ``(speech.envelopes() * carrier.tfs()).synthesize()``.
    """

    __array_ufunc__ = None

    def __init__(self, data: np.ndarray, fs: float, filterbank: ERBFilterbank):
        self.data = data
        self.fs = fs
        self.filterbank = filterbank

    def __len__(self) -> int:
        return self.data.shape[1]

    def __getitem__(self, i: int) -> Sound:
        return Sound(self.data[:, i, :], self.fs)

    def __repr__(self) -> str:
        n, b, c = self.data.shape
        return f"Subbands({b} filters, {n / self.fs:.3f} s, {self.fs:g} Hz, {c} ch)"

    @property
    def cfs(self) -> np.ndarray:
        return self.filterbank.cfs

    def _new(self, data) -> Subbands:
        return Subbands(data, self.fs, self.filterbank)

    def _op(self, other, op):
        o = other.data if isinstance(other, Subbands) else other
        return self._new(op(self.data, o))

    def __mul__(self, other):
        return self._op(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._op(other, np.divide)

    def analytic(self) -> np.ndarray:
        return hilbert(self.data, axis=0)

    def envelopes(self, lowpass: float | None = None) -> Subbands:
        """Hilbert envelopes, optionally smoothed by a zero-phase lowpass [Hz]."""
        env = np.abs(self.analytic())
        if lowpass is not None:
            sos = butter(4, lowpass, fs=self.fs, output="sos")
            env = np.maximum(sosfiltfilt(sos, env, axis=0), 0)
        return self._new(env)

    def tfs(self) -> Subbands:
        """Temporal fine structure: ``cos`` of the instantaneous phase."""
        a = self.analytic()
        return self._new(np.cos(np.angle(a)))

    def synthesize(self) -> Sound:
        """Re-filter each band and sum. Exact inverse of :meth:`ERBFilterbank.analyze`."""
        n = self.data.shape[0]
        H = self.filterbank.rfft_response(n, self.fs)
        X = np.fft.rfft(self.data, axis=0) * H[:, :, None]
        return Sound(np.fft.irfft(X.sum(axis=1), n=n, axis=0), self.fs)

    def plot(self, axes=None, channel: int = 0, **kwargs):
        from audstim.plotting import plot_subbands

        return plot_subbands(self, axes=axes, channel=channel, **kwargs)


def noise_vocode(
    sound: Sound,
    n_bands: int = 16,
    f_lo: float = 80.0,
    f_hi: float = 8000.0,
    carrier: Sound | str = "noise",
    env_lowpass: float | None = 50.0,
    rng=None,
) -> Sound:
    """Channel vocoder (Shannon et al., 1995).

    Band envelopes of ``sound`` (lowpassed at ``env_lowpass`` Hz) modulate the
    fine structure of the ``carrier`` (``"noise"``, ``"tone"`` for sine carriers
    at the band centers, or any Sound of the same length).
    """
    fb = ERBFilterbank(n_bands, f_lo, min(f_hi, 0.95 * sound.fs / 2))
    sb = fb.analyze(sound)
    if isinstance(carrier, str):
        from audstim.generators import gaussian_noise

        if carrier == "noise":
            carrier = gaussian_noise(sound.duration, sound.fs, n_channels=sound.n_channels, rng=as_rng(rng))
            carrier_tfs = fb.analyze(carrier).tfs()
        elif carrier == "tone":
            t = sound.t[:, None, None]
            carrier_tfs = sb._new(np.cos(2 * np.pi * fb.cfs[None, :, None] * t) * np.ones_like(sb.data))
        else:
            raise ValueError("carrier must be 'noise', 'tone', or a Sound")
    else:
        if len(carrier) < len(sound):
            raise ValueError("carrier is shorter than the sound")
        carrier_tfs = fb.analyze(Sound(carrier.data[: len(sound)], carrier.fs)).tfs()
    env = sb.envelopes(lowpass=env_lowpass)
    env.data[:, [0, -1], :] = 0  # drop edge filters (outside f_lo..f_hi)
    out = (env * carrier_tfs).synthesize()
    return out.normalize(sound.rms)
