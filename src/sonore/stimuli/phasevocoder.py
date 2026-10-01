"""Phase vocoder (Gordon & Strawn, 1985; Dolson, 1986).

The phase vocoder treats each STFT channel as a slowly varying sinusoid with an
amplitude and an *instantaneous frequency*, estimated from how much the
channel's phase advances between frames beyond what its center frequency
predicts. With amplitudes and frequencies in hand you can:

- change duration without changing pitch (:func:`time_stretch`), by
  re-accumulating phase at a different hop and overlap-adding;
- change pitch without changing duration (:func:`pitch_shift`), by stretching
  and then resampling;
- resynthesize through an oscillator bank with any frequency remapping
  (:meth:`PVAnalysis.resynthesize`), e.g. to shift or stretch partials.

Time stretching uses identity phase locking (Laroche & Dolson, 1999) by default:
bins around each spectral peak keep their original phase relationship to the
peak, which removes most of the classic "phasiness".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import ShortTimeFFT, resample_poly
from scipy.signal.windows import hann

from sonore.core.sound import Sound

__all__ = ["PVAnalysis", "pv_analyze", "time_stretch", "pitch_shift"]


def _wrap(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi


def _sft(n_win: int, hop: int, fs: float) -> ShortTimeFFT:
    return ShortTimeFFT(hann(n_win, sym=False), hop=hop, fs=fs, fft_mode="onesided")


def _inst_freq(phase: np.ndarray, n_win: int, hop: int) -> np.ndarray:
    """Instantaneous frequency [rad/sample] of each bin from successive phases."""
    omega = 2 * np.pi * np.arange(phase.shape[-2]) / n_win  # bin centers
    dphi = _wrap(np.diff(phase, axis=-1) - omega[:, None] * hop)
    inst = omega[:, None] + dphi / hop
    return np.concatenate([inst[..., :1], inst], axis=-1)  # frame 0 borrows frame 1


def _win_len(win_dur: float, fs: float) -> int:
    n = int(round(win_dur * fs))
    return n + (n % 2)  # even, so the window has a single center sample


@dataclass(frozen=True)
class PVAnalysis:
    """Phase-vocoder analysis: per-channel magnitude, phase and instantaneous
    frequency, all shaped ``(n_channels, n_bins, n_frames)``."""

    magnitude: np.ndarray
    phase: np.ndarray
    freq: np.ndarray  # Hz
    t: np.ndarray  # frame center times [s]
    fs: float
    n_win: int
    hop: int
    n_samples: int

    def resynthesize(
        self,
        time_scale: float = 1.0,
        freq_map: float | Callable[[np.ndarray], np.ndarray] | None = None,
        floor_db: float = -80.0,
        chunk: int = 64,
    ) -> Sound:
        """Oscillator-bank resynthesis.

        Each bin drives a sinusoid whose amplitude and frequency are
        interpolated between frames and whose phase is the running integral of
        frequency. ``time_scale`` stretches the time axis; ``freq_map`` is a
        ratio (e.g. ``1.5``) or a function mapping frequencies in Hz to new
        frequencies. For example ``lambda f: f + 70`` makes a 220 Hz harmonic
        complex inharmonic; note that shifting by a multiple of half the f0
        keeps it harmonic (``f + 110`` gives odd harmonics of 110 Hz).
        Bins that never come within ``floor_db`` of the loudest
        bin are skipped, and partials mapped above Nyquist are dropped.

        This is designed for tonal sounds, which it reconstructs closely
        (r > 0.999 for harmonic complexes). For noise, neighbouring channels
        drift out of phase and partially cancel (about 2 dB low, r ~ 0.9); use
        :func:`time_stretch` for noisy material.
        """
        if freq_map is None:
            fmap = None
        elif callable(freq_map):
            fmap = freq_map
        else:
            ratio = float(freq_map)
            fmap = lambda f: ratio * f  # noqa: E731

        n_out = int(round(self.n_samples * time_scale))
        ts = np.arange(n_out) / self.fs
        tf = self.t * time_scale
        # filterbank-summation gain: sum over bins of |X| equals N*w(center)*amplitude
        gain = np.full(self.magnitude.shape[1], 2.0)
        gain[0] = 1.0
        if self.n_win % 2 == 0:
            gain[-1] = 1.0
        gain /= self.n_win * hann(self.n_win, sym=False)[self.n_win // 2]
        # Anchor each oscillator's phase at the first frame whose window lies
        # fully inside the signal; edge frames are truncated, which scrambles
        # the phase relationship between neighbouring bins.
        half = self.n_win / 2 / self.fs
        interior = np.flatnonzero((self.t - half >= 0) & (self.t + half <= self.n_samples / self.fs))
        i0 = int(interior[0]) if len(interior) else int(np.argmin(np.abs(self.t)))
        n0 = min(int(round(tf[i0] * self.fs)), n_out - 1)

        out = np.zeros((n_out, self.magnitude.shape[0]))
        for c in range(self.magnitude.shape[0]):
            mag = self.magnitude[c]
            active = np.flatnonzero(mag.max(axis=1) > mag.max() * 10 ** (floor_db / 20))
            for start in range(0, len(active), chunk):
                bins = active[start : start + chunk]
                amp = np.array([np.interp(ts, tf, mag[k]) for k in bins]) * gain[bins, None]
                f = np.array([np.interp(ts, tf, self.freq[c, k]) for k in bins])
                if fmap is not None:
                    f = fmap(f)
                    amp = np.where((f > 0) & (f < self.fs / 2), amp, 0.0)
                cum = np.cumsum(f, axis=1) / self.fs
                theta = self.phase[c, bins, i0][:, None] + 2 * np.pi * (cum - cum[:, n0 : n0 + 1])
                out[:, c] += np.sum(amp * np.cos(theta), axis=0)
        return Sound(out, self.fs)


def pv_analyze(sound: Sound, win_dur: float = 46e-3, hop_dur: float | None = None) -> PVAnalysis:
    """Phase-vocoder analysis with a Hann window of ``win_dur`` seconds and a hop
    of ``hop_dur`` (default: a quarter window, the largest hop at which
    instantaneous frequency is unambiguous across a Hann main lobe)."""
    n_win = _win_len(win_dur, sound.fs)
    hop = max(1, int(round(hop_dur * sound.fs))) if hop_dur else n_win // 4
    sft = _sft(n_win, hop, sound.fs)
    X = sft.stft(sound.data.T, axis=-1)
    phase = np.angle(X)
    freq = _inst_freq(phase, n_win, hop) * sound.fs / (2 * np.pi)
    return PVAnalysis(np.abs(X), phase, freq, sft.t(len(sound)), sound.fs, n_win, hop, len(sound))


def _lock_phases(mag: np.ndarray, phi: np.ndarray, psi_peaks: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Identity phase locking: every bin inherits its nearest peak's synthesis
    phase plus its original offset from that peak."""
    bounds = (peaks[:-1] + peaks[1:]) / 2
    owner = np.searchsorted(bounds, np.arange(len(mag)), side="right")
    return psi_peaks[owner] + (phi - phi[peaks[owner]])


def time_stretch(sound: Sound, factor: float, win_dur: float = 46e-3, phase_lock: bool = True) -> Sound:
    """Change duration by ``factor`` (2 = twice as long) without changing pitch.

    The synthesis hop is fixed at a quarter window; the analysis hop is
    ``synthesis_hop / factor`` (rounded), so the achieved factor can differ
    slightly from the requested one for extreme values. The output length is
    ``round(len(sound) * factor)``.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    fs = sound.fs
    n_win = _win_len(win_dur, fs)
    hop_s = n_win // 4
    hop_a = max(1, int(round(hop_s / factor)))
    sft_a, sft_s = _sft(n_win, hop_a, fs), _sft(n_win, hop_s, fs)

    X = sft_a.stft(sound.data.T, axis=-1)  # (C, F, T)
    mag, phi = np.abs(X), np.angle(X)
    inst = _inst_freq(phi, n_win, hop_a)

    # Seed synthesis phases from the first frame whose window lies fully inside
    # the signal. Earlier (zero-padded) frames have distorted phases, and any
    # error in the seed persists as lost phase coherence between partials.
    starts = np.round(sft_a.t(len(sound)) * fs).astype(int) - sft_a.m_num_mid
    interior = np.flatnonzero((starts >= 0) & (starts + n_win <= len(sound)))
    m0 = int(interior[0]) if len(interior) else 0
    psi = np.empty_like(phi)
    psi[..., : m0 + 1] = phi[..., : m0 + 1]
    for m in range(m0 + 1, phi.shape[-1]):
        advanced = psi[..., m - 1] + inst[..., m] * hop_s
        if not phase_lock:
            psi[..., m] = advanced
            continue
        for c in range(phi.shape[0]):
            a = mag[c, :, m]
            is_peak = np.zeros(len(a), bool)
            is_peak[1:-1] = (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > 1e-6 * a.max())
            peaks = np.flatnonzero(is_peak)
            if len(peaks) == 0:
                psi[c, :, m] = advanced[c]
            else:
                psi[c, :, m] = _lock_phases(a, phi[c, :, m], advanced[c, peaks], peaks)
    Y = mag * np.exp(1j * psi)

    # place analysis frame m (time (p_min_a + m)*hop_a) at synthesis slot p_min_a + m
    n_out = int(round(len(sound) * factor))
    n_slots = sft_s.p_max(n_out) - sft_s.p_min
    offset = sft_a.p_min - sft_s.p_min
    Z = np.zeros(Y.shape[:-1] + (n_slots,), complex)
    src = np.arange(Y.shape[-1])
    dst = src + offset
    ok = (dst >= 0) & (dst < n_slots)
    Z[..., dst[ok]] = Y[..., src[ok]]
    y = sft_s.istft(Z, k1=n_out)
    return Sound(np.real(y).T, fs)


def pitch_shift(sound: Sound, semitones: float, win_dur: float = 46e-3, phase_lock: bool = True) -> Sound:
    """Shift pitch by ``semitones`` without changing duration (time-stretch,
    then resample). Formants shift along with the pitch."""
    ratio = Fraction(2 ** (semitones / 12)).limit_denominator(1000)
    stretched = time_stretch(sound, float(ratio), win_dur, phase_lock)
    y = resample_poly(stretched.data, ratio.denominator, ratio.numerator, axis=0)
    n = len(sound)
    y = y[:n] if len(y) >= n else np.pad(y, ((0, n - len(y)), (0, 0)))
    return Sound(y, sound.fs)
