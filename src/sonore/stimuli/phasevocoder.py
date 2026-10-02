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
    phase_deviation = _wrap(np.diff(phase, axis=-1) - omega[:, None] * hop)
    inst_freq = omega[:, None] + phase_deviation / hop
    return np.concatenate([inst_freq[..., :1], inst_freq], axis=-1)  # frame 0 borrows frame 1


def _win_len(win_dur: float, fs: float) -> int:
    n_win = int(round(win_dur * fs))
    return n_win + (n_win % 2)  # even, so the window has a single center sample


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
            map_freqs = None
        elif callable(freq_map):
            map_freqs = freq_map
        else:
            ratio = float(freq_map)
            map_freqs = lambda freqs: ratio * freqs  # noqa: E731

        n_out = int(round(self.n_samples * time_scale))
        sample_times = np.arange(n_out) / self.fs
        frame_times = self.t * time_scale
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
        anchor_frame = int(interior[0]) if len(interior) else int(np.argmin(np.abs(self.t)))
        anchor_sample = min(int(round(frame_times[anchor_frame] * self.fs)), n_out - 1)

        out = np.zeros((n_out, self.magnitude.shape[0]))
        for channel in range(self.magnitude.shape[0]):
            magnitude = self.magnitude[channel]
            active = np.flatnonzero(magnitude.max(axis=1) > magnitude.max() * 10 ** (floor_db / 20))
            for start in range(0, len(active), chunk):
                bins = active[start : start + chunk]
                amplitude = (
                    np.array([np.interp(sample_times, frame_times, magnitude[k]) for k in bins])
                    * gain[bins, None]
                )
                freqs = np.array([np.interp(sample_times, frame_times, self.freq[channel, k]) for k in bins])
                if map_freqs is not None:
                    freqs = map_freqs(freqs)
                    amplitude = np.where((freqs > 0) & (freqs < self.fs / 2), amplitude, 0.0)
                cycles = np.cumsum(freqs, axis=1) / self.fs
                osc_phase = self.phase[channel, bins, anchor_frame][:, None] + 2 * np.pi * (
                    cycles - cycles[:, anchor_sample : anchor_sample + 1]
                )
                out[:, channel] += np.sum(amplitude * np.cos(osc_phase), axis=0)
        return Sound(out, self.fs)


def pv_analyze(sound: Sound, win_dur: float = 46e-3, hop_dur: float | None = None) -> PVAnalysis:
    """Phase-vocoder analysis with a Hann window of ``win_dur`` seconds and a hop
    of ``hop_dur`` (default: a quarter window, the largest hop at which
    instantaneous frequency is unambiguous across a Hann main lobe)."""
    n_win = _win_len(win_dur, sound.fs)
    hop = max(1, int(round(hop_dur * sound.fs))) if hop_dur else n_win // 4
    sft = _sft(n_win, hop, sound.fs)
    spectrum = sft.stft(sound.data.T, axis=-1)
    phase = np.angle(spectrum)
    freq = _inst_freq(phase, n_win, hop) * sound.fs / (2 * np.pi)
    return PVAnalysis(np.abs(spectrum), phase, freq, sft.t(len(sound)), sound.fs, n_win, hop, len(sound))


def _lock_phases(
    magnitude: np.ndarray, analysis_phase: np.ndarray, peak_synthesis_phase: np.ndarray, peaks: np.ndarray
) -> np.ndarray:
    """Identity phase locking: every bin inherits its nearest peak's synthesis
    phase plus its original offset from that peak."""
    bounds = (peaks[:-1] + peaks[1:]) / 2
    owner = np.searchsorted(bounds, np.arange(len(magnitude)), side="right")
    return peak_synthesis_phase[owner] + (analysis_phase - analysis_phase[peaks[owner]])


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
    synthesis_hop = n_win // 4
    analysis_hop = max(1, int(round(synthesis_hop / factor)))
    analysis_sft, synthesis_sft = _sft(n_win, analysis_hop, fs), _sft(n_win, synthesis_hop, fs)

    spectrum = analysis_sft.stft(sound.data.T, axis=-1)  # (C, F, T)
    magnitude, analysis_phase = np.abs(spectrum), np.angle(spectrum)
    inst_freq = _inst_freq(analysis_phase, n_win, analysis_hop)

    # Seed synthesis phases from the first frame whose window lies fully inside
    # the signal. Earlier (zero-padded) frames have distorted phases, and any
    # error in the seed persists as lost phase coherence between partials.
    starts = np.round(analysis_sft.t(len(sound)) * fs).astype(int) - analysis_sft.m_num_mid
    interior = np.flatnonzero((starts >= 0) & (starts + n_win <= len(sound)))
    first_frame = int(interior[0]) if len(interior) else 0
    synthesis_phase = np.empty_like(analysis_phase)
    synthesis_phase[..., : first_frame + 1] = analysis_phase[..., : first_frame + 1]
    for frame in range(first_frame + 1, analysis_phase.shape[-1]):
        advanced = synthesis_phase[..., frame - 1] + inst_freq[..., frame] * synthesis_hop
        if not phase_lock:
            synthesis_phase[..., frame] = advanced
            continue
        for channel in range(analysis_phase.shape[0]):
            frame_magnitude = magnitude[channel, :, frame]
            is_peak = np.zeros(len(frame_magnitude), bool)
            is_peak[1:-1] = (
                (frame_magnitude[1:-1] > frame_magnitude[:-2])
                & (frame_magnitude[1:-1] >= frame_magnitude[2:])
                & (frame_magnitude[1:-1] > 1e-6 * frame_magnitude.max())
            )
            peaks = np.flatnonzero(is_peak)
            if len(peaks) == 0:
                synthesis_phase[channel, :, frame] = advanced[channel]
            else:
                synthesis_phase[channel, :, frame] = _lock_phases(
                    frame_magnitude, analysis_phase[channel, :, frame], advanced[channel, peaks], peaks
                )
    stretched = magnitude * np.exp(1j * synthesis_phase)

    # place analysis frame m (time (analysis_sft.p_min + m)*analysis_hop) at synthesis
    # slot analysis_sft.p_min + m
    n_out = int(round(len(sound) * factor))
    n_slots = synthesis_sft.p_max(n_out) - synthesis_sft.p_min
    offset = analysis_sft.p_min - synthesis_sft.p_min
    slotted = np.zeros(stretched.shape[:-1] + (n_slots,), complex)
    source_frames = np.arange(stretched.shape[-1])
    target_slots = source_frames + offset
    in_range = (target_slots >= 0) & (target_slots < n_slots)
    slotted[..., target_slots[in_range]] = stretched[..., source_frames[in_range]]
    signal = synthesis_sft.istft(slotted, k1=n_out)
    return Sound(np.real(signal).T, fs)


def pitch_shift(sound: Sound, semitones: float, win_dur: float = 46e-3, phase_lock: bool = True) -> Sound:
    """Shift pitch by ``semitones`` without changing duration (time-stretch,
    then resample). Formants shift along with the pitch."""
    ratio = Fraction(2 ** (semitones / 12)).limit_denominator(1000)
    stretched = time_stretch(sound, float(ratio), win_dur, phase_lock)
    resampled = resample_poly(stretched.data, ratio.denominator, ratio.numerator, axis=0)
    length = len(sound)
    resampled = (
        resampled[:length]
        if len(resampled) >= length
        else np.pad(resampled, ((0, length - len(resampled)), (0, 0)))
    )
    return Sound(resampled, sound.fs)
