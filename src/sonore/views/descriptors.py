"""Timbre descriptors after Peeters et al. (2011), "The Timbre Toolbox": the
log attack time, the spectral centroid and the spectral flux.

Written from the paper's equations, not from the Timbre Toolbox's code,
whose licence forbids redistribution. Where the defaults differ from the
paper's, the reason is measured in ``tools/check_timbre_claims.py`` and set
out in ``docs/design/music/timbre-page.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lfilter

from sonore.core.sound import Sound
from sonore.views.view import View

__all__ = ["DescriptorTrack", "attack_segment", "log_attack_time", "spectral_centroid", "spectral_flux"]


WIN_DUR = 0.0232  # Hamming window of the paper's STFT [s]
HOP_DUR = 0.0058  # and its hop [s]
EFFORT_FACTOR = 3.0  # the paper's alpha: an effort counts as weak below this many mean efforts


@dataclass(frozen=True)
class DescriptorTrack(View):
    """One descriptor's value in every time window, for each channel. A view:
    it keeps one number per time window.

    ``values`` has shape ``(n_channels, n_windows)`` and is NaN where a time
    window is silent. ``median`` and ``iqr`` are the summaries Peeters et al.
    (2011) use, since silent and near-silent time windows make a mean or a
    standard deviation meaningless; both skip NaN.
    """

    discards = "DescriptorTrack keeps one number per time window."

    t: np.ndarray
    values: np.ndarray
    name: str
    unit: str

    @property
    def median(self) -> np.ndarray:
        """Median over time, one per channel."""
        return np.nanmedian(self.values, axis=1)

    @property
    def iqr(self) -> np.ndarray:
        """Interquartile range over time, one per channel."""
        upper, lower = np.nanpercentile(self.values, [75, 25], axis=1)
        return upper - lower

    def __repr__(self) -> str:
        n_channels, n_windows = self.values.shape
        unit = f" {self.unit}" if self.unit else ""
        median = f"median {self.median[0]:.4g}{unit}"
        return f"DescriptorTrack({self.name}, {n_windows} time windows, {n_channels} ch, {median})"

    def plot(self, ax=None, channel: int = 0, **kwargs):
        """The descriptor against time (see :func:`~sonore.plotting.plot_descriptor_track`)."""
        from sonore.plotting import plot_descriptor_track

        return plot_descriptor_track(self, ax=ax, channel=channel, **kwargs)


def _energy_envelope(sound: Sound, channel: int, cutoff: float, zero_phase: bool) -> np.ndarray:
    """Peeters et al. (2011) II B 1: the amplitude of the analytic signal,
    low-passed by a third-order Butterworth filter at ``cutoff`` Hz, forward
    and backward when ``zero_phase``."""
    b, a = butter(3, cutoff / (sound.fs / 2))
    amplitude = np.abs(hilbert(sound.data[:, channel]))
    return filtfilt(b, a, amplitude) if zero_phase else lfilter(b, a, amplitude)


def attack_segment(
    sound: Sound, channel: int = 0, cutoff: float = 20.0, zero_phase: bool = True
) -> tuple[float, float]:
    """Start and end of the attack [s], by the weakest-effort method of
    Peeters et al. (2011), III A 2 a.

    The energy envelope (the amplitude of the analytic signal, low-passed by
    a third-order Butterworth filter at ``cutoff`` Hz) is cut at 0.1, 0.2,
    ..., 1 times its maximum, and the "efforts" are the times it takes to
    climb from one level to the next. The attack starts within the first
    effort shorter than three times the mean effort and ends within the last
    one, at the envelope's minimum and maximum inside those two efforts.

    The paper computes its descriptors with a 5 Hz filter applied once
    (``cutoff=5, zero_phase=False``), which measures a 5 ms attack as
    about 80 ms. The default here is the paper's onset setting (20 Hz,
    forward and backward), which measures it as about 16 ms; see
    ``tools/check_timbre_claims.py``.
    """
    envelope = _energy_envelope(sound, channel, cutoff, zero_phase)
    peak = int(np.argmax(envelope))
    if envelope[peak] <= 0:
        raise ValueError("the sound is silent")
    normalized = envelope[: peak + 1] / envelope[peak]
    levels = np.arange(1, 11) / 10
    crossings = np.array([np.argmax(normalized >= level) for level in levels])
    efforts = np.diff(crossings)
    weak = np.flatnonzero(efforts <= EFFORT_FACTOR * efforts.mean())
    first, last = weak[0], weak[-1]
    start = crossings[first] + np.argmin(envelope[crossings[first] : crossings[first + 1] + 1])
    end = crossings[last] + np.argmax(envelope[crossings[last] : crossings[last + 1] + 1])
    return start / sound.fs, end / sound.fs


def log_attack_time(sound: Sound, channel: int = 0, cutoff: float = 20.0, zero_phase: bool = True) -> float:
    """The base-10 logarithm of the attack's duration in seconds,
    ``log10(end - start)`` with the start and end of :func:`attack_segment`
    (Peeters et al., 2011, eq. 4). One sample is the shortest duration."""
    start, end = attack_segment(sound, channel, cutoff, zero_phase)
    return float(np.log10(max(end - start, 1 / sound.fs)))


def _spectra(sound: Sound, win_dur: float, hop_dur: float, scale: str) -> tuple[np.ndarray, ...]:
    """Times [s], frequencies [Hz] and spectra (channels, time windows, bins) of a
    Hamming STFT with no padding, each spectrum a magnitude or a power."""
    if scale not in ("magnitude", "power"):
        raise ValueError("scale must be 'magnitude' or 'power'")
    n_window, n_hop = round(win_dur * sound.fs), round(hop_dur * sound.fs)
    if sound.n_samples < n_window:
        raise ValueError(f"the sound is shorter than one time window ({win_dur * 1000:g} ms)")
    frames = np.lib.stride_tricks.sliding_window_view(sound.data.T, n_window, axis=1)[:, ::n_hop]
    spectra = np.abs(np.fft.rfft(frames * np.hamming(n_window), axis=-1))
    if scale == "power":
        spectra = spectra**2
    t = (np.arange(spectra.shape[1]) * n_hop + n_window / 2) / sound.fs
    return t, np.fft.rfftfreq(n_window, 1 / sound.fs), spectra


def spectral_centroid(
    sound: Sound, scale: str = "power", win_dur: float = WIN_DUR, hop_dur: float = HOP_DUR
) -> DescriptorTrack:
    """The centre of gravity of the spectrum in each time window,
    ``sum(f_k a_k) / sum(a_k)`` (Peeters et al., 2011, eq. 7), on a Hamming
    STFT of 23.2 ms windows every 5.8 ms by default.

    ``scale`` is ``"power"`` (the default) or ``"magnitude"``, the two
    spectra the paper offers; they differ a lot (2.5 times for a tone with
    harmonics falling as 1/n), so say which one you report. The power
    centroid of a steady harmonic tone matches the one computed from its
    partials. The magnitude centroid also counts the window's sidelobes in
    every bin up to Nyquist, which for a dull tone can outweigh the weak
    high partials: for harmonics falling as n^-3 at E-flat 4 it measures
    810 Hz at 44.1 kHz and 592 Hz at 22.05 kHz, against 414 Hz from the
    partials (``tools/check_timbre_claims.py``).
    """
    t, freqs, spectra = _spectra(sound, win_dur, hop_dur, scale)
    totals = spectra.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(totals > 0, (spectra * freqs).sum(-1) / totals, np.nan)
    return DescriptorTrack(t, values, "spectral centroid", "Hz")


def spectral_flux(
    sound: Sound,
    spacing: float | None = 0.1,
    scale: str = "magnitude",
    win_dur: float = WIN_DUR,
    hop_dur: float = HOP_DUR,
) -> DescriptorTrack:
    """How much the spectrum changes: 1 minus the normalized correlation of
    two spectra (Peeters et al., 2011, eq. 24, "spectral variation"), 0 for
    spectra of the same shape and up to 1.

    The paper correlates successive time windows, one hop (5.8 ms) apart:
    ``spacing=None``. At that spacing a steady tone and a tone whose spectral
    slope glides over a second measure almost alike; ``spacing`` correlates
    spectra that far apart instead (100 ms by default), at which the two
    differ 25 times (``tools/check_timbre_claims.py``). Each value is placed
    at the later of its two time windows.
    """
    t, _, spectra = _spectra(sound, win_dur, hop_dur, scale)
    lag = 1 if spacing is None else max(1, round(spacing / hop_dur))
    if spectra.shape[1] <= lag:
        raise ValueError("the sound is too short for this spacing")
    later, earlier = spectra[:, lag:], spectra[:, :-lag]
    norms = np.sqrt((later**2).sum(-1) * (earlier**2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(norms > 0, 1 - (later * earlier).sum(-1) / norms, np.nan)
    return DescriptorTrack(t[lag:], values, "spectral flux", "")
