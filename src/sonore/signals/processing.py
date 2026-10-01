"""Operations on sounds and lists of sounds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from sonore.core.sound import Sound
from sonore.core.utils import amp_to_db, time_axis

__all__ = [
    "match_fs",
    "match_channels",
    "pad",
    "truncate",
    "normalize",
    "concat",
    "mix",
    "relative_db",
    "butter_filter",
    "bandpass",
    "amplitude_modulate",
]


def match_fs(sounds: Sequence[Sound], fs: float | None = None, mode: str = "down") -> list[Sound]:
    """Resample so all sounds share a rate: ``fs`` if given, else the lowest
    (``mode="down"``) or highest (``mode="up"``) rate in the list."""
    if fs is None:
        rates = [s.fs for s in sounds]
        fs = {"down": min, "up": max}[mode](rates)
    return [s.resample(fs) for s in sounds]


def match_channels(sounds: Sequence[Sound]) -> list[Sound]:
    """Upmix mono sounds to the channel count of the others."""
    n = max(s.n_channels for s in sounds)
    return [s.to_channels(n) for s in sounds]


def pad(sounds: Sequence[Sound], align: str = "start") -> list[Sound]:
    """Zero-pad to the longest length. ``align``: ``start``, ``center``, ``end``."""
    n = max(len(s) for s in sounds)
    return [s.pad_to(n, align) for s in sounds]


def truncate(sounds: Sequence[Sound]) -> list[Sound]:
    """Cut all sounds to the shortest length."""
    n = min(len(s) for s in sounds)
    return [Sound(s.data[:n], s.fs) for s in sounds]


def normalize(sounds: Sequence[Sound], rms: float = 1.0) -> list[Sound]:
    """Set each sound's RMS independently."""
    return [s.normalize(rms) for s in sounds]


def _check_fs(sounds: Sequence[Sound]) -> float:
    rates = {s.fs for s in sounds}
    if len(rates) > 1:
        raise ValueError(f"sampling rates differ: {sorted(rates)}; use match_fs()")
    return rates.pop()


def concat(sounds: Sequence[Sound]) -> Sound:
    """Join end to end (mono sounds are upmixed to match)."""
    fs = _check_fs(sounds)
    return Sound(np.concatenate([s.data for s in match_channels(sounds)]), fs)


def mix(sounds: Sequence[Sound], align: str = "start") -> Sound:
    """Sum sounds of possibly different lengths (zero-padded per ``align``)."""
    _check_fs(sounds)
    return sum(pad(match_channels(sounds), align), start=0)


def relative_db(sounds: Sequence[Sound], ref: int = 0) -> list[float]:
    """Level of each sound in dB relative to ``sounds[ref]``."""
    r = sounds[ref].rms
    return [float(amp_to_db(s.rms / r)) for s in sounds]


def butter_filter(
    sound: Sound,
    cutoff: float | tuple[float, float],
    btype: str = "bandpass",
    order: int = 4,
    zero_phase: bool = True,
) -> Sound:
    """Butterworth filter along time (each channel separately).

    ``btype`` is ``lowpass``, ``highpass``, ``bandpass``, or ``bandstop``.
    ``zero_phase=True`` runs it forward and backward (doubling the order in dB).
    """
    sos = butter(order, cutoff, btype=btype, fs=sound.fs, output="sos")
    f = sosfiltfilt if zero_phase else sosfilt
    return Sound(f(sos, sound.data, axis=0), sound.fs)


def bandpass(sound: Sound, f_lo: float, f_hi: float, order: int = 4, zero_phase: bool = True) -> Sound:
    return butter_filter(sound, (f_lo, f_hi), "bandpass", order, zero_phase)


def amplitude_modulate(sound: Sound, f_mod: float, depth: float = 1.0, phase: float = 0.0) -> Sound:
    """Multiply by ``1 + depth*sin(2*pi*f_mod*t + phase)``."""
    t = time_axis(len(sound), sound.fs)
    return sound * (1 + depth * np.sin(2 * np.pi * f_mod * t + phase))
