"""Operations on sounds and lists of sounds."""

from __future__ import annotations

import functools
import numbers
import operator
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt

from sonore.core.sound import Sound
from sonore.core.utils import Align, amp_to_db, time_axis

__all__ = [
    "match_fs",
    "match_channels",
    "match_lengths",
    "normalize",
    "concat",
    "splice",
    "mix",
    "relative_db",
    "butter_filter",
    "bandpass",
    "amplitude_modulate",
    "resonator",
    "antiresonator",
]

# A value that may change over time: a number, or a ``(times, values)`` pair.
Track = float | tuple[ArrayLike, ArrayLike]


def match_fs(
    sounds: Sequence[Sound], fs: float | None = None, mode: Literal["down", "up"] = "down"
) -> list[Sound]:
    """Resample so all sounds share a rate: ``fs`` if given, else the lowest
    (``mode="down"``) or highest (``mode="up"``) rate in the list."""
    _check_not_empty(sounds)
    if fs is None:
        if mode not in ("down", "up"):
            raise ValueError(f"mode must be 'down' or 'up', not {mode!r}")
        rates = [s.fs for s in sounds]
        fs = {"down": min, "up": max}[mode](rates)
    return [s.resample(fs) for s in sounds]


def match_channels(sounds: Sequence[Sound]) -> list[Sound]:
    """Upmix mono sounds to the channel count of the others."""
    _check_not_empty(sounds)
    n_channels = max(s.n_channels for s in sounds)
    return [s.to_channels(n_channels) for s in sounds]


def match_lengths(
    sounds: Sequence[Sound], *, mode: Literal["pad", "truncate"] = "pad", align: Align = "start"
) -> list[Sound]:
    """Bring sounds to one length: zero-pad to the longest (``mode="pad"``)
    or cut to the shortest (``mode="truncate"``).

    ``align`` (``start``, ``center`` or ``end``) is where each sound sits in
    the common length: padding goes after, around or before it, and cutting
    keeps its start, middle or end. With ``center``, an odd number of extra
    samples puts the odd one at the end."""
    _check_not_empty(sounds)
    if align not in ("start", "center", "end"):
        raise ValueError(f"align must be 'start', 'center' or 'end', not {align!r}")
    match mode:
        case "pad":
            length = max(len(s) for s in sounds)
            return [s.pad_to(length, align=align) for s in sounds]
        case "truncate":
            length = min(len(s) for s in sounds)

            def keep(sound: Sound) -> Sound:
                extra = len(sound) - length
                start = {"start": 0, "center": extra // 2, "end": extra}[align]
                return Sound(sound.data[start : start + length], sound.fs)

            return [keep(s) for s in sounds]
        case _:
            raise ValueError(f"mode must be 'pad' or 'truncate', not {mode!r}")


def normalize(sounds: Sequence[Sound], *, rms: float = 1.0, peak: float | None = None) -> list[Sound]:
    """Scale each sound on its own to a target RMS (default 1) or, if
    ``peak`` is given, a target peak, as :meth:`Sound.normalize` does.

    Each sound gets its own gain, so their level differences are lost: with
    ``rms`` they all come out equally loud, with ``peak`` they all peak at
    the same value."""
    return [s.normalize(rms=rms, peak=peak) for s in sounds]


def _check_not_empty(sounds: Sequence[Sound]) -> None:
    if len(sounds) == 0:
        raise ValueError("need at least one sound")


def _check_fs(sounds: Sequence[Sound]) -> float:
    _check_not_empty(sounds)
    rates = {s.fs for s in sounds}
    if len(rates) > 1:
        raise ValueError(f"sampling rates differ: {sorted(rates)}; use match_fs()")
    return rates.pop()


def concat(sounds: Sequence[Sound]) -> Sound:
    """Join end to end (mono sounds are upmixed to match)."""
    fs = _check_fs(sounds)
    return Sound(np.concatenate([s.data for s in match_channels(sounds)]), fs)


def splice(
    sound: Sound,
    starts: ArrayLike,
    segment_dur: float,
    *,
    crossfade: float = 0.03,
    max_shift: float = 0.015,
    fade: Literal["adaptive", "equal_gain"] = "adaptive",
) -> Sound:
    """Segments of ``sound`` joined in a new order, with aligned cross-fades.

    Segment ``i`` of the output is the stretch of ``sound`` that begins at
    ``starts[i]`` [s] and lasts ``segment_dur`` [s]. The output is
    ``len(starts)`` segments long, and the join between segments ``i`` and
    ``i + 1`` sits at ``(i + 1) * segment_dur``. Each join is a raised-cosine
    cross-fade ``crossfade`` long, centered on the join, so a segment brings
    half a cross-fade of the sound beyond each of its ends. The first
    segment starts and the last one ends at full level.

    Before a segment is added, the point it is read from may move by up to
    ``max_shift`` either way, to where its waveform best matches what is
    already written: the shift that maximizes the cross-correlation over the
    cross-fade, weighted by both ramps so that the middle of the cross-fade
    counts most. The correlation is not normalized, so a shift where the new
    segment is louder can win. Shifts are limited to reads inside the sound;
    when none is (a segment at the very start with ``max_shift`` shorter
    than half the cross-fade), the segment is read at its start and the
    samples outside the sound count as silence. ``max_shift=0`` joins at the
    given starts exactly, and segments that follow each other in the sound
    then join back into the sound itself.

    The ramps are raised cosines whose sum is 1 (``fade="equal_gain"``),
    which keeps the level of two segments that match, but loses power where
    they do not: two unrelated stretches of noise keep on average 3/4 of
    their power over the cross-fade, 1.2 dB less. ``fade="adaptive"`` (the
    default) measures the correlation ``r`` between the two segments over
    the cross-fade, the outgoing one as it would have continued and the
    incoming one, unfaded, and divides both ramps by
    ``sqrt(rise**2 + fall**2 + 2 * r * rise * fall)``, which keeps the
    expected power constant across the join; ``r`` below 0 counts as 0.
    Matching segments (``r = 1``) are joined as with equal gain, so
    contiguous segments still give back the sound.

    This is the joining step of sound quilts (:func:`sonore.quilt`).
    Overath et al. (2015) call it pitch-synchronous overlap-add (PSOLA), but
    no pitch marks are used: it is overlap-add aligned by waveform
    similarity, the idea of WSOLA (Verhelst & Roelands, 1993).

    Mono only, because one shift per join cannot align several channels at
    once.
    """
    if sound.n_channels != 1:
        raise ValueError(f"splice needs a mono sound, not {sound.n_channels} channels")
    fs = sound.fs
    n_segment = int(round(segment_dur * fs))
    n_fade = int(round(crossfade * fs))
    n_shift = int(round(max_shift * fs))
    if n_segment < 1:
        raise ValueError("segment_dur is shorter than one sample")
    if not 0 <= n_fade <= n_segment:
        raise ValueError("crossfade must be between 0 and segment_dur")
    if n_shift < 0:
        raise ValueError("max_shift must not be negative")
    start_samples = np.round(np.atleast_1d(np.asarray(starts, dtype=float)) * fs).astype(int)
    if start_samples.ndim != 1 or len(start_samples) == 0:
        raise ValueError("starts must be a non-empty list of times")
    n_source = len(sound)
    if start_samples.min() < 0 or start_samples.max() + n_segment > n_source:
        raise ValueError(f"every segment must lie inside the sound ({n_source / fs:g} s)")

    if fade not in ("adaptive", "equal_gain"):
        raise ValueError(f'fade must be "adaptive" or "equal_gain", not {fade!r}')
    joined, _ = _aligned_overlap_add(
        sound.data[:, 0], start_samples, n_segment, n_fade, n_shift, adaptive=fade == "adaptive"
    )
    return Sound(joined, fs)


def _aligned_overlap_add(
    signal: np.ndarray,
    start_samples: np.ndarray,
    n_segment: int,
    n_fade: int,
    n_shift: int,
    *,
    adaptive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`splice` on a 1-D array, in samples; also returns the shift
    chosen for each segment."""
    before = n_fade // 2  # the part of a cross-fade before its join
    after = n_fade - before
    rise = 0.5 * (1 - np.cos(np.pi * np.arange(n_fade) / n_fade)) if n_fade else np.zeros(0)
    fall = 1 - rise
    # Zeros around the sound, so that any read the search can ask for exists.
    margin = before + after + n_shift
    source = np.pad(signal, margin)
    n_out = len(start_samples) * n_segment
    # The output with room for the first segment's lead-in and the last one's tail.
    written = np.zeros(before + n_out + after)
    shifts = np.arange(-n_shift, n_shift + 1)
    chosen = np.zeros(len(start_samples), dtype=int)
    outgoing = np.zeros(n_fade)  # the previous segment's tail over the coming join, unfaded
    for i, start in enumerate(start_samples):
        shift = 0
        if i > 0 and n_fade > 0 and n_shift > 0:
            # what is already written over this join, weighted by the coming ramp
            target = written[i * n_segment : i * n_segment + n_fade] * rise
            block = source[margin + start - before - n_shift : margin + start - before + n_fade + n_shift]
            score = np.correlate(block, target, mode="valid")
            inside = (start + shifts - before >= 0) & (start + shifts + n_segment + after <= len(signal))
            if inside.any():
                score[~inside] = -np.inf
                shift = int(shifts[np.argmax(score)])
        chosen[i] = shift
        read = margin + start + shift
        piece = source[read - before : read + n_segment + after].copy()
        join = slice(i * n_segment, i * n_segment + n_fade)
        if i > 0 and adaptive and n_fade > 0:
            incoming = piece[:n_fade]
            norm = np.sqrt(np.dot(outgoing, outgoing) * np.dot(incoming, incoming))
            r = min(max(np.dot(outgoing, incoming) / norm, 0.0), 1.0) if norm > 0 else 1.0
            gain = 1 / np.sqrt(rise**2 + fall**2 + 2 * r * rise * fall)
            written[join] *= gain  # only the outgoing segment's fall is written there
            piece[:n_fade] *= rise * gain
        elif i > 0:
            piece[:n_fade] *= rise
        if i < len(start_samples) - 1:
            outgoing = piece[len(piece) - n_fade :].copy()
            piece[len(piece) - n_fade :] *= fall
        written[i * n_segment : i * n_segment + len(piece)] += piece
    return written[before : before + n_out], chosen


def mix(sounds: Sequence[Sound], *, align: Align = "start") -> Sound:
    """Sum sounds of possibly different lengths (zero-padded per ``align``)."""
    _check_fs(sounds)
    return functools.reduce(operator.add, match_lengths(match_channels(sounds), align=align))


def relative_db(sounds: Sequence[Sound], *, ref: int = 0) -> list[float]:
    """Level of each sound in dB relative to ``sounds[ref]``."""
    ref_rms = sounds[ref].rms
    if ref_rms == 0:
        raise ValueError(f"the reference sounds[{ref}] is silent, so levels relative to it are undefined")
    return [float(amp_to_db(s.rms / ref_rms)) for s in sounds]


def butter_filter(
    sound: Sound,
    cutoff: float | tuple[float, float],
    *,
    btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "bandpass",
    order: int = 4,
    zero_phase: bool = True,
) -> Sound:
    """Butterworth filter along time (each channel separately).

    ``btype`` is ``lowpass``, ``highpass``, ``bandpass``, or ``bandstop``.
    ``zero_phase=True`` runs it forward and backward: no phase shift, and the
    magnitude response is squared (twice the attenuation in dB, so the gain at
    the cutoff is -6 dB instead of -3 dB). This needs a sound longer than
    SciPy's padding (15 samples for a 4th-order lowpass).
    """
    sos = butter(order, cutoff, btype=btype, fs=sound.fs, output="sos")
    run_filter = sosfiltfilt if zero_phase else sosfilt
    return Sound(run_filter(sos, sound.data, axis=0), sound.fs)


def bandpass(sound: Sound, f_lo: float, f_hi: float, *, order: int = 4, zero_phase: bool = True) -> Sound:
    """Butterworth band-pass from ``f_lo`` to ``f_hi`` [Hz].

    Shorthand for ``butter_filter(sound, (f_lo, f_hi), btype="bandpass",
    order=order, zero_phase=zero_phase)``; see :func:`butter_filter`. The band edges are where the
    gain is -3 dB (one way) or -6 dB (``zero_phase=True``). ``order`` is
    SciPy's, so the band-pass has ``2 * order`` poles."""
    return butter_filter(sound, (f_lo, f_hi), btype="bandpass", order=order, zero_phase=zero_phase)


def amplitude_modulate(sound: Sound, f_mod: float, *, depth: float = 1.0, phase: float = 0.0) -> Sound:
    """Multiply by ``1 + depth*sin(2*pi*f_mod*t + phase)``.

    ``phase`` is in radians. ``depth`` 1 takes the envelope down to zero;
    above 1 the envelope changes sign (overmodulation)."""
    t = time_axis(len(sound), sound.fs)
    return sound * (1 + depth * np.sin(2 * np.pi * f_mod * t + phase))


# ------------------------------------------------------------ resonators
def _track(value: Track, t: np.ndarray, name: str = "value") -> np.ndarray | float:
    """A number as it is, or a ``(times, values)`` pair interpolated linearly
    to the times ``t`` (values held before the first and after the last)."""
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        times, values = cast(tuple, value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number or a (times, values) pair") from None
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if times.ndim != 1 or values.shape != times.shape or len(times) == 0:
        raise ValueError(f"{name} needs one value per time: got {times.shape} times, {values.shape} values")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}'s times must increase")
    return np.interp(t, times, values)


def _resonator_coefs(f, bw, fs: float):
    """Klatt's (1980) coefficients: y[n] = A x[n] + B y[n-1] + C y[n-2]."""
    c = -np.exp(-2 * np.pi * bw / fs)
    b = 2 * np.exp(-np.pi * bw / fs) * np.cos(2 * np.pi * f / fs)
    return 1 - b - c, b, c


def _check_resonance(f, bw, fs: float) -> None:
    if np.any(np.asarray(f) < 0) or np.any(np.asarray(f) >= fs / 2):
        raise ValueError(f"resonance frequencies must be in [0, fs/2) = [0, {fs / 2:g}) Hz")
    if np.any(np.asarray(bw) <= 0):
        raise ValueError("bandwidths must be positive")


def _two_pole(x: np.ndarray, a, b, c) -> np.ndarray:
    """y[n] = a x[n] + b y[n-1] + c y[n-2] with per-sample coefficients."""
    y = np.empty_like(x)
    y1 = y2 = 0.0
    for n, (xn, an, bn, cn) in enumerate(zip(x.tolist(), a.tolist(), b.tolist(), c.tolist(), strict=True)):
        y0 = an * xn + bn * y1 + cn * y2
        y[n] = y0
        y2, y1 = y1, y0
    return y


def _two_zero(x: np.ndarray, a, b, c) -> np.ndarray:
    """y[n] = a x[n] + b x[n-1] + c x[n-2] with per-sample coefficients."""
    x1 = np.concatenate([[0.0], x[:-1]])
    x2 = np.concatenate([[0.0, 0.0], x[:-2]])[: len(x)]
    return a * x + b * x1 + c * x2


def _filter(sound: Sound, f: Track, bw: Track, inverse: bool) -> Sound:
    fs = sound.fs
    t = time_axis(len(sound), fs)
    f_values, bw_values = _track(f, t, "f"), _track(bw, t, "bw")
    _check_resonance(f_values, bw_values, fs)
    a, b, c = _resonator_coefs(f_values, bw_values, fs)
    data = sound.data
    if np.ndim(a) == 0:
        numerator, denominator = ([1 / a, -b / a, -c / a], [1.0]) if inverse else ([a], [1.0, -b, -c])
        return Sound(lfilter(numerator, denominator, data, axis=0), fs)
    a, b, c = (np.broadcast_to(coef, t.shape) for coef in (a, b, c))
    out = np.empty_like(data)
    for channel in range(data.shape[1]):
        x = data[:, channel]
        out[:, channel] = _two_zero(x, 1 / a, -b / a, -c / a) if inverse else _two_pole(x, a, b, c)
    return Sound(out, fs)


def resonator(sound: Sound, f: Track, bw: Track) -> Sound:
    """A formant: Klatt's (1980) second-order digital resonator.

    ``y[n] = A x[n] + B y[n-1] + C y[n-2]`` with ``C = -exp(-2 pi bw / fs)``,
    ``B = 2 exp(-pi bw / fs) cos(2 pi f / fs)`` and ``A = 1 - B - C``.

    This is the damped harmonic oscillator ``x'' + 2 sigma x' + w0^2 x =
    w0^2 u`` sampled: its poles are the oscillator's, mapped by ``z =
    exp(s / fs)``, with ``f`` the ringing frequency and ``bw = sigma / pi``
    the decay rate, and its gain at 0 Hz is exactly 1, as the oscillator's
    is (so ``f = 0`` gives a low-pass filter). ``f`` and ``bw`` are the peak
    and the -3 dB bandwidth only when ``bw`` is much smaller than ``f``, in
    the oscillator too. Far above the resonance the gain falls more slowly
    than the oscillator's, since a digital response repeats every ``fs``.

    ``f`` and ``bw`` are numbers, or ``(times, values)`` pairs that are
    interpolated linearly to every sample (held beyond their ends), so a
    formant can glide. The filter keeps its state while its coefficients
    change, which is what keeps a moving formant free of clicks.
    """
    return _filter(sound, f, bw, inverse=False)


def antiresonator(sound: Sound, f: Track, bw: Track) -> Sound:
    """An antiformant: the exact inverse of :func:`resonator` with the same
    ``f`` and ``bw``, a spectral zero (Klatt, 1980). Its gain at 0 Hz is 1.

    ``y[n] = (x[n] - B x[n-1] - C x[n-2]) / A``; ``f`` and ``bw`` may change
    over time as in :func:`resonator`.
    """
    return _filter(sound, f, bw, inverse=True)
