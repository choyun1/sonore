"""Small numerical helpers: levels and decibels, the ERB and mel scales and
the frequency scales built on them, cents and note names, time axes, random generators, and
internal array helpers used across sonore."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import resample_poly

__all__ = [
    "rms",
    "amp_to_db",
    "power_to_db",
    "db_to_power",
    "db_to_amp",
    "freq_to_erb",
    "erb_to_freq",
    "erb_bandwidth",
    "freq_to_mel",
    "mel_to_freq",
    "ratio_to_cents",
    "cents_to_ratio",
    "note_to_freq",
    "FrequencyScale",
    "FREQUENCY_SCALES",
    "cents_scale",
    "n_samples",
    "time_axis",
    "as_rng",
]


def rms(x: ArrayLike, axis: int | None = None) -> np.ndarray | float:
    """Root-mean-square value."""
    return np.sqrt(np.mean(np.square(x), axis=axis))


def amp_to_db(x: ArrayLike, ref: float = 1.0, floor_db: float = -300.0) -> np.ndarray:
    """Amplitude (not power) to decibels: ``20*log10(|x|/ref)``, floored."""
    x = np.abs(np.asarray(x)).astype(float) / ref
    with np.errstate(divide="ignore"):
        return np.maximum(20 * np.log10(x), floor_db)


def power_to_db(x: ArrayLike, ref: float = 1.0, floor_db: float = -300.0) -> np.ndarray:
    """Power (not amplitude) to decibels: ``10*log10(x/ref)``, floored at
    ``floor_db``.

    A power is never negative. A negative value is treated as zero and
    returns ``floor_db``, never the level of its absolute value: a tiny
    negative from round-off (a power of -1e-17 where the true value is 0)
    lands where true zeros do, and a negative from a bug upstream shows up
    at the floor instead of as a plausible level. (``amp_to_db`` takes the
    absolute value instead, because amplitudes are signed.)"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0) / ref
    with np.errstate(divide="ignore"):
        return np.maximum(10 * np.log10(x), floor_db)


def db_to_amp(db: ArrayLike) -> np.ndarray:
    """Decibels to amplitude factor: ``10**(db/20)``."""
    return np.power(10.0, np.asarray(db, dtype=float) / 20)


def db_to_power(db: ArrayLike) -> np.ndarray:
    """Decibels to power factor: ``10**(db/10)``."""
    return np.power(10.0, np.asarray(db, dtype=float) / 10)


def freq_to_erb(freq: ArrayLike) -> np.ndarray:
    """Frequency [Hz] to ERB-number [Cams] (Glasberg & Moore, 1990).

    The exact integral of ``1 / erb_bandwidth``, ``9.265 ln(1 + f / 228.8)``.
    The paper prints the rounded form ``21.4 log10(1 + 0.00437 f)``, which
    is about 0.3% higher (15.62 rather than 15.57 Cams at 1 kHz)."""
    return 9.265 * np.log1p(np.asarray(freq, dtype=float) / (24.7 * 9.265))


def erb_to_freq(erb: ArrayLike) -> np.ndarray:
    """ERB-number [Cams] to frequency [Hz]; inverse of :func:`freq_to_erb`."""
    return 24.7 * 9.265 * np.expm1(np.asarray(erb, dtype=float) / 9.265)


def erb_bandwidth(freq: ArrayLike) -> np.ndarray:
    """Equivalent rectangular bandwidth [Hz] of the auditory filter at ``freq``."""
    return 24.7 * (4.37 * np.asarray(freq, dtype=float) / 1000 + 1)


_SLANEY_HZ_PER_MEL = 200 / 3  # linear part: 15 mel at 1000 Hz
_SLANEY_LOG_STEP = np.log(6.4) / 27  # logarithmic part: 27 mel per factor of 6.4


def freq_to_mel(freq, scale: str = "htk") -> np.ndarray:
    """Frequency [Hz] to mel.

    ``scale="htk"`` is ``2595 log10(1 + f / 700)``, the formula HTK uses,
    as printed in O'Shaughnessy (2000, Eq. 4.2, p. 128), which gives no
    earlier source for it. ``"slaney"`` is the scale of
    Slaney's Auditory Toolbox and librosa: linear below 1 kHz (15 mel at
    1000 Hz) and logarithmic above it (27 mel per factor of 6.4)."""
    freq = np.asarray(freq, dtype=float)
    if scale == "htk":
        return 2595 * np.log10(1 + freq / 700)
    if scale == "slaney":
        above = 15 + np.log(np.maximum(freq, 1000) / 1000) / _SLANEY_LOG_STEP
        return np.where(freq < 1000, freq / _SLANEY_HZ_PER_MEL, above)
    raise ValueError(f"scale must be 'htk' or 'slaney', not {scale!r}")


def mel_to_freq(mel, scale: str = "htk") -> np.ndarray:
    """Mel to frequency [Hz]; the inverse of :func:`freq_to_mel`."""
    mel = np.asarray(mel, dtype=float)
    if scale == "htk":
        return 700 * (10 ** (mel / 2595) - 1)
    if scale == "slaney":
        above = 1000 * np.exp(_SLANEY_LOG_STEP * (np.maximum(mel, 15) - 15))
        return np.where(mel < 15, mel * _SLANEY_HZ_PER_MEL, above)
    raise ValueError(f"scale must be 'htk' or 'slaney', not {scale!r}")


def ratio_to_cents(ratio: ArrayLike) -> np.ndarray:
    """The size of a frequency ratio in cents, ``1200 log2(ratio)``: an
    octave (2) is 1200 cents, an equal-tempered semitone 100, a just fifth
    (3/2) 702.0."""
    return 1200 * np.log2(np.asarray(ratio, float))


def cents_to_ratio(cents: ArrayLike) -> np.ndarray:
    """The frequency ratio of an interval ``cents`` wide, ``2 ** (cents / 1200)``."""
    return np.exp2(np.asarray(cents, float) / 1200)


# Semitones above C within an octave, for note names
_NOTE_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTALS = {"#": 1, "\u266f": 1, "b": -1, "\u266d": -1}


def note_to_freq(note: str, a4: float = 440.0) -> float:
    """The equal-tempered frequency [Hz] of a note name such as ``"A4"``,
    ``"C#3"`` or ``"Bb2"`` (scientific pitch notation: octave 4 starts at
    middle C; ``#`` or ``b``, or ♯ and ♭, for sharp and flat, repeatable), with A4 at
    ``a4`` Hz."""
    if not a4 > 0:
        raise ValueError(f"a4 must be a frequency above 0 Hz, not {a4!r}")
    name = note.strip()
    letter, rest = name[:1].upper(), name[1:]
    if letter not in _NOTE_STEPS:
        raise ValueError(f"note must start with a letter A-G, as in 'A4' or 'C#3', not {note!r}")
    step = _NOTE_STEPS[letter]
    while rest[:1] in _ACCIDENTALS:
        step += _ACCIDENTALS[rest[:1]]
        rest = rest[1:]
    try:
        octave = int(rest)
    except ValueError:
        raise ValueError(f"note must end with an octave number, as in 'A4' or 'C#3', not {note!r}") from None
    semitones_from_a4 = step - 9 + 12 * (octave - 4)
    return float(a4 * cents_to_ratio(100 * semitones_from_a4))


def _identity(value):
    return value


@dataclass(frozen=True)
class FrequencyScale:
    """A frequency axis, such as the one filterbank centers are equally
    spaced on: its ``name``, the ``unit`` of one step on it, and the
    conversions to and from Hz."""

    name: str
    unit: str
    to_scale: Callable[[np.ndarray], np.ndarray]
    from_scale: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self):
        # A mismatched or mistyped pair would still build a bank, with every
        # filter in the wrong place; check a round trip where the scale is defined.
        freqs = np.array([100.0, 1000.0, 10000.0])
        with np.errstate(all="ignore"):
            back = np.asarray(self.from_scale(self.to_scale(freqs)), float)
        defined = np.isfinite(back)
        if not np.allclose(back[defined], freqs[defined], rtol=1e-9):
            raise ValueError(
                f"scale {self.name!r}: from_scale(to_scale(f)) does not give f back "
                f"({freqs[defined]} Hz -> {back[defined]} Hz): the two conversions must invert each other"
            )


def cents_scale(reference: float = 440.0) -> FrequencyScale:
    """Cents from ``reference`` [Hz] (A4 by default): ``1200 log2(f / reference)``.
    A factor of 2 is 1200 cents and an equal-tempered semitone 100, so a
    filterbank on this scale with ``spacing=100`` has one band per semitone.
    The reference only shifts the positions: the bank is the one built on
    octaves with ``spacing=1/12``."""
    if not reference > 0:
        raise ValueError(f"reference must be a frequency above 0 Hz, not {reference!r}")
    return FrequencyScale(
        "cents",
        "cent",
        lambda freq: ratio_to_cents(np.asarray(freq, float) / reference),
        lambda cents: reference * cents_to_ratio(cents),
    )


#: The scales a filterbank can be built on by name: the ERB-number scale
#: (Glasberg & Moore, 1990), octaves (log2 frequency), cents from A4 at 440 Hz,
#: the HTK mel scale, and Hz.
FREQUENCY_SCALES = {
    "erb": FrequencyScale("erb", "ERB", freq_to_erb, erb_to_freq),
    "octave": FrequencyScale("octave", "oct", np.log2, np.exp2),
    "cents": cents_scale(),
    "mel": FrequencyScale("mel", "mel", freq_to_mel, mel_to_freq),
    "linear": FrequencyScale("linear", "Hz", _identity, _identity),
}


def _as_frequency_scale(scale: str | FrequencyScale) -> FrequencyScale:
    if isinstance(scale, FrequencyScale):
        return scale
    try:
        return FREQUENCY_SCALES[scale.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"scale must be one of {sorted(FREQUENCY_SCALES)} or a FrequencyScale, not {scale!r}"
        ) from None


def n_samples(duration: float, fs: float) -> int:
    """Number of samples in ``duration`` seconds, rounded to the nearest whole
    number rather than floored (exact halves round to even, as Python's
    ``round`` does)."""
    return int(round(duration * fs))


def time_axis(n: int, fs: float) -> np.ndarray:
    """Sample times ``k/fs``. Use this instead of ``np.linspace(0, dur, n)``,
    whose spacing is ``dur/(n-1)`` rather than ``1/fs``."""
    return np.arange(n) / fs


def as_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """Accept a seed, a Generator, or None and return a Generator."""
    return np.random.default_rng(rng)


# Internal helpers on plain arrays, imported by other modules of sonore. They
# are not in processing.py because Sound uses two of them and processing
# imports Sound; an underscore name is never API (docs/design/layout.md).


def _parabola_vertex(before: np.ndarray, at: np.ndarray, after: np.ndarray, dip: bool = False) -> np.ndarray:
    """Offset, in samples, of the vertex of the parabola through three equally
    spaced values (``at`` in the middle): the usual sub-sample refinement of a
    peak or a dip. The offset is 0 where the parabola is flat, and with
    ``dip=True`` also where it opens downward (no dip to refine)."""
    curvature = before - 2 * at + after
    curved = curvature > 0 if dip else curvature != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(curved, 0.5 * (before - after) / curvature, 0.0)


def _resample_poly(data: np.ndarray, fs: float, fs_new: float, axis: int = 0, **kwargs) -> np.ndarray:
    """Polyphase resampling of ``data`` along ``axis`` from ``fs`` to ``fs_new``, the
    rate ratio approximated by a fraction with denominator at most 10000.
    ``kwargs`` go to ``scipy.signal.resample_poly`` (``padtype``)."""
    ratio = Fraction(fs_new / fs).limit_denominator(10000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=axis, **kwargs)


def _fit_length(data: np.ndarray, length: int, mode: str = "constant") -> np.ndarray:
    """``data`` cut or padded at the end (axis 0) to ``length`` samples; ``mode``
    is ``np.pad``'s (zeros by default, ``"edge"`` repeats the last sample)."""
    if data.shape[0] >= length:
        return data[:length]
    return np.pad(data, [(0, length - data.shape[0])] + [(0, 0)] * (data.ndim - 1), mode=mode)


def _phase_ramp_delay(data: np.ndarray, shift: ArrayLike, n_fft: int, n_out: int) -> np.ndarray:
    """Delay signals along the last axis by ``shift`` samples (fractional, one per
    signal, broadcast over the leading axes) with an FFT phase ramp: band-limited
    interpolation. ``n_fft`` must leave room for the delay and the sinc tails, which
    otherwise wrap around; the first ``n_out`` samples are returned."""
    spectrum = np.fft.rfft(data, n=n_fft, axis=-1)
    freqs = np.fft.rfftfreq(n_fft)
    spectrum *= np.exp(-2j * np.pi * freqs * np.asarray(shift)[..., None])
    return np.fft.irfft(spectrum, n=n_fft, axis=-1)[..., :n_out]
