"""Stimulus generators.

Every generator returns a :class:`~sonore.Sound` normalized to RMS = 1.
Anything random takes an ``rng`` argument (a seed or ``np.random.Generator``)
so stimuli can be regenerated exactly.
"""

from __future__ import annotations

import inspect
import numbers
import warnings
from collections.abc import Callable
from typing import Protocol, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import chirp

from sonore.core.sound import Sound
from sonore.core.utils import as_rng, n_samples, time_axis

__all__ = [
    "silence",
    "pure_tone",
    "harmonic_complex",
    "schroeder_complex",
    "square_wave",
    "sawtooth_wave",
    "pulse_train",
    "linear_chirp",
    "exponential_chirp",
    "gaussian_noise",
    "correlated_noise",
    "iterated_ripple_noise",
]

RNG = int | np.random.Generator | None


def _finish(data: np.ndarray, fs: float) -> Sound:
    sound = Sound(data, fs)
    return sound.normalize() if sound.rms > 0 else sound


def silence(duration: float, fs: float, n_channels: int = 1) -> Sound:
    return Sound(np.zeros((n_samples(duration, fs), n_channels)), fs)


def pure_tone(duration: float, fs: float, freq: float, phase: float = 0.0) -> Sound:
    """``cos(2*pi*freq*t + phase)``."""
    t = time_axis(n_samples(duration, fs), fs)
    return _finish(np.cos(2 * np.pi * freq * t + phase), fs)


# ----------------------------------------------------------- harmonic sounds
_PHASE_PRESETS = ("cosine", "sine", "alternating", "random", "schroeder+", "schroeder-")


class F0Contour(Protocol):
    """Anything with window times ``t`` [s] and F0 values ``f0`` [Hz], 0 where
    unvoiced, such as an :class:`~sonore.analysis.f0.F0Track`. ``f0`` is one
    row of values per channel, or a single row."""

    t: ArrayLike
    f0: ArrayLike


Amplitudes = (
    ArrayLike
    | Callable[[np.ndarray, np.ndarray], ArrayLike]
    | Callable[[np.ndarray, np.ndarray, int], ArrayLike]
    | None
)


def _harmonic_phases(harmonics: np.ndarray, phases, rng: RNG) -> np.ndarray:
    if not isinstance(phases, str):
        phases = np.asarray(phases, float)
        if phases.ndim > 0 and phases.shape != harmonics.shape:
            raise ValueError(
                f"phases has {phases.size} values; this sound has {harmonics.size} harmonics "
                "and needs one starting phase for each"
            )
        return np.broadcast_to(phases, harmonics.shape)
    n_harmonics = len(harmonics)
    match phases:
        case "cosine":
            return np.zeros(n_harmonics)
        case "sine":
            return np.full(n_harmonics, -np.pi / 2)
        case "alternating":
            return np.where(harmonics % 2 == 1, 0.0, -np.pi / 2)
        case "random":
            return as_rng(rng).uniform(-np.pi, np.pi, n_harmonics)
        case "schroeder+" | "schroeder-":
            sign = 1 if phases.endswith("+") else -1
            return sign * np.pi * harmonics * (harmonics + 1) / n_harmonics
    raise ValueError(f"phases must be an array or one of {_PHASE_PRESETS}")


def _contour(f0) -> tuple[np.ndarray, np.ndarray] | None:
    """``(window times, values of shape (n_channels, n_windows))`` for an F0
    contour, or None for a fixed F0 (a number)."""
    match f0:
        case numbers.Real():
            return None
        case (t, values):
            pass
        case object(t=t, f0=values):
            pass
        case _:
            raise TypeError(
                "f0 must be a number, a (times, values) pair, or an object with .t and .f0 such as an F0Track"
            )
    t = np.asarray(t, float)
    values = np.atleast_2d(np.asarray(values, float))
    if t.ndim != 1 or len(t) == 0 or values.ndim != 2 or values.shape[1] != len(t):
        raise ValueError(
            f"an F0 contour needs one value per time: got {np.shape(t)} times and {values.shape} values"
        )
    if np.any(np.diff(t) <= 0):
        raise ValueError("an F0 contour's times must increase")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("an F0 contour's values must be finite and >= 0 (0 where unvoiced)")
    return t, values


def _max_harmonic(values: np.ndarray, f_max: float) -> int:
    """The highest harmonic number that is below ``f_max`` somewhere on the contour."""
    voiced = values[values > 0]
    return int(f_max // voiced.min()) if voiced.size else 0


def _n_max(f0, fs: float, f_max: float | None) -> int:
    """The highest harmonic number a harmonic waveform on ``f0`` can hold:
    below Nyquist for a fixed F0, below ``f_max`` somewhere for a contour."""
    contour = _contour(f0)
    if contour is None:
        return int(np.ceil(fs / 2 / f0)) - 1
    return _max_harmonic(contour[1], 0.45 * fs if f_max is None else f_max)


def _taper(freq: np.ndarray, f_max: float) -> np.ndarray:
    """1 below 0.9 f_max, falling as cos^2 to 0 at f_max and above."""
    fade_start = 0.9 * f_max
    fade_position = np.clip((freq - fade_start) / (f_max - fade_start), 0.0, 1.0)
    return np.cos(np.pi / 2 * fade_position) ** 2


def _voicing_gate(
    t: np.ndarray, window_times: np.ndarray, voiced: np.ndarray, ramp: float, fs: float
) -> np.ndarray:
    """1 where the nearest time window is voiced, 0 elsewhere, every step smoothed by
    a Hann window ``ramp`` seconds long. The ends are extended, not faded."""
    gate = (np.interp(t, window_times, voiced.astype(float)) >= 0.5).astype(float)
    n_ramp = int(round(ramp * fs))
    if n_ramp < 2:
        return gate
    window = np.hanning(n_ramp + 2)[1:-1]
    padded = np.pad(gate, n_ramp, mode="edge")
    return np.convolve(padded, window / window.sum(), mode="same")[n_ramp:-n_ramp]


def _amplitude_function(amplitudes: Callable) -> Callable[[np.ndarray, np.ndarray, int], ArrayLike]:
    """``amplitudes`` as a function of time, frequency and harmonic number,
    whether it takes the harmonic number or not."""
    try:
        inspect.signature(amplitudes).bind(None, None, None)
    except (TypeError, ValueError):  # takes only (t, f), or has no signature to read
        return lambda t, freq, number: amplitudes(t, freq)
    return amplitudes


def _add_harmonic(total: np.ndarray, amplitude, argument: np.ndarray) -> None:
    """Add ``amplitude * cos(argument)`` to ``total``; a complex amplitude
    adds ``Re(amplitude * exp(i * argument))``, its angle shifting the phase."""
    if np.iscomplexobj(amplitude):
        total += amplitude.real * np.cos(argument) - amplitude.imag * np.sin(argument)
    else:
        total += amplitude * np.cos(argument)


def _gains(amplitudes: Amplitudes, harmonics: np.ndarray) -> np.ndarray | None:
    """Per-harmonic amplitudes as an array, or None when they are a function."""
    if callable(amplitudes):
        return None
    if amplitudes is None:
        return np.ones(len(harmonics))
    amplitudes = np.asarray(amplitudes, float)
    if amplitudes.shape != harmonics.shape:
        raise ValueError("amplitudes must match harmonics")
    return amplitudes


@overload
def harmonic_complex(
    duration: float,
    fs: float,
    f0: float,
    harmonics: ArrayLike | None = None,
    amplitudes: Amplitudes = None,
    phases: ArrayLike | str = "cosine",
    rng: RNG = None,
) -> Sound: ...


@overload
def harmonic_complex(
    duration: float,
    fs: float,
    f0: tuple[ArrayLike, ArrayLike] | F0Contour,
    harmonics: ArrayLike | None = None,
    amplitudes: Amplitudes = None,
    phases: ArrayLike | str = "cosine",
    rng: RNG = None,
    *,
    f_max: float | None = None,
    ramp: float = 0.005,
    unvoiced: str = "silence",
) -> Sound: ...


def harmonic_complex(
    duration,
    fs,
    f0,
    harmonics=None,
    amplitudes=None,
    phases="cosine",
    rng=None,
    *,
    f_max=None,
    ramp=0.005,
    unvoiced="silence",
):
    """Sum of harmonics ``a_n cos(n * Phi(t) + phi_n)`` of a fixed or moving F0.

    ``f0`` is either a number, for ``Phi = 2*pi*f0*t``, or an F0 contour:
    a ``(times, values)`` pair or anything with ``.t`` and ``.f0``, such as
    an :class:`~sonore.analysis.f0.F0Track`, with values in Hz and 0 where
    unvoiced (one row per channel for a multichannel sound).

    ``harmonics`` are the harmonic numbers; by default every one below
    Nyquist (fixed F0) or below ``f_max`` (contour). ``amplitudes`` is one
    value per harmonic, or a function ``amplitudes(t, f)`` giving the gain
    at times ``t`` and frequencies ``f`` (a spectral envelope, evaluated at
    every harmonic's frequency). The function may also take the harmonic
    number as a third argument, ``amplitudes(t, f, n)``, and may return
    complex values: harmonic ``n`` is then ``Re(a_n exp(i(n Phi + phi_n)))``,
    so the angle of ``a_n`` adds to its phase and can change over time (an
    LF glottal pulse whose shape changes, as in :func:`glottal_source`).
    ``phases`` are starting phases: an array,
    one per harmonic, or one of ``"cosine"``, ``"sine"``, ``"alternating"``,
    ``"random"``, ``"schroeder+"``, ``"schroeder-"``.

    A fixed F0 drops harmonics at or above Nyquist, with a warning.

    A contour (``docs/design/harmonic-source.md``) is filled across its
    unvoiced gaps and interpolated linearly to the sample rate, and the phase
    is its exact running integral, so every harmonic follows ``n`` times the
    contour with no jumps. Each harmonic fades out (``cos^2``) between
    ``0.9 * f_max`` and ``f_max`` (default ``0.45 * fs``), so nothing aliases
    as the pitch rises. Harmonics are switched off where the contour is
    unvoiced, with Hann ramps ``ramp`` seconds long; ``unvoiced="noise"``
    fills those stretches with white noise of the harmonics' power instead
    of silence. Values are held beyond the first and last times. With a
    contour, ``phases`` has one value per harmonic number up to the highest
    that fits below ``f_max`` at the lowest voiced F0, unless ``harmonics``
    is given.
    """
    contour = _contour(f0)
    if contour is not None:
        return _contour_complex(
            duration, fs, *contour, harmonics, amplitudes, phases, rng, f_max, ramp, unvoiced
        )
    if harmonics is None:
        if f0 <= 0:
            raise ValueError(f"f0 must be positive, got {f0:g}")
        harmonics = np.arange(1, int(np.ceil(fs / 2 / f0)))
    harmonics = np.atleast_1d(np.asarray(harmonics))
    gains = _gains(amplitudes, harmonics)
    start_phases = _harmonic_phases(harmonics, phases, rng)

    keep = harmonics * f0 < fs / 2
    if not keep.all():
        warnings.warn(f"dropping {np.sum(~keep)} harmonic(s) at or above Nyquist", stacklevel=2)
    t = time_axis(n_samples(duration, fs), fs)
    data = np.zeros_like(t)
    if gains is None:
        gain_function = _amplitude_function(amplitudes)
        gains = [gain_function(t, np.full_like(t, number * f0), number) for number in harmonics[keep]]
    else:
        gains = gains[keep]
    for number, amplitude, start_phase in zip(harmonics[keep], gains, start_phases[keep], strict=True):
        _add_harmonic(data, amplitude, 2 * np.pi * number * f0 * t + start_phase)
    return _finish(data, fs)


def _contour_complex(
    duration, fs, window_times, values, harmonics, amplitudes, phases, rng, f_max, ramp, unvoiced
) -> Sound:
    f_max = 0.45 * fs if f_max is None else float(f_max)
    if not 0 < f_max <= fs / 2:
        raise ValueError(f"f_max must be in (0, fs/2], got {f_max:g}")
    if unvoiced not in ("silence", "noise"):
        raise ValueError(f"unvoiced must be 'silence' or 'noise', got {unvoiced!r}")
    rng = as_rng(rng)
    if harmonics is None:
        harmonics = np.arange(1, _max_harmonic(values, f_max) + 1)
    harmonics = np.atleast_1d(np.asarray(harmonics))
    gains = _gains(amplitudes, harmonics)
    gain_function = _amplitude_function(amplitudes) if gains is None else None
    start_phases = _harmonic_phases(harmonics, phases, rng)

    length = n_samples(duration, fs)
    t = time_axis(length, fs)
    data = np.zeros((length, len(values)))
    for channel, channel_f0 in enumerate(values):
        voiced = channel_f0 > 0
        harmonic_sum = np.zeros(length)
        gate = np.zeros(length)
        if voiced.any():
            window_index = np.arange(len(channel_f0))
            f0_at_sample = np.interp(
                t, window_times, np.interp(window_index, window_index[voiced], channel_f0[voiced])
            )
            # the trapezoid rule: exact for an F0 linear between samples
            cumulative_f0 = np.concatenate([[0.0], np.cumsum((f0_at_sample[1:] + f0_at_sample[:-1]) / 2)])
            phase = 2 * np.pi / fs * cumulative_f0
            for index, (number, start_phase) in enumerate(zip(harmonics, start_phases, strict=True)):
                freq = number * f0_at_sample
                amplitude = gain_function(t, freq, number) if gains is None else gains[index]
                _add_harmonic(harmonic_sum, amplitude * _taper(freq, f_max), number * phase + start_phase)
            gate = _voicing_gate(t, window_times, voiced, ramp, fs)
        data[:, channel] = gate * harmonic_sum
        if unvoiced == "noise":
            weight = gate.sum()
            power = np.sum(gate * harmonic_sum**2) / weight if weight > 0 else 1.0
            data[:, channel] += (1 - gate) * np.sqrt(power) * rng.standard_normal(length)
    return _finish(data, fs)


def schroeder_complex(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    n_harmonics: int | None = None,
    sign: int = 1,
    **contour,
) -> Sound:
    """Equal-amplitude harmonic complex with Schroeder phases
    ``sign * pi * n(n+1)/N``. ``n_harmonics`` defaults to all below Nyquist
    (or below ``f_max`` for an F0 contour). ``f0`` and the keyword arguments
    for a contour are as in :func:`harmonic_complex`."""
    n_max = _n_max(f0, fs, contour.get("f_max"))
    count = n_max if n_harmonics is None else min(n_harmonics, n_max)
    return harmonic_complex(
        duration,
        fs,
        f0,
        np.arange(1, count + 1),
        phases="schroeder+" if sign > 0 else "schroeder-",
        **contour,
    )


def _waveform(duration, fs, f0, phase, harmonics, amplitudes, kind, contour) -> Sound:
    """A named band-limited waveform as a harmonic complex: harmonic ``n`` has
    amplitude ``amplitudes[i]`` and phase ``n * phase``, as a sine or a cosine."""
    phases = harmonics * phase - (np.pi / 2 if kind == "sin" else 0.0)
    return harmonic_complex(duration, fs, f0, harmonics, amplitudes, phases, **contour)


def _fixed_only(f0, name: str) -> None:
    if _contour(f0) is not None:
        raise ValueError(
            f"a {name} that is not band-limited needs a fixed f0; use bandlimited=True for a contour"
        )


def square_wave(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    phase: float = 0.0,
    bandlimited: bool = True,
    **contour,
) -> Sound:
    """Square wave. Band-limited (odd harmonics below Nyquist) unless
    ``bandlimited=False``, which aliases badly at high f0. A band-limited
    square wave also takes an F0 contour, as :func:`harmonic_complex` does."""
    if not bandlimited:
        _fixed_only(f0, "square wave")
        t = time_axis(n_samples(duration, fs), fs)
        return _finish(np.sign(np.sin(2 * np.pi * f0 * t + phase)), fs)
    numbers = np.arange(1, _n_max(f0, fs, contour.get("f_max")) + 1, 2)
    return _waveform(duration, fs, f0, phase, numbers, 1 / numbers, "sin", contour)


def sawtooth_wave(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    phase: float = 0.0,
    bandlimited: bool = True,
    **contour,
) -> Sound:
    """Rising sawtooth. Band-limited unless ``bandlimited=False``. A
    band-limited sawtooth also takes an F0 contour, as
    :func:`harmonic_complex` does."""
    if not bandlimited:
        from scipy.signal import sawtooth

        _fixed_only(f0, "sawtooth")
        t = time_axis(n_samples(duration, fs), fs)
        return _finish(sawtooth(2 * np.pi * f0 * t + phase + np.pi), fs)
    numbers = np.arange(1, _n_max(f0, fs, contour.get("f_max")) + 1)
    return _waveform(duration, fs, f0, phase, numbers, (-1.0) ** (numbers + 1) / numbers, "sin", contour)


def pulse_train(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    phase: float = 0.0,
    bandlimited: bool = True,
    **contour,
) -> Sound:
    """Periodic pulse (click) train.

    Band-limited: equal-amplitude cosine-phase harmonics up to Nyquist; this
    also takes an F0 contour, as :func:`harmonic_complex` does. Otherwise
    single-sample impulses at the nearest sample to each period.
    ``phase`` (radians of the fundamental) shifts the pulses in time.
    """
    if bandlimited:
        numbers = np.arange(1, _n_max(f0, fs, contour.get("f_max")) + 1)
        return _waveform(duration, fs, f0, phase, numbers, np.ones(len(numbers)), "cos", contour)
    _fixed_only(f0, "pulse train")
    length = n_samples(duration, fs)
    times = (np.arange(0, duration * f0 + 1) - phase / (2 * np.pi)) / f0
    pulse_samples = np.round(times * fs).astype(int)
    data = np.zeros(length)
    data[pulse_samples[(pulse_samples >= 0) & (pulse_samples < length)]] = 1.0
    return _finish(data, fs)


def linear_chirp(duration: float, fs: float, f0: float, f1: float, phase: float = 0.0) -> Sound:
    """Linear frequency sweep from ``f0`` to ``f1``."""
    t = time_axis(n_samples(duration, fs), fs)
    return _finish(chirp(t, f0, duration, f1, "linear", phi=np.degrees(phase)), fs)


def exponential_chirp(duration: float, fs: float, f0: float, f1: float, phase: float = 0.0) -> Sound:
    """Exponential (log-frequency) sweep from ``f0`` to ``f1``."""
    t = time_axis(n_samples(duration, fs), fs)
    return _finish(chirp(t, f0, duration, f1, "logarithmic", phi=np.degrees(phase)), fs)


# ------------------------------------------------------------------- noises
SpectrumLike = Callable[[np.ndarray], np.ndarray] | tuple[ArrayLike, ArrayLike]


def _spectral_gain(
    freqs: np.ndarray,
    band: tuple[float, float] | None,
    tilt: float,
    spectrum,
    tilt_ref: float,
) -> np.ndarray:
    gain = np.ones_like(freqs)
    if band is not None:
        f_lo, f_hi = band
        gain[(freqs < f_lo) | (freqs > f_hi)] = 0.0
    if tilt:
        with np.errstate(divide="ignore"):
            gain *= np.where(freqs > 0, (freqs / tilt_ref) ** (tilt / (20 * np.log10(2))), 0.0)
    if spectrum is not None:
        if hasattr(spectrum, "level_at"):  # a sonore Spectrum
            level_db = spectrum.level_at(freqs)
        elif callable(spectrum):
            level_db = spectrum(freqs)
        else:
            table_freqs, table_db = map(np.asarray, spectrum)
            level_db = np.interp(freqs, table_freqs, table_db)
        gain *= 10 ** (np.asarray(level_db) / 20)
    return gain


def gaussian_noise(
    duration: float,
    fs: float,
    band: tuple[float, float] | None = None,
    tilt: float = 0.0,
    spectrum: SpectrumLike | None = None,
    n_channels: int = 1,
    rng: RNG = None,
    tilt_ref: float = 1000.0,
) -> Sound:
    """Gaussian noise, optionally spectrally shaped.

    Parameters
    ----------
    band
        ``(f_lo, f_hi)`` passband in Hz (brick-wall).
    tilt
        Spectral slope in dB/octave (``-3`` = pink, ``-6`` = brown).
    spectrum
        Target spectrum level in dB: a :class:`~sonore.Spectrum`, a
        function ``f -> dB``, or a ``(freqs, dB)`` pair to interpolate.
    n_channels
        Independent noise in each channel.
    """
    rng = as_rng(rng)
    length = n_samples(duration, fs)
    freqs = np.fft.rfftfreq(length, 1 / fs)
    gain = _spectral_gain(freqs, band, tilt, spectrum, tilt_ref)
    noise_spectrum = np.fft.rfft(rng.standard_normal((length, n_channels)), axis=0) * gain[:, None]
    data = np.fft.irfft(noise_spectrum, n=length, axis=0)
    data -= data.mean(axis=0)
    return _finish(data, fs)


def correlated_noise(duration: float, fs: float, corr: float = 1.0, rng: RNG = None, **noise_kwargs) -> Sound:
    """Two-channel noise with interaural correlation ``corr`` (in [-1, 1]).
    Extra keyword arguments go to :func:`gaussian_noise`."""
    if not -1 <= corr <= 1:
        raise ValueError("corr must be in [-1, 1]")
    noise = gaussian_noise(duration, fs, n_channels=2, rng=rng, **noise_kwargs).data
    shared_weight, difference_weight = np.sqrt((1 + corr) / 2), np.sqrt((1 - corr) / 2)
    left = shared_weight * noise[:, 0] + difference_weight * noise[:, 1]
    right = shared_weight * noise[:, 0] - difference_weight * noise[:, 1]
    return Sound(np.column_stack([left, right]), fs)


def iterated_ripple_noise(
    duration: float,
    fs: float,
    delay: float,
    gain: float = 1.0,
    iterations: int = 16,
    network: str = "add-same",
    rng: RNG = None,
    **noise_kwargs,
) -> Sound:
    """Iterated rippled noise (Yost, 1996). Pitch is at ``1/delay``.

    ``network="add-same"`` (IRNS) iterates ``y <- y + g*delay(y)``;
    ``"add-original"`` (IRNO) iterates ``y <- x + g*delay(y)``. Implemented
    exactly in the frequency domain (fractional delays are fine), with a
    warm-up segment discarded so the output is stationary.
    """
    warmup = int(np.ceil(iterations * delay * fs))
    length = n_samples(duration, fs)
    noise = gaussian_noise((length + warmup) / fs, fs, rng=rng, **noise_kwargs).data[:, 0]
    freqs = np.fft.rfftfreq(len(noise), 1 / fs)
    one_pass = gain * np.exp(-2j * np.pi * freqs * delay)
    if network == "add-same":
        transfer = (1 + one_pass) ** iterations
    elif network == "add-original":
        transfer = sum(one_pass**i for i in range(iterations + 1))
    else:
        raise ValueError("network must be 'add-same' or 'add-original'")
    output = np.fft.irfft(np.fft.rfft(noise) * transfer, n=len(noise))[warmup:]
    return _finish(output, fs)
