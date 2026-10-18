"""A voice rebuilt from a source-filter description: an F0 track, a spectral
envelope and an aperiodicity, as the views measure them.

The sound is harmonics plus noise. The harmonics follow the F0 track and carry
the part of the envelope that is not noise, ``S (1 - a)``, with ``S`` the
envelope's power and ``a`` the aperiodicity's noise share; white noise, shaped
over time and frequency, carries the rest, ``S a``. The checks behind it are in
``tools/check_voice_recipe_claims.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import RNG, as_rng
from sonore.frames.gabor import STFT
from sonore.sources.waveforms import ContourLike, _contour, gaussian_noise, harmonic_complex
from sonore.views.spectral_envelope import GridEnvelope

__all__ = ["harmonics_and_noise"]


_FADED_FROM = 0.9  # harmonic_complex fades its harmonics out from 0.9 f_max to f_max


def harmonics_and_noise(
    f0: ContourLike,
    envelope: Any,
    aperiodicity: Any,
    *,
    duration: float | None = None,
    rng: RNG = None,
    win_dur: float = 0.020,
) -> Sound:
    """A voice from an F0 track, a spectral envelope and an aperiodicity.

    On the aperiodicity's grid of times and frequencies, with ``S`` the
    envelope's power and ``a`` the noise share:

    1. **Harmonics.** :func:`~sonore.sources.waveforms.harmonic_complex` on the
       F0 track, harmonic ``n`` carrying the power ``S (1 - a)`` over one
       harmonic spacing: an envelope reads unit white noise as 1, so a power
       density ``S`` with a harmonic every F0 is an amplitude ``a_n`` with
       ``a_n^2 / 2 = 2 F0 S / fs``. ``harmonic_complex`` returns RMS 1, as
       every generator does, so its output is scaled back to that power.
    2. **Noise.** White noise of variance 1, which the envelope reads as 1,
       its STFT (``win_dur`` long, on a 5 ms hop) multiplied by
       ``sqrt(S a)`` and inverted. Unvoiced time windows have ``a = 1``, so
       there the noise carries all of ``S``.
    3. The two added.

    The harmonics start in cosine phase and keep their phases as the pitch
    moves, so each period's waveform is a pulse, peakier than a voice's.

    Parameters
    ----------
    f0
        An F0 track (an :class:`~sonore.views.f0.F0Track` or a ``(times, f0)``
        pair, one row per channel or one for all), usually the one the
        envelope and aperiodicity were measured with. It may be changed
        first (:func:`~sonore.views.f0.scale_f0`, a pitch change); the
        envelope stays, so the formants stay.
    envelope
        A :class:`~sonore.views.spectral_envelope.SpectralEnvelope`, or any
        envelope read as ``envelope(t, f)`` (power, shape ``(n_channels,
        len(f), len(t))``), such as a
        :class:`~sonore.views.spectral_envelope.GridEnvelope` or one moved by
        :func:`~sonore.views.spectral_envelope.warp_frequency`. It is read at
        the aperiodicity's times and frequencies.
    aperiodicity
        An :class:`~sonore.views.aperiodicity.Aperiodicity`, or anything with
        its grid: ``.t``, ``.f``, ``.fs`` and ``.data`` (the amplitude ratio,
        whose square is the noise share). It sets the grid and the sampling
        rate.
    duration
        In seconds; by default one hop past the aperiodicity's last time
        window.
    rng
        A seed or Generator for the noise.
    win_dur
        The length of the STFT that shapes the noise, in seconds.
    """
    fs = float(aperiodicity.fs)
    times, freqs = np.asarray(aperiodicity.t, dtype=float), np.asarray(aperiodicity.f, dtype=float)
    hop = times[1] - times[0] if len(times) > 1 else 0.005
    if duration is None:
        duration = times[-1] + hop
    share = np.asarray(aperiodicity.data, dtype=float) ** 2
    power = _on_grid_of(envelope, times, freqs, share.shape)
    contour = _contour(f0)
    if contour is None:
        raise TypeError("f0 must be an F0 track: a (times, values) pair or an object with .t and .f0")
    track_times, track_values = contour
    n_channels = share.shape[0]
    if len(track_values) == 1:
        track_values = np.repeat(track_values, n_channels, axis=0)
    if len(track_values) != n_channels:
        raise ValueError(f"f0 has {len(track_values)} channels, the aperiodicity {n_channels}")
    generator = as_rng(rng)
    channels = []
    for channel in range(n_channels):
        track = (track_times, track_values[channel])
        harmonic_power = GridEnvelope(power[channel] * (1 - share[channel]), times, freqs)
        harmonics = _harmonics(duration, fs, track, harmonic_power)
        noise_power = GridEnvelope(power[channel] * share[channel], times, freqs)
        channels.append(
            harmonics + _shaped_noise(duration, fs, noise_power, generator, win_dur, hop)[: len(harmonics)]
        )
    return Sound(np.column_stack(channels), fs)


def _harmonics(duration, fs, track, harmonic_power):
    """Step 1: the harmonics, scaled to the power their amplitudes ask for."""

    def amplitudes(t, f, number):
        return np.sqrt(4 * (f / number) / fs) * harmonic_power.amplitude(t, f)

    sound = harmonic_complex(duration, fs, track, amplitudes=amplitudes)
    # the power asked for, sum_n a_n^2 / 2, at each time window of the track (faded as
    # harmonic_complex fades the top harmonics), averaged over the sound with unvoiced
    # time windows counting 0; harmonic_complex returned RMS 1
    track_times, track_f0 = (np.asarray(values, dtype=float) for values in track)
    f_max = 0.45 * fs
    asked = np.zeros(len(track_times))
    for index, (time, f0) in enumerate(zip(track_times, track_f0, strict=True)):
        if f0 > 0:
            numbers = np.arange(1, int(f_max / f0) + 1)
            harmonic_freqs = numbers * f0
            position = (harmonic_freqs - _FADED_FROM * f_max) / ((1 - _FADED_FROM) * f_max)
            fade = np.cos(np.pi / 2 * np.clip(position, 0, 1)) ** 2
            levels = amplitudes(np.full(len(numbers), time), harmonic_freqs, numbers)
            asked[index] = np.sum(levels**2 / 2 * fade**2)
    sample_times = np.arange(sound.n_samples) / fs
    return sound.data[:, 0] * np.sqrt(np.mean(np.interp(sample_times, track_times, asked)))


def _shaped_noise(duration, fs, noise_power, generator, win_dur, hop):
    """Step 2: white noise of variance 1 (which an envelope reads as 1), shaped to
    ``noise_power`` by its STFT."""
    noise = STFT(gaussian_noise(duration, fs, rng=generator), win_dur=win_dur, hop_dur=hop)
    noise.data = noise.data * np.sqrt(noise_power(noise.t, noise.f))
    return noise.to_sound().data[:, 0]


def _on_grid_of(envelope, times, freqs, shape) -> np.ndarray:
    """The envelope's power at the aperiodicity's times and frequencies, shape
    ``(n_channels, n_freqs, n_windows)``: its own data if it is held on that grid
    already (an envelope measured with the same track), otherwise read there as
    ``envelope(t, f)``."""
    held = getattr(envelope, "data", None)
    if (
        held is not None
        and np.shape(held) == shape
        and np.array_equal(getattr(envelope, "t", None), times)
        and np.array_equal(getattr(envelope, "f", None), freqs)
    ):
        return np.asarray(held, dtype=float)
    if not callable(envelope):
        raise TypeError("envelope must be a SpectralEnvelope or be read as envelope(t, f)")
    power = np.asarray(envelope(times, freqs), dtype=float)
    n_channels = shape[0]
    if power.ndim != 3 or power.shape[1:] != shape[1:] or power.shape[0] not in (1, n_channels):
        raise ValueError(
            f"envelope(t, f) gave shape {power.shape}; expected (n_channels, {shape[1]}, {shape[2]})"
        )
    return np.repeat(power, n_channels, axis=0) if power.shape[0] != n_channels else power
