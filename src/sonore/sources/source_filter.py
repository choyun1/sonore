"""A voice rebuilt from a source-filter description: an F0 track, a spectral
envelope and, optionally, an aperiodicity, as the views measure them.

The sound is harmonics plus noise. The harmonics follow the F0 track and carry
the part of the envelope that is not noise, ``S (1 - a)``, with ``S`` the
envelope's power and ``a`` the aperiodicity's noise share; white noise, shaped
over time and frequency, carries the rest, ``S a``. Without an aperiodicity
voicing is all or nothing, the classic source-filter resynthesis: ``a = 0``
where the track is voiced and ``a = 1`` where it is not. The checks behind it
are in ``tools/check_voice_recipe_claims.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import RNG, as_rng
from sonore.frames.gabor import STFT
from sonore.sources.waveforms import ContourLike, _contour_per_channel, gaussian_noise, harmonic_complex
from sonore.views.spectral_envelope import GridEnvelope

__all__ = ["source_filter_synthesize"]


_FADED_FROM = 0.9  # harmonic_complex fades its harmonics out from 0.9 f_max to f_max


def source_filter_synthesize(
    f0: ContourLike,
    envelope: Any,
    aperiodicity: Any = None,
    *,
    fs: float | None = None,
    duration: float | None = None,
    rng: RNG = None,
    win_dur: float = 0.020,
) -> Sound:
    """A voice from an F0 track, a spectral envelope and an aperiodicity.

    On the aperiodicity's grid of times and frequencies (without one, the
    envelope's), with ``S`` the envelope's power and ``a`` the noise share:

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
        the aperiodicity's times and frequencies. Without an aperiodicity it
        must be held on a grid (``.t`` and ``.f``, as every envelope view is).
    aperiodicity
        An :class:`~sonore.views.aperiodicity.Aperiodicity`, or anything with
        its grid: ``.t``, ``.f``, ``.fs`` and ``.data`` (the amplitude ratio,
        whose square is the noise share). It sets the grid and the sampling
        rate. None (the default) is all-or-nothing voicing: no noise in the
        time windows the track calls voiced (the nearest given F0 above 0),
        all noise in the others.
    fs
        The sampling rate, needed only without an aperiodicity when the
        envelope has no ``.fs`` (a :class:`~sonore.views.spectral_envelope.GridEnvelope`).
    duration
        In seconds; by default one hop past the grid's last time window.
    rng
        A seed or Generator for the noise.
    win_dur
        The length of the STFT that shapes the noise, in seconds.
    """
    if aperiodicity is None:
        fs, times, freqs, n_channels = _envelope_grid(envelope, fs)
        track_times, track_values = _contour_per_channel(f0, n_channels, "the envelope")
        share = _all_or_nothing(track_times, track_values, times, len(freqs))
    else:
        if fs is not None and fs != aperiodicity.fs:
            raise ValueError(f"fs is {fs:g} Hz but the aperiodicity's is {aperiodicity.fs:g} Hz")
        fs = float(aperiodicity.fs)
        times, freqs = np.asarray(aperiodicity.t, dtype=float), np.asarray(aperiodicity.f, dtype=float)
        share = np.asarray(aperiodicity.data, dtype=float) ** 2
        track_times, track_values = _contour_per_channel(f0, share.shape[0], "the aperiodicity")
    hop = times[1] - times[0] if len(times) > 1 else 0.005
    if duration is None:
        duration = times[-1] + hop
    power = _on_grid_of(envelope, times, freqs, share.shape)
    n_channels = share.shape[0]
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


def _envelope_grid(envelope, fs):
    """The sampling rate, times, frequencies and channel count of an envelope
    held on a grid, for a synthesis without an aperiodicity."""
    if not (hasattr(envelope, "t") and hasattr(envelope, "f") and hasattr(envelope, "data")):
        raise TypeError(
            "without an aperiodicity the envelope must be held on a grid (.t, .f and .data, as"
            " SpectralEnvelope and GridEnvelope are); pass an aperiodicity to read any other envelope"
        )
    own_fs = getattr(envelope, "fs", None)
    if fs is None:
        if own_fs is None:
            raise TypeError("the envelope has no .fs, so pass fs")
        fs = own_fs
    elif own_fs is not None and fs != own_fs:
        raise ValueError(f"fs is {fs:g} Hz but the envelope's is {own_fs:g} Hz")
    times, freqs = np.asarray(envelope.t, dtype=float), np.asarray(envelope.f, dtype=float)
    return float(fs), times, freqs, np.shape(envelope.data)[0]


def _all_or_nothing(track_times, track_values, times, n_freqs):
    """The noise share of all-or-nothing voicing, shape ``(n_channels, n_freqs,
    len(times))``: 0 where the nearest given F0 is above 0, 1 elsewhere."""
    upper = np.clip(np.searchsorted(track_times, times), 0, len(track_times) - 1)
    lower = np.clip(upper - 1, 0, len(track_times) - 1)
    nearest = np.where(np.abs(track_times[lower] - times) <= np.abs(track_times[upper] - times), lower, upper)
    unvoiced = (track_values[:, nearest] <= 0).astype(float)
    return np.repeat(unvoiced[:, None, :], n_freqs, axis=1)


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
        and hasattr(envelope, "t")
        and hasattr(envelope, "f")
        and np.array_equal(envelope.t, times)
        and np.array_equal(envelope.f, freqs)
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
