"""WORLD's synthesis (Morise, Yokomori & Ozawa, 2016), ported exactly: a
sound from an F0 track, a spectral envelope and an aperiodicity, as made by
:func:`~sonore.analysis.vocoder.cheaptrick` and
:func:`~sonore.analysis.vocoder.d4c` (or
:func:`~sonore.analysis.vocoder.harmonic_aperiodicity`)."""

from __future__ import annotations

import numpy as np

from sonore.analysis.vocoder import (
    _DEFAULT_F0,
    _SAFEGUARD,
    Aperiodicity,
    SpectralEnvelope,
    _Stream,
    _time_windows,
)
from sonore.analysis.voice import _contour_on_grid
from sonore.core.sound import Sound

__all__ = ["world_synthesize"]


class _GaussianNoise:
    """Fresh noise in place of WORLD's stream, from a seed or Generator."""

    def __init__(self, rng):
        self.rng = np.random.default_rng(rng)

    def draw(self, n: int) -> np.ndarray:
        return self.rng.standard_normal(n)


def world_synthesize(f0, envelope, aperiodicity: Aperiodicity, *, rng=None) -> Sound:
    """WORLD's synthesis, ported exactly.

    Pulses are placed where the running phase of the F0 track (linearly
    interpolated to every sample) crosses a multiple of 2 pi, the fraction
    of a sample kept as a linear phase shift; unvoiced stretches get a pulse
    every 1/500 s. At each pulse the envelope ``S`` and the amplitude ratio
    ``A`` are interpolated between time windows. The periodic part is the
    minimum-phase response of ``S (1 - A^2)``, scaled by the square root of
    the interval to the next pulse, with its DC removed; the aperiodic part
    is noise as long as that interval through the minimum-phase response of
    ``S A^2`` (of ``S`` where unvoiced). The responses are overlap-added.

    Minimum phase is an approximation: the waveform within each period is
    not the original's, which WORLD's authors note is audible at low F0.

    Parameters
    ----------
    f0
        An F0 track (an :class:`~sonore.analysis.f0.F0Track` or a ``(times,
        f0)`` pair), usually the one the envelope and aperiodicity were
        measured with, from any tracker. It is used as it is when its time
        windows are the aperiodicity's; otherwise it is read onto them
        (voiced where the nearest time window is voiced, linear between
        voiced values). Those time windows must be evenly spaced from time
        0, as WORLD assumes; the spacing is the hop (WORLD's frame period).
        F0 can be changed before synthesis
        (:func:`~sonore.analysis.voice.scale_f0`, a pitch change); the
        envelope stays, so the formants stay.
    envelope
        A :class:`~sonore.analysis.vocoder.SpectralEnvelope` on the
        aperiodicity's time windows and frequencies, used as it is; or any
        other envelope read as ``envelope(t, f)`` (power, shape
        ``(n_channels, len(f), len(t))``), such as a
        :class:`~sonore.analysis.voice.GridEnvelope` from the cepstrum or
        MFCCs, or one moved by :func:`~sonore.analysis.voice.warp_frequency`,
        which is read at the aperiodicity's time windows and frequencies
        first. WORLD's synthesis computes a minimum-phase response on its
        own FFT length, which suits a smooth envelope like CheapTrick's; an
        envelope with deep, narrow valleys (tens of dB) comes out a few dB
        off at the harmonics, so smooth such an envelope first, or give it
        to :func:`~sonore.signals.generators.harmonic_complex` as its
        ``amplitudes``, which reads the envelope at each harmonic exactly.
    aperiodicity
        An :class:`~sonore.analysis.vocoder.Aperiodicity`, which sets the
        time windows and frequencies.
    rng
        ``None`` (default): WORLD's own noise stream, restarted at every
        call, so the same inputs give WORLD's output, sample for sample.
        A seed or ``numpy.random.Generator``: fresh Gaussian noise instead,
        for independent tokens (listed in ``DIFFERENCES_FROM_WORLD``).

    Returns
    -------
    Sound
        ``int(n_windows * hop * fs)`` samples, one channel per
        channel of the envelope.
    """
    envelope = _on_grid_of(envelope, aperiodicity)
    fs = int(envelope.fs)
    n_channels = envelope.data.shape[0]
    times, f0_values = _time_windows(Sound(np.zeros((1, n_channels)), fs), f0)
    grid = aperiodicity.t
    if len(times) != len(grid) or not np.allclose(times, grid, rtol=0, atol=1e-9):
        f0_values = _contour_on_grid(times, f0_values, grid)
        times = grid
    if len(times) < 2:
        raise ValueError("WORLD's synthesis needs at least two time windows")
    hop_ms = round((times[1] - times[0]) * 1000, 9)
    if times[0] != 0 or not np.allclose(np.diff(times), hop_ms / 1000, rtol=0, atol=1e-9):
        raise ValueError("f0 time windows must be evenly spaced from time 0")
    # WORLD takes the hop (its frame period) in ms, as pyworld does
    hop = hop_ms / 1000.0
    n_samples = int(len(times) * hop_ms * fs / 1000)
    channels = [
        _synthesize_channel(
            f0_values[channel],
            envelope.data[channel].T,
            aperiodicity.data[channel].T,
            hop,
            fs,
            n_samples,
            _Stream() if rng is None else _GaussianNoise(rng),
        )
        for channel in range(n_channels)
    ]
    return Sound(np.column_stack(channels), fs)


def _on_grid_of(envelope, aperiodicity: Aperiodicity) -> SpectralEnvelope:
    """The envelope as a SpectralEnvelope on the aperiodicity's time windows
    and frequencies: itself if it already is one, otherwise read there."""
    if (
        isinstance(envelope, SpectralEnvelope)
        and envelope.data.shape == aperiodicity.data.shape
        and envelope.fs == aperiodicity.fs
        and np.array_equal(envelope.t, aperiodicity.t)
    ):
        return envelope
    if not callable(envelope):
        raise TypeError("envelope must be a SpectralEnvelope or be read as envelope(t, f)")
    power = np.asarray(envelope(aperiodicity.t, aperiodicity.f), dtype=float)
    n_channels = aperiodicity.data.shape[0]
    expected = aperiodicity.data.shape[1:]
    if power.ndim != 3 or power.shape[1:] != expected or power.shape[0] not in (1, n_channels):
        raise ValueError(
            f"envelope(t, f) gave shape {power.shape}; expected (n_channels, {expected[0]}, {expected[1]})"
        )
    if power.shape[0] != n_channels:
        power = np.repeat(power, n_channels, axis=0)
    return SpectralEnvelope(power, aperiodicity.t, aperiodicity.fs, q1=float("nan"))


def _minimum_phase(log_amplitude: np.ndarray, n_fft: int) -> np.ndarray:
    """WORLD's GetMinimumPhaseSpectrum: half-spectrum log amplitudes in,
    the minimum-phase spectrum on ``n_fft // 2 + 1`` bins out."""
    cepstrum = np.fft.rfft(np.concatenate([log_amplitude, log_amplitude[-2:0:-1]]))
    folded = np.zeros(n_fft, dtype=complex)
    folded[0] = np.conj(cepstrum[0])
    folded[1 : n_fft // 2] = 2 * np.conj(cepstrum[1 : n_fft // 2])
    folded[n_fft // 2] = np.conj(cepstrum[n_fft // 2])
    return np.exp(np.fft.fft(folded)[: n_fft // 2 + 1] / n_fft)


def _inverse_centred(spectrum: np.ndarray, n_fft: int) -> np.ndarray:
    """FFTW's unnormalized complex-to-real inverse, then fftshift."""
    waveform = np.fft.irfft(spectrum, n_fft) * n_fft
    return np.concatenate([waveform[n_fft // 2 :], waveform[: n_fft // 2]])


def _synthesize_channel(f0, spectrogram, ratio_windows, hop, fs, n_samples, noise):
    """WORLD's Synthesis on one channel; spectrogram and ratio_windows are
    ``(n_windows, n_freqs)``."""
    n_windows, n_half = spectrogram.shape
    n_fft = 2 * (n_half - 1)
    lowest_f0 = fs // n_fft + 1.0
    # F0 and voicing at every sample, the last time window extrapolated one step
    coarse_times = np.arange(n_windows + 1) * hop
    coarse_f0 = np.where(f0 < lowest_f0, 0.0, f0)
    coarse_voicing = (coarse_f0 != 0).astype(float)
    coarse_f0 = np.append(coarse_f0, 2 * coarse_f0[-1] - coarse_f0[-2])
    coarse_voicing = np.append(coarse_voicing, 2 * coarse_voicing[-1] - coarse_voicing[-2])
    sample_times = np.arange(n_samples) / fs
    sample_f0 = np.interp(sample_times, coarse_times, coarse_f0)
    voicing = (np.interp(sample_times, coarse_times, coarse_voicing) > 0.5).astype(float)
    sample_f0 = np.where(voicing == 0, _DEFAULT_F0, sample_f0)
    # pulses where the wrapped phase jumps
    total_phase = np.cumsum(2 * np.pi * sample_f0 / fs)
    wrapped = np.fmod(total_phase, 2 * np.pi)
    pulses = np.flatnonzero(np.abs(np.diff(wrapped)) > np.pi)
    before = wrapped[pulses] - 2 * np.pi
    time_shifts = -before / (wrapped[pulses + 1] - before) / fs
    # WORLD's DC remover: a Hann-shaped correction summing to one half
    half_index = np.arange(n_fft // 2)
    dc_remover = 0.5 - 0.5 * np.cos(2 * np.pi * (half_index + 1.0) / (1.0 + n_fft))
    dc_remover = np.concatenate([dc_remover, dc_remover[::-1]])
    dc_remover /= 2 * dc_remover[: n_fft // 2].sum()
    safe_ratios = np.clip(ratio_windows, 0.001, 0.999999999999)
    bins = np.arange(n_half)
    output = np.zeros(n_samples)
    for pulse, sample in enumerate(pulses):
        noise_size = pulses[min(len(pulses) - 1, pulse + 1)] - sample
        window_position = sample_times[sample] / hop
        lower = min(n_windows - 1, int(np.floor(window_position)))
        upper = min(n_windows - 1, int(np.ceil(window_position)))
        weight = window_position - lower
        if lower == upper:
            envelope = np.abs(spectrogram[lower])
            noise_share = safe_ratios[lower] ** 2
        else:
            envelope = (1 - weight) * np.abs(spectrogram[lower]) + weight * np.abs(spectrogram[upper])
            noise_share = ((1 - weight) * safe_ratios[lower] + weight * safe_ratios[upper]) ** 2
        # periodic part
        if voicing[sample] <= 0.5 or noise_share[0] > 0.999:
            periodic = np.zeros(n_fft)
        else:
            spectrum = _minimum_phase(np.log(envelope * (1 - noise_share) + _SAFEGUARD) / 2, n_fft)
            shift_cos = np.cos(2 * np.pi * time_shifts[pulse] * fs / n_fft * bins)
            shift_sin = np.sqrt(1 - shift_cos**2)
            spectrum = (spectrum.real * shift_cos + spectrum.imag * shift_sin) + 1j * (
                spectrum.imag * shift_cos - spectrum.real * shift_sin
            )
            periodic = _inverse_centred(spectrum, n_fft)
            dc = periodic[n_fft // 2 :].sum()
            # WORLD sets the first half (before the pulse) to the correction
            # alone rather than subtracting it
            periodic = np.concatenate([np.zeros(n_fft // 2), periodic[n_fft // 2 :]]) - dc * dc_remover
        # aperiodic part
        noise_segment = np.zeros(n_fft)
        if noise_size:
            values = noise.draw(noise_size)
            noise_segment[:noise_size] = values - values.mean()
        if voicing[sample] != 0:
            log_amplitude = np.log(envelope * noise_share) / 2
        else:
            log_amplitude = np.log(envelope) / 2
        aperiodic = _inverse_centred(_minimum_phase(log_amplitude, n_fft) * np.fft.rfft(noise_segment), n_fft)
        response = (periodic * np.sqrt(noise_size) + aperiodic) / n_fft
        offset = sample - n_fft // 2 + 1
        first, last = max(0, -offset), min(n_fft, n_samples - offset)
        output[offset + first : offset + last] += response[first:last]
    return output
