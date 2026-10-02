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
    _frames,
    _Stream,
)
from sonore.core.sound import Sound

__all__ = ["world_synthesize"]


class _GaussianNoise:
    """Fresh noise in place of WORLD's stream, from a seed or Generator."""

    def __init__(self, rng):
        self.rng = np.random.default_rng(rng)

    def draw(self, n: int) -> np.ndarray:
        return self.rng.standard_normal(n)


def world_synthesize(f0, envelope: SpectralEnvelope, aperiodicity: Aperiodicity, *, rng=None) -> Sound:
    """WORLD's synthesis, ported exactly.

    Pulses are placed where the running phase of the F0 track (linearly
    interpolated to every sample) crosses a multiple of 2 pi, the fraction
    of a sample kept as a linear phase shift; unvoiced stretches get a pulse
    every 1/500 s. At each pulse the envelope ``S`` and the amplitude ratio
    ``A`` are interpolated between frames. The periodic part is the
    minimum-phase response of ``S (1 - A^2)``, scaled by the square root of
    the interval to the next pulse, with its DC removed; the aperiodic part
    is noise as long as that interval through the minimum-phase response of
    ``S A^2`` (of ``S`` where unvoiced). The responses are overlap-added.

    Minimum phase is an approximation: the waveform within each period is
    not the original's, which WORLD's authors note is audible at low F0.

    Parameters
    ----------
    f0
        The F0 track the envelope and aperiodicity were measured with (an
        :class:`~sonore.analysis.f0.F0Track` or a ``(times, f0)`` pair). Its
        frames must be evenly spaced from time 0, as WORLD assumes; the
        spacing is WORLD's frame period. F0 can be changed before synthesis
        (a pitch shift); the envelope and aperiodicity stay as measured.
    envelope, aperiodicity
        On the same frames and frequencies.
    rng
        ``None`` (default): WORLD's own noise stream, restarted at every
        call, so the same inputs give WORLD's output, sample for sample.
        A seed or ``numpy.random.Generator``: fresh Gaussian noise instead,
        for independent tokens (listed in ``DIFFERENCES_FROM_WORLD``).

    Returns
    -------
    Sound
        ``int(n_frames * frame_period * fs)`` samples, one channel per
        channel of the envelope.
    """
    if envelope.data.shape != aperiodicity.data.shape or envelope.fs != aperiodicity.fs:
        raise ValueError("the envelope and aperiodicity must share frames and frequencies")
    if not (np.array_equal(envelope.t, aperiodicity.t)):
        raise ValueError("the envelope and aperiodicity must share frame times")
    fs = int(envelope.fs)
    n_channels = envelope.data.shape[0]
    times, f0_values = _frames(Sound(np.zeros((1, n_channels)), fs), f0)
    if len(times) != len(envelope.t):
        raise ValueError(f"f0 has {len(times)} frames, the envelope {len(envelope.t)}")
    if len(times) < 2:
        raise ValueError("WORLD's synthesis needs at least two frames")
    frame_period_ms = round((times[1] - times[0]) * 1000, 9)
    if times[0] != 0 or not np.allclose(np.diff(times), frame_period_ms / 1000, rtol=0, atol=1e-9):
        raise ValueError("f0 frames must be evenly spaced from time 0")
    # WORLD takes the frame period in ms, as pyworld does
    frame_period = frame_period_ms / 1000.0
    n_samples = int(len(times) * frame_period_ms * fs / 1000)
    channels = [
        _synthesize_channel(
            f0_values[channel],
            envelope.data[channel].T,
            aperiodicity.data[channel].T,
            frame_period,
            fs,
            n_samples,
            _Stream() if rng is None else _GaussianNoise(rng),
        )
        for channel in range(n_channels)
    ]
    return Sound(np.column_stack(channels), fs)


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


def _synthesize_channel(f0, spectrogram, ratio_frames, frame_period, fs, n_samples, noise):
    """WORLD's Synthesis on one channel; spectrogram and ratio_frames are
    ``(n_frames, n_freqs)``."""
    n_frames, n_half = spectrogram.shape
    n_fft = 2 * (n_half - 1)
    lowest_f0 = fs // n_fft + 1.0
    # F0 and voicing at every sample, the last frame extrapolated one step
    coarse_times = np.arange(n_frames + 1) * frame_period
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
    safe_ratios = np.clip(ratio_frames, 0.001, 0.999999999999)
    bins = np.arange(n_half)
    output = np.zeros(n_samples)
    for pulse, sample in enumerate(pulses):
        noise_size = pulses[min(len(pulses) - 1, pulse + 1)] - sample
        frame_position = sample_times[sample] / frame_period
        lower = min(n_frames - 1, int(np.floor(frame_position)))
        upper = min(n_frames - 1, int(np.ceil(frame_position)))
        weight = frame_position - lower
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
