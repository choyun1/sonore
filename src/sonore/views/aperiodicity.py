"""Aperiodicity, the share of noise in a voice at each frequency: WORLD's D4C
(Morise, 2016), ported exactly, and a harmonic-residual aperiodicity that
measures the share of noise directly.

Every step named after WORLD reproduces WORLD's C++ code
(github.com/mmorise/World, commit d625e76) to floating-point precision, with
WORLD's own noise in WORLD's order; options that depart from WORLD are listed
in :data:`~sonore.views.world.DIFFERENCES_FROM_WORLD`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import db_to_amp, time_axis
from sonore.views.spectral_envelope import (
    SpectralEnvelope,
    _dc_correction,
    _FrequencyView,
    _linear_smoothing,
    _windowed_waveform,
)
from sonore.views.world import (
    _FLOOR_F0,
    _SAFEGUARD,
    _integer_fs,
    _matlab_round,
    _periods_fft_size,
    _Stream,
    _time_windows,
    world_fft_size,
)

__all__ = ["Aperiodicity", "d4c", "harmonic_aperiodicity"]


_FLOOR_F0_D4C = 47.0


_D4C_BAND_SPACING = 3000.0


_D4C_UPPER_LIMIT = 15000.0


_D4C_SAFEGUARD = 1e-6


class Aperiodicity(_FrequencyView):
    """An aperiodicity, from :func:`d4c` or :func:`harmonic_aperiodicity`:
    per frequency and time window, how much of the power is noise rather than
    harmonics. Stored as WORLD stores it, an amplitude ratio between 0 and
    1 (``data``, shape ``(n_channels, n_freqs, n_windows)``): its square
    :attr:`share` is the share of the power that is noise, which is what
    :func:`~sonore.views.world.world_synthesize` uses. ``method`` names
    the measure that made it ("D4C" or "harmonic residual").
    """

    discards = (
        "Aperiodicity keeps only the share of noise in each frequency and time window: it discards the "
        "spectrum, the pitch and the phase."
    )
    back_to_sound = (
        "so.world_synthesize rebuilds an approximation of the voice from it together with an F0 track and "
        "a spectral envelope."
    )

    def __init__(self, data: np.ndarray, t: np.ndarray, fs: float, method: str):
        super().__init__(data, t, fs)
        self.method = method

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        return f"Aperiodicity({self.method}, {n_freqs} freqs x {n_windows} time windows, {n_channels} ch)"

    @property
    def share(self) -> np.ndarray:
        """The noise share of the power, ``data ** 2``."""
        return self.data**2

    @property
    def db(self) -> np.ndarray:
        """The noise share in dB, ``20 log10 data``."""
        return 20 * np.log10(self.data)

    def bands(self, edges: Sequence[float], envelope: SpectralEnvelope | None = None) -> np.ndarray:
        """The noise share averaged over each band ``[edges[i], edges[i+1])``,
        shape ``(n_channels, n_bands, n_windows)``. With an ``envelope``, the
        average is weighted by its power, so the result is the band's noise
        power over its total power."""
        freqs = self.f
        weights = np.ones_like(self.data) if envelope is None else envelope.data
        out = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            in_band = (freqs >= lower) & (freqs < upper)
            band_weights = weights[:, in_band]
            out.append(np.sum(self.share[:, in_band] * band_weights, axis=1) / np.sum(band_weights, axis=1))
        return np.stack(out, axis=1)

    def plot(self, ax=None, channel: int = 0, db_range: float = 60.0, fmax: float | None = None):
        """The noise share in dB as a time-frequency image (0 dB is all noise)."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(
            self.db[channel],
            self.t,
            self.f,
            ax=ax,
            db_range=db_range,
            fmax=fmax,
            title=f"Aperiodicity ({self.method})",
        )


# ------------------------------------------------------------------- D4C
def d4c(sound: Sound, f0, *, threshold: float = 0.85, f0_floor: float = _FLOOR_F0) -> Aperiodicity:
    """WORLD's D4C aperiodicity (Morise, 2016), ported exactly.

    At each voiced time window: a "static group delay" from two Blackman-windowed
    spectra four periods long, a quarter period either side of the window center,
    divided by a smoothed power spectrum, smoothed over F0 / 2, less itself
    smoothed over F0. Around each multiple of 3 kHz up to
    ``min(15 kHz, fs / 2 - 3 kHz)``, a Nuttall window over 3 kHz of that
    group delay is transformed, its power sorted, and the band's
    aperiodicity is the share outside the largest bins, in dB, plus
    ``(F0 - 100) / 50`` dB, capped at 0. The curve is then linear in dB
    from -60 dB at 0 Hz through those values to 0 dB at Nyquist. At
    16 kHz that is one measured value per time window, at 3 kHz.

    A time window is left fully aperiodic (amplitude ratio ``1 - 1e-12``) where
    F0 is 0, or where less than ``threshold`` of its power between 100 Hz
    and 7.9 kHz lies below 4 kHz (WORLD's "LoveTrain" test; ``threshold=0``
    keeps every voiced time window voiced).

    D4C was tuned so that WORLD's resynthesis sounds natural; it is robust
    to F0 errors but does not report the share of noise below 3 kHz, which
    the -60 dB anchor sets. :func:`harmonic_aperiodicity` measures that
    share directly.

    Parameters
    ----------
    sound, f0
        As :func:`cheaptrick`.
    threshold
        The LoveTrain voicing threshold.
    f0_floor
        Sets the output's frequency grid, which must match the envelope's
        for synthesis (WORLD's 71 Hz, as :func:`cheaptrick`).
    """
    fs = _integer_fs(sound)
    times, f0_values = _time_windows(sound, f0)
    n_fft_out = world_fft_size(fs, f0_floor)
    n_fft = _periods_fft_size(fs, 4.0, _FLOOR_F0_D4C)
    n_bands = int(min(_D4C_UPPER_LIMIT, fs / 2.0 - _D4C_BAND_SPACING) / _D4C_BAND_SPACING)
    window_length = int(_D4C_BAND_SPACING * n_fft / fs) * 2 + 1
    position = np.arange(window_length) / (window_length - 1.0)
    nuttall = (
        0.355768
        - 0.487396 * np.cos(2 * np.pi * position)
        + 0.144232 * np.cos(4 * np.pi * position)
        - 0.012604 * np.cos(6 * np.pi * position)
    )
    coarse_freqs = np.append(np.arange(n_bands + 1) * _D4C_BAND_SPACING, fs / 2.0)
    freqs = np.arange(n_fft_out // 2 + 1) * fs / n_fft_out
    data = np.full((sound.n_channels, n_fft_out // 2 + 1, len(times)), 1.0 - _SAFEGUARD)
    for channel in range(sound.n_channels):
        samples = sound.data[:, channel]
        noise = _Stream()
        voiced = [
            window_f0 != 0 and _love_train(samples, fs, time, window_f0, noise) > threshold
            for time, window_f0 in zip(times, f0_values[channel], strict=True)
        ]
        for window_index, (time, window_f0) in enumerate(zip(times, f0_values[channel], strict=True)):
            if not voiced[window_index]:
                continue
            bands = _d4c_bands(
                samples, fs, time, max(_FLOOR_F0_D4C, window_f0), n_fft, n_bands, nuttall, noise
            )
            coarse = np.concatenate([[-60.0], bands, [-_SAFEGUARD]])
            data[channel, :, window_index] = db_to_amp(np.interp(freqs, coarse_freqs, coarse))
    return Aperiodicity(data, times, fs, "D4C")


def _love_train(samples, fs, time, f0, noise):
    """The share of power (100 Hz to 7.9 kHz) below 4 kHz."""
    f0 = max(f0, 40.0)
    n_fft = _periods_fft_size(fs, 3.0, 40.0)
    segment = _windowed_waveform(samples, fs, f0, time, "blackman", 3.0, noise, _D4C_SAFEGUARD)
    power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
    lowest, middle, highest = (int(np.ceil(edge * n_fft / fs)) for edge in (100.0, 4000.0, 7900.0))
    power[: lowest + 1] = 0
    cumulative = np.cumsum(power)
    return cumulative[middle] / cumulative[highest]


def _centroid(samples, fs, time, f0, n_fft, noise):
    segment = _windowed_waveform(samples, fs, f0, time, "blackman", 4.0, noise, _D4C_SAFEGUARD)
    buffer = np.zeros(n_fft)
    buffer[: len(segment)] = segment
    n_normalized = _matlab_round(2.0 * fs / f0) * 2 + 1
    buffer /= np.sqrt(np.sum(buffer[:n_normalized] ** 2))
    spectrum = np.fft.rfft(buffer)
    weighted = np.fft.rfft(buffer * (np.arange(n_fft) + 1.0))
    return weighted.real * spectrum.real + spectrum.imag * weighted.imag


def _d4c_bands(samples, fs, time, f0, n_fft, n_bands, nuttall, noise):
    """D4C's coarse aperiodicities [dB], one per multiple of 3 kHz."""
    centroid = _centroid(samples, fs, time - 0.25 / f0, f0, n_fft, noise)
    centroid = centroid + _centroid(samples, fs, time + 0.25 / f0, f0, n_fft, noise)
    centroid = _dc_correction(centroid, f0, fs, n_fft)
    segment = _windowed_waveform(samples, fs, f0, time, "hann", 4.0, noise, _D4C_SAFEGUARD)
    power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
    power = _linear_smoothing(_dc_correction(power, f0, fs, n_fft), f0, fs, n_fft)
    # WORLD divides without a guard; where the power is 0 the C code gets inf or NaN as here
    with np.errstate(divide="ignore", invalid="ignore"):
        group_delay = _linear_smoothing(centroid / power, f0 / 2.0, fs, n_fft)
        group_delay = group_delay - _linear_smoothing(group_delay, f0, fs, n_fft)
    window_length = len(nuttall)
    boundary = _matlab_round(n_fft * 8.0 / window_length)
    bands = np.empty(n_bands)
    for band in range(n_bands):
        center = int(_D4C_BAND_SPACING * (band + 1) * n_fft / fs)
        start = center - window_length // 2
        buffer = np.zeros(n_fft)
        buffer[:window_length] = group_delay[start : start + window_length] * nuttall
        cumulative = np.cumsum(np.sort(np.abs(np.fft.rfft(buffer)) ** 2))
        bands[band] = 10 * np.log10(cumulative[n_fft // 2 - boundary - 1] / cumulative[n_fft // 2])
    return np.minimum(0.0, bands + (f0 - 100) / 50)


# ------------------------------------------------- harmonic residual
def harmonic_aperiodicity(
    sound: Sound, f0, *, periods: float = 4.0, f0_floor: float = _FLOOR_F0, cell_harmonics: float = 2.0
) -> Aperiodicity:
    """The share of noise, measured by fitting the harmonics and keeping
    what is left. Not part of WORLD; :func:`d4c` is WORLD's measure.

    The F0 track is interpolated to every sample and integrated to a
    running phase ``Phi(t)``. At each voiced time window, under a Hann window
    ``periods`` periods long, a weighted least-squares fit of the harmonics
    ``cos(k Phi)``, ``sin(k Phi)`` below Nyquist, each with a linear change
    of amplitude across the window, is subtracted. The residual's power,
    over the windowed sound's power, is the noise share. The fit also
    absorbs a little of the noise near every harmonic; how much is known
    exactly from the fit (the share of white noise the residual keeps at
    each frequency), and is divided out. Both are summed over cells
    ``cell_harmonics`` harmonics wide, and the cells' shares are
    interpolated in dB onto the grid of :func:`cheaptrick`. Unvoiced time windows
    are all noise.

    The measure is what the word means, so it reads a known share of noise
    within a few tenths of a dB on a steady vowel; but it needs F0 to about
    0.1%, since a harmonic that drifts out of phase with the fit over the
    window reads as noise, and so do jitter and shimmer.

    Parameters
    ----------
    sound, f0
        As :func:`cheaptrick`.
    periods
        The window's length in periods of the time window's F0.
    f0_floor
        Sets the output's frequency grid, as :func:`cheaptrick`.
    cell_harmonics
        The width of the cells, in harmonics.
    """
    fs = _integer_fs(sound)
    times, f0_values = _time_windows(sound, f0)
    n_fft_out = world_fft_size(fs, f0_floor)
    freqs_out = np.arange(n_fft_out // 2 + 1) * fs / n_fft_out
    data = np.ones((sound.n_channels, n_fft_out // 2 + 1, len(times)))
    sample_times = time_axis(sound.n_samples, fs)
    for channel in range(sound.n_channels):
        voiced = f0_values[channel] > 0
        if not voiced.any():
            continue
        f0_per_sample = np.interp(sample_times, times[voiced], f0_values[channel, voiced])
        phase = np.cumsum(2 * np.pi * f0_per_sample / fs)
        longest = int(np.ceil(periods / 2 * fs / f0_values[channel, voiced].min())) + 1
        padded = np.pad(sound.data[:, channel], longest)
        padded_phase = np.pad(phase, longest, mode="edge")
        for window_index in np.flatnonzero(voiced):
            share, cell_freqs = _residual_share(
                padded,
                padded_phase,
                fs,
                times[window_index],
                f0_values[channel, window_index],
                periods,
                longest,
                cell_harmonics,
            )
            share_db = np.interp(freqs_out, cell_freqs, 10 * np.log10(share))
            data[channel, :, window_index] = db_to_amp(share_db)
    return Aperiodicity(data, times, fs, "harmonic residual")


def _residual_share(samples, phase, fs, time, f0, periods, padding, cell_harmonics):
    """One time window: the noise share per cell and the cells' center frequencies."""
    center = int(np.round(time * fs)) + padding
    half_length = int(np.round(periods / 2 * fs / f0))
    offsets = np.arange(-half_length, half_length + 1)
    window = 0.5 + 0.5 * np.cos(np.pi * offsets / (half_length + 1))
    harmonic_numbers = np.arange(1, int((fs / 2) / f0 - 1e-9) + 1)
    harmonic_phases = np.outer(phase[center + offsets], harmonic_numbers)
    ramp = (offsets / half_length)[:, None]
    columns = np.column_stack(
        [
            np.ones(len(offsets)),
            np.cos(harmonic_phases),
            np.sin(harmonic_phases),
            ramp * np.cos(harmonic_phases),
            ramp * np.sin(harmonic_phases),
        ]
    )
    root_window = np.sqrt(window)
    basis, _ = np.linalg.qr(columns * root_window[:, None])
    weighted = samples[center + offsets] * root_window
    residual = (weighted - basis @ (basis.T @ weighted)) * root_window
    # white noise e becomes (W - S B B' S) e in the windowed residual (S the
    # root window, B the fit's basis); its power at each frequency, over the
    # windowed noise's, is the share the residual keeps
    n_fft = int(2 ** np.ceil(np.log2(len(offsets))))
    residual_operator = np.diag(window) - (root_window[:, None] * basis) @ (basis.T * root_window[None, :])
    kept = np.sum(np.abs(np.fft.rfft(residual_operator, n_fft, axis=0)) ** 2, axis=1) / np.sum(window**2)
    signal_power = np.abs(np.fft.rfft(window * samples[center + offsets], n_fft)) ** 2
    residual_power = np.abs(np.fft.rfft(residual, n_fft)) ** 2
    freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    cell = (freqs // (cell_harmonics * f0)).astype(int)
    noise_power = np.bincount(cell, residual_power) / np.bincount(cell, kept) * np.bincount(cell)
    share = noise_power / np.maximum(np.bincount(cell, signal_power), np.finfo(float).tiny)
    share = np.clip(share, 1e-12, 1.0)
    return share, np.bincount(cell, freqs) / np.bincount(cell)
