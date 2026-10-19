"""Aperiodicity, the share of noise in a voice at each frequency: D4C
(Morise, 2016), written from the paper, and a harmonic-residual aperiodicity
that measures the share of noise directly.

:func:`band_aperiodicity` follows the D4C paper's steps; where the paper leaves
a choice open, the choice is stated in its docstring. It is not WORLD's code,
and it reads differently from WORLD's: on harmonics plus white noise it
follows the true noise share with a slope of 1 but about 8 dB low, close to
the bias the paper reports for itself, while WORLD's code moves about half as
much as the true share does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import db_to_amp, time_axis
from sonore.sources.waveforms import _contour_per_channel
from sonore.views.spectral_envelope import (
    _FLOOR_F0,
    _POWER_FLOOR,
    _envelope_fft_size,
    _fft_frequencies,
    _GridView,
    _hann,
    _power_of_two_above,
    _power_on_grid,
    _smooth_rectangular,
    _unit_energy_power,
    _windowed_segment,
)
from sonore.views.view import TimeFrequencyGrid

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from sonore.sources.waveforms import ContourLike

__all__ = ["Aperiodicity", "band_aperiodicity", "harmonic_aperiodicity"]


_BAND_SPACING = 3000.0  # band centers every 3 kHz, as in the paper's evaluation


_HIGHEST_BAND = 15000.0


_BAND_WIDTH = 6000.0  # the length of the Nuttall window over frequency


_ZERO_HZ_DB = -60.0  # the paper's anchor at 0 Hz


_CORRECTION_DB = 8.0  # the bias measured by tools/check_paper_vocoder_claims.py


# the 4-term Nuttall window (Nuttall, 1981), side lobes about 98 dB down
_NUTTALL = (0.3635819, 0.4891775, 0.1365995, 0.0106411)


class Aperiodicity(_GridView):
    """An aperiodicity, from :func:`band_aperiodicity` or :func:`harmonic_aperiodicity`:
    per frequency and time window, how much of the power is noise rather than
    harmonics. Stored as WORLD stores it, an amplitude ratio between 0 and
    1 (``data``, shape ``(n_channels, n_freqs, n_windows)``) on the FFT grid
    of ``n_fft // 2 + 1`` frequencies :attr:`f` from 0 to ``fs / 2``: its
    square :attr:`share` is the share of the power that is noise, which is
    what :func:`~sonore.sources.source_filter.source_filter_synthesize` uses.
    ``method`` names the measure that made it ("D4C" or "harmonic residual").
    Calling it, ``ap(t, f)``, reads the ratio at any times and frequencies.
    """

    discards = (
        "Aperiodicity keeps only the share of noise in each frequency and time window: it discards the "
        "spectrum, the pitch and the phase."
    )
    back_to_sound = (
        "so.source_filter_synthesize rebuilds an approximation of the voice from it together with an F0 "
        "track and a spectral envelope."
    )

    def __init__(self, data: np.ndarray, t: np.ndarray, fs: float, method: str):
        data = np.asarray(data, dtype=float)
        self.n_fft = 2 * (data.shape[-2] - 1)
        self.fs = float(fs)
        super().__init__(data, t, _fft_frequencies(self.fs, self.n_fft))
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

    def bands(self, edges: Sequence[float], *, envelope: TimeFrequencyGrid | None = None) -> np.ndarray:
        """The noise share averaged over each band ``[edges[i], edges[i+1])``,
        shape ``(n_channels, n_bands, n_windows)``. With an ``envelope`` (any
        spectral envelope, read on this aperiodicity's grid), the average is
        weighted by its power, so the result is the band's noise power over
        its total power."""
        freqs = self.f
        if envelope is None:
            weights = np.ones_like(self.data)
        else:
            weights = _power_on_grid(envelope, self.t, self.f, self.data.shape)
        out = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            in_band = (freqs >= lower) & (freqs < upper)
            band_weights = weights[:, in_band]
            out.append(np.sum(self.share[:, in_band] * band_weights, axis=1) / np.sum(band_weights, axis=1))
        return np.stack(out, axis=1)

    def plot(
        self, ax: Axes | None = None, *, channel: int = 0, db_range: float = 60.0, fmax: float | None = None
    ) -> Axes:
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
def band_aperiodicity(
    sound: Sound, f0: ContourLike, *, f0_floor: float = _FLOOR_F0, correction_db: float = _CORRECTION_DB
) -> Aperiodicity:
    """The D4C aperiodicity (Morise, 2016), written from the paper.

    At each voiced time window:

    1. **A group delay that does not move with the window.** The group delay of
       the sound under a Blackman window four periods long, taken a quarter
       period before and a quarter period after the window's time: their sum
       cancels the terms that change with the window's position (the paper's
       Eqs. 7, 8 and 19). It is divided by the power spectrum under a Hann window
       four periods long, smoothed over F0 (Eqs. 21 and 22).
    2. **Shaped into a sinusoid.** That group delay smoothed over F0 / 2, less
       its own trend smoothed over F0 (Eqs. 23 to 25). For a periodic sound what
       is left is close to a sinusoid along frequency with period F0; noise adds
       everything else.
    3. **The share that is not the sinusoid.** For each band, centered every
       3 kHz from 3 to 15 kHz where its 6 kHz window fits below Nyquist, the
       shaped group delay under a Nuttall window is transformed back (Eq. 26),
       the power of the result sorted, and the share of the energy outside its
       strongest ``2 w_bw`` (``w_bw`` the width of the window's main lobe) is the
       band's noise share (Eqs. 27 and 28).
    4. **Across frequency.** Linear in dB from -60 dB at 0 Hz through the band
       values, held at the last band's value up to Nyquist.

    Unvoiced time windows (F0 of 0) are all noise; voicing is the F0 track's
    decision.

    As the paper's own evaluation finds, the band values read too periodic by a
    steady amount: on harmonics plus white noise from 5 to 40 dB SNR they follow
    the true noise share with a slope of 1 but about 8 dB low (7 dB at an F0 of
    125 Hz, 9 dB at 250 Hz; tools/check_paper_vocoder_claims.py). The paper
    reports about 6 dB and suggests subtracting it; ``correction_db`` adds the
    8 dB measured here to every band value, capped at 0 dB (all noise).

    Choices the paper leaves open: F0 below ``f0_floor`` is raised to it; the FFT
    size is the smallest power of two above ``4 fs / f0_floor``; samples beyond
    the ends of the sound are zeros; spectra are mirrored about 0 Hz and Nyquist
    where a smoothing reaches past them; ``1e-12`` guards the division; only the
    positive half of each band's transform is kept (it holds the lobe at one
    period once); ``w_bw`` is the first zero of the Nuttall window's transform.

    Parameters
    ----------
    sound, f0
        As :func:`~sonore.views.spectral_envelope.pitch_adaptive_envelope`. The
        sampling rate must be at least 12 kHz, so that one band fits.
    f0_floor
        The lowest F0 analyzed. It also sets the output's frequency grid, which
        must match the envelope's for synthesis.
    correction_db
        Added to every band value [dB]; ``0`` gives the paper's values as they are.
    """
    fs = float(sound.fs)
    times, f0_values = _contour_per_channel(f0, sound.n_channels, "the sound")
    n_fft = _power_of_two_above(4.0 * fs / f0_floor)
    bin_width = fs / n_fft
    band_centers = np.array(
        [
            center
            for center in np.arange(_BAND_SPACING, _HIGHEST_BAND + 1.0, _BAND_SPACING)
            if center + _BAND_WIDTH / 2.0 <= fs / 2.0
        ]
    )
    if len(band_centers) == 0:
        raise ValueError(f"band_aperiodicity needs a sampling rate of at least 12 kHz, not {fs:g}")
    band_window = _nuttall(int(np.round(_BAND_WIDTH / bin_width)))
    main_lobe = _first_zero_of_transform(band_window, bin_width)
    n_fft_out = _envelope_fft_size(fs, f0_floor)
    freqs_out = np.arange(n_fft_out // 2 + 1) * fs / n_fft_out
    anchors = np.concatenate([[0.0], band_centers])
    data = np.ones((sound.n_channels, n_fft_out // 2 + 1, len(times)))
    for channel in range(sound.n_channels):
        samples = sound.data[:, channel].astype(float)
        for window_index, (time, window_f0) in enumerate(zip(times, f0_values[channel], strict=True)):
            if window_f0 == 0:
                continue
            center_sample = int(np.round(time * fs))
            shaped = _shaped_group_delay(samples, fs, center_sample, max(window_f0, f0_floor), n_fft)
            shares = [
                _band_noise_share(shaped, bin_width, center, band_window, main_lobe)
                for center in band_centers
            ]
            bands_db = np.minimum(10 * np.log10(np.maximum(shares, 1e-12)) + correction_db, 0.0)
            share_db = np.interp(freqs_out, anchors, np.concatenate([[_ZERO_HZ_DB], bands_db]))
            data[channel, :, window_index] = np.minimum(db_to_amp(share_db), 1.0)
    return Aperiodicity(data, times, fs, "D4C")


def _blackman(length):
    """A Blackman window ``length`` samples long, as a function of the offset from its center."""
    return lambda offset: (
        0.42 + 0.5 * np.cos(2.0 * np.pi * offset / length) + 0.08 * np.cos(4.0 * np.pi * offset / length)
    )


def _group_delay_numerator(samples, fs, center, f0, n_fft):
    """``E_cs`` of the paper's Eq. 8 under a unit-energy Blackman window four periods
    long: ``Re(conj(X) X_t)``, with ``X`` the spectrum and ``X_t`` the spectrum of
    the time-weighted sound. Time is measured from the window's start (``tau0`` of
    Eq. 18 is half the window); measured from its center, the periodic part
    would vanish."""
    window_length = 4.0 * fs / f0
    half_length = window_length / 2.0
    offsets, window, segment = _windowed_segment(samples, center, _blackman(window_length), half_length)
    segment = segment / np.sqrt(np.sum(window**2))
    time = offsets / fs + 0.5 * window_length / fs
    spectrum = np.fft.rfft(segment, n_fft)
    return np.real(np.conj(spectrum) * np.fft.rfft(time * segment, n_fft))


def _shaped_group_delay(samples, fs, center, f0, n_fft):
    """Steps 1 and 2 of :func:`band_aperiodicity` at one time window: ``tau_D`` of Eq. 24."""
    bin_width = fs / n_fft
    quarter_period = fs / (4.0 * f0)
    # Eq. 19: two windows half a period apart, around the window's center
    before = _group_delay_numerator(samples, fs, center - quarter_period, f0, n_fft)
    after = _group_delay_numerator(samples, fs, center + quarter_period, f0, n_fft)
    numerator = before + after
    window_length = 4.0 * fs / f0
    power = _unit_energy_power(samples, center, _hann(window_length), window_length / 2.0, n_fft)
    smoothed_power = _smooth_rectangular(power, (f0 / 2.0) / bin_width)  # Eq. 21
    group_delay = numerator / (smoothed_power + _POWER_FLOOR)  # Eq. 22
    smoothed = _smooth_rectangular(group_delay, (f0 / 4.0) / bin_width)  # Eq. 23
    trend = _smooth_rectangular(smoothed, (f0 / 2.0) / bin_width)  # Eq. 25
    return smoothed - trend  # Eq. 24


def _nuttall(length):
    """The symmetric 4-term Nuttall window, ``length`` points."""
    if length == 1:
        return np.ones(1)
    phase = 2.0 * np.pi * np.arange(length) / (length - 1)
    a0, a1, a2, a3 = _NUTTALL
    return a0 - a1 * np.cos(phase) + a2 * np.cos(2 * phase) - a3 * np.cos(3 * phase)


def _first_zero_of_transform(window, step):
    """``w_bw`` of Eq. 28: where the transform of ``window`` (sampled every ``step``
    Hz) first reaches zero, in seconds. Found on a fine grid, then refined by
    golden-section search."""
    positions = np.arange(len(window))

    def magnitude(reciprocal):
        return np.abs(np.sum(window * np.exp(-2j * np.pi * positions * step * reciprocal)))

    padding = 64
    fine = np.abs(np.fft.rfft(window, padding * len(window)))
    first_minimum = np.argmax((fine[1:-1] <= fine[:-2]) & (fine[1:-1] <= fine[2:])) + 1
    grid_step = 1.0 / (padding * len(window) * step)
    low, high = (first_minimum - 1) * grid_step, (first_minimum + 1) * grid_step
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        lower_probe = high - golden * (high - low)
        upper_probe = low + golden * (high - low)
        if magnitude(lower_probe) < magnitude(upper_probe):
            high = upper_probe
        else:
            low = lower_probe
    return 0.5 * (low + high)


def _band_noise_share(shaped, bin_width, center, band_window, main_lobe):
    """Step 3 of :func:`band_aperiodicity` for one band: ``p_c(2 w_bw)`` of Eqs. 26 to 28."""
    first_bin = int(np.round((center - _BAND_WIDTH / 2.0) / bin_width))
    waveform = np.fft.ifft(band_window * shaped[first_bin : first_bin + len(band_window)])  # Eq. 26
    # the shaped group delay is real, so |p(t)|^2 is the same at t and -t: the
    # positive half holds the lobe at one period once (Eq. 27 integrates from 0)
    power = np.abs(waveform[: len(waveform) // 2 + 1]) ** 2
    power = power / (np.sum(power) + _POWER_FLOOR)
    sorted_power = np.sort(power)[::-1]
    # Eq. 27 at t = 2 w_bw, counting the last sample by the part of it inside
    time_step = 1.0 / (len(band_window) * bin_width)
    covered = 2.0 * main_lobe / time_step
    whole = int(np.floor(covered))
    captured = np.sum(sorted_power[:whole])
    if whole < len(sorted_power):
        captured += (covered - whole) * sorted_power[whole]
    return max(1.0 - captured, 0.0)


# ------------------------------------------------- harmonic residual
def harmonic_aperiodicity(
    sound: Sound,
    f0: ContourLike,
    *,
    periods: float = 4.0,
    f0_floor: float = _FLOOR_F0,
    cell_harmonics: float = 2.0,
) -> Aperiodicity:
    """The share of noise, measured by fitting the harmonics and keeping
    what is left. Not part of D4C or WORLD; it is sonore's own.

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
    interpolated in dB onto the grid of :func:`pitch_adaptive_envelope`. Unvoiced time windows
    are all noise.

    The measure is what the word means, so it reads a known share of noise
    within a few tenths of a dB on a steady vowel; but it needs F0 to about
    0.1%, since a harmonic that drifts out of phase with the fit over the
    window reads as noise, and so do jitter and shimmer.

    Parameters
    ----------
    sound, f0
        As :func:`pitch_adaptive_envelope`.
    periods
        The window's length in periods of the time window's F0.
    f0_floor
        Sets the output's frequency grid, as :func:`pitch_adaptive_envelope`.
    cell_harmonics
        The width of the cells, in harmonics.
    """
    fs = float(sound.fs)
    times, f0_values = _contour_per_channel(f0, sound.n_channels, "the sound")
    n_fft_out = _envelope_fft_size(fs, f0_floor)
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
