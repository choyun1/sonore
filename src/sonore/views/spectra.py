"""Spectra and spectrograms that keep power and drop the phase: the spectrum
of a sound, a TANDEM-STRAIGHT-style power spectrogram, and the reassigned
spectrogram.

A :class:`Spectrum`'s levels are a one-sided power spectral density in dB re
1 per Hz, as Welch's method gives it, so :meth:`Spectrum.from_sound` and
:func:`long_term_spectrum` agree for a steady noise (the first scatters, the
second is smooth)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import ShortTimeFFT, welch

from sonore.core.sound import Sound
from sonore.core.utils import db_to_power, n_samples, power_to_db
from sonore.frames.gabor import _FLOOR_DB, GaborFrame, TVGaborFrame
from sonore.views.view import View

__all__ = [
    "Spectrum",
    "long_term_spectrum",
    "TFPower",
    "tandem_power",
    "ReassignedSpectrogram",
    "reassigned_spectrogram",
]


# ---------------------------------------------------------------- spectrum
@dataclass(frozen=True)
class Spectrum(View):
    """A magnitude spectrum: frequencies [Hz] and levels [dB].

    A view: it keeps one level per frequency and drops the phase, and with
    it all timing, so no sound can be read back from it.
    """

    discards = (
        "Spectrum keeps only the level at each frequency: it discards the phase, and with it all timing, so "
        "many sounds share one spectrum."
    )
    back_to_sound = (
        "Spectrum.to_sound makes a sound with this spectrum from a carrier that supplies the phase "
        "(a new noise, another sound's phase, or the minimum phase), which is not the analyzed sound."
    )

    f: np.ndarray
    level: np.ndarray

    @classmethod
    def from_sound(cls, sound: Sound) -> Spectrum:
        """The power spectral density of the whole sound from one FFT with no
        window, ``|X|**2 / (fs N)`` with every bin but 0 Hz and Nyquist
        counted twice (one-sided), in dB re 1 per Hz. The sound is first
        mixed to mono by averaging its channels, so channels in antiphase
        cancel. Each bin of a noise scatters about its expected level; see
        :func:`long_term_spectrum` for a smooth estimate on the same scale."""
        samples = sound.mono().data[:, 0]
        length = len(samples)
        density = np.abs(np.fft.rfft(samples)) ** 2 * _one_sided(length) / (sound.fs * length)
        return cls(np.fft.rfftfreq(length, 1 / sound.fs), power_to_db(density, floor_db=_FLOOR_DB))

    def level_at(self, freqs: np.ndarray) -> np.ndarray:
        """Levels [dB] linearly interpolated at ``freqs``, held at the end
        values outside the spectrum's frequencies."""
        return np.interp(freqs, self.f, self.level)

    def _level_in_range(self, freqs: np.ndarray) -> np.ndarray:
        """``level_at`` inside the spectrum's frequencies and the floor
        outside them: a spectrum says nothing about the frequencies it was
        not measured at, so a sound made from it is silent there."""
        freqs = np.asarray(freqs, dtype=float)
        level = np.maximum(self.level_at(freqs), _FLOOR_DB)
        return np.where((freqs >= self.f[0]) & (freqs <= self.f[-1]), level, _FLOOR_DB)

    def relative(self) -> Spectrum:
        """Levels re the maximum (0 dB peak)."""
        return Spectrum(self.f, self.level - self.level.max())

    def smooth(self, fraction: float = 1 / 3) -> Spectrum:
        """Fractional-octave smoothing (power average over ``fraction`` octave)."""
        power = db_to_power(self.level)
        cumulative = np.concatenate([[0], np.cumsum(power)])
        half = 2 ** (fraction / 2)
        start = np.searchsorted(self.f, self.f / half, side="left")
        stop = np.searchsorted(self.f, self.f * half, side="right")
        stop = np.maximum(stop, start + 1)
        mean_power = (cumulative[stop] - cumulative[start]) / (stop - start)
        return Spectrum(self.f, power_to_db(mean_power, floor_db=_FLOOR_DB))

    def to_sound(
        self, duration: float, fs: float, carrier: str | Sound = "noise", rng=None, **noise_kwargs
    ) -> Sound:
        """A sound with this magnitude spectrum, its phase supplied by ``carrier``.

        ``"noise"`` draws Gaussian noise with this spectral shape (extra
        keyword arguments go to :func:`~sonore.sources.waveforms.gaussian_noise`).
        A :class:`~sonore.Sound` lends its phase: its first ``duration``
        seconds, each channel's spectrum given this magnitude, so
        ``Spectrum.from_sound(x).to_sound(x.duration, x.fs, carrier=x)``
        gives back ``x`` at RMS 1. ``"minimum"`` gives the minimum-phase
        impulse response with this magnitude, from the folded cepstrum as in
        :meth:`Cepstrum.to_stft <sonore.views.cepstrum.Cepstrum.to_stft>`.
        The level is read at the FFT bins of ``duration`` by linear
        interpolation, and is silent outside the spectrum's frequencies (a
        spectrum measured at 16 kHz gives nothing above 8 kHz at any ``fs``).
        Every result has RMS 1, as the generators do. None is the analyzed
        sound: the spectrum keeps no phase to give back.
        """
        length = n_samples(duration, fs)
        density = db_to_power(self._level_in_range(np.fft.rfftfreq(length, 1 / fs)))
        magnitude = np.sqrt(density / _one_sided(length))
        match carrier:  # str("...") so that an array is not compared elementwise
            case str("noise"):
                from sonore.sources.waveforms import gaussian_noise

                return gaussian_noise(duration, fs, spectrum=self, rng=rng, **noise_kwargs)
            case _ if noise_kwargs or rng is not None:
                raise TypeError("rng and noise keyword arguments apply only to carrier='noise'")
            case Sound():
                if carrier.fs != fs:
                    raise ValueError(f"carrier fs {carrier.fs} differs from fs {fs}")
                if len(carrier) < length:
                    raise ValueError("carrier is shorter than duration")
                phase = np.angle(np.fft.rfft(carrier.data[:length], axis=0))
                data = np.fft.irfft(magnitude[:, None] * np.exp(1j * phase), n=length, axis=0)
            case str("minimum"):
                from sonore.views.cepstrum import _minimum_phase

                cepstrum = np.fft.irfft(np.log(magnitude), n=length)[: length // 2 + 1]
                data = np.fft.irfft(_minimum_phase(cepstrum, length, axis=0), n=length)
            case _:
                raise ValueError(f"carrier must be 'noise', 'minimum' or a Sound, not {carrier!r}")
        sound = Sound(data, fs)
        return sound.normalize() if sound.rms > 0 else sound

    def plot(self, ax=None, **kwargs):
        """Level against frequency, on a log frequency axis and re the peak by
        default (see :func:`~sonore.plotting.plot_spectrum`)."""
        from sonore.plotting import plot_spectrum

        return plot_spectrum(self, ax=ax, **kwargs)


def _one_sided(length: int) -> np.ndarray:
    """How many times each bin of a ``length``-point rfft counts in a one-sided
    spectrum: twice, except 0 Hz and (for an even length) Nyquist, which
    have no negative twin."""
    weights = np.ones(length // 2 + 1)
    weights[1 : (length + 1) // 2] = 2.0
    return weights


def long_term_spectrum(sounds: Sound | Sequence[Sound], win_dur: float = 0.1) -> Spectrum:
    """Long-term average spectrum via Welch's method, averaged over sounds
    (weighted by duration), in dB re 1 per Hz. This is the right input for
    speech-shaped noise.

    ``win_dur`` [s] is Welch's segment length (Hann windows, half
    overlapping); the frequency spacing is ``1 / win_dur``, 10 Hz by
    default. A sound shorter than ``win_dur`` is one segment, its own
    length, zero-padded to ``win_dur``, which puts it on the same
    frequencies without changing its spectrum. Sounds at another rate are
    resampled to the first one's, and each is mixed to mono by averaging
    its channels."""
    if isinstance(sounds, Sound):
        sounds = [sounds]
    if len(sounds) == 0:
        raise ValueError("long_term_spectrum needs at least one sound")
    fs = sounds[0].fs
    n_fft = n_samples(win_dur, fs)
    total, weight = 0.0, 0
    for sound in sounds:
        if sound.fs != fs:
            sound = sound.resample(fs)
        freqs, sound_psd = welch(sound.mono().data[:, 0], fs, nperseg=min(n_fft, len(sound)), nfft=n_fft)
        total = total + sound_psd * len(sound)
        weight += len(sound)
    psd = total / weight
    return Spectrum(freqs, power_to_db(psd, floor_db=_FLOOR_DB))


# ------------------------------------------------- magnitude-only analyses
@dataclass(frozen=True)
class TFPower(View):
    """A time-frequency power that is not a frame's coefficients, so it has
    no synthesis: for example :func:`tandem_power`. It keeps the power of
    each cell and drops the phase.

    ``power`` has shape ``(n_channels, n_freqs, n_windows)``, on frequencies
    :attr:`f` [Hz] and window times :attr:`t` [s], which need not be uniform.
    """

    discards = (
        "TFPower keeps only the power of each cell: it discards the phase, and the cells are not a frame's "
        "coefficients, so no synthesis undoes them."
    )
    back_to_sound = ""

    power: np.ndarray
    t: np.ndarray
    f: np.ndarray

    @property
    def db(self) -> np.ndarray:
        """``10*log10(power)``, floored like the other representations."""
        return power_to_db(self.power, floor_db=_FLOOR_DB)

    def plot(self, ax=None, channel: int = 0, **kwargs):
        """Power in dB (see :func:`~sonore.plotting.plot_tf_db`)."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(self.db[channel], self.t, self.f, ax=ax, **kwargs)


def tandem_power(
    sound: Sound,
    f0_times: Sequence[float],
    f0: Sequence[float],
    periods: float = 2.5,
    overlap: float = 4,
    window: str | tuple = "blackman",
) -> TFPower:
    """A TANDEM-STRAIGHT-style power spectrogram: the average of two
    pitch-adaptive spectrograms whose windows sit a quarter period before
    and after each window center.

    For a periodic sound, the power through a window centered at ``t``
    fluctuates with period T0 as the window slides across the glottal
    pulses. Averaging the powers at ``t - T0/4`` and ``t + T0/4`` (half a
    period apart) cancels the odd harmonics of that fluctuation, including
    the largest, so a short window shows the spectral envelope steadily.
    The defaults, a Blackman window 2.5 periods long, are those of
    Kawahara et al. (2011). Only this averaging is implemented, not
    TANDEM-STRAIGHT's smoothing or aperiodicity analysis.

    The schedule is :meth:`~sonore.frames.gabor.TVGaborFrame.pitch_adaptive` over
    the sound, with F0 bridged across unvoiced stretches. The result is
    magnitude only: synthesizing from the pair would need a union of two
    frames, which sonore does not provide.
    """
    f0_values = np.asarray(f0, dtype=float)
    longest_dur = periods / np.min(f0_values[np.isfinite(f0_values) & (f0_values > 0)], initial=np.inf)
    base = TVGaborFrame.pitch_adaptive(
        f0_times, f0, t_end=sound.duration + longest_dur / 2, periods=periods, overlap=overlap, window=window
    )
    t, durs = np.asarray(base.times), np.asarray(base.win_durs)
    quarter = durs / periods / 4
    n_fft = int(base.layout(sound.fs).n_fft)
    early = TVGaborFrame(t - quarter, durs, n_fft=n_fft, window=window).analyze(sound)
    late = TVGaborFrame(t + quarter, durs, n_fft=n_fft, window=window).analyze(sound)
    power = (np.abs(early.data) ** 2 + np.abs(late.data) ** 2) / 2
    return TFPower(power, t, early.f)


# ---------------------------------------------------------------- reassignment
@dataclass(frozen=True)
class ReassignedSpectrogram(View):
    """Spectrogram cells moved to their reassigned times and frequencies.

    ``t_hat``, ``f_hat`` and ``power`` have shape ``(n_channels, n_freqs,
    n_windows)``, one entry per STFT cell; ``keep`` marks the cells within
    the threshold of the maximum. :meth:`binned` sums the kept power onto a
    grid for display. There is no synthesis: reassignment is not linear.
    """

    discards = (
        "ReassignedSpectrogram discards the phase and moves each cell's power to a new time and frequency, "
        "many cells to one point, so different sounds give the same picture."
    )
    back_to_sound = ""
    no_plot = (
        "ReassignedSpectrogram has no plot: its points lie off any grid until one is chosen, so plot "
        "binned(t_edges, f_edges)."
    )

    t_hat: np.ndarray
    f_hat: np.ndarray
    power: np.ndarray
    keep: np.ndarray

    def binned(self, t_edges: np.ndarray, f_edges: np.ndarray) -> TFPower:
        """Kept power summed into the cells of ``t_edges`` x ``f_edges``
        [s, Hz]; points outside the edges are dropped."""
        t_edges, f_edges = np.asarray(t_edges, float), np.asarray(f_edges, float)
        binned_power = np.stack(
            [
                np.histogram2d(freqs[kept], times[kept], bins=(f_edges, t_edges), weights=powers[kept])[0]
                for times, freqs, powers, kept in zip(
                    self.t_hat, self.f_hat, self.power, self.keep, strict=True
                )
            ]
        )
        return TFPower(binned_power, (t_edges[:-1] + t_edges[1:]) / 2, (f_edges[:-1] + f_edges[1:]) / 2)


def reassigned_spectrogram(
    sound: Sound, frame: GaborFrame, threshold_db: float = -60.0
) -> ReassignedSpectrogram:
    """The reassigned spectrogram (Kodera et al., 1978; Auger & Flandrin, 1995).

    Each cell of ``frame``'s spectrogram is moved from its window time ``t``
    and bin frequency ``f`` to

    - ``t_hat = t + Re(X_tw conj X) / |X|**2``,
    - ``f_hat = f - Im(X_dw conj X) / (2 pi |X|**2)``,

    where ``X`` uses the window ``w``, ``X_tw`` the time-weighted window
    ``tau w(tau)`` and ``X_dw`` its derivative ``w'(tau)``, with ``tau`` in
    seconds from the window's middle sample, which is where SciPy (and so
    ``GaborFrame``) references each time window's phase. A tone off the bin grid,
    an impulse and a linear chirp land on their true frequency, time and
    instantaneous-frequency line.

    The derivative is taken from the window's formula, so only ``"hann"`` and
    ``("gaussian", std)`` windows are accepted, ``std`` in samples as in
    SciPy. Cells more than
    ``threshold_db`` below the maximum (per channel) are marked not kept:
    their positions are mostly noise.
    """
    n_win, hop, n_fft = frame.lengths(sound.fs)
    window = frame.window_samples(sound.fs)
    tau, window_deriv = _window_tau_and_derivative(frame.window, n_win, sound.fs)
    # GaborFrame builds these windows from the same formulas today; this guards against a change there
    if not np.allclose(window, _window_from_formula(frame.window, n_win)):
        raise ValueError(f"window {frame.window!r} does not match its formula")

    def stft(window_values):
        short_time_fft = ShortTimeFFT(
            window_values, hop=hop, fs=sound.fs, mfft=n_fft, fft_mode="onesided", dual_win=window
        )
        return short_time_fft, short_time_fft.stft(sound.data.T)

    short_time_fft, X = stft(window)
    X_tw = stft(tau * window)[1]
    X_dw = stft(window_deriv)[1]
    power = np.abs(X) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hat = short_time_fft.t(len(sound))[None, None, :] + np.real(X_tw * X.conj()) / power
        f_hat = short_time_fft.f[None, :, None] - np.imag(X_dw * X.conj()) / power / (2 * np.pi)
    peak = power.max(axis=(1, 2), keepdims=True)
    keep = (power > peak * db_to_power(threshold_db)) & np.isfinite(t_hat) & np.isfinite(f_hat)
    return ReassignedSpectrogram(t_hat, f_hat, power, keep)


def _window_from_formula(spec, n: int) -> np.ndarray:
    k = np.arange(n)
    match spec:
        case "hann":
            return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
        case tuple(("gaussian", std)):
            return np.exp(-0.5 * ((k - n / 2) / std) ** 2)
        case _:
            raise ValueError(f"reassignment needs a 'hann' or ('gaussian', std) window, not {spec!r}")


def _window_tau_and_derivative(spec, n: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Time from the middle sample [s] and the window's derivative [1/s],
    both at the periodic window's samples."""
    k = np.arange(n)
    tau = (k - n // 2) / fs
    window = _window_from_formula(spec, n)
    match spec:
        case "hann":
            derivative = 0.5 * (2 * np.pi / n) * np.sin(2 * np.pi * k / n) * fs
        case ("gaussian", std):
            derivative = -(k - n / 2) / std**2 * window * fs
    return tau, derivative
