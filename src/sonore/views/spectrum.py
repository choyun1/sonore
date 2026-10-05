"""Magnitude spectra: the long-term spectrum of a sound, and the power of
time-frequency coefficients. Both drop the phase. All levels are in dB of
*power* (``20*log10|X|``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from sonore.core.sound import Sound
from sonore.core.utils import amp_to_db, db_to_amp, db_to_power, n_samples, power_to_db
from sonore.frames.gabor import _FLOOR_DB, TVGaborFrame
from sonore.views.view import View

__all__ = ["Spectrum", "long_term_spectrum", "TFPower", "tandem_power"]


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
        "(a new noise, another sound's phase, or the minimum phase), which is not the analysed sound."
    )

    f: np.ndarray
    level: np.ndarray

    @classmethod
    def from_sound(cls, sound: Sound) -> Spectrum:
        """Exact FFT magnitude of the (mono mixdown of the) sound. Noisy for
        noise-like signals; see :func:`long_term_spectrum` for a smooth estimate."""
        samples = sound.mono().data[:, 0]
        return cls(
            np.fft.rfftfreq(len(samples), 1 / sound.fs), amp_to_db(np.fft.rfft(samples), floor_db=_FLOOR_DB)
        )

    def level_at(self, freqs: np.ndarray) -> np.ndarray:
        """Levels [dB] linearly interpolated at ``freqs``."""
        return np.interp(freqs, self.f, self.level)

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
        The magnitude is read at the FFT bins of ``duration`` by linear
        interpolation, and every result has RMS 1, as the generators do. None
        is the analysed sound: the spectrum keeps no phase to give back.
        """
        length = n_samples(duration, fs)
        magnitude = db_to_amp(self.level_at(np.fft.rfftfreq(length, 1 / fs)))
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


def long_term_spectrum(sounds: Sound | Sequence[Sound], win_dur: float = 0.1) -> Spectrum:
    """Long-term average spectrum via Welch's method, averaged over sounds
    (weighted by duration). This is the right input for speech-shaped noise.

    ``win_dur`` [s] is Welch's segment length (Hann windows, half
    overlapping), cut to the sound's length if that is shorter; the
    frequency spacing is ``1 / win_dur``, 10 Hz by default."""
    if isinstance(sounds, Sound):
        sounds = [sounds]
    fs = sounds[0].fs
    total, weight = 0.0, 0
    for sound in sounds:
        if sound.fs != fs:
            sound = sound.resample(fs)
        freqs, sound_psd = welch(sound.mono().data[:, 0], fs, nperseg=min(n_samples(win_dur, fs), len(sound)))
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
