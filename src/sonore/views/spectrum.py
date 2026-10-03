"""Magnitude spectra: the long-term spectrum of a sound, and the power of
time-frequency coefficients. Both drop the phase. All levels are in dB of
*power* (``20*log10|X|``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from sonore.core.sound import Sound
from sonore.core.utils import amp_to_db
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
    back_to_sound = "Spectrum.to_noise draws a new noise with this spectrum, which is not the analysed sound."

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
        power = 10 ** (self.level / 10)
        cumulative = np.concatenate([[0], np.cumsum(power)])
        half = 2 ** (fraction / 2)
        start = np.searchsorted(self.f, self.f / half, side="left")
        stop = np.searchsorted(self.f, self.f * half, side="right")
        stop = np.maximum(stop, start + 1)
        mean_power = (cumulative[stop] - cumulative[start]) / (stop - start)
        return Spectrum(self.f, 10 * np.log10(np.maximum(mean_power, 10 ** (_FLOOR_DB / 10))))

    def to_noise(self, duration: float, fs: float, rng=None, **kwargs) -> Sound:
        """Gaussian noise with this spectral shape: a new draw, not an inverse
        of the spectrum."""
        from sonore.signals.generators import gaussian_noise

        return gaussian_noise(duration, fs, spectrum=self, rng=rng, **kwargs)

    def plot(self, ax=None, **kwargs):
        from sonore.plotting import plot_spectrum

        return plot_spectrum(self, ax=ax, **kwargs)


def long_term_spectrum(sounds: Sound | Sequence[Sound], nperseg: int = 4096) -> Spectrum:
    """Long-term average spectrum via Welch's method, averaged over sounds
    (weighted by duration). This is the right input for speech-shaped noise."""
    if isinstance(sounds, Sound):
        sounds = [sounds]
    fs = sounds[0].fs
    total, weight = 0.0, 0
    for sound in sounds:
        if sound.fs != fs:
            sound = sound.resample(fs)
        freqs, sound_psd = welch(sound.mono().data[:, 0], fs, nperseg=min(nperseg, len(sound)))
        total = total + sound_psd * len(sound)
        weight += len(sound)
    psd = total / weight
    return Spectrum(freqs, 10 * np.log10(np.maximum(psd, 10 ** (_FLOOR_DB / 10))))


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
        with np.errstate(divide="ignore"):
            return np.maximum(10 * np.log10(self.power), _FLOOR_DB)

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
