"""Spectral representations: magnitude spectra, STFTs, T-F masks, and
modulation spectra. All levels are in dB of *power* (``20*log10|X|``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import ShortTimeFFT, welch

from sonore.analysis.frames import GaborFrame, TVGaborFrame
from sonore.core.sound import Sound
from sonore.core.utils import amp_to_db, as_rng

__all__ = [
    "Spectrum",
    "long_term_spectrum",
    "STFT",
    "TVSTFT",
    "TFPower",
    "tandem_power",
    "ReassignedSpectrogram",
    "reassigned_spectrogram",
    "Mask",
    "ideal_binary_mask",
    "ideal_ratio_mask",
    "ModulationSpectrum",
]

_FLOOR_DB = -200.0


# ---------------------------------------------------------------- spectrum
@dataclass(frozen=True)
class Spectrum:
    """A magnitude spectrum: frequencies [Hz] and levels [dB]."""

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
        """Gaussian noise with this spectral shape."""
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


# -------------------------------------------------------------------- STFT
class STFT:
    """Short-time Fourier transform (wraps :class:`scipy.signal.ShortTimeFFT`).

    ``data`` has shape ``(n_channels, n_freqs, n_windows)``. Resynthesis with
    :meth:`to_sound` is exact for an unmodified STFT, and the least-squares
    signal for a modified one.

    ``STFT(sound, win_dur, hop_dur)`` is
    ``GaborFrame(win_dur, hop_dur).analyze(sound)``; the frame is kept as
    :attr:`frame` and SciPy's transform as :attr:`sft`.

    Parameters
    ----------
    win_dur
        Window length [s] (periodic Hann).
    hop_dur
        Hop [s]; defaults to a quarter window (75% overlap).
    frame
        A :class:`~sonore.analysis.frames.GaborFrame` to use instead (other windows,
        zero-padded FFTs); ``win_dur`` and ``hop_dur`` are then ignored.
    """

    __array_ufunc__ = None

    def __init__(
        self,
        sound: Sound,
        win_dur: float = 20e-3,
        hop_dur: float | None = None,
        *,
        frame: GaborFrame | None = None,
    ):
        self.frame = GaborFrame(win_dur, hop_dur) if frame is None else frame
        self.fs = sound.fs
        self.sft = self.frame.sft(sound.fs)
        self.n_samples = len(sound)
        self.data = self.sft.stft(sound.data.T, axis=-1)

    @classmethod
    def _from(cls, template: STFT, data: np.ndarray) -> STFT:
        new = cls.__new__(cls)
        new.frame, new.fs, new.sft = template.frame, template.fs, template.sft
        new.n_samples, new.data = template.n_samples, data
        return new

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        win_ms, hop_ms = self.sft.m_num / self.fs * 1e3, self.sft.hop / self.fs * 1e3
        return (
            f"STFT({n_freqs} freqs x {n_windows} time windows, {n_channels} ch, "
            f"win {win_ms:.1f} ms, hop {hop_ms:.1f} ms)"
        )

    @property
    def f(self) -> np.ndarray:
        return self.sft.f

    @property
    def t(self) -> np.ndarray:
        """Window center times [s]."""
        return self.sft.t(self.n_samples)

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self.data, floor_db=_FLOOR_DB)

    def __mul__(self, other):
        gains = other.values if isinstance(other, Mask) else other
        return STFT._from(self, self.data * gains)

    __rmul__ = __mul__

    def to_sound(self) -> Sound:
        """Inverse STFT (least-squares overlap-add); see
        :meth:`~sonore.analysis.frames.GaborFrame.synthesize`."""
        return self.frame.synthesize(self)

    def griffin_lim(self, n_iter: int = 100, momentum: float = 0.99, rng=None) -> Sound:
        """Reconstruct a signal from the magnitude only (fast Griffin-Lim,
        Perraudin et al., 2013). ``momentum=0`` gives classic Griffin-Lim."""
        target_mag = self.magnitude
        rng = as_rng(rng)
        coefs = target_mag * np.exp(2j * np.pi * rng.random(target_mag.shape))
        prev_projected = coefs
        for _ in range(n_iter):
            signal = self.sft.istft(coefs, k1=self.n_samples)
            projected = self.sft.stft(np.real(signal), axis=-1)
            coefs = projected + momentum * (projected - prev_projected)
            prev_projected = projected
            coefs = target_mag * np.exp(1j * np.angle(coefs))
        return STFT._from(self, coefs).to_sound()

    def plot(self, ax=None, channel: int = 0, **kwargs):
        from sonore.plotting import plot_stft

        return plot_stft(self, ax=ax, channel=channel, **kwargs)


class TVSTFT:
    """Coefficients of a :class:`~sonore.analysis.frames.TVGaborFrame`: a short-time
    Fourier transform whose window changes over time.

    ``data`` has shape ``(n_channels, n_freqs, n_windows)`` like
    :class:`STFT`, on one frequency grid :attr:`f` (every window is
    zero-padded to the same FFT length) and at non-uniform window centers
    :attr:`t`. :meth:`to_sound` is exact for unmodified coefficients and the
    least-squares signal for modified ones. Multiply by an array to mask.
    """

    __array_ufunc__ = None

    def __init__(self, data: np.ndarray, fs: float, n_samples: int, frame: TVGaborFrame):
        self.data, self.fs, self.n_samples, self.frame = data, fs, int(n_samples), frame

    @classmethod
    def _from(cls, template: TVSTFT, data: np.ndarray) -> TVSTFT:
        return cls(data, template.fs, template.n_samples, template.frame)

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        win_ms = self.frame.layout(self.fs).lengths / self.fs * 1e3
        return (
            f"TVSTFT({n_freqs} freqs x {n_windows} time windows, {n_channels} ch, "
            f"win {win_ms.min():.1f}-{win_ms.max():.1f} ms, n_fft {self.frame.layout(self.fs).n_fft})"
        )

    @property
    def f(self) -> np.ndarray:
        return np.fft.rfftfreq(self.frame.layout(self.fs).n_fft, 1 / self.fs)

    @property
    def t(self) -> np.ndarray:
        """Window center times [s], rounded to samples."""
        return self.frame.layout(self.fs).centers / self.fs

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def db(self) -> np.ndarray:
        return amp_to_db(self.data, floor_db=_FLOOR_DB)

    def __mul__(self, other):
        return TVSTFT._from(self, self.data * other)

    __rmul__ = __mul__

    def to_sound(self) -> Sound:
        """Least-squares resynthesis; see
        :meth:`~sonore.analysis.frames.TVGaborFrame.synthesize`."""
        return self.frame.synthesize(self)

    def plot(self, ax=None, channel: int = 0, **kwargs):
        """Spectrogram in dB at the window centers (see :func:`~sonore.plotting.plot_tf_db`)."""
        from sonore.plotting import plot_tf_db

        kwargs.setdefault("title", "Time-varying spectrogram")
        return plot_tf_db(self.db[channel], self.t, self.f, ax=ax, **kwargs)


# ------------------------------------------------- magnitude-only analyses
@dataclass(frozen=True)
class TFPower:
    """A time-frequency power that is not a frame's coefficients, so it has
    no synthesis: for example :func:`tandem_power`.

    ``power`` has shape ``(n_channels, n_freqs, n_windows)``, on frequencies
    :attr:`f` [Hz] and window times :attr:`t` [s], which need not be uniform.
    """

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

    The schedule is :meth:`~sonore.analysis.frames.TVGaborFrame.pitch_adaptive` over
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
class ReassignedSpectrogram:
    """Spectrogram cells moved to their reassigned times and frequencies.

    ``t_hat``, ``f_hat`` and ``power`` have shape ``(n_channels, n_freqs,
    n_windows)``, one entry per STFT cell; ``keep`` marks the cells within
    the threshold of the maximum. :meth:`binned` sums the kept power onto a
    grid for display. There is no synthesis: reassignment is not linear.
    """

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
    ``("gaussian", std)`` windows are accepted. Cells more than
    ``threshold_db`` below the maximum (per channel) are marked not kept:
    their positions are mostly noise.
    """
    n_win, hop, n_fft = frame.lengths(sound.fs)
    window = frame.window_samples(sound.fs)
    tau, window_deriv = _window_tau_and_derivative(frame.window, n_win, sound.fs)
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
    keep = (power > peak * 10 ** (threshold_db / 10)) & np.isfinite(t_hat) & np.isfinite(f_hat)
    return ReassignedSpectrogram(t_hat, f_hat, power, keep)


def _window_from_formula(spec, n: int) -> np.ndarray:
    k = np.arange(n)
    if spec == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "gaussian":
        return np.exp(-0.5 * ((k - n / 2) / spec[1]) ** 2)
    raise ValueError(f"reassignment needs a 'hann' or ('gaussian', std) window, not {spec!r}")


def _window_tau_and_derivative(spec, n: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Time from the middle sample [s] and the window's derivative [1/s],
    both at the periodic window's samples."""
    k = np.arange(n)
    tau = (k - n // 2) / fs
    window = _window_from_formula(spec, n)
    if spec == "hann":
        derivative = 0.5 * (2 * np.pi / n) * np.sin(2 * np.pi * k / n) * fs
    else:
        derivative = -(k - n / 2) / spec[1] ** 2 * window * fs
    return tau, derivative


# ------------------------------------------------------------------- masks
@dataclass(frozen=True)
class Mask:
    """A time-frequency mask (binary or soft) aligned with an STFT."""

    values: np.ndarray
    t: np.ndarray
    f: np.ndarray

    def __mul__(self, other):
        if isinstance(other, STFT):
            return other * self
        return NotImplemented

    def plot(self, ax=None, channel: int = 0, **kwargs):
        from sonore.plotting import plot_mask

        return plot_mask(self, ax=ax, channel=channel, **kwargs)


def ideal_binary_mask(target: STFT, masker: STFT, lc_db: float = 0.0) -> Mask:
    """1 where the target exceeds the masker by more than ``lc_db`` (local SNR criterion)."""
    return Mask((target.db - masker.db > lc_db).astype(float), target.t, target.f)


def ideal_ratio_mask(target: STFT, masker: STFT, beta: float = 0.5) -> Mask:
    """``(|T|^2 / (|T|^2 + |M|^2))**beta``."""
    target_power, masker_power = target.magnitude**2, masker.magnitude**2
    with np.errstate(invalid="ignore", divide="ignore"):
        irm = np.nan_to_num((target_power / (target_power + masker_power)) ** beta)
    return Mask(irm, target.t, target.f)


# ------------------------------------------------------ modulation spectrum
class ModulationSpectrum:
    """2-D Fourier transform of a time-frequency envelope (Singh & Theunissen,
    2003; Chi et al., 1999).

    ``ModulationSpectrum(stft)`` uses a dB spectrogram, so spectral modulation
    is w.r.t. *linear* frequency (cycles/kHz). :meth:`octave` uses subband
    envelopes on a log-frequency axis (cycles/octave), the axis on which
    ripples (:mod:`sonore.stimuli.ripples`) are defined; more generally, any
    :class:`~sonore.analysis.envelopes.Envelopes` has ``.modulation_spectrum()``.

    Sign convention: a ripple ``sin(2*pi*(rate*t + density*x))`` appears at
    ``(+rate, +density)``. Only non-negative spectral modulations are kept
    (the other half is the complex conjugate).
    """

    def __init__(self, stft: STFT, channel: int = 0):
        spectrogram_db = stft.db[channel]
        bin_spacing = stft.f[1] - stft.f[0]
        dt = stft.sft.hop / stft.fs
        self._compute(spectrogram_db, dt=dt, dx=bin_spacing / 1000, spectral_unit="cyc/kHz")

    @classmethod
    def from_array(cls, env: np.ndarray, dt: float, dx: float, spectral_unit: str) -> ModulationSpectrum:
        """From a (frequency, time) envelope array sampled every ``dt`` seconds
        and ``dx`` scale units."""
        new = cls.__new__(cls)
        new._compute(env, dt, dx, spectral_unit)
        return new

    def _compute(self, env: np.ndarray, dt: float, dx: float, spectral_unit: str) -> None:
        # Taper in time: the 2-D FFT treats the envelope as periodic, and the jump
        # between its end and start otherwise leaks energy across modulation rates
        # (enough to misplace the peak when a modulation isn't a whole number of
        # cycles long). Removing each band's own mean instead would also work but
        # would erase static spectral structure (rate-0 ripples).
        env = (env - env.mean()) * np.hanning(env.shape[1])[None, :]
        spectrum = np.fft.fft2(env)
        w_f = np.fft.fftfreq(env.shape[0], d=dx)
        w_t = np.fft.fftfreq(env.shape[1], d=dt)
        keep = w_f >= 0
        self.w_f = w_f[keep]
        self.w_t = np.fft.fftshift(w_t)
        self.level = amp_to_db(np.fft.fftshift(spectrum[keep], axes=1), floor_db=_FLOOR_DB)
        self.spectral_unit = spectral_unit

    @classmethod
    def octave(
        cls,
        sound: Sound,
        bands_per_octave: float = 12,
        f_lo: float = 125.0,
        f_hi: float = 8000.0,
        env_fs: float = 1000.0,
        scale: str = "linear",
    ) -> ModulationSpectrum:
        """Modulation spectrum on a log-frequency axis [cycles/octave].

        Shorthand for::

            fb = OctaveFilterbank.per_octave(bands_per_octave, f_lo, f_hi)
            fb.analyze(sound.mono()).envelopes(fs=env_fs).modulation_spectrum(scale)
        """
        from sonore.analysis.filterbank import OctaveFilterbank

        fb = OctaveFilterbank.per_octave(bands_per_octave, f_lo, min(f_hi, 0.95 * sound.fs / 2))
        return fb.analyze(sound.mono()).envelopes(fs=env_fs).modulation_spectrum(scale)

    def peak(self, exclude_dc: bool = True) -> tuple[float, float]:
        """``(temporal Hz, spectral)`` coordinates of the largest component.

        At zero spectral modulation, ``+rate`` and ``-rate`` are mirror images
        (the envelope is the same at every frequency, so it has no direction),
        and the rate is reported as non-negative.
        """
        level = self.level.copy()
        if exclude_dc:
            level[np.ix_(self.w_f == 0, self.w_t == 0)] = -np.inf
        i, j = np.unravel_index(np.argmax(level), level.shape)
        rate = float(self.w_t[j])
        return (abs(rate) if self.w_f[i] == 0 else rate), float(self.w_f[i])

    def plot(self, ax=None, **kwargs):
        from sonore.plotting import plot_modulation_spectrum

        return plot_modulation_spectrum(self, ax=ax, **kwargs)
