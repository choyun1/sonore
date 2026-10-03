"""Modulation filterbanks: bandpass filters applied to envelopes.

Two banks from McDermott & Simoncelli (2011), usable on their own (e.g. for
Dau-style modulation models) and by :mod:`sonore.texture`:

* :class:`ConstantQModulationFilterbank`: half-cycle cosines on a *linear*
  frequency axis, each spanning ``cf +/- cf/Q``, with centers log-spaced. Used
  for the modulation power spectrum.
* :class:`OctaveModulationFilterbank`: half-cycle cosines on a log2 axis, each
  two octaves wide (``cf/2 .. 2*cf``), centers one octave apart. Used for the
  C1/C2 modulation correlations.

Filtering is done in the frequency domain and is therefore *circular*: the
envelope is treated as one period of a periodic signal. That is exact for
texture synthesis (seamless loops); for other uses pad the envelope first.

A third bank, :class:`HannModulationFilterbank`, is defined in time instead:
Hann-windowed complex exponentials of finite length, applied by direct
(zero-padded, not circular) correlation, centred or causal. It is the bank
behind :class:`~sonore.views.modspectrogram.ModulationSpectrogram`, and
its finite kernels are what make a causal, block-by-block version possible.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import fftconvolve

from sonore.core.fft import threads
from sonore.core.sound import Sound
from sonore.core.utils import _below_nyquist, amp_to_db, as_rng, db_to_amp
from sonore.frames.gabor import _FLOOR_DB, STFT
from sonore.views.view import View

if TYPE_CHECKING:
    from sonore.views.envelopes import Envelopes

__all__ = [
    "ModulationFilterbank",
    "ConstantQModulationFilterbank",
    "OctaveModulationFilterbank",
    "HannModulationFilterbank",
    "ModulationSpectrum",
    "ModulationBlob",
]


@dataclass(frozen=True)
class ModulationFilterbank:
    """Base class. Subclasses define ``n_bands``, :attr:`cfs` and :meth:`response`.

    Despite the name, a modulation filterbank is not a
    :class:`~sonore.frames.filterbank.Filterbank`: it filters envelopes to
    make views such as :class:`~sonore.views.modspectrogram.ModulationSpectrogram`,
    and has no synthesis.
    """

    @property
    def cfs(self) -> np.ndarray:
        raise NotImplementedError

    def response(self, freqs) -> np.ndarray:
        """Magnitude responses, shape ``(len(freqs), n_bands)`` (``freqs`` >= 0)."""
        raise NotImplementedError

    def filter(self, x, fs: float, axis: int = 0, analytic: bool = False) -> np.ndarray:
        """Filter ``x`` (circularly) along ``axis`` with every band.

        The band index is appended as a new *last* axis. With ``analytic=True``
        the output is the complex analytic signal of each band (negative
        frequencies removed, positive ones doubled), whose real part equals the
        ordinary filtered output.
        """
        x = np.moveaxis(np.asarray(x, float), axis, 0)
        n_samples = x.shape[0]
        trailing_axes = (None,) * (x.ndim - 1)
        if not analytic:
            transfer = self.response(np.fft.rfftfreq(n_samples, 1 / fs))  # (F, M)
            spectrum = np.fft.rfft(x, axis=0)[..., None]
            filtered = np.fft.irfft(spectrum * transfer[(slice(None),) + trailing_axes], n=n_samples, axis=0)
        else:
            freqs = np.fft.fftfreq(n_samples, 1 / fs)
            transfer = self.response(np.abs(freqs))
            gain = np.where(freqs > 0, 2.0, np.where(freqs == 0, 1.0, 0.0))
            if n_samples % 2 == 0:
                gain[n_samples // 2] = 1.0  # Nyquist bin is its own conjugate
            transfer = transfer * gain[:, None]
            spectrum = np.fft.fft(x, axis=0)[..., None]
            filtered = np.fft.ifft(spectrum * transfer[(slice(None),) + trailing_axes], axis=0)
        return np.moveaxis(filtered, 0, axis) if axis != 0 else filtered


@dataclass(frozen=True)
class ConstantQModulationFilterbank(ModulationFilterbank):
    """``n_bands`` constant-Q filters with centers log-spaced from ``f_lo`` to
    ``f_hi`` [Hz]. Filter ``k`` is a half-cycle cosine on a linear frequency
    axis, nonzero on ``cf_k*(1 - 1/Q) .. cf_k*(1 + 1/Q)``.

    Responses are scaled so that the summed squared response, averaged over
    the middle of the bank (between the 4th and the 4th-from-last center),
    is 1. That makes band powers sum to about the envelope's variance.
    """

    n_bands: int = 20
    f_lo: float = 0.5
    f_hi: float = 200.0
    Q: float = 2.0

    @property
    def cfs(self) -> np.ndarray:
        return np.geomspace(self.f_lo, self.f_hi, self.n_bands)

    @property
    def scale(self) -> float:
        cfs = self.cfs
        cf_lo, cf_hi = cfs[min(3, len(cfs) - 1)], cfs[max(len(cfs) - 4, 0)]
        freqs = np.linspace(cf_lo, cf_hi, 4097) if cf_hi > cf_lo else np.array([cf_lo])
        return float(1 / np.sqrt(np.mean(np.sum(self._raw(freqs) ** 2, axis=1))))

    def _raw(self, freqs) -> np.ndarray:
        freq_col = np.asarray(freqs, float)[:, None]
        cf_row = self.cfs[None, :]
        position = (freq_col - cf_row) / (2 * cf_row / self.Q)  # -1/2 .. 1/2 across the support
        return np.where(np.abs(position) < 0.5, np.cos(np.pi * np.clip(position, -0.5, 0.5)), 0.0)

    def response(self, freqs) -> np.ndarray:
        return self._raw(freqs) * self.scale


@dataclass(frozen=True)
class OctaveModulationFilterbank(ModulationFilterbank):
    """``n_bands`` filters with centers one octave apart, the highest at
    ``f_hi`` [Hz] (default: 1.5625, 3.125, ..., 100 Hz). Filter ``k`` is a
    half-cycle cosine on log2 frequency spanning ``cf_k/2 .. 2*cf_k``. Peak
    gain is 1 (the C1/C2 correlations are scale-invariant, so no further
    normalization is applied)."""

    n_bands: int = 7
    f_hi: float = 100.0

    @property
    def cfs(self) -> np.ndarray:
        return self.f_hi / 2.0 ** np.arange(self.n_bands - 1, -1, -1)

    def response(self, freqs) -> np.ndarray:
        freq_col = np.asarray(freqs, float)[:, None]
        with np.errstate(divide="ignore"):
            octave_offset = (np.log2(freq_col) - np.log2(self.cfs)[None, :]) / 2  # octaves / width
        return np.where(np.abs(octave_offset) < 0.5, np.cos(np.pi * np.clip(octave_offset, -0.5, 0.5)), 0.0)


@dataclass(frozen=True)
class HannModulationFilterbank(ModulationFilterbank):
    """Hann-windowed complex exponentials, defined in time.

    Band ``k`` correlates the envelope with ``h_k[j] = w_k[j] exp(i 2 pi f_k
    (j - (L_k - 1)/2) / fs)``, where ``w_k`` is a Hann window of ``L_k``
    samples normalized to sum 1. Two ways to set the window:

    * ``cycles`` (the default, 3): every window holds that many cycles of its
      own rate, ``L_k = cycles * fs / f_k``, with centres ``per_octave`` to
      the octave from ``f_lo`` to ``f_hi``. A constant-Q bank (Q = cycles /
      1.44, about 2.1 for 3 cycles), long windows for slow rates and short
      ones for fast rates.
    * ``window`` [s]: every band uses the same window T, with centres on the
      linear grid ``k / T`` from the first one at or above ``max(f_lo, 2/T)``
      to ``f_hi``. This is the STFT of the envelope.

    Either way each window holds a whole number (at least 2) of cycles of its
    rate, so the bank ignores the envelope's mean: the Hann window's transform
    is zero at every whole bin from 2 on.

    Unlike the other two banks, :meth:`filter` works in time and is not
    circular: the envelope is taken to be zero outside its extent. The
    analytic output ``2 y`` has magnitude ``A * m`` for an envelope component
    ``A * m * cos(2 pi f_k t)``, and its real part is the real filtered
    output, whose frequency response is :meth:`response`.
    """

    f_lo: float = 0.5
    f_hi: float = 64.0
    per_octave: float = 2
    cycles: int = 3
    window: float | None = None

    def __post_init__(self):
        if self.window is None:
            if self.cycles != int(self.cycles) or self.cycles < 2:
                raise ValueError("cycles must be a whole number of at least 2, so the bank ignores the mean")
            if not 0 < self.f_lo <= self.f_hi:
                raise ValueError("need 0 < f_lo <= f_hi")
        elif self.window <= 0 or self.f_hi < max(self.f_lo, 2 / self.window):
            raise ValueError("need window > 0 and f_hi >= max(f_lo, 2 / window)")

    @property
    def n_bands(self) -> int:
        return len(self.cfs)

    @property
    def cfs(self) -> np.ndarray:
        """Centre rates [Hz]."""
        if self.window is None:
            n_cfs = int(np.floor(np.log2(self.f_hi / self.f_lo) * self.per_octave + 1e-9)) + 1
            return self.f_lo * 2.0 ** (np.arange(n_cfs) / self.per_octave)
        k_lo = max(2, int(np.ceil(self.f_lo * self.window - 1e-9)))
        k_hi = int(np.floor(self.f_hi * self.window + 1e-9))
        return np.arange(k_lo, k_hi + 1) / self.window

    def durations(self) -> np.ndarray:
        """Window length [s] of each band."""
        if self.window is None:
            return self.cycles / self.cfs
        return np.full(self.n_bands, float(self.window))

    def lengths(self, fs: float) -> np.ndarray:
        """Window length in samples at ``fs`` (rounded, at least 3)."""
        return np.maximum(np.round(self.durations() * fs).astype(int), 3)

    def check_fs(self, fs: float) -> None:
        """The envelope rate must be at least 3 * f_hi: the top band reaches
        about 1.25 * f_hi, and its window must hold enough samples."""
        if fs < 3 * self.cfs.max():
            raise ValueError(
                f"envelope rate {fs:g} Hz is too low for modulation rates up to {self.cfs.max():g} Hz; "
                f"resample the envelopes to at least {3 * self.cfs.max():g} Hz (1000 Hz is a good default)"
            )

    def kernels(self, fs: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(h_k, w_k)`` for every band at ``fs``: the complex kernel and its
        normalized Hann window."""
        kernels = []
        for rate, length in zip(self.cfs, self.lengths(fs), strict=True):
            hann = np.sin(np.pi * (np.arange(length) + 0.5) / length) ** 2
            hann = hann / hann.sum()
            t_centred = (np.arange(length) - (length - 1) / 2) / fs
            kernels.append((hann * np.exp(2j * np.pi * rate * t_centred), hann))
        return kernels

    def response(self, freqs) -> np.ndarray:
        """Magnitude response of the real filter (the real part of
        :meth:`filter`'s analytic output), shape ``(len(freqs), n_bands)``,
        for continuous-time windows. Peak gain is about 1; zero at 0 Hz."""
        freq_col = np.asarray(freqs, float)[:, None]
        durations, rates = self.durations()[None, :], self.cfs[None, :]
        return np.abs(_hann_ft((freq_col - rates) * durations) + _hann_ft((freq_col + rates) * durations))

    def filter(
        self, x, fs: float, axis: int = 0, analytic: bool = False, align: str = "center"
    ) -> np.ndarray:
        """Filter ``x`` along ``axis`` with every band (zero outside ``x``).

        The band index is appended as a new *last* axis. ``align="center"``
        puts each window's middle on the output sample; ``"causal"`` ends it
        there, which is the centred output delayed by ``(L_k - 1) // 2``
        samples and is what a live, block-by-block analysis would produce.
        """
        if align not in ("center", "causal"):
            raise ValueError("align must be 'center' or 'causal'")
        self.check_fs(fs)
        x = np.moveaxis(np.asarray(x, float), axis, 0)
        filtered = np.empty(x.shape + (self.n_bands,), complex)
        for k, (kernel, _) in enumerate(self.kernels(fs)):
            filtered[..., k] = 2 * _correlate(x, kernel, align)
        filtered = filtered if analytic else filtered.real
        return np.moveaxis(filtered, 0, axis) if axis != 0 else filtered


def _hann_ft(u: np.ndarray) -> np.ndarray:
    """Fourier transform of a unit-area Hann window of length 1, at ``u``
    cycles per window length: sinc(u) / (1 - u^2), with its limit at |u| = 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        transform = np.sinc(u) / (1 - u * u)
    return np.where(np.isclose(np.abs(u), 1.0), 0.5, transform)


def _correlate(x: np.ndarray, h: np.ndarray, align: str) -> np.ndarray:
    """``y[n] = sum_j x[n + j - c] conj(h[j])`` along axis 0, with ``x`` zero
    outside its extent; ``c = L - 1`` (causal) or ``L - 1 - (L - 1) // 2``
    (centred)."""
    n_samples, L = x.shape[0], len(h)
    reversed_conj_kernel = np.conj(h)[::-1].reshape((L,) + (1,) * (x.ndim - 1))
    with threads():
        full = fftconvolve(x, reversed_conj_kernel, axes=0)  # full[m] = sum_j x[m - L + 1 + j] conj(h[j])
    start = 0 if align == "causal" else (L - 1) // 2
    return full[start : start + n_samples]


# ------------------------------------------------------ modulation spectrum
@dataclass(frozen=True)
class ModulationBlob:
    """A patch of modulation power, for drawing a target spectrum in code
    (:meth:`ModulationSpectrum.from_blobs`).

    A Gaussian bump centred on ``rate`` [Hz] and ``density`` [cycles/octave]:
    ``rate_width`` is its standard deviation in octaves of rate,
    ``density_width`` in cycles/octave, and ``level`` its peak power [dB]
    relative to the other blobs. The signs follow :class:`~sonore.Ripple`:
    positive rate and density are downward sweeps, a negative rate upward
    ones, and a blob at density 0 is temporal modulation alone. A ripple is a
    blob of zero width.
    """

    rate: float
    density: float
    rate_width: float = 0.5
    density_width: float = 0.25
    level: float = 0.0

    def __post_init__(self):
        if self.rate == 0:
            raise ValueError("a blob needs a nonzero rate (its width is in octaves of rate)")
        if self.rate_width <= 0 or self.density_width <= 0:
            raise ValueError("widths must be positive")

    def power(self, rate: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Relative power at every ``(rate, density)`` (broadcast), zero on
        the other side of the rate axis."""
        same_side = np.sign(rate) == np.sign(self.rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            octaves = np.log2(np.abs(rate) / abs(self.rate))
        bump = np.exp(
            -(octaves**2) / (2 * self.rate_width**2)
            - (density - self.density) ** 2 / (2 * self.density_width**2)
        )
        return np.where(same_side, 10 ** (self.level / 10) * bump, 0.0)


class ModulationSpectrum(View):
    """2-D Fourier transform of a time-frequency envelope (Singh & Theunissen,
    2003; Chi et al., 1999).

    ``ModulationSpectrum(stft)`` uses a dB spectrogram, so spectral modulation
    is w.r.t. *linear* frequency (cycles/kHz). :meth:`octave` uses subband
    envelopes on a log-frequency axis (cycles/octave), the axis on which
    ripples (:mod:`sonore.sources.ripples`) are defined; more generally, any
    :class:`~sonore.views.envelopes.Envelopes` has ``.modulation_spectrum()``.

    Sign convention: a ripple ``sin(2*pi*(rate*t + density*x))`` appears at
    ``(+rate, +density)``. Only non-negative spectral modulations are stored
    (the other half is the complex conjugate).

    A view: before the transform the envelope's mean is removed and a Hann
    taper applied in time, and only the level (dB, floored) is kept, so the
    phase of the modulations is dropped and no envelope can be read back.

    A spectrum made from :class:`~sonore.views.envelopes.Envelopes` (by
    :meth:`octave` or ``env.modulation_spectrum()``) also keeps, for
    :meth:`to_sound`, the magnitude of the untapered transform on the whole
    plane, the envelope's mean, and how the envelopes were made. Edit it with
    :meth:`with_gain`, and hear it with :meth:`to_sound`, whose carrier
    supplies the phases it lacks (see ``docs/design/views/modulation-targets.md``).
    """

    discards = (
        "ModulationSpectrum keeps only the magnitude of the 2-D Fourier transform of an envelope: it "
        "discards the phase of the modulations, which holds the timing of every event, and the fine "
        "structure under the envelope."
    )
    back_to_sound = (
        "ModulationSpectrum.to_sound(carrier=...) takes both from a carrier: a Sound lends its own "
        "modulation phase and fine structure, 'tones' and 'noise' draw a random modulation phase."
    )
    _analysis = None  # an _EnvelopeAnalysis when made from Envelopes or blobs
    _rms_depth = None  # set by from_blobs, which refuses to clip

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
        self._mean = float(env.mean())
        self._magnitude = np.abs(np.fft.fft2(env - self._mean))  # untapered, whole plane, for to_sound
        env = (env - self._mean) * np.hanning(env.shape[1])[None, :]
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

            fb = cosine_filterbank(f_lo=f_lo, f_hi=f_hi, spacing=1 / bands_per_octave, scale="octave")
            fb.analyze(sound.mono()).envelopes(fs=env_fs).modulation_spectrum(scale)
        """
        from sonore.frames.filterbank import cosine_filterbank

        f_hi = min(f_hi, _below_nyquist(sound.fs))
        fb = cosine_filterbank(f_lo=f_lo, f_hi=f_hi, spacing=1 / bands_per_octave, scale="octave")
        return fb.analyze(sound.mono()).envelopes(fs=env_fs).modulation_spectrum(scale)

    @classmethod
    def from_blobs(
        cls,
        blobs,
        duration: float,
        f_lo: float = 250.0,
        f_hi: float = 8000.0,
        bands_per_octave: float = 12,
        env_fs: float = 1000.0,
        rms_depth: float = 0.2,
    ) -> ModulationSpectrum:
        """A target spectrum drawn in code: the sum of :class:`ModulationBlob`
        powers, on the grid :meth:`octave` would measure for a sound of
        ``duration`` seconds, ready for :meth:`to_sound`.

        A drawing sets a shape, not a depth, and no long-term spectrum: every
        band gets the same mean envelope, and ``rms_depth`` scales the
        modulation (the rms of the envelope array about its mean, relative to
        the mean). Because envelopes cannot go below zero, a random draw
        reaches only a limited depth: about 0.28 for a one-blob target in
        ``docs/design/views/modulation-targets.md`` (C2), against 0.71 for one
        full ripple. :meth:`to_envelopes` refuses a draw that would need
        clipping and names the largest depth that fits it. :attr:`level`
        shows the drawn power itself (no taper).
        """
        from sonore.core.utils import n_samples
        from sonore.frames.filterbank import cosine_filterbank
        from sonore.views.envelopes import _EnvelopeAnalysis

        if isinstance(blobs, ModulationBlob):
            blobs = [blobs]
        if not blobs:
            raise ValueError("give at least one ModulationBlob")
        if rms_depth <= 0:
            raise ValueError("rms_depth must be positive")
        bank = cosine_filterbank(f_lo=f_lo, f_hi=f_hi, spacing=1 / bands_per_octave, scale="octave")
        n_bands, n_times = bank.n_filters - 2, n_samples(duration, env_fs)
        new = cls.__new__(cls)
        new._analysis = _EnvelopeAnalysis(bank, env_fs, n_times, "linear", True)
        rate, density = new._full_axes(shape=(n_bands, n_times))
        power = sum(blob.power(rate, density) for blob in blobs)
        power = (power + np.roll(power[::-1, ::-1], 1, axis=(0, 1))) / 2  # the same at (-rate, -density)
        magnitude = np.sqrt(power)
        if not np.any(magnitude > 0):
            raise ValueError("the blobs fall outside this grid's rates and densities")
        # Parseval: the rms of an array is the norm of its unnormalized 2-D DFT over the cell count
        new._mean = 1.0
        new._magnitude = magnitude * (rms_depth * magnitude.size / np.linalg.norm(magnitude))
        new._rms_depth = rms_depth
        keep = np.fft.fftfreq(n_bands, bank.spacing) >= 0
        new.w_f = np.fft.fftfreq(n_bands, bank.spacing)[keep]
        new.w_t = np.fft.fftshift(np.fft.fftfreq(n_times, 1 / env_fs))
        new.level = amp_to_db(np.fft.fftshift(new._magnitude[keep], axes=1), floor_db=_FLOOR_DB)
        new.spectral_unit = f"cyc/{bank.unit}"
        return new

    def _full_axes(self, shape=None) -> tuple[np.ndarray, np.ndarray]:
        """Rate [Hz] (row) and density (column) of every cell of the
        untapered transform, in FFT order."""
        n_density, n_times = self._magnitude.shape if shape is None else shape
        analysis = self._analysis
        rate = np.fft.fftfreq(n_times, 1 / analysis.fs)[None, :]
        density = np.fft.fftfreq(n_density, analysis.filterbank.spacing)[:, None]
        return rate, density

    def _require_envelopes(self, what: str) -> None:
        if self._analysis is None:
            raise TypeError(
                f"{what} needs a modulation spectrum made from Envelopes (ModulationSpectrum.octave or "
                "env.modulation_spectrum()); this one was made from an STFT or an array"
            )

    def with_gain(self, gain) -> ModulationSpectrum:
        """A copy with every cell's amplitude multiplied by ``gain``.

        ``gain`` is a function ``g(rate, density)`` of rate [Hz] and spectral
        modulation, evaluated with ``rate`` as a row and ``density`` as a
        column (as ripple patterns are), or a number. A sound's spectrum is
        the same at ``(rate, density)`` and ``(-rate, -density)``, so the gain
        is averaged over each such pair; ``g = lambda r, d: abs(r) <= 4``
        removes every modulation faster than 4 Hz, ``lambda r, d: r * d <= 0``
        every downward sweep. The displayed :attr:`level` (Hann-tapered) gets
        the same gain, which is exact where the gain is smooth across the
        taper's rate resolution.
        """
        self._require_envelopes("with_gain")
        rate, density = self._full_axes()
        values = gain(rate, density) if callable(gain) else gain
        values = np.broadcast_to(np.asarray(values, float), self._magnitude.shape)
        if np.any(values < 0):
            raise ValueError("gains must be non-negative")
        mirrored = np.roll(values[::-1, ::-1], 1, axis=(0, 1))  # the value at (-rate, -density)
        values = (values + mirrored) / 2
        new = copy.copy(self)
        new._magnitude = self._magnitude * values
        shown = np.fft.fftshift(values[: len(self.w_f)], axes=1)  # the level's rows and column order
        new.level = np.maximum(self.level + amp_to_db(shown, floor_db=-400.0), _FLOOR_DB)
        return new

    def to_envelopes(self, carrier: Sound | None = None, rng=None) -> Envelopes:
        """The envelopes this spectrum describes, given a modulation phase.

        The stored magnitudes and mean fix everything but the phase of the
        2-D transform, which holds when each event happens and how the bands
        line up. A :class:`~sonore.Sound` ``carrier`` lends the phase of its
        own envelopes (analysed as this spectrum's were), so the spectrum of
        ``x`` with ``carrier=x`` gives back ``x``'s envelopes; with no carrier
        the phase is drawn at random from ``rng``. Envelopes rebuilt from a
        linear-scale spectrum can go below zero; they are clipped, with a
        warning saying how much, which changes their spectrum. Edge bands
        that the spectrum dropped are zero.
        """
        from sonore.views.envelopes import Envelopes

        self._require_envelopes("to_envelopes")
        analysis = self._analysis
        bank, n_env = analysis.filterbank, analysis.n_samples
        kept = slice(1, -1) if analysis.drop_edges else slice(None)
        if carrier is None:
            phase = np.angle(np.fft.fft2(as_rng(rng).standard_normal(self._magnitude.shape)))
        else:
            if rng is not None:
                raise TypeError("rng applies only when no carrier is given")
            n_audio = int(round(n_env * carrier.fs / analysis.fs))
            if carrier.n_samples < n_audio:
                raise ValueError(f"the carrier is shorter than the analysed {n_env / analysis.fs:g} s")
            own_sound = Sound(carrier.mono().data[:n_audio], carrier.fs)
            own = bank.analyze(own_sound).envelopes(fs=analysis.fs).data.mean(axis=2)[:, kept]
            if own.shape[0] != n_env:
                raise ValueError("the carrier's envelopes do not fit the analysed grid")
            if analysis.scale == "db":
                own = amp_to_db(own + 1e-12 * (own.max() or 1.0))
            phase = np.angle(np.fft.fft2(own.T - own.mean()))
        rebuilt = self._mean + np.real(np.fft.ifft2(self._magnitude * np.exp(1j * phase)))
        if analysis.scale == "db":
            values = db_to_amp(rebuilt)
        else:
            clipped = float(np.mean(rebuilt < -1e-9 * (np.max(np.abs(rebuilt)) or 1.0)))  # not round-off
            if clipped > 0 and self._rms_depth is not None:
                fits = self._rms_depth * self._mean / np.max(self._mean - rebuilt)
                raise ValueError(
                    f"rms_depth {self._rms_depth:g} would push {clipped:.1%} of this draw's envelope values "
                    f"below zero; at most {fits:.3g} fits it (or draw again with another rng)"
                )
            if clipped > 0:
                warnings.warn(
                    f"{clipped:.1%} of the rebuilt envelope values were below zero and were clipped, "
                    "which changes their modulation spectrum",
                    stacklevel=3,
                )
            values = np.maximum(rebuilt, 0.0)
        bands = np.zeros((n_env, bank.n_filters))
        bands[:, kept] = values.T
        return Envelopes(bands, analysis.fs, bank)

    def to_sound(self, carrier: str | Sound = "tones", fs: float | None = None, rng=None) -> Sound:
        """A sound whose envelopes have this modulation spectrum.

        A modulation spectrum lacks two kinds of phase, and the ``carrier``
        supplies both:

        * the **modulation phase** (when each event happens, how the bands
          line up; see :meth:`to_envelopes`): a :class:`~sonore.Sound` lends
          its own, ``"tones"`` and ``"noise"`` draw a random one from ``rng``;
        * the **fine structure** under each band's envelope: a Sound's own
          (``(envelopes * subbands.tfs()).to_sound()``, the vocoder's route),
          narrowband noise (``"noise"``, the same route), or a steady tone at
          each band's centre (``"tones"``), added without re-filtering.

        ``to_sound(carrier=x)`` on the spectrum of ``x`` itself rebuilds
        ``x``'s envelopes, so an edit made with :meth:`with_gain` keeps the
        sound's timing wherever the gain is 1. ``fs`` is the audio rate,
        needed unless the carrier is a Sound, which must be at least as long
        as the analysed envelopes. The result has RMS 1.

        A fine structure that fluctuates within a band (a sound's own, or
        noise) adds modulation of its own when the result is analysed again,
        so an edit survives best on ``"tones"``: removing every rate above
        4 Hz from a sentence leaves about 15 dB less power at 6-40 Hz on
        tones, but only 3-5 dB less on noise or the sentence's own fine
        structure (``docs/design/views/modulation-targets.md``, C6).
        """
        from sonore.frames.filterbank import Subbands
        from sonore.sources.waveforms import gaussian_noise

        if isinstance(carrier, Sound):
            if rng is not None:
                raise TypeError("rng applies only to carrier='tones' or 'noise'")
            if fs is not None and fs != carrier.fs:
                raise ValueError(f"fs {fs} differs from the carrier's {carrier.fs}")
            envelopes = self.to_envelopes(carrier)
            fs = carrier.fs
        elif carrier in ("tones", "noise"):
            if fs is None:
                raise TypeError("fs is needed for carrier='tones' or 'noise'")
            rng = as_rng(rng)
            envelopes = self.to_envelopes(rng=rng)
        else:
            raise ValueError(f"carrier must be 'tones', 'noise' or a Sound, not {carrier!r}")
        analysis = self._analysis
        bank = analysis.filterbank
        n_audio = int(round(analysis.n_samples * fs / analysis.fs))

        if isinstance(carrier, Sound):
            fine = bank.analyze(Sound(carrier.mono().data[:n_audio], fs)).tfs()
            sound = (envelopes * fine).to_sound()
        elif carrier == "noise":
            fine = bank.analyze(gaussian_noise(n_audio / fs, fs, rng=rng)).tfs()
            sound = (envelopes * fine).to_sound()
        else:
            t = np.arange(n_audio)[:, None] / fs
            tones = np.cos(2 * np.pi * bank.cfs[None, :] * t + rng.uniform(0, 2 * np.pi, bank.n_filters))
            sound = (envelopes * Subbands(tones[:, :, None], fs, bank)).sum()
        sound = Sound(sound.data[:n_audio], fs)
        return sound.normalize() if sound.rms > 0 else sound

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
