"""The modulation spectrogram: how strongly each band's envelope is modulated
at each rate, time window by time window.

An STFT shows how a sound's power spectrum changes over time; a
:class:`ModulationSpectrogram` shows how its modulation spectrum changes over
time. It is built from a filterbank's :class:`~sonore.views.temporal_envelope.TemporalEnvelope` (any
filterbank) by passing every band's envelope through a
:class:`~sonore.views.modulation.HannModulationFilterbank` and sampling the
result every ``hop`` seconds. docs/design/views/modulation-spectrogram.md has the
design and the numbers behind it. :class:`ModulationSpectrogramStream`
computes the causal one a block of envelopes at a time
(docs/design/streaming.md).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from sonore.core.stream import SlidingWindowStream, WindowState
from sonore.core.utils import n_samples
from sonore.views.modulation import HannModulationFilterbank, _correlate
from sonore.views.temporal_envelope import TemporalEnvelope
from sonore.views.view import View

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from sonore.core.sound import Sound
    from sonore.frames.filterbank import Filterbank

__all__ = ["ModulationSpectrogram", "ModulationSpectrogramStream"]


class ModulationSpectrogram(View):
    """Modulation power, local mean and depth for every time window, acoustic band
    and modulation band.

    For band envelope ``e_b`` and modulation band ``k`` with complex kernel
    ``h_k`` and window ``w_k`` (see
    :class:`~sonore.views.modulation.HannModulationFilterbank`), at each
    time window::

        y    = sum_j e_b[n + j - c] conj(h_k[j])     (c: centered or causal)
        mean = sum_j e_b[n + j - c] w_k[j]

    and the stored arrays are ``power = |2 y|**2`` and ``mean``, both of shape
    ``(n_channels, n_bands, n_mod, n_windows)``. :attr:`depth` is
    ``2 |y| / mean``: a sinusoidal AM of depth ``m`` at the band's rate reads
    ``m``, so 100% modulation is 0 dB. Power says how much of the sound a
    modulation is; depth says how modulated the band is.

    The acoustic axis :attr:`f` is the envelopes' filterbank without its edge
    bands, :attr:`fm` holds the modulation rates and :attr:`t` the window
    times. :attr:`valid` (``(n_bands, n_mod, n_windows)``) is False where a
    number can't be trusted: where the modulation rate exceeds the band's
    -3 dB width (a band's envelope can't move faster than the band is wide),
    and where the window runs past either end of the envelopes (outside them
    the envelope is taken to be zero, so an abrupt start reads as
    modulation).

    The representation is not invertible: the phase of ``y`` is dropped, as
    a magnitude spectrogram drops an STFT's phase (the local mean is kept,
    and :attr:`depth` divides by it).
    For modulation filtering, filter the envelopes with a modulation bank
    directly.

    Parameters
    ----------
    envelopes
        Band envelopes, for example ``fb.analyze(snd).envelopes(fs=1000)``.
        Their rate must be at least 3 times the highest modulation rate.
    f_lo, f_hi, per_octave, cycles, window
        The modulation bank (see
        :class:`~sonore.views.modulation.HannModulationFilterbank`):
        ``cycles`` sets a window of that many cycles of each band's rate
        (constant Q); ``window`` [s] one window for every rate (the STFT of
        each envelope).
    hop
        Spacing between time windows [s], independent of the window, as in an STFT.
    align
        ``"center"`` (windows centered on the window times) or ``"causal"`` (windows
        ending at them: what a live analysis would see).
    bank
        A ready-made bank, instead of ``f_lo`` .. ``window``.
    """

    discards = (
        "ModulationSpectrogram discards the fine structure under the envelopes and the phase of every "
        "modulation band, as a magnitude spectrogram drops an STFT's phase."
    )
    back_to_sound = ""

    first_window = 0  # a part of a stream starts later

    def __init__(
        self,
        envelopes: TemporalEnvelope,
        *,
        f_lo: float = 0.5,
        f_hi: float = 64.0,
        per_octave: float = 2,
        cycles: int = 3,
        window: float | None = None,
        hop: float = 0.010,
        align: Literal["center", "causal"] = "center",
        bank: HannModulationFilterbank | None = None,
    ):
        if not isinstance(envelopes, TemporalEnvelope):
            raise TypeError(f"expected a TemporalEnvelope, not {type(envelopes).__name__}")
        if envelopes.filterbank is None:
            raise TypeError(
                "expected a filterbank's envelopes, one per band; this envelope has no filterbank"
            )
        if align not in ("center", "causal"):
            raise ValueError("align must be 'center' or 'causal'")
        if bank is None:
            bank = HannModulationFilterbank(f_lo, f_hi, per_octave, cycles, window)
        fs = envelopes.fs
        bank.check_fs(fs)
        hop_samples = n_samples(hop, fs)
        if hop_samples < 1:
            raise ValueError(f"hop {hop:g} s is shorter than one envelope sample at {fs:g} Hz")

        filterbank = envelopes.filterbank
        band_env = envelopes.data[:, _kept_bands(filterbank), :]  # (n, B, C)
        n_env_samples = band_env.shape[0]
        positions = np.arange(0, n_env_samples, hop_samples)

        y = np.empty((len(positions),) + band_env.shape[1:] + (bank.n_bands,), complex)
        mean = np.empty(y.shape)
        for k, (kernel, hann) in enumerate(bank.kernels(fs)):
            y[..., k] = _correlate(band_env, kernel, align)[positions]
            mean[..., k] = _correlate(band_env, hann, align)[positions].real
        self._fill(y, mean, bank, filterbank, align, fs, hop_samples, positions, n_env_samples)

    def _fill(self, y, mean, bank, filterbank, align, fs, hop_samples, positions, n_env_samples) -> None:
        """Set the arrays and axes from ``y`` and ``mean`` of shape ``(F, B, C, K)``."""
        keep = _kept_bands(filterbank)
        # (F, B, C, K) -> (C, B, K, F)
        self.power = np.transpose(np.abs(2 * y) ** 2, (2, 1, 3, 0))
        self.mean = np.maximum(np.transpose(mean, (2, 1, 3, 0)), 0.0)

        self.bank, self.filterbank, self.align = bank, filterbank, align
        self.fs, self.hop = fs, hop_samples / fs
        self.f = np.asarray(filterbank.cfs)[keep]
        self.fm = bank.cfs
        self.t = positions / fs
        self.valid = self._valid(filterbank, keep, n_env_samples, positions)

    @classmethod
    def concatenate(cls, parts: Sequence[ModulationSpectrogram]) -> ModulationSpectrogram:
        """Join the parts of a stream (:class:`ModulationSpectrogramStream`)
        in order. When the parts cover the whole envelope, the result equals
        ``ModulationSpectrogram(envelopes, align="causal", ...)`` to within
        1e-12 of its peak (a stream computes each time window directly, the
        batch path by FFT)."""
        parts = list(parts)
        if not parts:
            raise ValueError("no parts to join")
        first = parts[0]
        position = first.first_window
        for part in parts:
            if not isinstance(part, ModulationSpectrogram):
                raise TypeError(f"concatenate joins ModulationSpectrograms, not {type(part).__name__}")
            same = (part.bank, part.filterbank, part.align, part.fs, part.hop) == (
                first.bank,
                first.filterbank,
                first.align,
                first.fs,
                first.hop,
            )
            if not same or part.power.shape[:3] != first.power.shape[:3]:
                raise ValueError("the parts come from different analyses")
            if part.first_window != position:
                raise ValueError(
                    f"parts are not consecutive: time window {part.first_window} after {position}"
                )
            position += part.power.shape[-1]
        joined = object.__new__(cls)
        joined.__dict__.update(first.__dict__)
        for name in ("power", "mean", "valid", "t"):
            setattr(joined, name, np.concatenate([getattr(part, name) for part in parts], axis=-1))
        return joined

    def _valid(self, fb, keep, n: int, positions: np.ndarray) -> np.ndarray:
        bandwidth = _band_widths(fb)[keep]  # (B,)
        rate_ok = self.fm[None, :] <= bandwidth[:, None]  # (B, K)
        lengths = self.bank.lengths(self.fs)[:, None]  # (K, 1)
        if self.align == "causal":
            time_ok = positions[None, :] >= lengths - 1
        else:
            before = lengths - 1 - (lengths - 1) // 2  # samples before the time window's own
            after = (lengths - 1) // 2
            time_ok = (positions[None, :] >= before) & (positions[None, :] + after <= n - 1)
        return rate_ok[:, :, None] & time_ok[None, :, :]

    def __repr__(self) -> str:
        n_channels, n_bands, n_mod, n_windows = self.power.shape
        return (
            f"ModulationSpectrogram({n_bands} bands x {n_mod} rates ({self.fm[0]:.3g}-{self.fm[-1]:.3g} Hz), "
            f"{n_windows} time windows every {1000 * self.hop:g} ms, {n_channels} ch, {self.align})"
        )

    @property
    def depth(self) -> np.ndarray:
        """Modulation depth ``2 |y| / mean``, same shape as :attr:`power`;
        NaN where the window holds only digital silence."""
        tiny = 1e-12 * self.mean.max(axis=(1, 2, 3), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.mean > tiny, np.sqrt(self.power) / self.mean, np.nan)

    def at(self, t: float, *, kind: Literal["depth", "power"] = "depth") -> np.ndarray:
        """The acoustic band x modulation rate image at the time window nearest
        ``t`` [s], shape ``(n_channels, n_bands, n_mod)``: Atlas and Shamma's
        joint acoustic and modulation frequency display at one moment.
        ``kind`` is ``"depth"`` or ``"power"``."""
        window_index = int(np.argmin(np.abs(self.t - t)))
        return self._kind(kind)[..., window_index]

    def average(self, *, kind: Literal["depth", "power"] = "power") -> np.ndarray:
        """Average over the time windows where the cell is :attr:`valid`,
        shape ``(n_channels, n_bands, n_mod)``; NaN where a band and rate has
        no valid window (its rate exceeds the band's width, or no window of
        that rate fits inside the sound). The average power is the per-band
        modulation power spectrum, the same kind of quantity as the texture
        statistics' ``mod_power``."""
        values = self._kind(kind)
        counted = self.valid[None] & ~np.isnan(values)
        n_counted = counted.sum(axis=-1)
        total = np.sum(np.where(counted, values, 0.0), axis=-1)
        return np.where(n_counted > 0, total / np.maximum(n_counted, 1), np.nan)

    def pooled_depth(self) -> np.ndarray:
        """Depth pooled over acoustic bands, shape ``(n_channels, n_mod,
        n_windows)``: ``sqrt(sum_b power) / sqrt(sum_b mean**2)`` over the
        bands whose cell is :attr:`valid`, which weights bands by their level.
        NaN where no band is valid."""
        valid = self.valid[None]
        power_sum = np.sum(np.where(valid, self.power, 0.0), axis=1)
        mean_sq_sum = np.sum(np.where(valid, self.mean**2, 0.0), axis=1)
        tiny = 1e-24 * mean_sq_sum.max(axis=(1, 2), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(mean_sq_sum > tiny, np.sqrt(power_sum / mean_sq_sum), np.nan)

    def plot(
        self, ax: Axes | None = None, *, band: float | None = None, rate: float | None = None, **kwargs
    ) -> Axes:
        """Depth in dB as an image, invalid cells in gray. By default,
        modulation rate against time, pooled over bands
        (:meth:`pooled_depth`); ``band=`` [Hz] shows the acoustic band
        nearest that frequency instead; ``rate=`` [Hz] shows acoustic band
        against time at the modulation band nearest that rate."""
        from sonore.plotting import plot_modulation_spectrogram

        return plot_modulation_spectrogram(self, ax=ax, band=band, rate=rate, **kwargs)

    def slices(self, t: float, *, rate: float = 4.0, **kwargs) -> Figure:
        """Three linked cuts through the time x band x rate cube with a cursor
        at ``t`` [s]: rate against time (pooled), band against time at
        ``rate``, and band against rate at ``t``. Returns the figure."""
        from sonore.plotting import plot_modulation_slices

        return plot_modulation_slices(self, t, rate=rate, **kwargs)

    def animate(
        self,
        *,
        path: str | os.PathLike | None = None,
        sound: Sound | None = None,
        fps: float = 25.0,
        **kwargs,
    ) -> FuncAnimation:
        """The band x rate image (depth, dB) time window by time window, as a matplotlib
        animation. With ``path``, it is written to a video file (needs
        ffmpeg), and with ``sound`` as well, the sound becomes its audio
        track. Returns the animation."""
        from sonore.plotting import animate_modulation_spectrogram

        return animate_modulation_spectrogram(self, path=path, sound=sound, fps=fps, **kwargs)

    def _kind(self, kind: Literal["depth", "power"]) -> np.ndarray:
        match kind:
            case "depth":
                return self.depth
            case "power":
                return self.power
            case _:
                raise ValueError("kind must be 'depth' or 'power'")


@dataclass(frozen=True)
class ModulationSpectrogramStream(SlidingWindowStream):
    """The causal :class:`ModulationSpectrogram` (``align="causal"``), a
    block of envelopes at a time. The parameters are the spectrogram's, and
    ``filterbank`` is the one whose envelopes the blocks are. Each step
    returns a ModulationSpectrogram of the time windows that block
    completed, and :meth:`ModulationSpectrogram.concatenate` joins them::

        bank = so.gammatone_filterbank(n_bands=32, edges=False)
        pooling = so.EnvelopePooling(2e-3, 1e-3, bank)     # envelopes at 1 kHz
        modulation = so.ModulationSpectrogramStream(bank)
        subbands = so.stream(bank, so.blocks(snd, 441))
        envelopes = so.stream(pooling, (part.envelopes() for part in subbands))
        parts = list(so.stream(modulation, envelopes))
        so.ModulationSpectrogram.concatenate(parts)
        # == so.ModulationSpectrogram(envelopes_of_snd, align="causal"), within 1e-12 of its peak

    The time window at ``t`` holds the envelope up to ``t``, so the stream
    keeps the last samples of the longest kernel (6 s for the default 0.5 Hz
    band) and gives each time window as soon as its sample arrives. It
    computes each kept time window directly, ``sum_j e[n + j - L + 1]
    conj(h[j])``, where the batch path convolves by FFT and keeps every
    hop-th sample: the two agree to about 1e-15 of the peak, not bit for bit
    (computing every batch time window directly would be about 70 times
    slower; docs/design/streaming.md). Only the causal alignment streams:
    the centered one needs half a kernel of the future.
    """

    filterbank: Filterbank
    f_lo: float = 0.5
    f_hi: float = 64.0
    per_octave: float = 2
    cycles: int = 3
    window: float | None = None
    hop: float = 0.010
    bank: HannModulationFilterbank | None = None
    _modulation_bank: HannModulationFilterbank = field(init=False, repr=False, compare=False)

    block_type: ClassVar[type[Any]] = TemporalEnvelope

    def __post_init__(self):
        bank = self.bank or HannModulationFilterbank(
            self.f_lo, self.f_hi, self.per_octave, self.cycles, self.window
        )
        object.__setattr__(self, "_modulation_bank", bank)

    def window_layout(self, fs: float) -> tuple[int, int, int]:
        self._modulation_bank.check_fs(fs)
        hop = n_samples(self.hop, fs)
        if hop < 1:
            raise ValueError(f"hop {self.hop:g} s is shorter than one envelope sample at {fs:g} Hz")
        longest = int(self._modulation_bank.lengths(fs).max())
        return longest, hop, -(longest - 1)  # time window i ends at sample i * hop

    def window_count(self, n_samples: int, fs: float) -> int:
        _, hop, _ = self.window_layout(fs)
        return -(-n_samples // hop)

    def time_windows(
        self, samples: np.ndarray, fs: float, first: int, count: int, n_samples: int
    ) -> ModulationSpectrogram:
        longest, hop, _ = self.window_layout(fs)
        bank = self._modulation_bank
        # (B, C, n), so that each stretch of samples is contiguous for the products below
        band_env = np.ascontiguousarray(np.moveaxis(samples[:, _kept_bands(self.filterbank), :], 0, -1))
        y = np.empty((count, *band_env.shape[:2], bank.n_bands), complex)
        mean = np.empty(y.shape)
        for k, weights in enumerate(_real_kernels(bank, fs) if count else []):
            # time window i holds the kernel's length of samples ending at the window's own sample
            length = len(weights)
            tail = band_env[..., longest - length : longest + (count - 1) * hop]
            stretches = np.lib.stride_tricks.sliding_window_view(tail, length, axis=-1)[..., ::hop, :]
            sums = np.moveaxis(stretches @ weights, -2, 0)  # (count, B, C, 3)
            y[..., k] = sums[..., 0] + 1j * sums[..., 1]
            mean[..., k] = sums[..., 2]
        part = object.__new__(ModulationSpectrogram)
        positions = (first + np.arange(count)) * hop
        part._fill(y, mean, bank, self.filterbank, "causal", fs, hop, positions, n_samples)
        part.first_window = first
        return part

    def step(self, state: WindowState, block: TemporalEnvelope) -> tuple[WindowState, ModulationSpectrogram]:
        """Take ``block``, envelopes of this stream's filterbank, and return
        every time window now complete."""
        if isinstance(block, self.block_type):
            if block.filterbank != self.filterbank:
                raise ValueError("the block's envelopes are of a different filterbank than this stream's")
            if block.pad:
                raise ValueError("a block of a stream has no padding")
        return super().step(state, block)


def _kept_bands(filterbank) -> slice:
    """The bands a modulation spectrogram keeps: all of them, or all but the edge filters."""
    return slice(None) if getattr(filterbank, "edges", True) is False else slice(1, -1)


@lru_cache(maxsize=16)
def _real_kernels(bank: HannModulationFilterbank, fs: float) -> list[np.ndarray]:
    """Each of the bank's kernels at ``fs`` as the real ``(length, 3)``
    matrix ``[Re conj(h), Im conj(h), w]``: envelopes are real, so one real
    product gives ``y`` and the mean together, with no complex copy of the
    samples. Made once for every step of a stream."""
    return [np.stack([np.conj(h).real, np.conj(h).imag, w], axis=1) for h, w in bank.kernels(fs)]


@lru_cache(maxsize=16)
def _band_widths(fb) -> np.ndarray:
    """-3 dB width [Hz] of every filter of ``fb``, from its own responses
    (computed once per filterbank, read-only)."""
    cfs = np.asarray(fb.cfs, float)
    # a log-spaced grid: fine steps (0.03%) at every center frequency
    freqs = np.geomspace(max(cfs.min(), 1.0) / 4, 2.0 * cfs.max(), 20001)
    freq_step = np.gradient(freqs)
    magnitude = np.abs(np.asarray(fb.response(freqs)))  # (F, n_filters)
    widths = np.sum((magnitude >= magnitude.max(axis=0) / np.sqrt(2)) * freq_step[:, None], axis=0)
    widths.flags.writeable = False
    return widths
