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
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

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

__all__ = ["ModulationSpectrogram", "ModulationSpectrogramStream", "ModulationState"]


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
        1e-12 of its peak (a stream carries sliding sums, the batch path
        convolves by FFT)."""
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
class ModulationSpectrogramStream:
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

    The time window at ``t`` holds the envelope up to ``t``, so each time
    window is ready as soon as its sample arrives. Only the causal alignment
    streams: the centered one needs half a kernel of the future.

    How it is computed: a kernel is a Hann window times a complex
    exponential, and ``sin(pi (j + 1/2) / L)**2 = 1/2 - (e**(i theta (j +
    1/2)) + e**(-i theta (j + 1/2))) / 4`` with ``theta = 2 pi / L``, so
    ``y`` and the mean are each a few sliding sums ``S_n(a) = sum_j e[n - L
    + 1 + j] a**j`` with ``|a| = 1``. A sliding sum moves on by one sample
    with ``S_n = conj(a) (S_(n-1) - e[n - L]) + e[n] a**(L - 1)``, so a
    sample costs the same for every kernel length, where the direct sum
    costs the kernel's length for every time window. To keep rounding from
    building up, every RESTART_EVERY samples (counted from the start of the
    stream) the sums are computed again directly from the last samples, and
    in between they are carried from that restart. The result agrees with
    the batch path, which convolves by FFT, to within 3.2e-14 of its peak
    over 10 minutes, without growing, but not bit for bit
    (tools/check_streaming_claims.py, C12).
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

    RESTART_EVERY: ClassVar[int] = 1024
    PIECE: ClassVar[int] = 256  # divides RESTART_EVERY

    def __post_init__(self):
        bank = self.bank or HannModulationFilterbank(
            self.f_lo, self.f_hi, self.per_octave, self.cycles, self.window
        )
        object.__setattr__(self, "_modulation_bank", bank)

    def start(self, fs: float) -> ModulationState:
        """Nothing seen yet: zeros before the first sample, as in batch."""
        fs = float(fs)
        self._modulation_bank.check_fs(fs)
        if n_samples(self.hop, fs) < 1:
            raise ValueError(f"hop {self.hop:g} s is shorter than one envelope sample at {fs:g} Hz")
        return ModulationState(fs, None, 0, None, None, -1)

    def step(
        self, state: ModulationState, block: TemporalEnvelope
    ) -> tuple[ModulationState, ModulationSpectrogram]:
        """Take ``block``, envelopes of this stream's filterbank, and return
        every time window now complete."""
        if not isinstance(block, TemporalEnvelope):
            raise TypeError(f"a block is a TemporalEnvelope, not {type(block).__name__}")
        if block.fs != state.fs:
            raise ValueError(f"this stream runs at {state.fs:g} Hz; the block is at {block.fs:g} Hz")
        if block.filterbank != self.filterbank:
            raise ValueError("the block's envelopes are of a different filterbank than this stream's")
        if block.pad:
            raise ValueError("a block of a stream has no padding")
        fs, layout = state.fs, _sliding_sums(self._modulation_bank, state.fs)
        hop = n_samples(self.hop, fs)
        samples = block.data[:, _kept_bands(self.filterbank), :]  # (m, B, C)
        first_window = -(-state.n_seen // hop)
        if state.history is None:
            if not len(samples):  # an empty block does not fix the bands and channels
                return state, self._part(
                    np.zeros((0, *samples.shape[1:], len(layout))), None, fs, first_window
                )
            shape = samples.shape[1:]
            state = ModulationState(
                fs,
                np.zeros((layout.longest, *shape)),
                0,
                np.zeros((len(layout), 5, *shape), complex),
                np.zeros((len(layout), 5, *shape), complex),
                -1,
            )
        assert state.history is not None and state.restart_sums is not None and state.carried is not None
        if samples.shape[1:] != state.history.shape[1:]:
            raise ValueError(
                f"this stream's envelopes have {state.history.shape[1:]} bands and channels; "
                f"the block's {samples.shape[1:]}"
            )
        buffer = np.concatenate([state.history, samples])
        origin = state.n_seen - layout.longest  # the sample buffer[0] is
        restart_sums, carried, restart = state.restart_sums, state.carried, state.restart
        y_parts, mean_parts = [], []
        position, end = state.n_seen, state.n_seen + len(samples)
        while position < end:
            if position % self.RESTART_EVERY == 0:
                restart = position - 1
                restart_sums = layout.direct(buffer, restart - origin)
                carried = np.zeros_like(carried)
            # at most PIECE samples at a time, which bounds the arrays a long block needs
            stop = min(end, (position // self.PIECE + 1) * self.PIECE)
            windows = np.arange(-(-position // hop) * hop, stop, hop)
            y, mean, carried = layout.carry(
                buffer, origin, position, stop, restart, restart_sums, carried, windows
            )
            y_parts.append(y)
            mean_parts.append(mean)
            position = stop
        new_state = ModulationState(fs, buffer[-layout.longest :], end, restart_sums, carried, restart)
        y = np.concatenate(y_parts) if y_parts else np.zeros((0, *samples.shape[1:], len(layout)), complex)
        mean = np.concatenate(mean_parts) if mean_parts else np.zeros(y.shape)
        return new_state, self._part(y, mean, fs, first_window)

    def finish(self, state: ModulationState) -> ModulationSpectrogram:
        """Nothing: every time window ends at a sample already seen."""
        n_bands = len(self.filterbank.cfs[_kept_bands(self.filterbank)])
        n_channels = 1 if state.history is None else state.history.shape[-1]
        n_mod = self._modulation_bank.n_bands
        hop = n_samples(self.hop, state.fs)
        return self._part(np.zeros((0, n_bands, n_channels, n_mod)), None, state.fs, -(-state.n_seen // hop))

    def _part(self, y, mean, fs, first_window) -> ModulationSpectrogram:
        """The part holding time windows ``first_window`` on, from ``y`` and ``mean`` ``(F, B, C, K)``."""
        hop = n_samples(self.hop, fs)
        mean = np.zeros(y.shape) if mean is None else mean
        part = object.__new__(ModulationSpectrogram)
        positions = (first_window + np.arange(len(y))) * hop
        part._fill(y, mean, self._modulation_bank, self.filterbank, "causal", fs, hop, positions, 0)
        part.first_window = first_window
        return part


@dataclass(frozen=True)
class ModulationState:
    """The state of a :class:`ModulationSpectrogramStream`: the last samples
    (as many as the longest kernel, ``(L, n_bands, n_channels)``; ``None``
    before the first block), how many have arrived, and each kernel's five
    sliding sums at the last restart and what has been carried since."""

    fs: float
    history: np.ndarray | None
    n_seen: int
    restart_sums: np.ndarray | None  # (n_kernels, 5, n_bands, n_channels) complex
    carried: np.ndarray | None
    restart: int  # the sample the restart sums end at


class _SlidingSumLayout:
    """Each kernel's five sliding sums: their bases ``a = e**(i phi)``, and
    the weights that turn them into ``y`` and the mean (see
    :class:`ModulationSpectrogramStream`)."""

    def __init__(self, bank: HannModulationFilterbank, fs: float):
        self.lengths = [int(length) for length in bank.lengths(fs)]
        self.longest = max(self.lengths)
        self.phis, self.y_weights, self.mean_weights = [], [], []
        for rate, length in zip(bank.cfs, self.lengths, strict=True):
            omega, theta, center = 2 * np.pi * rate / fs, 2 * np.pi / length, (length - 1) / 2
            hann_sum = np.sum(np.sin(np.pi * (np.arange(length) + 0.5) / length) ** 2)
            self.phis.append(np.array([0.0, theta, -omega, theta - omega, -(theta + omega)]))
            turn = np.exp(1j * omega * center)
            self.y_weights.append(
                np.array(
                    [0, 0, turn / 2, -np.exp(0.5j * theta) * turn / 4, -np.exp(-0.5j * theta) * turn / 4]
                )
                / hann_sum
            )
            self.mean_weights.append(np.array([0.5, -0.5 * np.exp(0.5j * theta), 0, 0, 0]) / hann_sum)

        self.length_array = np.array(self.lengths)
        self.phi = np.array(self.phis)  # (K, 5)
        self.y_weight_array, self.mean_weight_array = np.array(self.y_weights), np.array(self.mean_weights)

    def __len__(self) -> int:
        return len(self.lengths)

    def direct(self, buffer: np.ndarray, last: int) -> np.ndarray:
        """Every sliding sum of the stretch ending at ``buffer[last]``, summed directly."""
        sums = np.empty((len(self), 5, *buffer.shape[1:]), complex)
        for k, (length, phi) in enumerate(zip(self.lengths, self.phis, strict=True)):
            stretch = buffer[last - length + 1 : last + 1]
            powers = np.exp(1j * phi[:, None] * np.arange(length)[None, :])  # (5, L)
            sums[k] = np.tensordot(powers, stretch, axes=(1, 0))
        return sums

    def carry(self, buffer, origin, start, stop, restart, restart_sums, carried, windows):
        """``y`` and the mean at the time windows ``windows`` (samples in
        ``start .. stop - 1``), and the carried sums after sample
        ``stop - 1``, from the sums at ``restart``. All kernels at once."""
        phi = self.phi[:, :, None]  # (K, 5, 1)
        arriving = buffer[start - origin : stop - origin]  # (m, B, C)
        leaving = buffer[
            np.arange(start, stop)[None, :] - origin - self.length_array[:, None]
        ]  # (K, m, B, C)
        since_restart = np.arange(start, stop) - restart  # 1 .. RESTART_EVERY
        # u_n = e[n] a**(L - 1) - conj(a) e[n - L], weighted by a**(n - restart)
        turn = np.exp(1j * phi * since_restart[None, None, :])  # (K, 5, m)
        update = (turn * np.exp(1j * phi * (self.length_array[:, None, None] - 1)))[
            ..., None, None
        ] * arriving
        update -= (turn * np.exp(-1j * phi))[..., None, None] * leaving[:, None]
        running = carried[:, :, None] + np.cumsum(update, axis=2)  # (K, 5, m, B, C)
        at = windows - start
        sums = (restart_sums[:, :, None] + running[:, :, at]) / turn[:, :, at, None, None]  # (K, 5, F, B, C)
        y = np.einsum("kf,kfwbc->wbck", self.y_weight_array, sums)
        mean = np.einsum("kf,kfwbc->wbck", self.mean_weight_array, sums).real
        return y, mean, running[:, :, -1]


@lru_cache(maxsize=16)
def _sliding_sums(bank: HannModulationFilterbank, fs: float) -> _SlidingSumLayout:
    return _SlidingSumLayout(bank, fs)


def _kept_bands(filterbank) -> slice:
    """The bands a modulation spectrogram keeps: all of them, or all but the edge filters."""
    return slice(None) if getattr(filterbank, "edges", True) is False else slice(1, -1)


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
