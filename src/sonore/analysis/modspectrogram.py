"""The modulation spectrogram: how strongly each band's envelope is modulated
at each rate, frame by frame.

An STFT shows how a sound's power spectrum changes over time; a
:class:`ModulationSpectrogram` shows how its modulation spectrum changes over
time. It is built from :class:`~sonore.analysis.envelopes.Envelopes` (any
filterbank) by passing every band's envelope through a
:class:`~sonore.analysis.modulation.HannModulationFilterbank` and sampling the
result every ``hop`` seconds. docs/design/modulation-spectrogram.md has the
design and the numbers behind it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sonore.analysis.modulation import HannModulationFilterbank, _correlate

if TYPE_CHECKING:
    from sonore.analysis.envelopes import Envelopes

__all__ = ["ModulationSpectrogram"]


class ModulationSpectrogram:
    """Modulation power, local mean and depth for every frame, acoustic band
    and modulation band.

    For band envelope ``e_b`` and modulation band ``k`` with complex kernel
    ``h_k`` and window ``w_k`` (see
    :class:`~sonore.analysis.modulation.HannModulationFilterbank`), at each
    frame::

        y    = sum_j e_b[n + j - c] conj(h_k[j])     (c: centred or causal)
        mean = sum_j e_b[n + j - c] w_k[j]

    and the stored arrays are ``power = |2 y|**2`` and ``mean``, both of shape
    ``(n_channels, n_bands, n_mod, n_frames)``. :attr:`depth` is
    ``2 |y| / mean``: a sinusoidal AM of depth ``m`` at the band's rate reads
    ``m``, so 100% modulation is 0 dB. Power says how much of the sound a
    modulation is; depth says how modulated the band is.

    The acoustic axis :attr:`f` is the envelopes' filterbank without its edge
    bands, :attr:`fm` holds the modulation rates and :attr:`t` the frame
    times. :attr:`valid` (``(n_bands, n_mod, n_frames)``) is False where a
    number can't be trusted: where the modulation rate exceeds the band's
    -3 dB width (a band's envelope can't move faster than the band is wide),
    and where the window runs past either end of the envelopes (outside them
    the envelope is taken to be zero, so an abrupt start reads as
    modulation).

    The representation is not invertible: the phase of ``y`` is dropped and
    the mean divided out, as a magnitude spectrogram drops an STFT's phase.
    For modulation filtering, filter the envelopes with a modulation bank
    directly.

    Parameters
    ----------
    envelopes
        Band envelopes, for example ``fb.analyze(snd).envelopes(fs=1000)``.
        Their rate must be at least 3 times the highest modulation rate.
    f_lo, f_hi, per_octave, cycles, window
        The modulation bank (see
        :class:`~sonore.analysis.modulation.HannModulationFilterbank`):
        ``cycles`` sets a window of that many cycles of each band's rate
        (constant Q); ``window`` [s] one window for every rate (the STFT of
        each envelope).
    hop
        Frame step [s], independent of the window, as in an STFT.
    align
        ``"center"`` (windows centred on the frames) or ``"causal"`` (windows
        ending at the frames: what a live analysis would see).
    bank
        A ready-made bank, instead of ``f_lo`` .. ``window``.
    """

    def __init__(
        self,
        envelopes: Envelopes,
        f_lo: float = 0.5,
        f_hi: float = 64.0,
        per_octave: float = 2,
        cycles: int = 3,
        window: float | None = None,
        hop: float = 0.010,
        align: str = "center",
        bank: HannModulationFilterbank | None = None,
    ):
        from sonore.analysis.envelopes import Envelopes

        if not isinstance(envelopes, Envelopes):
            raise TypeError(f"expected Envelopes, not {type(envelopes).__name__}")
        if align not in ("center", "causal"):
            raise ValueError("align must be 'center' or 'causal'")
        if bank is None:
            bank = HannModulationFilterbank(f_lo, f_hi, per_octave, cycles, window)
        fs = envelopes.fs
        bank.check_fs(fs)
        hop_samples = int(round(hop * fs))
        if hop_samples < 1:
            raise ValueError(f"hop {hop:g} s is shorter than one envelope sample at {fs:g} Hz")

        filterbank = envelopes.filterbank
        band_env = envelopes.data  # (n, B, C)
        keep = slice(None) if getattr(filterbank, "edges", True) is False else slice(1, -1)
        band_env = band_env[:, keep, :]
        n_samples = band_env.shape[0]
        frames = np.arange(0, n_samples, hop_samples)

        y = np.empty((len(frames),) + band_env.shape[1:] + (bank.n_bands,), complex)
        mean = np.empty(y.shape)
        for k, (kernel, hann) in enumerate(bank.kernels(fs)):
            y[..., k] = _correlate(band_env, kernel, align)[frames]
            mean[..., k] = _correlate(band_env, hann, align)[frames].real
        # (F, B, C, K) -> (C, B, K, F)
        self.power = np.transpose(np.abs(2 * y) ** 2, (2, 1, 3, 0))
        self.mean = np.maximum(np.transpose(mean, (2, 1, 3, 0)), 0.0)

        self.bank, self.filterbank, self.align = bank, filterbank, align
        self.fs, self.hop = fs, hop_samples / fs
        self.f = np.asarray(filterbank.cfs)[keep]
        self.fm = bank.cfs
        self.t = frames / fs
        self.valid = self._valid(filterbank, keep, n_samples, frames)

    def _valid(self, fb, keep, n: int, frames: np.ndarray) -> np.ndarray:
        bandwidth = _band_widths(fb)[keep]  # (B,)
        rate_ok = self.fm[None, :] <= bandwidth[:, None]  # (B, K)
        lengths = self.bank.lengths(self.fs)[:, None]  # (K, 1)
        if self.align == "causal":
            time_ok = frames[None, :] >= lengths - 1
        else:
            before = lengths - 1 - (lengths - 1) // 2  # samples before the frame's own
            after = (lengths - 1) // 2
            time_ok = (frames[None, :] >= before) & (frames[None, :] + after <= n - 1)
        return rate_ok[:, :, None] & time_ok[None, :, :]

    def __repr__(self) -> str:
        n_channels, n_bands, n_mod, n_frames = self.power.shape
        return (
            f"ModulationSpectrogram({n_bands} bands x {n_mod} rates ({self.fm[0]:.3g}-{self.fm[-1]:.3g} Hz), "
            f"{n_frames} frames every {1000 * self.hop:g} ms, {n_channels} ch, {self.align})"
        )

    @property
    def depth(self) -> np.ndarray:
        """Modulation depth ``2 |y| / mean``, same shape as :attr:`power`;
        NaN where the window holds only digital silence."""
        tiny = 1e-12 * self.mean.max(axis=(1, 2, 3), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.mean > tiny, np.sqrt(self.power) / self.mean, np.nan)

    def at(self, t: float, kind: str = "depth") -> np.ndarray:
        """The acoustic band x modulation rate image at the frame nearest
        ``t`` [s], shape ``(n_channels, n_bands, n_mod)``: Atlas and Shamma's
        joint acoustic and modulation frequency display at one moment.
        ``kind`` is ``"depth"`` or ``"power"``."""
        frame_index = int(np.argmin(np.abs(self.t - t)))
        return self._kind(kind)[..., frame_index]

    def average(self, kind: str = "power") -> np.ndarray:
        """Average over frames, shape ``(n_channels, n_bands, n_mod)``. The
        average power is the per-band modulation power spectrum, the same kind
        of quantity as the texture statistics' ``mod_power``."""
        return np.nanmean(self._kind(kind), axis=-1)

    def pooled_depth(self) -> np.ndarray:
        """Depth pooled over acoustic bands, shape ``(n_channels, n_mod,
        n_frames)``: ``sqrt(sum_b power) / sqrt(sum_b mean**2)`` over the
        bands whose cell is :attr:`valid`, which weights bands by their level.
        NaN where no band is valid."""
        valid = self.valid[None]
        power_sum = np.sum(np.where(valid, self.power, 0.0), axis=1)
        mean_sq_sum = np.sum(np.where(valid, self.mean**2, 0.0), axis=1)
        tiny = 1e-24 * mean_sq_sum.max(axis=(1, 2), keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(mean_sq_sum > tiny, np.sqrt(power_sum / mean_sq_sum), np.nan)

    def plot(self, ax=None, band: float | None = None, rate: float | None = None, **kwargs):
        """Depth in dB as an image, invalid cells in grey. By default,
        modulation rate against time, pooled over bands
        (:meth:`pooled_depth`); ``band=`` [Hz] shows the acoustic band
        nearest that frequency instead; ``rate=`` [Hz] shows acoustic band
        against time at the modulation band nearest that rate."""
        from sonore.plotting import plot_modulation_spectrogram

        return plot_modulation_spectrogram(self, ax=ax, band=band, rate=rate, **kwargs)

    def slices(self, t: float, rate: float = 4.0, **kwargs):
        """Three linked cuts through the time x band x rate cube with a cursor
        at ``t`` [s]: rate against time (pooled), band against time at
        ``rate``, and band against rate at ``t``. Returns the figure."""
        from sonore.plotting import plot_modulation_slices

        return plot_modulation_slices(self, t, rate=rate, **kwargs)

    def animate(self, path=None, sound=None, fps: float = 25.0, **kwargs):
        """The band x rate image (depth, dB) frame by frame, as a matplotlib
        animation. With ``path``, it is written to a video file (needs
        ffmpeg), and with ``sound`` as well, the sound becomes its audio
        track. Returns the animation."""
        from sonore.plotting import animate_modulation_spectrogram

        return animate_modulation_spectrogram(self, path=path, sound=sound, fps=fps, **kwargs)

    def _kind(self, kind: str) -> np.ndarray:
        if kind == "depth":
            return self.depth
        if kind == "power":
            return self.power
        raise ValueError("kind must be 'depth' or 'power'")


def _band_widths(fb) -> np.ndarray:
    """-3 dB width [Hz] of every filter of ``fb``, from its own responses."""
    cfs = np.asarray(fb.cfs, float)
    # a log-spaced grid: fine steps (0.03%) at every centre frequency
    freqs = np.geomspace(max(cfs.min(), 1.0) / 4, 2.0 * cfs.max(), 20001)
    freq_step = np.gradient(freqs)
    magnitude = np.abs(np.asarray(fb.response(freqs)))  # (F, n_filters)
    return np.sum((magnitude >= magnitude.max(axis=0) / np.sqrt(2)) * freq_step[:, None], axis=0)
