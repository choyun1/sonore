"""Sound texture statistics (McDermott & Simoncelli, 2011).

A texture is summarized by time-averaged statistics of an auditory model:
a cochlear filterbank, compressed and downsampled subband envelopes, and
modulation filterbanks applied to those envelopes. ::

    stats = so.texture.TextureStats.measure(rain)
    stats.count()          # 1515 statistics, as in the paper
    stats.save("rain.npz")

Differences from the MATLAB toolbox (v1.7) are listed in one place,
:data:`DIFFERENCES_FROM_TOOLBOX`. This is a clean-room implementation from the
paper; the toolbox was consulted for behavior only.

All filtering here is circular (``pad=0``): the sound is treated as one period
of a periodic signal, which is what makes synthesized textures loop seamlessly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from scipy.signal import hilbert, resample

from sonore.core.fft import threads
from sonore.core.sound import Sound
from sonore.frames.filterbank import ERBFilterbank
from sonore.views.modulation import ConstantQModulationFilterbank, OctaveModulationFilterbank

__all__ = ["TextureModel", "TextureStats", "measurement_window", "STAT_CLASSES", "DIFFERENCES_FROM_TOOLBOX"]

STAT_CLASSES = (
    "env_mean",
    "env_var",
    "env_skew",
    "env_kurt",
    "env_corr",
    "mod_power",
    "c1",
    "c2",
    "subband_var",
)
PAPER_CLASSES = STAT_CLASSES[
    :-1
]  # subband variance is used by synthesis but isn't one of the paper's statistics

DIFFERENCES_FROM_TOOLBOX = """\
Deliberate differences from the MATLAB Sound Texture Synthesis Toolbox v1.7:

1. Envelope variance is var/mean^2, as in the paper (Eq. 2). The toolbox stores
   std/mean. The two carry the same information; var/mean^2 matches the paper.
2. All statistics use the measurement window consistently. The toolbox weights
   the cross-product in the envelope correlation C (and C1) by the window but
   normalizes by *unweighted* standard deviations and means; here means and
   standard deviations are weighted too, so C is a true weighted correlation
   bounded by +/-1.
3. C1 (Eq. 6) is computed as in the paper: no mean subtraction (the bands are
   bandpass, hence zero-mean), normalized by weighted RMS. The toolbox
   subtracts unweighted means.
4. Only the statistics the model uses are measured: C at channel offsets
   {1,2,3,5,8,11,16,21} and C1 at offsets {1,2}. The toolbox computes full
   channel-by-channel matrices.
5. The constant-Q modulation bank is normalized by the mean summed squared
   response between its 4th and 4th-from-last centers, computed on a fine
   frequency grid independent of signal length (the toolbox uses the FFT bins
   of the signal, so the scale depends slightly on duration).
6. Subband variance is weighted by the measurement window (toolbox: unweighted).
7. The sound is truncated to a whole number of envelope samples, so that
   downsampling to the envelope rate is exact.
8. Synthesis: after the conjugate-gradient steps for each channel, the
   envelope's mean and variance are set exactly by an affine map (see
   :func:`sonore.texture.synth.impose_channel`). All other statistics are
   invariant to it, and without it those two directions, nearly flat in the
   objective, converge very slowly (on AM noise: envelope-mean SNR 31 -> 59
   dB and envelope-variance SNR 15 -> 52 dB after 20 iterations).
"""


@dataclass(frozen=True)
class TextureModel:
    """Parameters of the texture model. Defaults are the paper's."""

    fs: float = 20000.0
    rms: float = 0.01  # sounds are normalized to this RMS before measurement
    n_bands: int = 30
    f_lo: float = 20.0
    f_hi: float = 10000.0
    compression: float = 0.3
    env_fs: float = 400.0
    n_mod: int = 20
    mod_lo: float = 0.5
    mod_hi: float = 200.0
    mod_Q: float = 2.0
    n_oct: int = 7
    oct_hi: float = 100.0
    c1_bands: tuple[int, ...] = (1, 2, 3, 4, 5, 6)  # 0-based indices into the octave bank
    c1_offsets: tuple[int, ...] = (1, 2)
    corr_offsets: tuple[int, ...] = (1, 2, 3, 5, 8, 11, 16, 21)

    @property
    def filterbank(self) -> ERBFilterbank:
        return ERBFilterbank(self.n_bands, self.f_lo, self.f_hi)

    @property
    def mod_bank(self) -> ConstantQModulationFilterbank:
        return ConstantQModulationFilterbank(self.n_mod, self.mod_lo, self.mod_hi, self.mod_Q)

    @property
    def oct_bank(self) -> OctaveModulationFilterbank:
        return OctaveModulationFilterbank(self.n_oct, self.oct_hi)

    @property
    def decimation(self) -> int:
        ratio = self.fs / self.env_fs
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("fs must be an integer multiple of env_fs")
        return int(round(ratio))

    def prepare(self, sound: Sound) -> np.ndarray:
        """Mono, resampled to :attr:`fs`, truncated to a whole number of
        envelope samples, RMS-normalized. Returns a 1-D array."""
        mono = sound.mono() if sound.n_channels > 1 else sound
        if mono.fs != self.fs:
            mono = mono.resample(self.fs)
        signal = mono.data[:, 0]
        signal = signal[: len(signal) // self.decimation * self.decimation]
        signal_rms = np.sqrt(np.mean(signal**2))
        if signal_rms == 0:
            raise ValueError("sound is silent")
        return signal * (self.rms / signal_rms)

    def subbands(self, x: np.ndarray) -> np.ndarray:
        """Circular cochlear subbands of a prepared signal, shape ``(n, n_bands + 2)``."""
        return self.filterbank.analyze(Sound(x, self.fs), pad=0).data[:, :, 0]

    def envelopes(self, subbands: np.ndarray) -> np.ndarray:
        """Compressed, downsampled envelopes, shape ``(n // decimation, n_bands + 2)``:
        Hilbert magnitude, raised to :attr:`compression` at the full rate,
        resampled (FFT, circular) to :attr:`env_fs`, clipped at 0."""
        with threads():
            env = np.abs(hilbert(subbands, axis=0)) ** self.compression
            return np.maximum(resample(env, subbands.shape[0] // self.decimation, axis=0), 0.0)


def measurement_window(n: int, n_seconds: int) -> np.ndarray:
    """The toolbox's window for measuring an original: flat, with a
    raised-cosine ramp of ``n // (n_seconds + 1)`` samples at each end (so
    about ``duration / (seconds + 1)``). Normalized to sum to 1."""
    ramp_len = n // (max(int(n_seconds), 1) + 1)
    window = np.ones(n)
    if ramp_len > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, ramp_len + 1) / ramp_len)
        window[:ramp_len] = np.minimum(window[:ramp_len], ramp)
        window[n - ramp_len :] = np.minimum(window[n - ramp_len :], ramp[::-1])
    return window / window.sum()


def _div(numerator, denominator):
    denominator = np.asarray(denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.broadcast(numerator, denominator).shape),
        where=denominator > 1e-300,
    )


def _pair_corr(x: np.ndarray, w: np.ndarray, offsets, centered: bool) -> np.ndarray:
    """Weighted correlation of columns ``j`` and ``j + d`` of ``x`` (n, B, ...)
    for each ``d`` in ``offsets``. Shape ``(B, ..., len(offsets))``; NaN where
    ``j + d`` is out of range."""
    if centered:
        x = x - np.tensordot(w, x, axes=(0, 0))[None]
    power = np.tensordot(w, x**2, axes=(0, 0))  # (B, ...)
    n_channels = x.shape[1]
    corr = np.full(x.shape[1:] + (len(offsets),), np.nan)
    for i, d in enumerate(offsets):
        if d >= n_channels:
            continue
        cross = np.tensordot(w, x[:, : n_channels - d] * x[:, d:], axes=(0, 0))
        corr[: n_channels - d, ..., i] = _div(cross, np.sqrt(power[: n_channels - d] * power[d:]))
    return corr


@dataclass(frozen=True, eq=False)
class TextureStats:
    """The statistics of one texture. Arrays are indexed by cochlear channel
    first (``n_bands + 2`` channels, including the lowpass and highpass
    edges, as in the paper's count of 1515 statistics).

    Attributes
    ----------
    env_mean, env_var, env_skew, env_kurt : (B,)
        Weighted moments of the compressed envelopes; ``env_var`` is var/mean^2.
    env_corr : (B, len(corr_offsets))
        ``env_corr[j, i]`` = correlation of channels ``j`` and ``j + corr_offsets[i]``.
    mod_power : (B, n_mod)
        Modulation power in each constant-Q band, relative to envelope variance.
    c1 : (B, len(c1_bands), len(c1_offsets))
        Correlation of channels ``j`` and ``j + d`` within an octave modulation band.
    c2 : (B, n_oct - 1), complex
        Correlation of each octave modulation band (frequency-doubled) with the
        next one up in the same channel; real and imaginary parts are the
        in-phase and quadrature components.
    subband_var : (B,)
        Variance of each cochlear subband (used to set levels in synthesis).
    """

    model: TextureModel
    env_mean: np.ndarray
    env_var: np.ndarray
    env_skew: np.ndarray
    env_kurt: np.ndarray
    env_corr: np.ndarray
    mod_power: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    subband_var: np.ndarray
    duration: float = field(default=0.0)

    @classmethod
    def measure(cls, sound: Sound, model: TextureModel | None = None, window: str = "ramped") -> TextureStats:
        """Measure the statistics of ``sound``. ``window="ramped"`` (the
        default, for recorded originals) downweights the ends with
        :func:`measurement_window`; ``"uniform"`` weights all samples
        equally (as for circular synthetic signals)."""
        model = TextureModel() if model is None else model
        x = model.prepare(sound)
        return cls.from_subbands(model.subbands(x), model, window)

    @classmethod
    def from_subbands(cls, sb: np.ndarray, model: TextureModel, window: str = "ramped") -> TextureStats:
        env = model.envelopes(sb)
        n_env = env.shape[0]
        env_weights = cls._window(model, n_env, window)
        # the window at the full rate
        full_rate_weights = np.repeat(env_weights, model.decimation) / model.decimation
        subband_variance = full_rate_weights @ (sb - full_rate_weights @ sb) ** 2
        return cls.from_envelopes(env, subband_variance, model, window)

    @staticmethod
    def _window(model: TextureModel, n_env: int, window: str) -> np.ndarray:
        if window == "ramped":
            weights = measurement_window(n_env, int(round(n_env / model.env_fs)))
        elif window == "uniform":
            weights = np.full(n_env, 1.0 / n_env)
        else:
            raise ValueError("window must be 'ramped' or 'uniform'")
        return weights

    @classmethod
    def from_envelopes(
        cls, env: np.ndarray, subband_var: np.ndarray, model: TextureModel, window: str = "uniform"
    ) -> TextureStats:
        """Statistics of compressed, downsampled envelopes ``(n_env, B)``
        (subband variances are passed through). Used during synthesis."""
        n_env = env.shape[0]
        weights = cls._window(model, n_env, window)
        n_samples = n_env * model.decimation

        mean = weights @ env
        deviation = env - mean
        variance = weights @ deviation**2
        env_var = _div(variance, mean**2)
        skewness = _div(weights @ deviation**3, variance**1.5)
        kurtosis = _div(weights @ deviation**4, variance**2)
        corr = _pair_corr(env, weights, model.corr_offsets, centered=True)

        mod_bands = model.mod_bank.filter(env, model.env_fs)  # (n_env, B, M)
        mod_power = _div(np.tensordot(weights, mod_bands**2, axes=(0, 0)), variance[:, None])

        oct_analytic = model.oct_bank.filter(env, model.env_fs, analytic=True)  # (n_env, B, K)
        oct_real = oct_analytic.real
        c1 = _pair_corr(oct_real[:, :, list(model.c1_bands)], weights, model.c1_offsets, centered=False)
        lower_magnitude = np.abs(oct_analytic[:, :, :-1])
        doubled = np.real(_div_complex(oct_analytic[:, :, :-1] ** 2, lower_magnitude))
        upper_band = oct_analytic[:, :, 1:]
        norm = np.sqrt(
            np.tensordot(weights, doubled**2, axes=(0, 0))
            * np.tensordot(weights, upper_band.real**2, axes=(0, 0))
        )
        c2 = _div(np.tensordot(weights, doubled * upper_band.real, axes=(0, 0)), norm) + 1j * _div(
            np.tensordot(weights, doubled * upper_band.imag, axes=(0, 0)), norm
        )

        subband_var_array = np.asarray(subband_var)
        return cls(
            model,
            mean,
            env_var,
            skewness,
            kurtosis,
            corr,
            mod_power,
            c1,
            c2,
            subband_var_array,
            n_samples / model.fs,
        )

    # -- bookkeeping -------------------------------------------------------

    def get(self, name: str) -> np.ndarray:
        if name not in STAT_CLASSES:
            raise KeyError(f"unknown statistic class {name!r}; choose from {STAT_CLASSES}")
        return getattr(self, name)

    def count(self, classes=PAPER_CLASSES) -> int:
        """Number of statistics (defined entries; a complex C2 value counts as
        one, as in the paper). The default counts the paper's classes: 1515."""
        if isinstance(classes, str):
            classes = (classes,)
        return int(sum(np.count_nonzero(~np.isnan(self.get(c))) for c in classes))

    def replace(self, **changes) -> TextureStats:
        """A copy with some classes replaced, e.g. for hybrid textures:
        ``a.replace(mod_power=b.mod_power)``."""
        from dataclasses import replace

        return replace(self, **changes)

    def channel_mask(self, range_db: float = 30.0) -> np.ndarray:
        """Channels whose subband variance is within ``range_db`` of the
        loudest. Quieter channels are ignored by :meth:`snr` (as in the
        toolbox): their statistics are dominated by noise and inaudible."""
        floored_var = np.maximum(self.subband_var, 1e-300)
        return 10 * np.log10(floored_var / floored_var.max()) > -range_db

    def snr(self, other: TextureStats, classes=PAPER_CLASSES, range_db: float = 30.0) -> dict[str, float]:
        """How well ``other`` matches these (target) statistics, per class:
        ``10*log10(sum |target|**2 / sum |target - other|**2)`` in dB, over
        channels within ``range_db`` of the loudest. Pairwise classes (C, C1)
        count a pair only if both channels qualify.

        The SNR is relative to the target's own magnitude, so classes whose
        targets are near zero (C, C1, C2 and skew of noise-like textures)
        score low even between two samples of the same texture: sampling
        fluctuation dominates the ratio."""
        if isinstance(classes, str):
            classes = (classes,)
        loud = self.channel_mask(range_db)
        n_channels = len(loud)
        out = {}
        for stat_class in classes:
            target_vals, other_vals = self.get(stat_class), other.get(stat_class)
            if stat_class in ("env_corr", "c1"):
                offsets = self.model.corr_offsets if stat_class == "env_corr" else self.model.c1_offsets
                pair_loud = np.array(
                    [
                        [loud[j] and j + d < n_channels and loud[j + d] for d in offsets]
                        for j in range(n_channels)
                    ]
                )
                selected = pair_loud[:, None, :] if stat_class == "c1" else pair_loud
                selected = np.broadcast_to(selected, target_vals.shape)
            else:
                selected = np.broadcast_to(
                    loud.reshape((n_channels,) + (1,) * (target_vals.ndim - 1)), target_vals.shape
                )
            target_sel, other_sel = target_vals[selected], other_vals[selected]
            error_energy = np.sum(np.abs(target_sel - other_sel) ** 2)
            out[stat_class] = (
                float("inf")
                if error_energy == 0
                else float(10 * np.log10(np.sum(np.abs(target_sel) ** 2) / error_energy))
            )
        return out

    def save(self, path) -> None:
        arrays = {name: getattr(self, name) for name in STAT_CLASSES}
        meta = {"model": asdict(self.model), "duration": self.duration, "version": 1}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path) -> TextureStats:
        with np.load(path) as npz:
            meta = json.loads(str(npz["meta"]))
            arrays = {name: npz[name] for name in STAT_CLASSES}
        model_fields = {model_field.name: model_field for model_field in fields(TextureModel)}
        model_kwargs = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in meta["model"].items()
            if key in model_fields
        }
        return cls(TextureModel(**model_kwargs), duration=meta["duration"], **arrays)

    def __repr__(self) -> str:
        return f"TextureStats({self.count()} stats, {self.env_mean.shape[0]} channels, {self.duration:.2f} s)"


def _div_complex(numerator, denominator):
    return np.divide(
        numerator, denominator, out=np.zeros(numerator.shape, complex), where=denominator > 1e-300
    )
