"""Synthetic room impulse responses from the statistics of natural
reverberation (Traer & McDermott, 2016)."""

from __future__ import annotations

from functools import cache
from importlib.resources import files

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import _below_nyquist, as_rng, db_to_amp
from sonore.frames.filterbank import ERBFilterbank
from sonore.signals.generators import gaussian_noise
from sonore.views.envelopes import Envelopes

__all__ = ["synth_ir", "band_rt60s", "measure_rt60", "DECAY_SHAPES", "RT60_PROFILES", "DRR_PROFILES"]


@cache
def _model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_root = files("sonore") / "data"
    load = lambda name: np.load(data_root / name)  # noqa: E731
    return load("fit_DRR.npy"), load("fit_RT60.npy"), load("fit_freqs.npy")


DECAY_SHAPES = ("exponential", "time_reversed", "linear_matched_start", "linear_matched_end")
RT60_PROFILES = ("ecological", "inverted", "exaggerated", "reduced")
DRR_PROFILES = ("ecological", "constant")


def band_rt60s(rt60: float, freqs: np.ndarray, profile: str = "ecological") -> np.ndarray:
    """Per-band RT60s [s] at ``freqs`` for a broadband (median) RT60, from the
    regression fits of Traer & McDermott (2016, Eq. S11), or one of the
    paper's atypical variants (Eq. S15 and following):

    ``"ecological"``: mid frequencies decay slowest, as in real rooms.
    ``"inverted"``: ``max + min - ecological``; slow where real rooms are fast.
    ``"exaggerated"``: the profile of a room twice as reverberant, scaled
    down to this length (more sharply peaked than real rooms).
    ``"reduced"``: the profile of a room half as reverberant, scaled up (flatter).
    """
    _, fit_rt60, fit_freqs = _model()

    def eco(median_rt60):
        return np.interp(freqs, fit_freqs, 10 ** (fit_rt60[:, 0] * np.log10(median_rt60) + fit_rt60[:, 1]))

    if profile == "ecological":
        return eco(rt60)
    if profile == "inverted":
        eco_rt60s = eco(rt60)
        return eco_rt60s.max() + eco_rt60s.min() - eco_rt60s
    if profile == "exaggerated":
        return eco(2 * rt60) / 2
    if profile == "reduced":
        return eco(rt60 / 2) * 2
    raise ValueError(f"rt60_profile must be one of {RT60_PROFILES}")


def _decay_envelopes(t, onset_db, taus, shape):
    """Amplitude envelopes (n, B). Linear variants keep each band's energy equal
    to that of the exponential decay they replace (Traer & McDermott, Eq. S14)."""
    onset_amp = db_to_amp(onset_db)[None, :]
    exp_env = onset_amp * 10 ** (-3 * t[:, None] / taus[None, :])  # -60 dB per RT60
    if shape in ("exponential", "time_reversed"):
        return exp_env
    energy = onset_amp**2 * taus[None, :] / (6 * np.log(10))  # integral of exp_env^2
    if shape == "linear_matched_start":
        t0 = 3 * energy / onset_amp**2  # same starting level; linear energy is onset_amp^2 t0 / 3
        return np.maximum(onset_amp * (1 - t[:, None] / t0), 0.0)
    if shape == "linear_matched_end":
        t0 = taus[None, :]  # reaches zero when the exponential is 60 dB down
        return np.maximum(np.sqrt(3 * energy / t0) * (1 - t[:, None] / t0), 0.0)
    raise ValueError(f"decay_shape must be one of {DECAY_SHAPES}")


def synth_ir(
    rt60: float,
    fs: float,
    drr_db: float | None = None,
    decay_db: float = 60.0,
    n_bands: int = 32,
    f_lo: float = 20.0,
    f_hi: float = 16000.0,
    n_channels: int = 1,
    decay_shape: str = "exponential",
    rt60_profile: str = "ecological",
    drr_profile: str = "ecological",
    rng=None,
) -> Sound:
    """Synthesize a room impulse response (Traer & McDermott, 2016).

    Gaussian noise is split into ERB bands, each band is multiplied by a decay
    envelope, and the bands are re-filtered and summed. With the defaults the
    decay is exponential with frequency-dependent RT60s and onset levels taken
    from the paper's regressions on 271 real-world rooms, scaled to the
    requested broadband ``rt60`` (the median across bands). Listeners can't
    tell such IRs from real ones.

    The paper's "atypical" IRs, which listeners readily hear as unnatural,
    are available through ``decay_shape``, ``rt60_profile`` and
    ``drr_profile``. Note that in the paper each atypical IR was further
    adjusted to distort sounds as much as the ecological one (equating
    cochleagram error); that psychophysical control isn't done here, so
    variants can differ somewhat in loudness and audible length.

    Parameters
    ----------
    rt60
        Median reverberation time [s].
    drr_db
        Direct-to-reverberant energy ratio [dB]. If given, a unit impulse is
        placed at t=0 and the tail is scaled to this DRR. If None, only the
        (unit-RMS) reverberant tail is returned.
    decay_db
        For exponential decays, truncate once the slowest band has decayed
        this many dB.
    n_channels
        Independent tails per channel (e.g. 2 for a decorrelated binaural
        tail); the direct impulse is identical in all channels.
    decay_shape
        ``"exponential"`` (natural), ``"time_reversed"``,
        ``"linear_matched_start"`` (linear from the natural starting level,
        same energy per band), or ``"linear_matched_end"`` (linear, reaching
        zero where the natural decay is 60 dB down, same energy per band).
    rt60_profile
        Frequency dependence of decay; see :func:`band_rt60s`.
    drr_profile
        ``"ecological"`` onset levels per band, or ``"constant"`` (their mean).
    """
    fit_drr, _, fit_freqs = _model()
    rng = as_rng(rng)
    f_hi = min(f_hi, _below_nyquist(fs))
    filterbank = ERBFilterbank(n_bands, f_lo, f_hi)
    cfs = filterbank.cfs

    taus = band_rt60s(rt60, cfs, rt60_profile)
    if drr_profile == "ecological":
        onset_db = fit_drr[:, 0] * np.log10(rt60) + fit_drr[:, 1]
        onset_db = np.interp(cfs, fit_freqs, onset_db - np.median(onset_db))
    elif drr_profile == "constant":
        onset_db = np.zeros(len(cfs))
    else:
        raise ValueError(f"drr_profile must be one of {DRR_PROFILES}")

    if decay_shape in ("exponential", "time_reversed"):
        duration = decay_db * taus.max() / 60
    elif decay_shape == "linear_matched_start":
        duration = taus.max() / (2 * np.log(10))
    else:
        duration = taus.max()
    noise = gaussian_noise(duration, fs, n_channels=n_channels, rng=rng)
    envelopes = _decay_envelopes(noise.t, onset_db, taus, decay_shape)
    if decay_shape == "time_reversed":
        envelopes = envelopes[::-1]
    decay = Envelopes(envelopes, fs, filterbank)  # one decay envelope per band
    # analyze pads by default, so re-filtering the decaying bands can't wrap the
    # loud onset around to the end of the IR
    tail = (decay * filterbank.analyze(noise)).synthesize().normalize()

    if drr_db is None:
        return tail
    # direct impulse has unit energy per channel; scale tail energy to match DRR
    tail_energy = np.sum(tail.data**2, axis=0).mean()
    tail = tail * np.sqrt(db_to_amp(-drr_db) ** 2 / tail_energy)
    data = tail.data.copy()
    data[0, :] += 1.0
    return Sound(data, fs)


def measure_rt60(
    ir: Sound, n_bands: int = 30, f_lo: float = 50.0, f_hi: float = 8000.0, fit_range_db=(-5.0, -25.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band RT60s of an impulse response: ``(center freqs [Hz], RT60s [s])``.

    Each ERB band's energy decay curve (Schroeder backward integration) is fit
    with a line over ``fit_range_db`` and extrapolated to -60 dB (so the
    default measures T20 x 3). The direct sound, if any, should be removed
    first; bands that never decay through the fit range give NaN.
    """
    filterbank = ERBFilterbank(n_bands, f_lo, min(f_hi, _below_nyquist(ir.fs)))
    bands = filterbank.analyze(ir.mono()).data[:, 1:-1, 0]  # bandpass bands only
    energy = np.cumsum(bands[::-1] ** 2, axis=0)[::-1]
    t = np.arange(len(ir)) / ir.fs
    rt60s = np.full(bands.shape[1], np.nan)
    hi_db, lo_db = fit_range_db
    for k in range(bands.shape[1]):
        edc = 10 * np.log10(energy[:, k] / energy[0, k] + 1e-300)
        in_range = (edc <= hi_db) & (edc >= lo_db)
        if in_range.sum() > 10:
            slope = np.polyfit(t[in_range], edc[in_range], 1)[0]
            rt60s[k] = -60 / slope if slope < 0 else np.nan
    return filterbank.cfs[1:-1], rt60s
