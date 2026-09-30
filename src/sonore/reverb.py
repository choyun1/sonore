"""Synthetic room impulse responses from the statistics of natural
reverberation (Traer & McDermott, 2016)."""

from __future__ import annotations

from functools import cache
from importlib.resources import files

import numpy as np

from sonore.envelopes import Envelopes
from sonore.filterbank import ERBFilterbank
from sonore.generators import gaussian_noise
from sonore.sound import Sound
from sonore.utils import as_rng, db_to_amp

__all__ = ["synth_ir"]


@cache
def _model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = files("sonore") / "data"
    load = lambda name: np.load(d / name)  # noqa: E731
    return load("fit_DRR.npy"), load("fit_RT60.npy"), load("fit_freqs.npy")


def synth_ir(
    rt60: float,
    fs: float,
    drr_db: float | None = None,
    decay_db: float = 60.0,
    n_bands: int = 32,
    f_lo: float = 20.0,
    f_hi: float = 16000.0,
    n_channels: int = 1,
    envelope: str = "exponential",
    rng=None,
) -> Sound:
    """Synthesize a room impulse response.

    Band-limited Gaussian noise gets an exponential decay in each ERB band,
    with frequency-dependent RT60s and onset levels taken from the regression
    fits of Traer & McDermott (2016), scaled to the requested broadband
    ``rt60`` (the median across bands).

    Parameters
    ----------
    rt60
        Median reverberation time [s].
    drr_db
        Direct-to-reverberant energy ratio [dB]. If given, a unit impulse is
        placed at t=0 and the tail is scaled to this DRR. If None, only the
        (unit-RMS) reverberant tail is returned.
    decay_db
        Truncate once the slowest band has decayed this many dB.
    n_channels
        Independent tails per channel (e.g. 2 for a decorrelated binaural
        tail); the direct impulse is identical in all channels.
    envelope
        ``"exponential"``, or ``"time_reversed"`` for a reversed-decay tail.
    """
    fit_drr, fit_rt60, fit_f = _model()
    rng = as_rng(rng)
    f_hi = min(f_hi, 0.95 * fs / 2)
    fb = ERBFilterbank(n_bands, f_lo, f_hi)
    cfs = fb.cfs

    # per-band RT60 and relative onset level, interpolated onto our band centers
    band_rt60 = 10 ** (fit_rt60[:, 0] * np.log10(rt60) + fit_rt60[:, 1])
    band_rt60 = np.interp(cfs, fit_f, band_rt60)
    onset_db = fit_drr[:, 0] * np.log10(rt60) + fit_drr[:, 1]
    onset_db = np.interp(cfs, fit_f, onset_db - np.median(onset_db))

    dur = decay_db * band_rt60.max() / 60
    noise = gaussian_noise(dur, fs, n_channels=n_channels, rng=rng)
    t = noise.t
    env_db = onset_db[None, :] - 60 * t[:, None] / band_rt60[None, :]  # (n, B)
    if envelope == "time_reversed":
        env_db = env_db[::-1]
    elif envelope != "exponential":
        raise ValueError("envelope must be 'exponential' or 'time_reversed'")
    decay = Envelopes(db_to_amp(env_db), fs, fb)  # one exponential decay per band
    tail = (decay * fb.analyze(noise)).synthesize().normalize()

    if drr_db is None:
        return tail
    # direct impulse has unit energy per channel; scale tail energy to match DRR
    tail_energy = np.sum(tail.data**2, axis=0).mean()
    tail = tail * np.sqrt(db_to_amp(-drr_db) ** 2 / tail_energy)
    data = tail.data.copy()
    data[0, :] += 1.0
    return Sound(data, fs)
