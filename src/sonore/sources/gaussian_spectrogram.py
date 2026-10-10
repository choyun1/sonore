"""Random spectrograms with natural correlations (McDermott, Wrobleski &
Oxenham, 2011).

A spectrogram of log amplitudes, one cell per ERB band and time window, is
drawn from a Gaussian whose correlations fall off exponentially in time and
in frequency, as those of spoken words and animal calls roughly do. The
result is :class:`~sonore.views.envelopes.Envelopes`; put it on a carrier
with :meth:`~sonore.views.envelopes.Envelopes.to_sound` to hear it. The
sounds share coarse statistics with natural sources without being any
recognizable one (docs/design/sources/gaussian-spectrogram.md).
"""

from __future__ import annotations

import numpy as np

from sonore.core.utils import as_rng, n_samples, time_axis
from sonore.frames.filterbank import cosine_filterbank
from sonore.views.envelopes import Envelopes

__all__ = ["gaussian_spectrogram"]


def _correlated_field(n_bands: int, n_windows: int, rho_band: float, rho_time: float, rng) -> np.ndarray:
    """Unit-variance Gaussian field (n_bands, n_windows) whose correlation is
    ``rho_band**|band lag| * rho_time**|window lag|``.

    An exponential correlation on a regular grid is a first-order
    autoregression, so the field is white noise run through that recursion
    along time and then along frequency, each run started in its stationary
    distribution. Its covariance equals the separable (Kronecker) one exactly,
    without factoring the full covariance matrix."""
    field = rng.standard_normal((n_bands, n_windows))
    innovation_time, innovation_band = np.sqrt(1 - rho_time**2), np.sqrt(1 - rho_band**2)
    for k in range(1, n_windows):
        field[:, k] = rho_time * field[:, k - 1] + innovation_time * field[:, k]
    for k in range(1, n_bands):
        field[k] = rho_band * field[k - 1] + innovation_band * field[k]
    return field


def gaussian_spectrogram(
    duration: float,
    fs: float,
    band_correlation_erb: float = 8.78,
    time_correlation: float = 0.154,
    sd_db: float = 14.1,
    n_bands: int = 39,
    f_lo: float = 20.0,
    f_hi: float = 4000.0,
    window: float = 0.020,
    rng=None,
) -> Envelopes:
    """A random spectrogram after McDermott, Wrobleski & Oxenham (2011).

    Each cell, one ERB band by one time window, holds a level in dB drawn
    from a multivariate Gaussian. Two cells correlate by
    ``exp(-band distance [ERB] / band_correlation_erb)`` times
    ``exp(-time distance [s] / time_correlation)``. The defaults are the
    paper's: 39 half-cosine ERB filters from 20 to 4000 Hz, 20 ms
    raised-cosine windows overlapping by half, and decay constants of
    0.075 per filter and 0.065 per window, which are about 8.8 ERB and
    154 ms. The paper gives no variance; ``sd_db`` (the standard deviation
    of each cell) defaults to 14.1 dB, a variance of 0.5 in log10 amplitude,
    as in the author's 2017 implementation.

    The mean level of each band rises with its bandwidth in Hz, so the
    long-term spectrum is flat on average, as in the paper. Each band's
    envelope is the sum over windows of the cell amplitude times that
    window's raised cosine; the windows sum to 1, so this is the paper's
    "scale each window to its cell", written as an envelope. The edge bands
    of the filterbank get zero envelopes.

    The sound comes from :meth:`Envelopes.to_sound`. A noise carrier's fine
    structure alone adds no envelope of its own::

        env = so.gaussian_spectrogram(0.4, fs, rng=1)
        sound = env.to_sound(so.gaussian_noise(0.4, fs, rng=2)).ramp(0.01)

    (the paper used 10 ms ramps). Re-filtering through overlapping bands and
    windows makes the sound's own spectrogram differ somewhat from the one
    drawn, as the paper notes.
    """
    if band_correlation_erb <= 0 or time_correlation <= 0:
        raise ValueError("correlation lengths must be positive")
    if sd_db < 0:
        raise ValueError("sd_db must be non-negative")
    if window <= 0:
        raise ValueError("window must be positive")
    rng = as_rng(rng)
    bank = cosine_filterbank(n_bands, f_lo, f_hi)
    hop = window / 2
    length = n_samples(duration, fs)
    t = time_axis(length, fs)
    n_windows = int(np.ceil(t[-1] / hop)) + 1 if length else 1  # centers 0, hop, ... cover the sound

    field = _correlated_field(
        n_bands, n_windows, np.exp(-bank.spacing / band_correlation_erb), np.exp(-hop / time_correlation), rng
    )
    # flat on average: band power in proportion to its width in Hz
    knots = bank.scale.to_scale(bank.band_cfs)
    widths = bank.scale.from_scale(knots + bank.spacing / 2) - bank.scale.from_scale(knots - bank.spacing / 2)
    mean_db = 10 * np.log10(widths / widths.mean())
    cell_amplitude = 10 ** ((mean_db[:, None] + sd_db * field) / 20)

    # raised cosines of width `window` centered every `hop`; they sum to 1
    offset = (t[:, None] - hop * np.arange(n_windows)[None, :]) / window
    windows = np.where(np.abs(offset) < 0.5, 0.5 * (1 + np.cos(2 * np.pi * offset)), 0.0)
    envelopes = np.zeros((length, n_bands + 2))
    envelopes[:, 1:-1] = windows @ cell_amplitude.T
    return Envelopes(envelopes, fs, bank)
