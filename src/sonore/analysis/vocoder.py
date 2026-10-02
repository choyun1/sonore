"""WORLD's analysis, ported exactly: the CheapTrick spectral envelope and the
D4C aperiodicity (Morise, 2015, 2016), plus a harmonic-residual
aperiodicity that measures the share of noise directly.

Every step named after WORLD reproduces WORLD's C++ code (github.com/mmorise/World,
commit d625e76) to floating-point precision, including the tiny noise WORLD
adds to keep logarithms and divisions finite, drawn from WORLD's own
generator in WORLD's order. Where an option departs from WORLD it is listed
in :data:`DIFFERENCES_FROM_WORLD`. The synthesis is
:func:`sonore.stimuli.vocoder.world_synthesize`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sonore.analysis.f0 import F0Track
from sonore.core.sound import Sound

__all__ = [
    "DIFFERENCES_FROM_WORLD",
    "Aperiodicity",
    "SpectralEnvelope",
    "cheaptrick",
    "d4c",
    "harmonic_aperiodicity",
    "world_fft_size",
    "world_randn",
]

DIFFERENCES_FROM_WORLD = """\
Differences from WORLD (C++ code at commit d625e76, as run by pyworld 0.3.5).
With every option at its default, sonore's output is WORLD's to
floating-point precision.

1. F0 is not estimated: Harvest and DIO are not ported, so every function
   takes an F0 track. so.f0_track is sonore's own tracker, not Harvest, so
   WORLD's numbers need WORLD's track.
2. cheaptrick(q1=...) accepts any value. WORLD's code uses -0.15 (the
   default here); its paper (Morise, 2015) gives -0.09.
3. world_synthesize(rng=...) accepts a seed or Generator, which replaces
   WORLD's noise stream with fresh Gaussian noise. The default, rng=None,
   is WORLD's stream, restarted at every call as WORLD does.
4. harmonic_aperiodicity is not part of WORLD. It measures the share of
   noise by fitting the harmonics; D4C is WORLD's measure.
5. Multichannel sounds are analysed one channel at a time; WORLD takes one
   channel.
"""

_WORLD_SEED = (123456789, 362436069, 521288629, 88675123)
_SAFEGUARD = 1e-12  # WORLD's kMySafeGuardMinimum
_EPS = 2.2204460492503131e-16  # WORLD's kEps
_DEFAULT_F0 = 500.0  # WORLD's kDefaultF0: unvoiced time windows are analysed at this F0
_FLOOR_F0 = 71.0  # WORLD's kFloorF0, which sets CheapTrick's FFT size
_FLOOR_F0_D4C = 47.0
_D4C_BAND_SPACING = 3000.0
_D4C_UPPER_LIMIT = 15000.0
_D4C_SAFEGUARD = 1e-6


# --------------------------------------------------------- WORLD's generator
_STEPS_PER_LANE = 2040  # a multiple of 12, so each lane yields whole draws
_stream_cache = {"draws": np.zeros(0), "next_lane": None, "jump": None}


def _step_words(x, y, z, w):
    """One step of the xorshift128 generator on four 32-bit words (x, y, z,
    w are Marsaglia's names for the state, as in WORLD's code)."""
    shifted = (x ^ (x << 11)) & 0xFFFFFFFF
    return y, z, w, (w ^ (w >> 19)) ^ (shifted ^ (shifted >> 8))


def _to_bits(words) -> np.ndarray:
    return np.array([(words[i // 32] >> (i % 32)) & 1 for i in range(128)], dtype=float)


def _lane_jump() -> np.ndarray:
    """The generator is linear over GF(2) on its 128-bit state, so a jump of
    a lane's length is one 128 x 128 binary matrix."""
    columns = []
    for bit in range(128):
        words = [0, 0, 0, 0]
        words[bit // 32] = 1 << (bit % 32)
        columns.append(_to_bits(_step_words(*words)))
    step = np.array(columns).T
    jump, power, remaining = np.eye(128), step, _STEPS_PER_LANE
    while remaining:
        if remaining & 1:
            jump = (jump @ power) % 2
        power = (power @ power) % 2
        remaining >>= 1
    return jump


def _extend_stream(n_draws: int) -> None:
    cache = _stream_cache
    if cache["jump"] is None:
        cache["jump"] = _lane_jump()
        cache["next_lane"] = _to_bits(_WORLD_SEED)
    draws_per_lane = _STEPS_PER_LANE // 12
    n_cached = len(cache["draws"])
    # at least double the cache, so that growing it step by step stays cheap
    n_lanes = max(-(-(n_draws - n_cached) // draws_per_lane), n_cached // draws_per_lane, 64)
    # start states of the new lanes, doubling: each block is the previous one jumped ahead
    starts = cache["next_lane"][None, :]
    jump = cache["jump"]
    while len(starts) < n_lanes + 1:
        starts = np.concatenate([starts, (starts @ jump.T) % 2])
        jump = (jump @ jump) % 2
    cache["next_lane"] = starts[n_lanes]
    words = (starts[:n_lanes].reshape(n_lanes, 4, 32) @ (2.0 ** np.arange(32))).astype(np.uint32)
    x, y, z, w = (words[:, k].copy() for k in range(4))
    steps = np.empty((_STEPS_PER_LANE, n_lanes), dtype=np.uint32)
    for i in range(_STEPS_PER_LANE):
        shifted = x ^ (x << np.uint32(11))
        x, y, z, w = y, z, w, (w ^ (w >> np.uint32(19))) ^ (shifted ^ (shifted >> np.uint32(8)))
        steps[i] = w
    sums = (steps.T >> np.uint32(4)).reshape(-1, 12).sum(axis=1, dtype=np.uint32)
    cache["draws"] = np.concatenate([cache["draws"], sums / 268435456.0 - 6.0])


def world_randn(n: int) -> np.ndarray:
    """The first ``n`` values of WORLD's ``randn`` after ``randn_reseed``:
    each is the sum of twelve xorshift128 draws (Marsaglia, 2003), shifted
    to 28 bits, scaled to [0, 12) and less 6, so approximately standard
    normal. WORLD restarts this stream at the start of every CheapTrick, D4C
    and Synthesis call, so the same values come back each time; they are
    computed once, many lanes at a time, and kept. Read-only."""
    if len(_stream_cache["draws"]) < n:
        _extend_stream(n)
    view = _stream_cache["draws"][:n]
    view.flags.writeable = False
    return view


class _Stream:
    """Reads WORLD's generator in order, as one analysis or synthesis call does."""

    def __init__(self):
        self.position = 0

    def draw(self, n: int) -> np.ndarray:
        values = world_randn(self.position + n)[self.position :]
        self.position += n
        return values


# ------------------------------------------------------ WORLD's helpers
def _matlab_round(value: float) -> int:
    """MATLAB's round (halves away from zero), as WORLD's matlab_round."""
    return int(value + 0.5) if value > 0 else int(value - 0.5)


def world_fft_size(fs: float, f0_floor: float = _FLOOR_F0) -> int:
    """CheapTrick's FFT size: ``2 ** (1 + floor(log2(3 fs / f0_floor + 1)))``,
    long enough for a window three periods of ``f0_floor`` long. 1024 at
    16 kHz and 2048 at 44.1 or 48 kHz with WORLD's floor of 71 Hz."""
    return int(2.0 ** (1 + int(np.log(3.0 * int(fs) / f0_floor + 1) / np.log(2))))


def _interp1q(x_start: float, x_step: float, values: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """WORLD's interp1Q: ``values`` sampled at ``x_start + k x_step``, read
    linearly at ``x_query`` (index truncated toward zero, last step flat)."""
    position = (np.asarray(x_query, float) - x_start) / x_step
    index = position.astype(int)
    steps = np.append(np.diff(values), 0.0)
    return values[index] + steps[index] * (position - index)


def _linear_smoothing(spectrum: np.ndarray, width: float, fs: int, n_fft: int) -> np.ndarray:
    """WORLD's LinearSmoothing: the mean over ``width`` Hz of a half
    spectrum, piecewise constant per bin, mirrored at 0 Hz and Nyquist."""
    n_half = n_fft // 2
    bin_width = fs / n_fft
    boundary = int(width * n_fft / fs) + 1
    mirrored = np.concatenate(
        [spectrum[boundary:0:-1], spectrum[:n_half], spectrum[n_half::-1][: boundary + 1]]
    )
    integral = np.cumsum(mirrored * bin_width)
    lower_edges = np.arange(n_half + 1) * bin_width - width / 2
    origin = -(boundary - 0.5) * bin_width
    upper = _interp1q(origin, bin_width, integral, lower_edges + width)
    lower = _interp1q(origin, bin_width, integral, lower_edges)
    return (upper - lower) / width


def _dc_correction(spectrum: np.ndarray, f0: float, fs: int, n_fft: int) -> np.ndarray:
    """WORLD's DCCorrection: the spectrum below F0 folded back about F0 / 2."""
    bin_width = fs / n_fft
    upper_index = 2 + int(f0 * n_fft / fs)
    below_f0 = np.arange(upper_index - 1) * bin_width
    corrected = spectrum.copy()
    corrected[: upper_index - 1] += _interp1q(f0, -bin_width, spectrum[: upper_index + 1], below_f0)
    return corrected


def _windowed_waveform(samples, fs, f0, time, window, periods, noise, noise_scale):
    """An F0-adaptive Hann or Blackman window ``periods`` long, centred at
    ``time`` (WORLD's rounding), samples beyond the ends repeated; plus
    WORLD's safety noise; less the window times the weighted mean."""
    half_length = _matlab_round(periods / 2 * fs / f0)
    offsets = np.arange(-half_length, half_length + 1)
    if window == "cheaptrick":  # Hann, scaled to unit energy
        shape = 0.5 * np.cos(np.pi * (offsets / 1.5 / fs) * f0) + 0.5
        shape = shape / np.sqrt(np.sum(shape * shape))
    else:
        position = 2.0 * offsets / periods / fs
        if window == "hann":
            shape = 0.5 * np.cos(np.pi * position * f0) + 0.5
        else:  # Blackman
            shape = 0.42 + 0.5 * np.cos(np.pi * position * f0) + 0.08 * np.cos(2 * np.pi * position * f0)
    origin = _matlab_round(time * fs + 0.001)
    indices = np.clip(origin + offsets, 0, len(samples) - 1)
    segment = samples[indices] * shape + noise.draw(len(offsets)) * noise_scale
    return segment - shape * (segment.sum() / shape.sum())


# ------------------------------------------------------------- F0 tracks
def _time_windows(sound: Sound, f0) -> tuple[np.ndarray, np.ndarray]:
    """Window times ``(n_windows,)`` and F0 values ``(n_channels, n_windows)``
    from an :class:`F0Track` or a ``(times, f0)`` pair."""
    if isinstance(f0, F0Track):
        times, values = f0.t, f0.f0
    else:
        try:
            times, values = f0
        except (TypeError, ValueError):
            raise TypeError("f0 must be an F0Track or a (times, f0) pair") from None
    times = np.asarray(times, dtype=float)
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if times.ndim != 1 or values.shape[1] != len(times):
        raise ValueError(f"f0 has {values.shape[-1]} values for {len(times)} times")
    if values.shape[0] == 1:
        values = np.repeat(values, sound.n_channels, axis=0)
    if values.shape[0] != sound.n_channels:
        raise ValueError(f"f0 has {values.shape[0]} channels, the sound {sound.n_channels}")
    if np.any(values < 0):
        raise ValueError("f0 must be 0 (unvoiced) or positive")
    return times, values


def _integer_fs(sound: Sound) -> int:
    if sound.fs != int(sound.fs):
        raise ValueError(f"WORLD works at whole-number sampling rates, not {sound.fs}")
    return int(sound.fs)


# ------------------------------------------------------------ reading a grid
def _positions(query: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query value, the grid index below it, the one above, and the
    weight of the upper one; held at the ends of the grid."""
    position = np.interp(query, grid, np.arange(len(grid)))
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(grid) - 1)
    return lower, upper, position - lower


def _pointwise(log_values: np.ndarray, grid_t: np.ndarray, grid_f: np.ndarray, t, f) -> np.ndarray:
    """``log_values`` (one channel, ``(len(grid_f), len(grid_t))``) read at
    the points ``(t[i], f[i])``: linear in time and frequency, held beyond
    the ends of the grid."""
    t, f = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(f, dtype=float))
    t_lower, t_upper, t_weight = _positions(t.ravel(), grid_t)
    f_lower, f_upper, f_weight = _positions(f.ravel(), grid_f)
    below = log_values[f_lower, t_lower] * (1 - t_weight) + log_values[f_lower, t_upper] * t_weight
    above = log_values[f_upper, t_lower] * (1 - t_weight) + log_values[f_upper, t_upper] * t_weight
    return (below * (1 - f_weight) + above * f_weight).reshape(t.shape)


# ------------------------------------------------------------ the views
class _FrequencyView:
    """Shared by the envelope and the aperiodicity: data of shape
    ``(n_channels, n_freqs, n_windows)`` on WORLD's grid."""

    data: np.ndarray
    t: np.ndarray
    fs: float
    n_fft: int

    @property
    def f(self) -> np.ndarray:
        """Frequencies [Hz]: ``n_fft // 2 + 1`` bins from 0 to ``fs / 2``."""
        return np.arange(self.n_fft // 2 + 1) * self.fs / self.n_fft

    def to_world(self, channel: int = 0) -> np.ndarray:
        """One channel as WORLD and pyworld hold it: shape ``(n_windows,
        n_freqs)``, C-contiguous, ready for ``pyworld.synthesize``."""
        return np.ascontiguousarray(self.data[channel].T)

    def _interpolate(self, log_values: np.ndarray, t, f) -> np.ndarray:
        """``log_values`` (same shape as data) read linearly at times ``t``
        and frequencies ``f``: shape ``(n_channels, len(f), len(t))``."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        freqs = np.atleast_1d(np.asarray(f, dtype=float))
        window_position = np.interp(times, self.t, np.arange(len(self.t)))
        lower = np.floor(window_position).astype(int)
        upper = np.minimum(lower + 1, len(self.t) - 1)
        weight = window_position - lower
        bin_pos = np.clip(freqs * self.n_fft / self.fs, 0, self.n_fft // 2)
        bin_lower = np.floor(bin_pos).astype(int)
        bin_upper = np.minimum(bin_lower + 1, self.n_fft // 2)
        bin_weight = (bin_pos - bin_lower)[:, None]
        at_freqs = log_values[:, bin_lower] * (1 - bin_weight) + log_values[:, bin_upper] * bin_weight
        return at_freqs[:, :, lower] * (1 - weight) + at_freqs[:, :, upper] * weight


class SpectralEnvelope(_FrequencyView):
    """A spectral envelope from :func:`cheaptrick`: power on ``n_fft // 2 + 1``
    frequencies :attr:`f` from 0 to ``fs / 2``, at the window times :attr:`t`
    of the F0 track. ``data`` has shape ``(n_channels, n_freqs, n_windows)``;
    :meth:`to_world` gives one channel in WORLD's layout.

    It is a view (it keeps the envelope, not the sound): calling it,
    ``env(t, f)``, reads the power at any times and frequencies, linearly
    in time and in dB over frequency.
    """

    def __init__(self, data: np.ndarray, t: np.ndarray, fs: float, q1: float):
        self.data = data
        self.t = t
        self.fs = fs
        self.n_fft = 2 * (data.shape[1] - 1)
        self.q1 = q1

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        return (
            f"SpectralEnvelope({n_freqs} freqs x {n_windows} time windows, {n_channels} ch, q1 {self.q1:g})"
        )

    @property
    def db(self) -> np.ndarray:
        """``10 log10`` of the power."""
        return 10 * np.log10(self.data)

    def __call__(self, t, f) -> np.ndarray:
        """Power at times ``t`` [s] and frequencies ``f`` [Hz], shape
        ``(n_channels, len(f), len(t))``: linear in time between time windows,
        linear in dB between bins."""
        return np.exp(self._interpolate(np.log(self.data), t, f))

    def amplitude(self, t, f, channel: int = 0) -> np.ndarray:
        """The amplitude (square root of the power) at the points ``(t[i],
        f[i])``, ``t`` and ``f`` of the same shape: the form
        :func:`~sonore.signals.generators.harmonic_complex` reads, one value
        per sample and harmonic. Linear in time and in dB over frequency,
        held beyond the first and last time windows."""
        return np.exp(0.5 * _pointwise(np.log(self.data[channel]), self.t, self.f, t, f))

    def plot(self, ax=None, channel: int = 0, db_range: float = 70.0, fmax: float | None = None):
        """The envelope in dB as a time-frequency image."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(
            self.db[channel], self.t, self.f, ax=ax, db_range=db_range, fmax=fmax, title="Spectral envelope"
        )


class Aperiodicity(_FrequencyView):
    """An aperiodicity, from :func:`d4c` or :func:`harmonic_aperiodicity`:
    per frequency and time window, how much of the power is noise rather than
    harmonics. Stored as WORLD stores it, an amplitude ratio between 0 and
    1 (``data``, shape ``(n_channels, n_freqs, n_windows)``): its square
    :attr:`share` is the share of the power that is noise, which is what
    :func:`~sonore.stimuli.vocoder.world_synthesize` uses. ``method`` names
    the measure that made it ("D4C" or "harmonic residual").
    """

    def __init__(self, data: np.ndarray, t: np.ndarray, fs: float, method: str):
        self.data = data
        self.t = t
        self.fs = fs
        self.n_fft = 2 * (data.shape[1] - 1)
        self.method = method

    def __repr__(self) -> str:
        n_channels, n_freqs, n_windows = self.data.shape
        return f"Aperiodicity({self.method}, {n_freqs} freqs x {n_windows} time windows, {n_channels} ch)"

    @property
    def share(self) -> np.ndarray:
        """The noise share of the power, ``data ** 2``."""
        return self.data**2

    @property
    def db(self) -> np.ndarray:
        """The noise share in dB, ``20 log10 data``."""
        return 20 * np.log10(self.data)

    def __call__(self, t, f) -> np.ndarray:
        """The amplitude ratio at times ``t`` [s] and frequencies ``f`` [Hz],
        shape ``(n_channels, len(f), len(t))``: linear in time between time windows,
        linear in dB between bins."""
        return np.exp(self._interpolate(np.log(self.data), t, f))

    def bands(self, edges: Sequence[float], envelope: SpectralEnvelope | None = None) -> np.ndarray:
        """The noise share averaged over each band ``[edges[i], edges[i+1])``,
        shape ``(n_channels, n_bands, n_windows)``. With an ``envelope``, the
        average is weighted by its power, so the result is the band's noise
        power over its total power."""
        freqs = self.f
        weights = np.ones_like(self.data) if envelope is None else envelope.data
        out = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            in_band = (freqs >= lower) & (freqs < upper)
            band_weights = weights[:, in_band]
            out.append(np.sum(self.share[:, in_band] * band_weights, axis=1) / np.sum(band_weights, axis=1))
        return np.stack(out, axis=1)

    def plot(self, ax=None, channel: int = 0, db_range: float = 60.0, fmax: float | None = None):
        """The noise share in dB as a time-frequency image (0 dB is all noise)."""
        from sonore.plotting import plot_tf_db

        return plot_tf_db(
            self.db[channel],
            self.t,
            self.f,
            ax=ax,
            db_range=db_range,
            fmax=fmax,
            title=f"Aperiodicity ({self.method})",
        )


# ------------------------------------------------------------ CheapTrick
def cheaptrick(sound: Sound, f0, *, q1: float = -0.15, f0_floor: float = _FLOOR_F0) -> SpectralEnvelope:
    """WORLD's CheapTrick spectral envelope (Morise, 2015), ported exactly.

    At each time window of the F0 track: the sound under a Hann window three
    periods long, scaled to unit energy, less its weighted mean; its power
    spectrum on :func:`world_fft_size` bins; the power below F0 folded back
    about F0 / 2; a moving average over 2 F0 / 3; then the log, liftered by
    ``sinc(F0 q)`` (smoothing over F0) times the recovery lifter
    ``(1 - 2 q1) + 2 q1 cos(2 pi F0 q)``, and back. The result follows the
    shape of the harmonic peaks a few dB below them and, because of the
    smoothing over 2 F0 / 3, barely changes with the time window's position
    within a period. Time windows with F0 at or below the floor
    ``3 fs / (n_fft - 3)`` (unvoiced ones included) are analysed at 500 Hz.

    Parameters
    ----------
    sound
        The sound. Its sampling rate must be a whole number of Hz.
    f0
        An :class:`~sonore.analysis.f0.F0Track` (one row per channel) or a
        ``(times, f0)`` pair; 0 marks unvoiced time windows. The envelope is
        computed at these times.
    q1
        The recovery lifter's parameter. ``-0.15`` is WORLD's code; the
        paper gives ``-0.09``; ``0`` is no recovery.
    f0_floor
        The lowest F0 the FFT size must hold (WORLD's 71 Hz).
    """
    fs = _integer_fs(sound)
    times, f0_values = _time_windows(sound, f0)
    n_fft = world_fft_size(fs, f0_floor)
    analysis_floor = 3.0 * fs / (n_fft - 3.0)
    quefrencies = np.arange(n_fft // 2 + 1) / fs
    data = np.empty((sound.n_channels, n_fft // 2 + 1, len(times)))
    for channel in range(sound.n_channels):
        samples = sound.data[:, channel]
        noise = _Stream()
        for window_index, (time, window_f0) in enumerate(zip(times, f0_values[channel], strict=True)):
            current_f0 = _DEFAULT_F0 if window_f0 <= analysis_floor else window_f0
            segment = _windowed_waveform(samples, fs, current_f0, time, "cheaptrick", 3.0, noise, _SAFEGUARD)
            power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
            power = _dc_correction(power, current_f0, fs, n_fft)
            power = _linear_smoothing(power, current_f0 * 2.0 / 3.0, fs, n_fft)
            power = power + np.abs(noise.draw(n_fft // 2 + 1)) * _EPS
            data[channel, :, window_index] = _smooth_with_recovery(power, current_f0, q1, quefrencies, n_fft)
    return SpectralEnvelope(data, times, fs, q1)


def _smooth_with_recovery(power, f0, q1, quefrencies, n_fft):
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothing = np.sin(np.pi * f0 * quefrencies) / (np.pi * f0 * quefrencies)
    smoothing[0] = 1.0
    recovery = (1.0 - 2.0 * q1) + 2.0 * q1 * np.cos(2.0 * np.pi * quefrencies * f0)
    log_power = np.log(power)
    cepstrum = np.fft.rfft(np.concatenate([log_power, log_power[-2:0:-1]])).real
    return np.exp(np.fft.irfft(cepstrum * smoothing * recovery, n_fft)[: n_fft // 2 + 1])


# ------------------------------------------------------------------- D4C
def d4c(sound: Sound, f0, *, threshold: float = 0.85, f0_floor: float = _FLOOR_F0) -> Aperiodicity:
    """WORLD's D4C aperiodicity (Morise, 2016), ported exactly.

    At each voiced time window: a "static group delay" from two Blackman-windowed
    spectra four periods long, a quarter period either side of the window center,
    divided by a smoothed power spectrum, smoothed over F0 / 2, less itself
    smoothed over F0. Around each multiple of 3 kHz up to
    ``min(15 kHz, fs / 2 - 3 kHz)``, a Nuttall window over 3 kHz of that
    group delay is transformed, its power sorted, and the band's
    aperiodicity is the share outside the largest bins, in dB, plus
    ``(F0 - 100) / 50`` dB, capped at 0. The curve is then linear in dB
    from -60 dB at 0 Hz through those values to 0 dB at Nyquist. At
    16 kHz that is one measured value per time window, at 3 kHz.

    A time window is left fully aperiodic (amplitude ratio ``1 - 1e-12``) where
    F0 is 0, or where less than ``threshold`` of its power between 100 Hz
    and 7.9 kHz lies below 4 kHz (WORLD's "LoveTrain" test; ``threshold=0``
    keeps every voiced time window voiced).

    D4C was tuned so that WORLD's resynthesis sounds natural; it is robust
    to F0 errors but does not report the share of noise below 3 kHz, which
    the -60 dB anchor sets. :func:`harmonic_aperiodicity` measures that
    share directly.

    Parameters
    ----------
    sound, f0
        As :func:`cheaptrick`.
    threshold
        The LoveTrain voicing threshold.
    f0_floor
        Sets the output's frequency grid, which must match the envelope's
        for synthesis (WORLD's 71 Hz, as :func:`cheaptrick`).
    """
    fs = _integer_fs(sound)
    times, f0_values = _time_windows(sound, f0)
    n_fft_out = world_fft_size(fs, f0_floor)
    n_fft = int(2.0 ** (1 + int(np.log(4.0 * fs / _FLOOR_F0_D4C + 1) / np.log(2))))
    n_bands = int(min(_D4C_UPPER_LIMIT, fs / 2.0 - _D4C_BAND_SPACING) / _D4C_BAND_SPACING)
    window_length = int(_D4C_BAND_SPACING * n_fft / fs) * 2 + 1
    position = np.arange(window_length) / (window_length - 1.0)
    nuttall = (
        0.355768
        - 0.487396 * np.cos(2 * np.pi * position)
        + 0.144232 * np.cos(4 * np.pi * position)
        - 0.012604 * np.cos(6 * np.pi * position)
    )
    coarse_freqs = np.append(np.arange(n_bands + 1) * _D4C_BAND_SPACING, fs / 2.0)
    freqs = np.arange(n_fft_out // 2 + 1) * fs / n_fft_out
    data = np.full((sound.n_channels, n_fft_out // 2 + 1, len(times)), 1.0 - _SAFEGUARD)
    for channel in range(sound.n_channels):
        samples = sound.data[:, channel]
        noise = _Stream()
        voiced = [
            window_f0 != 0 and _love_train(samples, fs, time, window_f0, noise) > threshold
            for time, window_f0 in zip(times, f0_values[channel], strict=True)
        ]
        for window_index, (time, window_f0) in enumerate(zip(times, f0_values[channel], strict=True)):
            if not voiced[window_index]:
                continue
            bands = _d4c_bands(
                samples, fs, time, max(_FLOOR_F0_D4C, window_f0), n_fft, n_bands, nuttall, noise
            )
            coarse = np.concatenate([[-60.0], bands, [-_SAFEGUARD]])
            data[channel, :, window_index] = 10 ** (np.interp(freqs, coarse_freqs, coarse) / 20)
    return Aperiodicity(data, times, fs, "D4C")


def _love_train(samples, fs, time, f0, noise):
    """The share of power (100 Hz to 7.9 kHz) below 4 kHz."""
    f0 = max(f0, 40.0)
    n_fft = int(2.0 ** (1 + int(np.log(3.0 * fs / 40.0 + 1) / np.log(2))))
    segment = _windowed_waveform(samples, fs, f0, time, "blackman", 3.0, noise, _D4C_SAFEGUARD)
    power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
    lowest, middle, highest = (int(np.ceil(edge * n_fft / fs)) for edge in (100.0, 4000.0, 7900.0))
    power[: lowest + 1] = 0
    cumulative = np.cumsum(power)
    return cumulative[middle] / cumulative[highest]


def _centroid(samples, fs, time, f0, n_fft, noise):
    segment = _windowed_waveform(samples, fs, f0, time, "blackman", 4.0, noise, _D4C_SAFEGUARD)
    buffer = np.zeros(n_fft)
    buffer[: len(segment)] = segment
    n_normalized = _matlab_round(2.0 * fs / f0) * 2 + 1
    buffer /= np.sqrt(np.sum(buffer[:n_normalized] ** 2))
    spectrum = np.fft.rfft(buffer)
    weighted = np.fft.rfft(buffer * (np.arange(n_fft) + 1.0))
    return weighted.real * spectrum.real + spectrum.imag * weighted.imag


def _d4c_bands(samples, fs, time, f0, n_fft, n_bands, nuttall, noise):
    """D4C's coarse aperiodicities [dB], one per multiple of 3 kHz."""
    centroid = _centroid(samples, fs, time - 0.25 / f0, f0, n_fft, noise)
    centroid = centroid + _centroid(samples, fs, time + 0.25 / f0, f0, n_fft, noise)
    centroid = _dc_correction(centroid, f0, fs, n_fft)
    segment = _windowed_waveform(samples, fs, f0, time, "hann", 4.0, noise, _D4C_SAFEGUARD)
    power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
    power = _linear_smoothing(_dc_correction(power, f0, fs, n_fft), f0, fs, n_fft)
    # WORLD divides without a guard; where the power is 0 the C code gets inf or NaN as here
    with np.errstate(divide="ignore", invalid="ignore"):
        group_delay = _linear_smoothing(centroid / power, f0 / 2.0, fs, n_fft)
        group_delay = group_delay - _linear_smoothing(group_delay, f0, fs, n_fft)
    window_length = len(nuttall)
    boundary = _matlab_round(n_fft * 8.0 / window_length)
    bands = np.empty(n_bands)
    for band in range(n_bands):
        centre = int(_D4C_BAND_SPACING * (band + 1) * n_fft / fs)
        start = centre - window_length // 2
        buffer = np.zeros(n_fft)
        buffer[:window_length] = group_delay[start : start + window_length] * nuttall
        cumulative = np.cumsum(np.sort(np.abs(np.fft.rfft(buffer)) ** 2))
        bands[band] = 10 * np.log10(cumulative[n_fft // 2 - boundary - 1] / cumulative[n_fft // 2])
    return np.minimum(0.0, bands + (f0 - 100) / 50)


# ------------------------------------------------- harmonic residual
def harmonic_aperiodicity(
    sound: Sound, f0, *, periods: float = 4.0, f0_floor: float = _FLOOR_F0, cell_harmonics: float = 2.0
) -> Aperiodicity:
    """The share of noise, measured by fitting the harmonics and keeping
    what is left. Not part of WORLD; :func:`d4c` is WORLD's measure.

    The F0 track is interpolated to every sample and integrated to a
    running phase ``Phi(t)``. At each voiced time window, under a Hann window
    ``periods`` periods long, a weighted least-squares fit of the harmonics
    ``cos(k Phi)``, ``sin(k Phi)`` below Nyquist, each with a linear change
    of amplitude across the window, is subtracted. The residual's power,
    over the windowed sound's power, is the noise share. The fit also
    absorbs a little of the noise near every harmonic; how much is known
    exactly from the fit (the share of white noise the residual keeps at
    each frequency), and is divided out. Both are summed over cells
    ``cell_harmonics`` harmonics wide, and the cells' shares are
    interpolated in dB onto the grid of :func:`cheaptrick`. Unvoiced time windows
    are all noise.

    The measure is what the word means, so it reads a known share of noise
    within a few tenths of a dB on a steady vowel; but it needs F0 to about
    0.1%, since a harmonic that drifts out of phase with the fit over the
    window reads as noise, and so do jitter and shimmer.

    Parameters
    ----------
    sound, f0
        As :func:`cheaptrick`.
    periods
        The window's length in periods of the time window's F0.
    f0_floor
        Sets the output's frequency grid, as :func:`cheaptrick`.
    cell_harmonics
        The width of the cells, in harmonics.
    """
    fs = _integer_fs(sound)
    times, f0_values = _time_windows(sound, f0)
    n_fft_out = world_fft_size(fs, f0_floor)
    freqs_out = np.arange(n_fft_out // 2 + 1) * fs / n_fft_out
    data = np.ones((sound.n_channels, n_fft_out // 2 + 1, len(times)))
    sample_times = np.arange(sound.n_samples) / fs
    for channel in range(sound.n_channels):
        voiced = f0_values[channel] > 0
        if not voiced.any():
            continue
        f0_per_sample = np.interp(sample_times, times[voiced], f0_values[channel, voiced])
        phase = np.cumsum(2 * np.pi * f0_per_sample / fs)
        longest = int(np.ceil(periods / 2 * fs / f0_values[channel, voiced].min())) + 1
        padded = np.pad(sound.data[:, channel], longest)
        padded_phase = np.pad(phase, longest, mode="edge")
        for window_index in np.flatnonzero(voiced):
            share, cell_freqs = _residual_share(
                padded,
                padded_phase,
                fs,
                times[window_index],
                f0_values[channel, window_index],
                periods,
                longest,
                cell_harmonics,
            )
            share_db = np.interp(freqs_out, cell_freqs, 10 * np.log10(share))
            data[channel, :, window_index] = 10 ** (share_db / 20)
    return Aperiodicity(data, times, fs, "harmonic residual")


def _residual_share(samples, phase, fs, time, f0, periods, padding, cell_harmonics):
    """One time window: the noise share per cell and the cells' centre frequencies."""
    centre = int(np.round(time * fs)) + padding
    half_length = int(np.round(periods / 2 * fs / f0))
    offsets = np.arange(-half_length, half_length + 1)
    window = 0.5 + 0.5 * np.cos(np.pi * offsets / (half_length + 1))
    harmonic_numbers = np.arange(1, int((fs / 2) / f0 - 1e-9) + 1)
    harmonic_phases = np.outer(phase[centre + offsets], harmonic_numbers)
    ramp = (offsets / half_length)[:, None]
    columns = np.column_stack(
        [
            np.ones(len(offsets)),
            np.cos(harmonic_phases),
            np.sin(harmonic_phases),
            ramp * np.cos(harmonic_phases),
            ramp * np.sin(harmonic_phases),
        ]
    )
    root_window = np.sqrt(window)
    basis, _ = np.linalg.qr(columns * root_window[:, None])
    weighted = samples[centre + offsets] * root_window
    residual = (weighted - basis @ (basis.T @ weighted)) * root_window
    # white noise e becomes (W - S B B' S) e in the windowed residual (S the
    # root window, B the fit's basis); its power at each frequency, over the
    # windowed noise's, is the share the residual keeps
    n_fft = int(2 ** np.ceil(np.log2(len(offsets))))
    residual_operator = np.diag(window) - (root_window[:, None] * basis) @ (basis.T * root_window[None, :])
    kept = np.sum(np.abs(np.fft.rfft(residual_operator, n_fft, axis=0)) ** 2, axis=1) / np.sum(window**2)
    signal_power = np.abs(np.fft.rfft(window * samples[centre + offsets], n_fft)) ** 2
    residual_power = np.abs(np.fft.rfft(residual, n_fft)) ** 2
    freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    cell = (freqs // (cell_harmonics * f0)).astype(int)
    noise_power = np.bincount(cell, residual_power) / np.bincount(cell, kept) * np.bincount(cell)
    share = noise_power / np.maximum(np.bincount(cell, signal_power), np.finfo(float).tiny)
    share = np.clip(share, 1e-12, 1.0)
    return share, np.bincount(cell, freqs) / np.bincount(cell)
