"""WORLD's synthesis (Morise, Yokomori & Ozawa, 2016), ported exactly: a
sound from an F0 track, a spectral envelope and an aperiodicity, as made by
:func:`~sonore.views.spectral_envelope.cheaptrick` and
:func:`~sonore.views.aperiodicity.d4c` (or
:func:`~sonore.views.aperiodicity.harmonic_aperiodicity`).

Also here are the pieces WORLD's analysis shares with its synthesis: WORLD's
own noise generator (:func:`world_randn`), its FFT size, MATLAB's rounding,
the reading of an F0 track onto time windows, and
:data:`DIFFERENCES_FROM_WORLD`. The synthesis reads its inputs by what they
provide, not by their type: an F0 track as ``.t`` and ``.f0`` or a ``(times,
f0)`` pair, an envelope as ``env(t, f)``, and an aperiodicity as its grid
``.t``, ``.f`` and ``.data``.
"""

from __future__ import annotations

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import as_rng

__all__ = ["DIFFERENCES_FROM_WORLD", "world_randn", "world_fft_size", "world_synthesize"]


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
    return _periods_fft_size(int(fs), 3.0, f0_floor)


def _periods_fft_size(fs: float, n_periods: float, f0_floor: float) -> int:
    """WORLD's FFT size for a window ``n_periods`` periods of ``f0_floor`` long:
    the power of two above twice that many samples. CheapTrick passes ``int(fs)``
    and D4C passes ``fs`` as is, as WORLD does, so the two agree only for whole
    sample rates."""
    return int(2.0 ** (1 + int(np.log(n_periods * fs / f0_floor + 1) / np.log(2))))


# ------------------------------------------------------------- F0 tracks
def _time_windows(sound: Sound, f0) -> tuple[np.ndarray, np.ndarray]:
    """Window times ``(n_windows,)`` and F0 values ``(n_channels, n_windows)``
    from anything with window times ``.t`` and values ``.f0`` (an
    :class:`~sonore.views.f0.F0Track`) or a ``(times, f0)`` pair. The track is
    read by what it provides, not by its type, so that this layer needs no
    import from the views above it."""
    if hasattr(f0, "t") and hasattr(f0, "f0"):
        times, values = f0.t, f0.f0
    else:
        try:
            times, values = f0
        except (TypeError, ValueError):
            raise TypeError("f0 must be an F0Track (.t and .f0) or a (times, f0) pair") from None
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


# ------------------------------------------------------------ a contour on a grid
def _contour_on_grid(times, f0_values, grid) -> np.ndarray:
    """An F0 contour (one row per channel) read at other window times: voiced
    where the nearest given time window is voiced, there linear between
    voiced values; 0 elsewhere."""
    times = np.asarray(times, dtype=float)
    grid = np.asarray(grid, dtype=float)
    rows = np.atleast_2d(np.asarray(f0_values, dtype=float))
    upper = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    lower = np.clip(upper - 1, 0, len(times) - 1)
    nearest = np.where(np.abs(times[lower] - grid) <= np.abs(times[upper] - grid), lower, upper)
    out = np.zeros((len(rows), len(grid)))
    for row_in, row_out in zip(rows, out, strict=True):
        voiced = row_in > 0
        if voiced.any():
            row_out[:] = np.where(voiced[nearest], np.interp(grid, times[voiced], row_in[voiced]), 0.0)
    return out


class _GaussianNoise:
    """Fresh noise in place of WORLD's stream, from a seed or Generator."""

    def __init__(self, rng):
        self.rng = as_rng(rng)

    def draw(self, n: int) -> np.ndarray:
        return self.rng.standard_normal(n)


def world_synthesize(f0, envelope, aperiodicity, *, rng=None) -> Sound:
    """WORLD's synthesis, ported exactly.

    Pulses are placed where the running phase of the F0 track (linearly
    interpolated to every sample) crosses a multiple of 2 pi, the fraction
    of a sample kept as a linear phase shift; unvoiced stretches get a pulse
    every 1/500 s. At each pulse the envelope ``S`` and the amplitude ratio
    ``A`` are interpolated between time windows. The periodic part is the
    minimum-phase response of ``S (1 - A^2)``, scaled by the square root of
    the interval to the next pulse, with its DC removed; the aperiodic part
    is noise as long as that interval through the minimum-phase response of
    ``S A^2`` (of ``S`` where unvoiced). The responses are overlap-added.

    Minimum phase is an approximation: the waveform within each period is
    not the original's, which WORLD's authors note is audible at low F0.

    Parameters
    ----------
    f0
        An F0 track (an :class:`~sonore.views.f0.F0Track` or a ``(times,
        f0)`` pair), usually the one the envelope and aperiodicity were
        measured with, from any tracker. It is used as it is when its time
        windows are the aperiodicity's; otherwise it is read onto them
        (voiced where the nearest time window is voiced, linear between
        voiced values). Those time windows must be evenly spaced from time
        0, as WORLD assumes; the spacing is the hop (WORLD's frame period).
        F0 can be changed before synthesis
        (:func:`~sonore.views.f0.scale_f0`, a pitch change); the
        envelope stays, so the formants stay.
    envelope
        A :class:`~sonore.views.spectral_envelope.SpectralEnvelope` on the
        aperiodicity's time windows and frequencies (any envelope whose
        ``.fs``, ``.t``, ``.f`` and ``.data`` match the aperiodicity's grid),
        used as it is; or any
        other envelope read as ``envelope(t, f)`` (power, shape
        ``(n_channels, len(f), len(t))``), such as a
        :class:`~sonore.views.spectral_envelope.GridEnvelope` from the cepstrum or
        MFCCs, or one moved by :func:`~sonore.views.spectral_envelope.warp_frequency`,
        which is read at the aperiodicity's time windows and frequencies
        first. WORLD's synthesis computes a minimum-phase response on its
        own FFT length, which suits a smooth envelope like CheapTrick's; an
        envelope with deep, narrow valleys (tens of dB) comes out a few dB
        off at the harmonics, so smooth such an envelope first, or give it
        to :func:`~sonore.signals.generators.harmonic_complex` as its
        ``amplitudes``, which reads the envelope at each harmonic exactly.
    aperiodicity
        An :class:`~sonore.views.aperiodicity.Aperiodicity` (or anything with
        its grid: ``.t``, ``.f``, ``.fs`` and ``.data``, the amplitude ratio),
        which sets the time windows and frequencies.
    rng
        ``None`` (default): WORLD's own noise stream, restarted at every
        call, so the same inputs give WORLD's output, sample for sample.
        A seed or ``numpy.random.Generator``: fresh Gaussian noise instead,
        for independent tokens (listed in ``DIFFERENCES_FROM_WORLD``).

    Returns
    -------
    Sound
        ``int(n_windows * hop * fs)`` samples, one channel per
        channel of the envelope.
    """
    envelope_power = _on_grid_of(envelope, aperiodicity)
    fs = int(aperiodicity.fs)
    n_channels = envelope_power.shape[0]
    times, f0_values = _time_windows(Sound(np.zeros((1, n_channels)), fs), f0)
    grid = aperiodicity.t
    if len(times) != len(grid) or not np.allclose(times, grid, rtol=0, atol=1e-9):
        f0_values = _contour_on_grid(times, f0_values, grid)
        times = grid
    if len(times) < 2:
        raise ValueError("WORLD's synthesis needs at least two time windows")
    hop_ms = round((times[1] - times[0]) * 1000, 9)
    if times[0] != 0 or not np.allclose(np.diff(times), hop_ms / 1000, rtol=0, atol=1e-9):
        raise ValueError("f0 time windows must be evenly spaced from time 0")
    # WORLD takes the hop (its frame period) in ms, as pyworld does
    hop = hop_ms / 1000.0
    n_samples = int(len(times) * hop_ms * fs / 1000)
    channels = [
        _synthesize_channel(
            f0_values[channel],
            envelope_power[channel].T,
            aperiodicity.data[channel].T,
            hop,
            fs,
            n_samples,
            _Stream() if rng is None else _GaussianNoise(rng),
        )
        for channel in range(n_channels)
    ]
    return Sound(np.column_stack(channels), fs)


def _on_grid_of(envelope, aperiodicity) -> np.ndarray:
    """The envelope's power on the aperiodicity's time windows and frequencies,
    shape ``(n_channels, n_freqs, n_windows)``: its own data if it is already
    held on that grid (a CheapTrick envelope measured with the same track),
    otherwise read there as ``envelope(t, f)``."""
    if (
        getattr(envelope, "fs", None) == aperiodicity.fs
        and np.shape(getattr(envelope, "data", None)) == aperiodicity.data.shape
        and np.array_equal(getattr(envelope, "t", None), aperiodicity.t)
        and np.array_equal(getattr(envelope, "f", None), aperiodicity.f)
    ):
        return envelope.data
    if not callable(envelope):
        raise TypeError("envelope must be a SpectralEnvelope or be read as envelope(t, f)")
    power = np.asarray(envelope(aperiodicity.t, aperiodicity.f), dtype=float)
    n_channels = aperiodicity.data.shape[0]
    expected = aperiodicity.data.shape[1:]
    if power.ndim != 3 or power.shape[1:] != expected or power.shape[0] not in (1, n_channels):
        raise ValueError(
            f"envelope(t, f) gave shape {power.shape}; expected (n_channels, {expected[0]}, {expected[1]})"
        )
    if power.shape[0] != n_channels:
        power = np.repeat(power, n_channels, axis=0)
    return power


def _minimum_phase(log_amplitude: np.ndarray, n_fft: int) -> np.ndarray:
    """WORLD's GetMinimumPhaseSpectrum: half-spectrum log amplitudes in,
    the minimum-phase spectrum on ``n_fft // 2 + 1`` bins out."""
    cepstrum = np.fft.rfft(np.concatenate([log_amplitude, log_amplitude[-2:0:-1]]))
    folded = np.zeros(n_fft, dtype=complex)
    folded[0] = np.conj(cepstrum[0])
    folded[1 : n_fft // 2] = 2 * np.conj(cepstrum[1 : n_fft // 2])
    folded[n_fft // 2] = np.conj(cepstrum[n_fft // 2])
    return np.exp(np.fft.fft(folded)[: n_fft // 2 + 1] / n_fft)


def _inverse_centred(spectrum: np.ndarray, n_fft: int) -> np.ndarray:
    """FFTW's unnormalized complex-to-real inverse, then fftshift."""
    waveform = np.fft.irfft(spectrum, n_fft) * n_fft
    return np.concatenate([waveform[n_fft // 2 :], waveform[: n_fft // 2]])


def _synthesize_channel(f0, spectrogram, ratio_windows, hop, fs, n_samples, noise):
    """WORLD's Synthesis on one channel; spectrogram and ratio_windows are
    ``(n_windows, n_freqs)``."""
    n_windows, n_half = spectrogram.shape
    n_fft = 2 * (n_half - 1)
    lowest_f0 = fs // n_fft + 1.0
    # F0 and voicing at every sample, the last time window extrapolated one step
    coarse_times = np.arange(n_windows + 1) * hop
    coarse_f0 = np.where(f0 < lowest_f0, 0.0, f0)
    coarse_voicing = (coarse_f0 != 0).astype(float)
    coarse_f0 = np.append(coarse_f0, 2 * coarse_f0[-1] - coarse_f0[-2])
    coarse_voicing = np.append(coarse_voicing, 2 * coarse_voicing[-1] - coarse_voicing[-2])
    sample_times = np.arange(n_samples) / fs
    sample_f0 = np.interp(sample_times, coarse_times, coarse_f0)
    voicing = (np.interp(sample_times, coarse_times, coarse_voicing) > 0.5).astype(float)
    sample_f0 = np.where(voicing == 0, _DEFAULT_F0, sample_f0)
    # pulses where the wrapped phase jumps
    total_phase = np.cumsum(2 * np.pi * sample_f0 / fs)
    wrapped = np.fmod(total_phase, 2 * np.pi)
    pulses = np.flatnonzero(np.abs(np.diff(wrapped)) > np.pi)
    before = wrapped[pulses] - 2 * np.pi
    time_shifts = -before / (wrapped[pulses + 1] - before) / fs
    # WORLD's DC remover: a Hann-shaped correction summing to one half
    half_index = np.arange(n_fft // 2)
    dc_remover = 0.5 - 0.5 * np.cos(2 * np.pi * (half_index + 1.0) / (1.0 + n_fft))
    dc_remover = np.concatenate([dc_remover, dc_remover[::-1]])
    dc_remover /= 2 * dc_remover[: n_fft // 2].sum()
    safe_ratios = np.clip(ratio_windows, 0.001, 0.999999999999)
    bins = np.arange(n_half)
    output = np.zeros(n_samples)
    for pulse, sample in enumerate(pulses):
        noise_size = pulses[min(len(pulses) - 1, pulse + 1)] - sample
        window_position = sample_times[sample] / hop
        lower = min(n_windows - 1, int(np.floor(window_position)))
        upper = min(n_windows - 1, int(np.ceil(window_position)))
        weight = window_position - lower
        if lower == upper:
            envelope = np.abs(spectrogram[lower])
            noise_share = safe_ratios[lower] ** 2
        else:
            envelope = (1 - weight) * np.abs(spectrogram[lower]) + weight * np.abs(spectrogram[upper])
            noise_share = ((1 - weight) * safe_ratios[lower] + weight * safe_ratios[upper]) ** 2
        # periodic part
        if voicing[sample] <= 0.5 or noise_share[0] > 0.999:
            periodic = np.zeros(n_fft)
        else:
            spectrum = _minimum_phase(np.log(envelope * (1 - noise_share) + _SAFEGUARD) / 2, n_fft)
            shift_cos = np.cos(2 * np.pi * time_shifts[pulse] * fs / n_fft * bins)
            shift_sin = np.sqrt(1 - shift_cos**2)
            spectrum = (spectrum.real * shift_cos + spectrum.imag * shift_sin) + 1j * (
                spectrum.imag * shift_cos - spectrum.real * shift_sin
            )
            periodic = _inverse_centred(spectrum, n_fft)
            dc = periodic[n_fft // 2 :].sum()
            # WORLD sets the first half (before the pulse) to the correction
            # alone rather than subtracting it
            periodic = np.concatenate([np.zeros(n_fft // 2), periodic[n_fft // 2 :]]) - dc * dc_remover
        # aperiodic part
        noise_segment = np.zeros(n_fft)
        if noise_size:
            values = noise.draw(noise_size)
            noise_segment[:noise_size] = values - values.mean()
        if voicing[sample] != 0:
            log_amplitude = np.log(envelope * noise_share) / 2
        else:
            log_amplitude = np.log(envelope) / 2
        aperiodic = _inverse_centred(_minimum_phase(log_amplitude, n_fft) * np.fft.rfft(noise_segment), n_fft)
        response = (periodic * np.sqrt(noise_size) + aperiodic) / n_fft
        offset = sample - n_fft // 2 + 1
        first, last = max(0, -offset), min(n_fft, n_samples - offset)
        output[offset + first : offset + last] += response[first:last]
    return output
