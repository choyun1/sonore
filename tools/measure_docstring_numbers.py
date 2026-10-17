"""Measures the numbers that docstrings quote about sonore's own behavior.

Unlike the check_*_claims.py scripts, which are independent of sonore, this one
runs sonore itself: each number it prints is what the library does, and the
docstring that quotes it names this script. Each line says what was measured.

    python tools/measure_docstring_numbers.py

It takes about a minute.
"""

import time

import numpy as np
import scipy.fft as sp_fft

import sonore as so
from sonore.core.fft import fast_padding
from sonore.sources.ripples import _evaluate
from sonore.views.phasevocoder import _win_len

FS = 16000


def correlation_and_level(x, y, fs):
    """Correlation and level difference [dB] of y against x, 0.1 s from each end."""
    inside = slice(int(0.1 * fs), len(x) - int(0.1 * fs))
    a, b = x.data[inside, 0], y.data[inside, 0]
    return np.corrcoef(a, b)[0, 1], 20 * np.log10(so.rms(b) / so.rms(a))


def phase_vocoder_resynthesis():
    """PVAnalysis.to_sound: harmonic complexes and noise, unchanged."""
    print("PVAnalysis.to_sound (views/phasevocoder.py), 1 s at 16 kHz:")
    worst = 1.0
    for f0 in (110.0, 220.0, 440.0):
        x = so.harmonic_complex(1.0, FS, f0, harmonics=np.arange(1, 8)).ramp(20e-3)
        r, level = correlation_and_level(x, so.pv_analyze(x).to_sound(), FS)
        worst = min(worst, r)
        print(f"  harmonic complex, f0 {f0:g} Hz, harmonics 1-7: r = {r:.5f}, level {level:+.2f} dB")
    print(f"  lowest r over harmonic complexes: {worst:.5f}")
    for seed in range(3):
        x = so.gaussian_noise(1.0, FS, rng=seed).ramp(20e-3)
        r, level = correlation_and_level(x, so.pv_analyze(x).to_sound(), FS)
        print(f"  Gaussian noise, seed {seed}: r = {r:.3f}, level {level:+.2f} dB")


def time_stretch_factor_error():
    """time_stretch: the content is stretched by synthesis_hop / analysis_hop,
    with the analysis hop rounded to whole samples; the worst gap to the
    requested factor over each range of factors."""
    print("time_stretch (views/phasevocoder.py), achieved against requested factor:")
    for fs in (16000, 44100):
        for win_dur in (0.046, 0.093):
            synthesis_hop = _win_len(win_dur, fs) // 4
            worst = []
            for low, high in ((1, 4), (4, 10), (10, 30)):
                factors = np.linspace(low, high, 5000)
                analysis_hop = np.maximum(1, np.round(synthesis_hop / factors))
                error = np.abs(synthesis_hop / analysis_hop / factors - 1).max()
                worst.append(f"{low}-{high}: {100 * error:.2f}%")
            print(f"  {fs} Hz, win_dur {win_dur:g} s (synthesis hop {synthesis_hop}): " + ", ".join(worst))


def ripple_energy_outside_rate_range():
    """DynamicRipple: share of the dB pattern's temporal-modulation power above
    the largest |rate| in rate_range, over 5 octaves."""
    print("DynamicRipple (sources/ripples.py), 5 octaves, 4 s, seeds 0-19:")
    grid_rate = 4000.0  # Hz, well above the 350 Hz rate limit and any fanned-out rate
    t = np.arange(int(4.0 * grid_rate)) / grid_rate
    octaves = np.linspace(0, 5, 101)
    for density_change in (6.0, 0.25):
        shares, top_shares = [], []
        top_octave = octaves >= 4
        for seed in range(20):
            pattern = so.DynamicRipple(density_change=density_change, seed=seed)
            db = 20 * np.log10(_evaluate(pattern, t, octaves))  # (octaves, times)
            power = np.abs(np.fft.rfft(db - db.mean(axis=1, keepdims=True), axis=1)) ** 2
            rates = np.fft.rfftfreq(len(t), 1 / grid_rate)
            limit = max(abs(bound) for bound in pattern.rate_range)
            shares.append(power[:, rates > limit].sum() / power.sum())
            top_shares.append(power[top_octave][:, rates > limit].sum() / power[top_octave].sum())
        for where, values in (("", shares), (" in the top octave", top_shares)):
            print(
                f"  density_change {density_change:g} Hz: {100 * np.mean(values):.1f}% of the power above "
                f"{limit:g} Hz{where} (seeds range {100 * min(values):.1f}-{100 * max(values):.1f}%)"
            )


def fft_padding():
    """fast_padding: how much the length grows, and how much faster the FFT is."""
    print("fast_padding (core/fft.py), lengths 10001-60000 with pad = 0:")
    lengths = np.arange(10001, 60001)
    growth = np.array([2 * fast_padding(int(n), 0) / n for n in lengths])
    for name, chosen in (("odd", lengths % 2 == 1), ("even", lengths % 2 == 0)):
        print(
            f"  {name} n: padded length grows by median {100 * np.median(growth[chosen]):.2f}%, "
            f"at most {100 * growth[chosen].max():.2f}%"
        )
    rng = np.random.default_rng(0)
    print("  rfft time of 30 bands, own length against the padded length (best of 5):")
    for n in (44101, 48017, 88211, 96001):  # primes, the worst case for an FFT
        padded = n + 2 * fast_padding(n, 0)
        times = []
        for length in (n, padded):
            data = rng.standard_normal((length, 30))
            best = np.inf
            for _ in range(5):
                start = time.perf_counter()
                sp_fft.rfft(data, axis=0)
                best = min(best, time.perf_counter() - start)
            times.append(best)
        print(
            f"    n = {n}: {1e3 * times[0]:.1f} ms, padded to {padded}: {1e3 * times[1]:.1f} ms, "
            f"{times[0] / times[1]:.1f}x faster"
        )


def filterbank_padding():
    """Filterbank pad='auto': how much fast_padding adds on top of the ringing time."""
    print("Filterbank pad='auto' (frames/filterbank.py), cosine_filterbank(30, 50, 7600), 16 kHz:")
    bank = so.cosine_filterbank(30, 50, 7600)
    ringing = bank.ringing(FS)
    extra = []
    for n in range(16000, 16000 * 10, 997):
        pad = bank._pad_samples("auto", FS, n)
        extra.append((pad - ringing) * 2 / n)
    print(
        f"  rounding adds median {100 * np.median(extra):.3f}%, at most {100 * max(extra):.3f}% "
        f"of the length (1 to 10 s, {len(extra)} lengths)"
    )


def gaussian_spectrogram_level():
    """The level of a sound made from so.gaussian_spectrogram with the defaults."""
    print("gaussian_spectrogram (sources/gaussian_spectrogram.py), 1 s at 16 kHz, a noise's fine structure:")
    levels = []
    for seed in range(100):
        env = so.gaussian_spectrogram(1.0, FS, rng=seed)
        sound = env.to_sound(carrier=so.gaussian_noise(1.0, FS, rng=1000 + seed))
        levels.append(20 * np.log10(sound.rms))
    low, median, high = np.percentile(levels, [5, 50, 95])
    print(
        f"  RMS in dB re 1: median {median:.1f}, 5th to 95th percentile {low:.1f} to {high:.1f} "
        f"({len(levels)} draws)"
    )


if __name__ == "__main__":
    phase_vocoder_resynthesis()
    time_stretch_factor_error()
    ripple_energy_outside_rate_range()
    fft_padding()
    filterbank_padding()
    gaussian_spectrogram_level()
