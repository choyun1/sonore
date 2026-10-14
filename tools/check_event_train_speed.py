"""Numbers behind docs/design/sources/event-train-speed.md: how long so.event_train
takes with a pitch spread, where the time goes, and what each faster way of making
the transposed copies would change in the output.

    python tools/check_event_train_speed.py

Times are wall-clock on whatever machine runs this, one run each, so read them as
rough sizes. The event is the gallery's chirp (30 ms, 2 to 6 kHz under a Hann
window, 44.1 kHz) and, where the high end of the spectrum matters, a 30 ms noise
burst under the same window. Nothing here changes sonore: the faster ways are
written out below and swapped in for the duration of a measurement.
"""

import cProfile
import pstats
import time
from fractions import Fraction

import numpy as np
from scipy import stats
from scipy.signal import resample_poly, welch
from scipy.special import i0

import sonore as so
from sonore.sources import events
from sonore.spatial.spatialization import _read_between_samples

FS = 44100
chirp = so.exponential_chirp(0.03, FS, 2000, 6000)
chirp = chirp * np.hanning(chirp.n_samples)[:, None]
burst = so.gaussian_noise(0.03, FS, rng=0)
burst = burst * np.hanning(burst.n_samples)[:, None]


def train(duration, rate, sd_cents, *, seed=0, pitch=None):
    pitch = pitch if pitch is not None else (stats.norm(0, sd_cents) if sd_cents else 0.0)
    return so.event_train(
        chirp,
        duration,
        stats.expon(scale=1 / rate),
        level_db=stats.norm(0, 3),
        pitch_cents=pitch,
        lead=0.03,
        rng=seed,
    )


def seconds(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return time.perf_counter() - start, result


def distinct_shifts(duration, rate, sd_cents, seed=0):
    """How many distinct whole-cent shifts event_train makes, from the same draws."""
    rng = np.random.default_rng(seed)
    n = len(events._warped_onsets(stats.expon(scale=1 / rate), duration + 0.03, rng))
    stats.norm(0, 3).rvs(size=n, random_state=rng)
    return len(np.unique(np.round(stats.norm(0, sd_cents).rvs(size=n, random_state=rng)))) if sd_cents else 1


# --- The faster ways ---------------------------------------------------------------------

HALF, BETA, PER_SAMPLE = 16, 8.0, 512  # the move_sound reader's windowed sinc, tabulated finely
_u = np.arange(-HALF * PER_SAMPLE, HALF * PER_SAMPLE + 1) / PER_SAMPLE
KERNEL = np.sinc(_u) * i0(BETA * np.sqrt(np.clip(1 - (_u / HALF) ** 2, 0, None))) / i0(BETA)


def shared_kernel_copy(x, ratio):
    """``x`` (one channel) read every ``ratio`` samples with the windowed sinc of
    move_sound's reader. Going up (ratio > 1) the kernel is stretched by the ratio,
    so its cutoff falls to fs / (2 ratio) and nothing folds back below it."""
    stretch = max(ratio, 1.0)
    positions = np.arange(int(np.ceil(len(x) / ratio))) * ratio
    reach = int(np.ceil(HALF * stretch))
    index = np.floor(positions).astype(int)[:, None] + np.arange(-reach + 1, reach + 1)
    u = (positions[:, None] - index) / stretch * PER_SAMPLE + HALF * PER_SAMPLE
    below = np.clip(u.astype(int), 0, len(KERNEL) - 2)
    share = u - below
    taps = (KERNEL[below] * (1 - share) + KERNEL[below + 1] * share) / stretch
    taps[(u < 0) | (u > 2 * HALF * PER_SAMPLE)] = 0
    padded = np.concatenate([x, [0.0]])
    return np.sum(taps * padded[np.where((index >= 0) & (index < len(x)), index, len(x))], axis=1)


def shared_kernel_transposed(event, ratio, fs):
    if callable(event) or ratio == 1:
        return original_transposed(event, ratio, fs)
    return np.stack([shared_kernel_copy(channel, ratio) for channel in event.data.T], axis=1)


original_transposed = events._transposed


class Rounded:
    """A distribution's draws rounded to a grid: what a slider tab could pass as
    pitch_cents without any change to sonore."""

    def __init__(self, distribution, step):
        self.distribution, self.step = distribution, step

    def rvs(self, size=None, random_state=None):
        return self.step * np.round(self.distribution.rvs(size=size, random_state=random_state) / self.step)


def relative_difference(a, b):
    n = max(len(a), len(b))
    a, b = np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b**2)))


def band_spectrum_db(sound):
    f, p = welch(sound.data[:, 0], FS, nperseg=4096)
    band = (f >= 2000) & (f <= 6000)
    return 10 * np.log10(p[band])


# --- 1. Now ------------------------------------------------------------------------------

train(1.0, 10, 0)  # warm up
print("1. event_train on main: seconds (distinct shifts), level spread 3 dB, Poisson timing")
print("   duration  rate   SD 0 cents     SD 10          SD 100         SD 300")
for duration in (5.0, 60.0):
    for rate in (30, 300, 1000):
        cells = []
        for sd in (0, 10, 100, 300):
            took, _ = seconds(train, duration, rate, sd)
            cells.append(f"{took:6.2f} ({distinct_shifts(duration, rate, sd):4d})")
        print(f"   {duration:5.0f} s  {rate:5d}  " + "  ".join(cells))

profile = cProfile.Profile()
profile.enable()
total, _ = seconds(train, 60.0, 1000, 100)
profile.disable()
by_name = {key[2]: value[3] for key, value in pstats.Stats(profile).stats.items()}
print(
    f"\n   60 s at 1000/s, SD 100 cents: {total:.2f} s, of which firwin (filter design) "
    f"{by_name['firwin']:.2f} s and its Kaiser window {by_name['kaiser']:.2f} s"
)
ratio = Fraction(1 / 2 ** (37 / 1200)).limit_denominator(10000)
print(
    f"   why: 37 cents is resampled {ratio.denominator}/{ratio.numerator}, "
    f"so SciPy designs a filter of {2 * 10 * max(ratio.numerator, ratio.denominator) + 1} taps"
)

# --- 2. Per copy -------------------------------------------------------------------------

shifts = [c for c in range(-400, 401) if c]
x = chirp.data[:, 0]
took_main, _ = seconds(lambda: [original_transposed(chirp, 2 ** (c / 1200), FS) for c in shifts[::8]])
took_shared, _ = seconds(lambda: [shared_kernel_copy(x, 2 ** (c / 1200)) for c in shifts])
coarse = [Fraction(1 / 2 ** (c / 1200)).limit_denominator(1000) for c in shifts]
took_den, _ = seconds(lambda: [resample_poly(x, f.numerator, f.denominator) for f in coarse])
den_error = max(abs(1200 * np.log2(float(f) * 2 ** (c / 1200))) for f, c in zip(coarse, shifts, strict=True))
print("\n2. One transposed copy of the chirp, -400 to 400 cents")
print(
    f"   main (resample_poly, fraction denominator up to 10000): {1e3 * took_main / len(shifts[::8]):.1f} ms"
)
print(f"   shared windowed-sinc kernel:                            {1e3 * took_shared / len(shifts):.2f} ms")
print(
    f"   resample_poly, denominator up to 1000:                  {1e3 * took_den / len(shifts):.2f} ms, "
    f"pitch off by up to {den_error:.2f} cents"
)

print("\n   shared kernel against main, RMS difference relative to main's copy")
print("   cents    chirp    noise burst   noise burst, kernel not stretched")
for c in (-400, -100, -7, 7, 100, 400):
    r = 2 ** (c / 1200)
    positions = np.arange(int(np.ceil(burst.n_samples / r))) * r
    row = [
        relative_difference(shared_kernel_copy(chirp.data[:, 0], r), original_transposed(chirp, r, FS)[:, 0]),
        relative_difference(shared_kernel_copy(burst.data[:, 0], r), original_transposed(burst, r, FS)[:, 0]),
        relative_difference(
            _read_between_samples(burst.data[:, 0], positions), original_transposed(burst, r, FS)[:, 0]
        ),
    ]
    print(f"   {c:5d}   {row[0]:.1e}  {row[1]:.1e}       {row[2]:.1e}")

r = 2 ** (-7 / 1200)
shared, main = shared_kernel_copy(burst.data[:, 0], r), original_transposed(burst, r, FS)[:, 0]
n = min(len(shared), len(main))
difference = np.abs(np.fft.rfft(shared[:n] - main[:n], 8192)) ** 2
power = np.sum(np.abs(np.fft.rfft(main[:n], 8192)) ** 2)
f = np.fft.rfftfreq(8192, 1 / FS)
print("\n   where the noise burst's difference lies (-7 cents), dB re the copy's power:")
for low, high in ((0, 18000), (18000, 20000), (20000, FS / 2 + 1)):
    band = (f >= low) & (f < high)
    level = 10 * np.log10(difference[band].sum() / power)
    print(f"   {low / 1000:4.0f} to {min(high, FS / 2) / 1000:5.2f} kHz: {level:6.1f}")

# --- 3. Whole trains ---------------------------------------------------------------------

print("\n3. Whole trains, against main")
print("   case                        main s   faster s   RMS diff   band spectrum diff (dB, max)")
cases = [(5.0, 300, 100), (5.0, 1000, 100), (60.0, 1000, 100), (60.0, 1000, 300)]
for duration, rate, sd in cases:
    took_main, reference = seconds(train, duration, rate, sd)
    events._transposed = shared_kernel_transposed
    try:
        took, shared = seconds(train, duration, rate, sd)
    finally:
        events._transposed = original_transposed
    spectrum_diff = np.max(np.abs(band_spectrum_db(shared) - band_spectrum_db(reference)))
    print(
        f"   shared kernel {duration:3.0f} s {rate:4d}/s SD {sd:3d}  {took_main:6.2f}   {took:6.2f}    "
        f"{relative_difference(shared.data[:, 0], reference.data[:, 0]):.1e}    {spectrum_diff:.3f}"
    )
    for step in (5, 10):
        took, rounded = seconds(train, duration, rate, sd, pitch=Rounded(stats.norm(0, sd), step))
        spectrum_diff = np.max(np.abs(band_spectrum_db(rounded) - band_spectrum_db(reference)))
        label = f"grid {step:2d} cents {duration:3.0f} s {rate:4d}/s SD {sd:3d}"
        print(
            f"   {label}   {took_main:6.2f}   {took:6.2f}    "
            f"{relative_difference(rounded.data[:, 0], reference.data[:, 0]):.1e}    {spectrum_diff:.3f}"
        )

reference = train(5.0, 300, 0)
events._transposed = shared_kernel_transposed
try:
    same = np.array_equal(train(5.0, 300, 0).data, reference.data)
finally:
    events._transposed = original_transposed
print(f"\n   without a pitch spread the copies are not made, so the output is the same as main's: {same}")
