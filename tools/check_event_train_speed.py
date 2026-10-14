"""Numbers behind docs/design/sources/event-train-speed.md: how long so.event_train
takes with a pitch spread, where the time went before, and how much the shared
windowed sinc (the option Cho chose) and the other options change the output.

    python tools/check_event_train_speed.py

Times are wall-clock on whatever machine runs this, one run each, so read them as
rough sizes. The event is the gallery's chirp (30 ms, 2 to 6 kHz under a Hann
window, 44.1 kHz) and, where the high end of the spectrum matters, a 30 ms noise
burst under the same window. The way copies were made before (Sound.resample) is
written out below and swapped in for the duration of a measurement.
"""

import cProfile
import pstats
import time
from fractions import Fraction

import numpy as np
from scipy import stats
from scipy.signal import resample_poly, welch

import sonore as so
from sonore.core.utils import _read_between_samples
from sonore.sources import events

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


# --- The ways compared ---------------------------------------------------------------------


def resampled_transposed(event, ratio, fs):
    """How event_train made a copy before: SciPy's polyphase resampling."""
    if callable(event) or ratio == 1:
        return shared_transposed(event, ratio, fs)
    return event.resample(fs / ratio).data


def before(function, *args, **kwargs):
    events._transposed = resampled_transposed
    try:
        return function(*args, **kwargs)
    finally:
        events._transposed = shared_transposed


def shared_copy(x, ratio):
    return shared_transposed(so.Sound(x, FS), ratio, FS)[:, 0]


shared_transposed = events._transposed


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
print("1. event_train: seconds before / now (distinct shifts), level spread 3 dB, Poisson timing")
print("   duration  rate   SD 0 cents            SD 10                 SD 100                SD 300")
for duration in (5.0, 60.0):
    for rate in (30, 300, 1000):
        cells = []
        for sd in (0, 10, 100, 300):
            took_before, _ = before(seconds, train, duration, rate, sd)
            took, _ = seconds(train, duration, rate, sd)
            cells.append(f"{took_before:6.2f} / {took:5.2f} ({distinct_shifts(duration, rate, sd):4d})")
        print(f"   {duration:5.0f} s  {rate:5d}  " + "  ".join(cells))

profile = cProfile.Profile()
profile.enable()
total, _ = before(seconds, train, 60.0, 1000, 100)
profile.disable()
by_name = {key[2]: value[3] for key, value in pstats.Stats(profile).stats.items()}
print(
    f"\n   before, 60 s at 1000/s, SD 100 cents: {total:.2f} s, of which firwin (filter design) "
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
took_main, _ = seconds(lambda: [resampled_transposed(chirp, 2 ** (c / 1200), FS) for c in shifts[::8]])
took_shared, _ = seconds(lambda: [shared_copy(x, 2 ** (c / 1200)) for c in shifts])
coarse = [Fraction(1 / 2 ** (c / 1200)).limit_denominator(1000) for c in shifts]
took_den, _ = seconds(lambda: [resample_poly(x, f.numerator, f.denominator) for f in coarse])
den_error = max(abs(1200 * np.log2(float(f) * 2 ** (c / 1200))) for f, c in zip(coarse, shifts, strict=True))
print("\n2. One transposed copy of the chirp, -400 to 400 cents")
print(f"   before (resample_poly, denominator up to 10000): {1e3 * took_main / len(shifts[::8]):.1f} ms")
print(f"   now (shared windowed sinc):                      {1e3 * took_shared / len(shifts):.2f} ms")
print(
    f"   resample_poly, denominator up to 1000:           {1e3 * took_den / len(shifts):.2f} ms, "
    f"pitch off by up to {den_error:.2f} cents"
)

print("\n   now against before, RMS difference relative to the resampled copy")
print("   cents    chirp    noise burst   noise burst, sinc not widened")
for c in (-400, -100, -7, 7, 100, 400):
    r = 2 ** (c / 1200)
    positions = np.arange(int(np.ceil(burst.n_samples / r))) * r
    row = [
        relative_difference(shared_copy(chirp.data[:, 0], r), resampled_transposed(chirp, r, FS)[:, 0]),
        relative_difference(shared_copy(burst.data[:, 0], r), resampled_transposed(burst, r, FS)[:, 0]),
        relative_difference(
            _read_between_samples(burst.data[:, 0], positions), resampled_transposed(burst, r, FS)[:, 0]
        ),
    ]
    print(f"   {c:5d}   {row[0]:.1e}  {row[1]:.1e}       {row[2]:.1e}")

r = 2 ** (-7 / 1200)
shared, main = shared_copy(burst.data[:, 0], r), resampled_transposed(burst, r, FS)[:, 0]
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

print("\n3. Whole trains, against the resampled ones")
print("   case                          before s   other s   RMS diff   band spectrum diff (dB, max)")
cases = [(5.0, 300, 100), (5.0, 1000, 100), (60.0, 1000, 100), (60.0, 1000, 300)]
for duration, rate, sd in cases:
    took_before, reference = before(seconds, train, duration, rate, sd)
    others = [("now", seconds(train, duration, rate, sd))]
    for step in (5, 10):
        rounded = before(seconds, train, duration, rate, sd, pitch=Rounded(stats.norm(0, sd), step))
        others.append((f"grid {step:2d} cents", rounded))
    for name, (took, sound) in others:
        spectrum_diff = np.max(np.abs(band_spectrum_db(sound) - band_spectrum_db(reference)))
        label = f"{name:13s}{duration:3.0f} s {rate:4d}/s SD {sd:3d}"
        print(
            f"   {label}   {took_before:6.2f}   {took:6.2f}    "
            f"{relative_difference(sound.data[:, 0], reference.data[:, 0]):.1e}    {spectrum_diff:.3f}"
        )

same = np.array_equal(train(5.0, 300, 0).data, before(train, 5.0, 300, 0).data)
print(f"\n   without a pitch spread no copies are made, so the output is unchanged: {same}")
