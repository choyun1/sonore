"""Numerical checks for the claims in docs/design/sources/harmonic-source.md (C1-C7).

Like the other claim checkers, this is independent of sonore: only NumPy,
SciPy and soundfile (to read the gallery sentence's stored F0 track), with
every step written out from its formula. It contains a small prototype of
the source the design describes, so the numbers in the document can be
reproduced; it is not the library code. Each line prints the claim number
and the number that supports it.

    python tools/check_harmonic_source_claims.py

It runs in a few seconds.
"""

import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

FS = 16000.0
HOP = 0.005  # hop of the F0 track [s], as in the stored track and docs/design/views/f0.md
ROOT = Path(__file__).resolve().parent.parent
SPEECH = ROOT / "docs" / "speech"


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


# ------------------------------------------------------------- the prototype
def fill_unvoiced(f0):
    """Unvoiced time windows (F0 = 0) take values interpolated linearly between the
    voiced time windows around them; leading and trailing ones hold the nearest."""
    v = f0 > 0
    i = np.arange(len(f0))
    return np.interp(i, i[v], f0[v])


def to_samples(t_c, values, n, fs=FS):
    """A contour given at the times t_c onto n samples by linear interpolation (ends held)."""
    return np.interp(np.arange(n) / fs, t_c, values)


def phase_trapezoid(f, fs=FS):
    """Running phase 2*pi*integral(f): exact for f linear between samples."""
    return 2 * np.pi * np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2)]) / fs


def phase_rectangle(f, fs=FS):
    """The running sum the gallery pages use today (pv.py, resynthesis.py)."""
    return 2 * np.pi * np.cumsum(f) / fs


def voicing_gate(t_c, f0_c, n, ramp, fs=FS):
    """1 where the nearest time window is voiced, 0 elsewhere, with every step
    smoothed by a Hann window of `ramp` seconds (none if ramp == 0)."""
    g = (to_samples(t_c, (f0_c > 0).astype(float), n, fs) >= 0.5).astype(float)
    m = int(round(ramp * fs))
    if m < 2:
        return g
    w = np.hanning(m + 2)[1:-1]
    return np.convolve(g, w / w.sum(), mode="same")


def taper(freq, f_max, width=0.1):
    """Harmonic gain: 1 below (1 - width) * f_max, cos^2 down to 0 at f_max."""
    lo = (1 - width) * f_max
    u = np.clip((freq - lo) / (f_max - lo), 0.0, 1.0)
    return np.cos(np.pi / 2 * u) ** 2


def harmonic_source(f, gate=None, f_max=None, harmonics=None, amplitudes=None, fs=FS, phase=phase_trapezoid):
    """sum_k a_k * taper(k f(t)) * cos(k Phi(t)), Phi = 2 pi integral f.

    f is the per-sample F0 [Hz]. Without `harmonics`, every k whose frequency
    is below f_max at some sample is synthesized, faded by the taper."""
    f_max = 0.45 * fs if f_max is None else f_max
    phi = phase(f, fs)
    ks = np.arange(1, int(f_max / f.min()) + 1) if harmonics is None else np.asarray(harmonics)
    amps = np.ones(len(ks)) if amplitudes is None else np.asarray(amplitudes, float)
    out = np.zeros(len(f))
    for k, a in zip(ks, amps, strict=True):
        g = taper(k * f, f_max) if harmonics is None else 1.0
        out += a * g * np.cos(k * phi)
    return out if gate is None else out * gate


def inst_freq(x, fs=FS):
    """Instantaneous frequency from the analytic signal, at sample midpoints."""
    return np.diff(np.unwrap(np.angle(hilbert(x)))) * fs / (2 * np.pi)


def power_above(x, f_cut, fs=FS, n_win=512):
    """Power above f_cut over total, from a Hann STFT with 75% overlap [dB]."""
    w = np.hanning(n_win)
    segments = np.lib.stride_tricks.sliding_window_view(x, n_win)[:: n_win // 4] * w
    p = np.abs(np.fft.rfft(segments, axis=1)) ** 2
    f = np.fft.rfftfreq(n_win, 1 / fs)
    return 10 * np.log10(p[:, f > f_cut].sum() / p.sum() + 1e-300)


def time_windows(f_of_t, dur):
    t_c = np.arange(0, dur + HOP / 2, HOP)
    return t_c, f_of_t(t_c)


# ---------------------------------------------------------------- C1
# A constant contour gives exactly the fixed-F0 harmonic complex.
dur = 0.5
n = int(dur * FS)
t = np.arange(n) / FS
t_c, f_c = time_windows(lambda t: np.full_like(t, 220.0), dur)
x = harmonic_source(to_samples(t_c, f_c, n), harmonics=range(1, 11))
ref = sum(np.cos(2 * np.pi * k * 220.0 * t) for k in range(1, 11))
report(
    "C1",
    "constant 220 Hz contour vs fixed harmonic complex (10 harmonics), max abs diff",
    np.max(np.abs(x - ref)),
)

# ---------------------------------------------------------------- C2
# A linear glide sampled every 5 ms is reproduced exactly: linear
# interpolation of a linear contour is exact, and the trapezoid integrates it
# exactly. The running sum leads by (f_0 + f_i) / (2 fs) cycles.
dur = 1.0
n = int(dur * FS)
t = np.arange(n) / FS
t_c, f_c = time_windows(lambda t: 100 + 200 * t, dur)
f = to_samples(t_c, f_c, n)
closed = np.cos(2 * np.pi * (100 * t + 100 * t**2))
x_trap = harmonic_source(f, harmonics=[1])
x_rect = harmonic_source(f, harmonics=[1], phase=phase_rectangle)
report(
    "C2",
    "glide 100->300 Hz sampled every 5 ms vs closed-form chirp, max abs diff (trapezoid)",
    np.max(np.abs(x_trap - closed)),
)
report(
    "C2", "same, running sum as in the gallery pages today (max abs diff)", np.max(np.abs(x_rect - closed))
)
report(
    "C2",
    "running sum minus trapezoid minus pi (f_0 + f_i) / fs, max abs [rad]",
    np.max(np.abs(phase_rectangle(f) - phase_trapezoid(f) - np.pi * (f[0] + f) / FS)),
)

# ---------------------------------------------------------------- C3
# The instantaneous frequency of every harmonic follows k times the contour.
# Vibrato: 150 Hz, +-4%, 5.5 Hz, sampled every 5 ms. Linear interpolation
# errs by at most h^2/8 max|f''| between time windows.
dur = 2.0
n = int(dur * FS)
t = np.arange(n) / FS


def vib(t):
    return 150 * (1 + 0.04 * np.sin(2 * np.pi * 5.5 * t))


t_c, f_c = time_windows(vib, dur)
f = to_samples(t_c, f_c, n)
mid = (np.arange(n - 1) + 0.5) / FS
inner = (mid > 0.1) & (mid < dur - 0.1)
bound = HOP**2 / 8 * 150 * 0.04 * (2 * np.pi * 5.5) ** 2 / (150 * 0.96)
report("C3", "vibrato sampled every 5 ms: bound h^2/8 max|f''| / min f (relative)", bound)
for k in (1, 10, 40):
    ifk = inst_freq(harmonic_source(f, harmonics=[k]))
    err = np.max(np.abs(ifk[inner] / (k * vib(mid[inner])) - 1))
    report("C3", f"  measured IF of harmonic {k} vs k * true contour, max relative error", err)
if1 = inst_freq(harmonic_source(f, harmonics=[1]))
if10 = inst_freq(harmonic_source(f, harmonics=[10]))
report(
    "C3",
    "IF of harmonic 10 / IF of harmonic 1, max |ratio - 10|",
    np.max(np.abs(if10[inner] / if1[inner] - 10)),
)

# ---------------------------------------------------------------- C4
# On a real track (the stored Harvest track of the gallery sentence), the
# accumulated phase has no jumps, while per-time-window synthesis with absolute
# time, the obvious shortcut, clicks at every time window boundary.
track = np.loadtxt(SPEECH / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
t_c, f0_c = track[:, 0], track[:, 1]
n = int(round(2.53 * FS))
F_MAX = 7200.0
f = to_samples(t_c, fill_unvoiced(f0_c), n)
gate = voicing_gate(t_c, f0_c, n, ramp=0.005)
x_acc = harmonic_source(f, gate, F_MAX)

t = np.arange(n) / FS
window_of = np.clip(np.round((t - t_c[0]) / HOP).astype(int), 0, len(t_c) - 1)
f_naive = fill_unvoiced(f0_c)[window_of]  # piecewise constant, phase from absolute time
x_naive = np.zeros(n)
for k in range(1, int(F_MAX / f_naive.min()) + 1):
    x_naive += taper(k * f_naive, F_MAX) * np.cos(2 * np.pi * k * f_naive * t)
x_naive *= gate
report(
    "C4",
    "bdl track, accumulated phase: max |phase step - 2 pi f/fs| over samples [rad]",
    np.max(np.abs(np.diff(phase_trapezoid(f)) - 2 * np.pi * (f[1:] + f[:-1]) / 2 / FS)),
)
report("C4", "bdl track, power above 7.6 kHz, accumulated phase [dB re total]", power_above(x_acc, 7600))
report(
    "C4",
    "bdl track, power above 7.6 kHz, phase per time window from abs. time [dB re total]",
    power_above(x_naive, 7600),
)

# ---------------------------------------------------------------- C5
# Voicing. Interpolating straight through the unvoiced zeros sweeps the pitch
# down toward 0 Hz at every voicing boundary; filling the gaps first keeps
# it in the voiced range. Ramped gates switch on without clicks.
f_through = to_samples(t_c, f0_c, n)
on = gate > 0.01
report("C5", "voiced F0 range of the track, lowest [Hz]", f0_c[f0_c > 0].min())
report("C5", "lowest F0 while the gate is open, gaps filled [Hz]", f[on].min())
report("C5", "lowest F0 while the gate is open, interpolated through zeros [Hz]", f_through[on].min())
hard = voicing_gate(t_c, f0_c, n, ramp=0.0)
x_hard = harmonic_source(f, hard, F_MAX)
report("C5", "power above 7.6 kHz with a hard 0/1 gate [dB re total]", power_above(x_hard, 7600))
report("C5", "power above 7.6 kHz with 5 ms Hann ramps [dB re total]", power_above(x_acc, 7600))

# ---------------------------------------------------------------- C6
# Band-limiting. An exponential glide 100 -> 1000 Hz at 16 kHz. A fixed set
# of harmonics chosen at the start (all below Nyquist at 100 Hz) puts most
# of its power above Nyquist by the end, where it aliases. The taper keeps
# every component below f_max.
dur = 1.0
n = int(dur * FS)
t_c, f_c = time_windows(lambda t: 100 * 10**t, dur)
f = to_samples(t_c, f_c, n)
K = int(FS / 2 / 100) - 1  # 79: every harmonic below Nyquist at 100 Hz
ks = np.arange(1, K + 1)
aliased = np.mean([(k * f > FS / 2).mean() for k in ks])
report("C6", f"glide 100->1000 Hz, fixed {K} harmonics: share of power above Nyquist (aliased)", aliased)
x_tap = harmonic_source(f, f_max=F_MAX)
report("C6", "same glide, tapered at 7.2 kHz: power above 7.6 kHz [dB re total]", power_above(x_tap, 7600))
report("C6", "  number of harmonics at the start", np.sum(ks * f[0] < F_MAX))
report("C6", "  number of harmonics at the end", np.sum(np.arange(1, 80) * f[-1] < F_MAX))

# ---------------------------------------------------------------- C7
# Speed: the gallery sentence's track, harmonics up to 7.2 kHz, at 16 kHz.
f = to_samples(track[:, 0], fill_unvoiced(f0_c), int(round(2.53 * FS)))
start = time.perf_counter()
for _ in range(3):
    harmonic_source(f, gate, F_MAX)
report(
    "C7",
    "bdl track, 2.53 s at 16 kHz, harmonics to 7.2 kHz: seconds per call (Python loop over k)",
    (time.perf_counter() - start) / 3,
)
