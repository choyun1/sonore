"""Numerical checks for the claims in docs/design/views/cepstrum.md (C1-C7).

Like the Frames checkers, this is deliberately independent of sonore: only
NumPy, SciPy and soundfile (to read the gallery sentence), with every window,
filter and transform written out from its formula. Each line prints the claim
number and the number that supports it.

    python tools/check_cepstrum_claims.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

FS = 16000.0
ROOT = Path(__file__).resolve().parent.parent
SPEECH = ROOT / "docs" / "speech"
rng = np.random.default_rng(0)


def report(claim, text, value):
    print(f"{claim:4s} {text:<74s} {value:.4g}")


def hann_periodic(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def real_cepstrum(segment, n_fft):
    """Inverse FFT of the natural log magnitude, full length n_fft."""
    return np.fft.ifft(np.log(np.abs(np.fft.fft(segment, n_fft))))


def fold(c):
    """The minimum-phase fold of a real cepstrum of even length N: keep c[0]
    and c[N/2], double 1..N/2-1, zero the rest."""
    n = len(c)
    out = np.zeros_like(c)
    out[0] = c[0]
    out[1 : n // 2] = 2 * c[1 : n // 2]
    out[n // 2] = c[n // 2]
    return out


# Vowel /a/ formants (Peterson & Barney male averages, rounded) and bandwidths.
FORMANTS = [(730, 60), (1090, 100), (2440, 120), (3400, 175)]


def vowel_filter():
    """All-pole cascade of second-order resonators: (b, a)."""
    a = np.array([1.0])
    for f, bw in FORMANTS:
        r = np.exp(-np.pi * bw / FS)
        a = np.convolve(a, [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r])
    return np.array([1.0]), a


def vowel_response(freqs):
    b, a = vowel_filter()
    z = np.exp(-2j * np.pi * np.asarray(freqs) / FS)
    return np.polyval(b[::-1], z) / np.polyval(a[::-1], z)


def impulse_vowel(f0, dur=0.5):
    """The steady state of a band-limited pulse train at f0 through the vowel
    filter: harmonics of f0 below 0.95 Nyquist with the filter's gain and phase."""
    t = np.arange(int(dur * FS)) / FS
    harm = np.arange(1, int(0.95 * FS / 2 // f0) + 1) * f0
    H = vowel_response(harm)
    return np.sum(np.abs(H)[:, None] * np.cos(2 * np.pi * harm[:, None] * t + np.angle(H)[:, None]), axis=0)


def cepstral_f0(segment, n_fft, f_lo=60.0, f_hi=400.0):
    """F0 from the largest real-cepstrum peak in 1/f_hi..1/f_lo s, refined by
    a parabola through the peak and its neighbors. Returns (f0, peak height)."""
    c = np.real(real_cepstrum(segment, n_fft))
    q_lo, q_hi = int(np.floor(FS / f_hi)), int(np.ceil(FS / f_lo))
    k = q_lo + int(np.argmax(c[q_lo : q_hi + 1]))
    y0, y1, y2 = c[k - 1], c[k], c[k + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return FS / (k + shift), y1


# ---------------------------------------------------------------- C1
x = rng.standard_normal(256)
c = real_cepstrum(x * hann_periodic(256), 512)
report(
    "C1",
    "real cepstrum of a real time window: max |imag| / max |real|",
    np.abs(c.imag).max() / np.abs(c.real).max(),
)
report("C1", "real cepstrum is even: max |c[n] - c[N-n]|", np.abs(c.real[1:] - c.real[1:][::-1]).max())
half = np.fft.irfft(np.log(np.abs(np.fft.rfft(x * hann_periodic(256), 512))), 512)
report(
    "C1", "irfft of the one-sided log magnitude equals the full ifft: max diff", np.abs(half - c.real).max()
)

# ---------------------------------------------------------------- C2
X = np.fft.rfft(x * hann_periodic(256), 512)
cq = np.fft.irfft(np.log(np.abs(X)), 512)
mag = np.exp(np.fft.rfft(cq, 512).real)
report("C2", "unliftered round trip, max relative magnitude error", np.max(np.abs(mag / np.abs(X) - 1)))
X_back = mag * np.exp(1j * np.angle(X))
report(
    "C2",
    "with the original phase restored, max |X_back - X| / max |X|",
    np.abs(X_back - X).max() / np.abs(X).max(),
)
c_scaled = np.fft.irfft(np.log(np.abs(3.7 * X)), 512)
d = c_scaled - cq
report("C2", "scaling by a = 3.7 shifts c[0] by ln a: |c0 shift - ln a|", abs(d[0] - np.log(3.7)))
report("C2", "and leaves every other quefrency unchanged: max |shift|", np.abs(d[1:]).max())

# ---------------------------------------------------------------- C3
# A minimum-phase FIR: 24 zeros, conjugate pairs, inside radius 0.9.
roots = 0.9 * rng.random(12) * np.exp(1j * np.pi * rng.random(12))
h_min = np.real(np.poly(np.concatenate([roots, roots.conj()])))
h_min /= np.abs(h_min).sum()
for n_fft in (64, 256, 1024, 4096):
    H = np.fft.fft(h_min, n_fft)
    H_rec = np.exp(np.fft.fft(fold(np.real(np.fft.ifft(np.log(np.abs(H)))))))
    report(
        "C3",
        f"minimum-phase FIR (25 taps), fold at n_fft {n_fft:4d}: max |H_rec - H| / max |H|",
        np.abs(H_rec - H).max() / np.abs(H).max(),
    )
# A mixed-phase FIR: reflect four of the zeros outside the unit circle.
mixed = np.concatenate([roots, roots.conj()])
mixed[:4] = 1 / mixed[:4].conj()
mixed[12:16] = 1 / mixed[12:16].conj()
h_mix = np.real(np.poly(mixed))
H = np.fft.fft(h_mix, 4096)
h_rec = np.real(np.fft.ifft(np.exp(np.fft.fft(fold(np.real(np.fft.ifft(np.log(np.abs(H)))))))))
rel = np.abs(np.fft.fft(h_rec)) / np.abs(H) - 1
report("C3", "mixed-phase FIR, fold: max relative magnitude error", np.max(np.abs(rel)))
e_mix, e_rec = np.cumsum(h_mix**2), np.cumsum(h_rec[:4096] ** 2)
report("C3", "mixed-phase FIR, fold: energy in first 5 taps, original / folded", e_mix[4] / e_mix[-1])
report("C3", "", e_rec[4] / e_rec[-1])

# ---------------------------------------------------------------- C4
for periods in (1.5, 2, 3, 4):
    worst, octave_errors = 0.0, 0
    for f0 in (80, 100, 120, 150, 200, 250, 300):
        sig = impulse_vowel(f0)
        n_win = int(round(periods / f0 * FS))
        n_fft = 1 << int(np.ceil(np.log2(max(n_win, FS / 60) * 2)))
        for start in range(2000, 6000, 397):
            est, _ = cepstral_f0(sig[start : start + n_win] * hann_periodic(n_win), n_fft)
            err = abs(est / f0 - 1)
            if abs(np.log2(est / f0)) > 0.5:
                octave_errors += 1
            else:
                worst = max(worst, err)
    report("C4", f"Hann {periods:g} periods, 7 F0s x 11 positions: worst relative error (non-octave)", worst)
    report("C4", f"Hann {periods:g} periods: time windows off by an octave or more (of 77)", octave_errors)
for win_ms in (20, 40):
    n_win = int(win_ms * FS / 1000)
    worst = 0.0
    for f0 in (80, 100, 120, 150, 200, 250, 300):
        sig = impulse_vowel(f0)
        est, _ = cepstral_f0(sig[3000 : 3000 + n_win] * hann_periodic(n_win), 2048)
        worst = max(worst, abs(est / f0 - 1))
    report("C4", f"fixed Hann {win_ms} ms, F0 80-300 Hz: worst relative error", worst)

# ---------------------------------------------------------------- C5
for f0 in (100, 200):
    sig = impulse_vowel(f0)
    n_win = int(round(3 / f0 * FS))  # pitch-adaptive, 3 periods
    n_fft = 4096
    segment = sig[3000 : 3000 + n_win] * hann_periodic(n_win)
    X = np.fft.rfft(segment, n_fft)
    cq = np.fft.irfft(np.log(np.abs(X)), n_fft)
    q_c = int(0.5 * FS / f0)  # half a period
    lif = np.zeros(n_fft)
    lif[:q_c] = 1
    lif[n_fft - q_c + 1 :] = 1
    env = np.exp(np.fft.rfft(cq * lif, n_fft).real)
    freqs = np.fft.rfftfreq(n_fft, 1 / FS)
    harm = np.arange(1, int(4000 // f0) + 1) * f0
    idx = np.round(harm / FS * n_fft).astype(int)
    true_db = 20 * np.log10(np.abs(vowel_response(harm)))
    env_db = 20 * np.log10(env[idx])
    pk_db = 20 * np.log10(np.abs(X[idx]))
    # Compare shapes: remove the mean offset (window gain, pulse amplitude).
    e_env = (env_db - true_db) - np.mean(env_db - true_db)
    rms_db = np.sqrt(np.mean(e_env**2))
    report("C5", f"F0 {f0} Hz, 3-period Hann, lifter < T0/2: env - true at harmonics, RMS dB", rms_db)
    report(
        "C5",
        f"F0 {f0} Hz: envelope minus harmonic peaks, mean dB (negative = below peaks)",
        np.mean(env_db - pk_db),
    )
    f1 = np.argmin(np.abs(harm - FORMANTS[0][0]))
    report("C5", f"F0 {f0} Hz: same at the harmonic nearest F1, dB", env_db[f1] - pk_db[f1])

# ---------------------------------------------------------------- C6
snd, fs = sf.read(SPEECH / "bdl_arctic_a0131.flac")
assert fs == FS
tab = np.loadtxt(SPEECH / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
times, harvest = tab[:, 0], tab[:, 1]
n_win, n_fft = int(0.040 * FS), 2048
w = hann_periodic(n_win)
est, peak = np.full(len(times), np.nan), np.full(len(times), np.nan)
for i, t in enumerate(times):
    s = int(round(t * FS)) - n_win // 2
    if s < 0 or s + n_win > len(snd):
        continue
    est[i], peak[i] = cepstral_f0(snd[s : s + n_win] * w, n_fft, 75, 400)
ok = np.isfinite(est)
voiced = ok & (harvest > 0)
unvoiced = ok & (harvest == 0)
ratio = est[voiced] / harvest[voiced]
report("C6", "bdl sentence, 40 ms Hann, Harvest-voiced time windows compared", voiced.sum())
report("C6", "agree with Harvest within 5%, fraction", np.mean(np.abs(ratio - 1) < 0.05))
report("C6", "about double Harvest (ratio 1.8-2.2), time windows", np.sum((ratio > 1.8) & (ratio < 2.2)))
report("C6", "about half Harvest (ratio 0.45-0.55), time windows", np.sum((ratio > 0.45) & (ratio < 0.55)))
report("C6", "median cepstral peak height, Harvest-voiced time windows", np.median(peak[voiced]))
report("C6", "median cepstral peak height, Harvest-unvoiced time windows", np.median(peak[unvoiced]))
thr = 0.1
vd = peak > thr
report("C6", f"peak > {thr}: fraction of Harvest-voiced time windows called voiced", np.mean(vd[voiced]))
report("C6", f"peak > {thr}: fraction of Harvest-unvoiced time windows called voiced", np.mean(vd[unvoiced]))
both = voiced & vd
report(
    "C6",
    f"peak > {thr} and Harvest-voiced: agree within 5%, fraction",
    np.mean(np.abs(est[both] / harvest[both] - 1) < 0.05),
)

# ---------------------------------------------------------------- C7
with np.errstate(divide="ignore"):
    report(
        "C7",
        "digital silence: log|X| of an all-zero time window is finite? (1 = yes)",
        float(np.isfinite(np.log(np.abs(np.fft.rfft(np.zeros(512))))).all()),
    )
lowest = []
for t in times:
    s = int(round(t * FS)) - n_win // 2
    if 0 <= s and s + n_win <= len(snd):
        m = np.abs(np.fft.rfft(snd[s : s + n_win] * w, n_fft))
        if m.max() > 0:
            lowest.append(20 * np.log10(m.min() / m.max()))
report(
    "C7",
    "bdl sentence, 40 ms time windows: lowest bin relative to the time window's maximum, dB",
    min(lowest),
)
