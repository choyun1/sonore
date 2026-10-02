"""Numerical checks for the claims in docs/design/frames.md, step 3 (C15-C20).

Like the step 1 and step 2 checkers, this is deliberately independent of
sonore: only NumPy and SciPy, with every window, filter and transform written
out from its formula. Each line prints the claim number and the number that
supports it.

    python tools/check_frames_step3_claims.py
"""

from math import factorial

import numpy as np
from scipy.signal import hilbert

FS = 16000.0


def report(claim, text, value):
    print(f"{claim:4s} {text:<74s} {value:.4g}")


def hann_periodic(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def pulse_train(f0, dur, fs=FS, f_max=None):
    """Sum of equal-amplitude cosine harmonics of f0 up to f_max (a band-limited
    pulse train, pulses at t = 0, 1/f0, 2/f0, ...)."""
    f_max = fs / 2 * 0.95 if f_max is None else f_max
    t = np.arange(int(round(dur * fs))) / fs
    return sum(np.cos(2 * np.pi * h * f0 * t) for h in range(1, int(f_max // f0) + 1))


def blackman_periodic(n):
    a = 2 * np.pi * np.arange(n) / n
    return 0.42 - 0.5 * np.cos(a) + 0.08 * np.cos(2 * a)


def segment_power(x, center, n_win, n_fft, fs=FS, window=hann_periodic):
    """|FFT|^2 of x windowed by a periodic window (Hann by default) of n_win
    samples centered at sample `center`, zero-padded to n_fft, plus its
    frequency grid."""
    seg = x[center - n_win // 2 : center - n_win // 2 + n_win] * window(n_win)
    return np.abs(np.fft.rfft(seg, n_fft)) ** 2, np.fft.rfftfreq(n_fft, 1 / fs)


def harmonic_dip_db(f0, n_win, f_around=1000.0):
    """Peak-to-dip ratio [dB] between the two harmonics either side of
    f_around in the power spectrum averaged over window positions in one
    period (so the result does not depend on where the window sits)."""
    x = pulse_train(f0, 1.0)
    period = int(round(FS / f0))
    centers = len(x) // 2 + np.arange(period)
    p = np.mean([segment_power(x, c, n_win, 16 * 4096)[0] for c in centers[:: max(1, period // 16)]], axis=0)
    f = np.fft.rfftfreq(16 * 4096, 1 / FS)
    h = int(f_around // f0)
    lo, hi = h * f0, (h + 1) * f0
    band = (f > lo) & (f < hi)
    peak = min(p[np.argmin(abs(f - lo))], p[np.argmin(abs(f - hi))])
    return 10 * np.log10(peak / p[band].min())


def pulse_depth_db(f0, n_win, band=(1000.0, 3000.0)):
    """Max-to-min ratio [dB], over window positions in one period, of the
    power summed over a frequency band: how clearly each glottal pulse shows
    as a vertical stripe."""
    x = pulse_train(f0, 1.0)
    period = int(round(FS / f0))
    n_fft = 4096
    f = np.fft.rfftfreq(n_fft, 1 / FS)
    sel = (f >= band[0]) & (f <= band[1])
    p = [segment_power(x, len(x) // 2 + c, n_win, n_fft)[0][sel].sum() for c in range(period)]
    return 10 * np.log10(max(p) / min(p))


# ---------------------------------------------- C15: equivalent noise bandwidth of Hann
# ENBW = N sum w^2 / (sum w)^2 bins = 1.5 / T Hz for a periodic Hann of length T.
for n in (80, 528):
    w = hann_periodic(n)
    report("C15", f"Hann {n} samples: ENBW in bins (expect 1.5)", n * np.sum(w**2) / np.sum(w) ** 2)
for bw in (300.0, 45.0):
    report("C15", f"window length [ms] for ENBW {bw:g} Hz (1.5 / ENBW)", 1.5 / bw * 1e3)

# ------------------------- C16: fixed windows vs F0; pitch-adaptive windows are F0-invariant
for f0 in (100.0, 130.0, 200.0):
    for ms in (33.3, 5.0):
        n = int(round(ms * 1e-3 * FS))
        report(
            "C16",
            f"F0 {f0:g} Hz, Hann {ms:g} ms: harmonic dip / pulse depth [dB]",
            harmonic_dip_db(f0, n),
        )
        report("", "", pulse_depth_db(f0, n))
for k in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    for f0 in (100.0, 200.0):
        n = int(round(k * FS / f0))
        report(
            "C16",
            f"Hann {k:g} periods, F0 {f0:g} Hz: harmonic dip / pulse depth [dB]",
            harmonic_dip_db(f0, n),
        )
        report("", "", pulse_depth_db(f0, n))

# --------------------------------- C17: TANDEM pair cancels the period-rate fluctuation
# For a periodic x, the power P(t, f) through a window centered at t is
# T0-periodic in t, so P(t) + P(t + T0/2) keeps only the even Fourier
# components of that fluctuation. Measured as (max - min) / mean over t, per
# frequency, then the worst case over 300-4000 Hz.
f0 = 125.0
x = pulse_train(f0, 1.0)
period = int(round(FS / f0))  # 128 samples, so T0/2 is exactly 64
for name, window, k in (
    ("Hann", hann_periodic, 2.0),
    ("Hann", hann_periodic, 2.5),
    ("Hann", hann_periodic, 3.0),
    ("Blackman", blackman_periodic, 2.5),  # the published TANDEM-STRAIGHT window
):
    n = int(round(k * period))
    n_fft = 4096
    f = np.fft.rfftfreq(n_fft, 1 / FS)
    sel = (f > 300) & (f < 4000)
    mid = len(x) // 2
    P = np.array([segment_power(x, mid + c, n, n_fft, window=window)[0] for c in range(period)])  # (t, f)
    Pt = P + np.roll(P, -period // 2, axis=0)
    single = ((P.max(0) - P.min(0)) / P.mean(0))[sel]
    pair = ((Pt.max(0) - Pt.min(0)) / Pt.mean(0))[sel]
    report("C17", f"{name} {k:g} T0, F0 125 Hz: worst fluctuation, single window", single.max())
    report("C17", f"{name} {k:g} T0, F0 125 Hz: worst fluctuation, TANDEM pair", pair.max())
    report("C17", f"{name} {k:g} T0, F0 125 Hz: median fluctuation, single / pair", np.median(single))
    report("", "", np.median(pair))


# -------------------------------------- C18: reassignment, with the window-centered phase
def stft_column(x, center, w, n_fft):
    """One STFT column with the phase referenced to the window's center
    (SciPy's and sonore's convention): sum_m x[center + m] w[m] e^{-i 2 pi f m},
    m = -n/2 .. n/2 - 1, with sample m stored at FFT index m mod n_fft."""
    n = len(w)
    m = np.arange(n) - n // 2
    buf = np.zeros(n_fft)
    buf[m % n_fft] = x[center + m] * w
    return np.fft.rfft(buf)


def reassign(x, center, n, n_fft, kind, sign=1.0):
    """Reassigned (t, f) [s, Hz] of every bin of one column, from the window w,
    the time-weighted window m*w and the derivative w', all sampled from their
    continuous formulas. kind: 'hann' (periodic) or 'gauss' (sigma = n/8).
    sign=-1 applies the frequency correction with the opposite sign."""
    m = np.arange(n) - n // 2  # samples relative to the center
    if kind == "hann":
        w = 0.5 + 0.5 * np.cos(2 * np.pi * m / n)  # periodic Hann, peak at m = 0
        dw = -0.5 * (2 * np.pi / n) * np.sin(2 * np.pi * m / n) * FS  # per second
    else:
        sig = n / 8
        w = np.exp(-0.5 * (m / sig) ** 2)
        dw = -m / sig**2 * w * FS
    X = stft_column(x, center, w, n_fft)
    Xt = stft_column(x, center, m / FS * w, n_fft)
    Xd = stft_column(x, center, dw, n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / FS)
    p = np.abs(X) ** 2
    t_hat = center / FS + np.real(Xt * X.conj()) / p
    f_hat = f - sign * np.imag(Xd * X.conj()) / p / (2 * np.pi)
    return t_hat, f_hat, p


n, n_fft, c = 512, 512, 8000  # 32 ms window centered at 0.5 s
t = np.arange(16000) / FS
f_tone = 1000.0 + 0.37 * FS / n_fft  # 0.37 bins off the grid
impulse = np.zeros_like(t)
impulse[c + 50] = 1.0  # 3.1 ms after the window center
rate = 3000.0  # a linear chirp, 500 Hz at t = 0, rising 3000 Hz/s
chirp = np.cos(2 * np.pi * (500 * t + rate / 2 * t**2))
for kind in ("hann", "gauss"):
    th, fh, p = reassign(np.cos(2 * np.pi * f_tone * t), c, n, n_fft, kind)
    top = p > p.max() * 1e-2  # bins within 20 dB of the peak
    report(
        "C18", f"{kind}: off-bin tone, max |f_hat - f_tone| within 20 dB [Hz]", np.abs(fh[top] - f_tone).max()
    )
    th, fh, p = reassign(np.cos(2 * np.pi * f_tone * t), c, n, n_fft, kind, sign=-1.0)
    report("C18", f"{kind}:   the same with the opposite sign [Hz]", np.abs(fh[top] - f_tone).max())
    th, fh, p = reassign(impulse, c, n, n_fft, kind)
    band = p > p.max() * 1e-4  # within 40 dB
    report(
        "C18",
        f"{kind}: impulse, max |t_hat - t_impulse| within 40 dB [ms]",
        1e3 * np.abs(th[band] - (c + 50) / FS).max(),
    )
    th, fh, p = reassign(chirp, c, n, n_fft, kind)
    top = p > p.max() * 1e-2
    report(
        "C18",
        f"{kind}: chirp, max |f_hat - IF(t_hat)| within 20 dB [Hz]",
        np.abs(fh[top] - (500 + rate * th[top])).max(),
    )
    report(
        "C18",
        f"{kind}: chirp, plain spectrogram, span of bins within 20 dB [Hz]",
        np.ptp(np.fft.rfftfreq(n_fft, 1 / FS)[top]),
    )


# ---------------------------------- C19: gammatone delay compensation for display
def erb(f):
    return 24.7 * (4.37e-3 * f + 1)  # Glasberg & Moore (1990)


def gammatone_ir(fc, n_samp, order=4, fs=FS):
    """Impulse response of the exact response (step 2, C10), computed on a long
    FFT grid so the causal response is not wrapped."""
    b = 1.019 * erb(fc)
    k = factorial(order - 1) / (2 * np.pi) ** order / 2
    f = np.fft.fftfreq(n_samp, 1 / fs)
    H = k * ((b + 1j * (f - fc)) ** -order + (b + 1j * (f + fc)) ** -order)
    return np.real(np.fft.ifft(H)), b


cfs = 100 * 2 ** np.arange(0, 5.7, 0.25)  # 100 Hz to about 5 kHz
peaks, gd, pk = [], [], []
for fc in cfs:
    h, b = gammatone_ir(fc, 1 << 15)
    env = np.abs(hilbert(h))
    peaks.append(np.argmax(env[: 1 << 14]) / FS)
    gd.append(4 / (2 * np.pi * b))
    pk.append(3 / (2 * np.pi * b))
peaks, gd, pk = map(np.array, (peaks, gd, pk))
report("C19", "click: spread of envelope peaks over 100-5000 Hz, uncompensated [ms]", 1e3 * np.ptp(peaks))
report("C19", "  after shifting each channel by its group delay n/(2 pi b) [ms]", 1e3 * np.ptp(peaks - gd))
report(
    "C19", "  after shifting each channel by its envelope peak (n-1)/(2 pi b) [ms]", 1e3 * np.ptp(peaks - pk)
)
report("C19", "  largest |measured peak - (n-1)/(2 pi b)| [ms]", 1e3 * np.abs(peaks - pk).max())
# the group delay is the envelope's centroid: int t a(t) / int a(t) for a = t^3 e^{-ct}
cc = 2 * np.pi * 1.019 * erb(1000.0)
tt = np.arange(1 << 15) / FS / 4
a = tt**3 * np.exp(-cc * tt)
report("C19", "1 kHz: envelope centroid / (n/(2 pi b)) (expect 1)", (tt @ a / a.sum()) / (4 / cc))


# ------------------------- C20: a pitch-adaptive schedule is a well-conditioned frame
def schedule(f0_of_t, k, overlap, t_end, unvoiced_dur):
    """Window centers and lengths [samples] stepping with hop = length/overlap;
    the window is k periods long where f0 > 0, else unvoiced_dur."""
    times, lens, tc = [], [], 0.0
    while tc <= t_end:
        f0 = f0_of_t(tc)
        d = k / f0 if f0 > 0 else unvoiced_dur
        times.append(int(round(tc * FS)))
        lens.append(int(round(d * FS)))
        tc += d / overlap
    return np.array(times), np.array(lens)


def s_of_t(times, lens, n_total):
    """Frame operator diagonal M * sum_q |w_q|^2 (step 2, C12), M = max length."""
    s = np.zeros(n_total)
    M = lens.max()
    for c0, L in zip(times, lens, strict=True):
        m = np.arange(L) - L // 2
        idx = c0 + m
        ok = (idx >= 0) & (idx < n_total)
        s[idx[ok]] += M * hann_periodic(L)[(m + L // 2)[ok]] ** 2
    return s


def contour(t):
    """A sentence-like F0 contour: voiced 0.1-0.9 s and 1.2-1.9 s, gliding
    180 -> 90 Hz and 140 -> 100 Hz; unvoiced elsewhere."""
    if 0.1 <= t < 0.9:
        return 180 - 90 * (t - 0.1) / 0.8
    if 1.2 <= t < 1.9:
        return 140 - 40 * (t - 1.2) / 0.7
    return 0.0


def bridged(t):
    """The same contour with F0 carried through the unvoiced stretches:
    log-linear between voiced neighbours, held constant before the first and
    after the last voiced time window."""
    tv = np.linspace(0, 2, 2001)
    fv = np.array([contour(u) for u in tv])
    v = fv > 0
    return float(np.exp(np.interp(t, tv[v], np.log(fv[v]))))


for name, f0_fn, k, ov, uv in (
    ("gaps 20 ms", contour, 3.0, 4, 0.020),
    ("gaps 10 ms", contour, 3.0, 4, 0.010),
    ("gaps 20 ms", contour, 3.0, 3, 0.020),
    ("gaps 20 ms", contour, 4.0, 4, 0.020),
    ("bridged", bridged, 3.0, 4, 0.0),
    ("bridged", bridged, 4.0, 4, 0.0),
):
    times, lens = schedule(f0_fn, k, ov, 2.0, uv)
    n_total = int(2.0 * FS)
    s = s_of_t(times, lens, n_total)
    inner = s[int(0.05 * FS) : int(1.95 * FS)]
    report(
        "C20",
        f"{name}, k {k:g}, overlap {ov}: A/B ({len(times)} time windows, n_fft {lens.max()})",
        inner.min() / inner.max(),
    )
times, lens = schedule(lambda t: 125.0, 3.0, 4, 2.0, 0.02)
s = s_of_t(times, lens, int(2.0 * FS))[800:-800]
report("C20", "constant F0 125 Hz, k 3, overlap 4: A/B (Hann^2 at hop L/4 is constant)", s.min() / s.max())
