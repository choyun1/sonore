"""Numerical checks for the claims in docs/design/modulation-spectrogram.md (C1-C10).

Like the other design checkers, this is deliberately independent of sonore:
only NumPy, SciPy and soundfile (to read the gallery sentence), with every
filter, window and transform written out
from its formula. Each line prints the claim number and the number that
supports it.

    python tools/check_modulation_spectrogram_claims.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

FS = 16000.0  # audio rate [Hz]
FE = 1000.0  # envelope rate [Hz]
HOP = 0.010  # frame hop [s]
CYCLES = 3  # constant-Q kernels hold this many cycles of their own rate
rng = np.random.default_rng(0)
SPEECH = Path(__file__).resolve().parent.parent / "docs" / "speech"


def report(claim, text, value):
    print(f"{claim:4s} {text:<76s} {value:.4g}")


# ------------------------------------------------------------------ kernels
def hann(n_len):
    """Hann window sampled at the midpoints of n_len equal steps, so its
    centre falls between samples for even n_len and on one for odd n_len."""
    return np.sin(np.pi * (np.arange(n_len) + 0.5) / n_len) ** 2


def kernel(f_mod, n_len):
    """Complex modulation kernel: a Hann window times exp(+i 2 pi f t),
    referenced to the window's centre, normalized to sum(window) = 1."""
    w = hann(n_len)
    w = w / w.sum()
    tc = (np.arange(n_len) - (n_len - 1) / 2) / FE
    return w * np.exp(2j * np.pi * f_mod * tc), w


def mod_bands(f_lo=0.5, f_hi=64.0, per_octave=2):
    n = int(round(np.log2(f_hi / f_lo) * per_octave)) + 1
    return f_lo * 2.0 ** (np.arange(n) / per_octave)


def cq_length(f_mod, cycles=CYCLES):
    return int(round(cycles * FE / f_mod))


def response(h, freqs):
    """Frequency response of a kernel applied as a correlation over its
    support, at modulation frequencies ``freqs`` [Hz]."""
    n = np.arange(len(h))
    return np.exp(2j * np.pi * np.outer(freqs, n) / FE) @ np.conj(h)


def sliding(env, h, w, centered=True):
    """Depth and modulation power of an envelope (1-D, at FE) for one kernel,
    at every sample. Correlation with the kernel over a window that ends at
    the sample (causal) or is centred on it. Returns (y, mean): y the complex
    output, mean the windowed mean of the envelope."""
    n_len = len(h)
    pad = np.concatenate([np.zeros(n_len - 1), env])
    y = fftconvolve(pad, np.conj(h)[::-1], mode="valid")  # sum_k env[n-L+1+k] conj(h[k])
    m = fftconvolve(pad, w[::-1], mode="valid")
    if centered:
        d = (n_len - 1) // 2
        y = np.concatenate([y[d:], np.zeros(d, complex)])
        m = np.concatenate([m[d:], np.zeros(d)])
    return y, m


def depth(y, m):
    """Modulation depth: 2|y| / windowed mean. A sinusoidal AM of depth m
    at the kernel's own rate reads m."""
    return 2 * np.abs(y) / np.maximum(m, 1e-300)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-300))


# ------------------------------------------------------------ audio bands
def erb_number(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def erb_freq(e):
    return (10 ** (e / 21.4) - 1) / 0.00437


def erb_bank(n_bands, f_lo, f_hi, n):
    """Half-cosine filters equally spaced on the ERB-number scale, each
    spanning one spacing either side of its centre (the shape sonore's
    cosine banks use). Returns (cfs, responses on rfft bins)."""
    knots = np.linspace(erb_number(f_lo), erb_number(f_hi), n_bands + 2)
    e = erb_number(np.fft.rfftfreq(n, 1 / FS))
    sp = knots[1] - knots[0]
    H = np.zeros((len(e), n_bands))
    for k in range(n_bands):
        u = (e - knots[k + 1]) / (2 * sp)
        H[:, k] = np.where(np.abs(u) < 0.5, np.cos(np.pi * u), 0.0)
    return erb_freq(knots[1:-1]), H


def gammatone_bank(n_bands, f_lo, f_hi, n):
    """4th-order gammatone filters (b = 1.019 ERB) at the same centres as
    erb_bank, causal phase, unit gain at the centre: the Fourier transform of
    t**3 exp(-2 pi b t) exp(i 2 pi cf t), kept for positive frequencies.
    Returns (cfs, complex responses on rfft bins)."""
    knots = np.linspace(erb_number(f_lo), erb_number(f_hi), n_bands + 2)
    cfs = erb_freq(knots[1:-1])
    f = np.fft.rfftfreq(n, 1 / FS)[:, None]
    b = 1.019 * 24.7 * (4.37 * cfs / 1000 + 1)
    return cfs, (1 + 1j * (f - cfs) / b) ** -4


def band_envelopes(x, n_bands=24, f_lo=100.0, f_hi=7000.0, shape="cosine", power=1.0):
    """Hilbert envelope of every band, raised to ``power`` and resampled to
    FE: shape (n, n_bands). The sound is padded by 0.25 s of zeros at each
    end while filtering, so filter tails can't wrap around."""
    pad = int(0.25 * FS)
    xp = np.pad(x, pad)
    n = len(xp)
    bank = erb_bank if shape == "cosine" else gammatone_bank
    cfs, H = bank(n_bands, f_lo, f_hi, n)
    X = np.fft.rfft(xp)[:, None] * H
    full = np.zeros((n, n_bands), complex)
    full[: X.shape[0]] = X
    full[1 : (n + 1) // 2] *= 2
    if n % 2 == 0:
        full[n // 2] = X[-1]
    env = np.abs(np.fft.ifft(full, axis=0))[pad : pad + len(x)] ** power
    env = resample_poly(env, int(FE), int(FS), axis=0)
    return cfs, np.maximum(env, 0.0)


def sam_tone(fc, f_mod, m, dur, on=(0.0, None)):
    t = np.arange(int(dur * FS)) / FS
    t_on, t_off = on[0], dur if on[1] is None else on[1]
    gate = (t >= t_on) & (t < t_off)
    return (1 + m * gate * np.sin(2 * np.pi * f_mod * t)) * np.sin(2 * np.pi * fc * t), t


# ===================================================================== C1
def c1():
    for T in (0.25, 1.0):
        L = int(T * FE)
        h, _ = kernel(0.0, L)
        f = np.linspace(0, 6 / T, 60001)
        a = np.abs(response(h, f))
        a /= a[0]
        bw3 = 2 * f[np.argmax(a < 2**-0.5)]
        null = f[np.argmax(np.diff(a) > 0)]  # first minimum
        report("C1", f"Hann kernel T = {T:g} s: full -3 dB bandwidth x T", bw3 * T)
        report("C1", f"Hann kernel T = {T:g} s: first null x T", null * T)
    report("C1", f"constant-Q with {CYCLES} cycles: Q = f / bandwidth", CYCLES / 1.4382)
    for f in (0.5, 4.0, 64.0):
        report("C1", f"kernel length at {f:g} Hz [s]", cq_length(f) / FE)


# ===================================================================== C2
def c2():
    # exact: lengths that hold a whole number of cycles
    worst = 0.0
    for c in (2, 3, 4):
        L = 600
        h, _ = kernel(c * FE / L, L)
        worst = max(worst, abs(response(h, [0.0])[0]))
    report("C2", "DC gain, whole number (2, 3, 4) of cycles per kernel", worst)
    for c in (1, 1.5, 2.5):
        L = 600
        h, _ = kernel(c * FE / L, L)
        report("C2", f"DC gain [dB], {c:g} cycles per kernel", db(response(h, [0.0])[0]))
    # rounded lengths at FE for the default half-octave bands
    worst = max(abs(response(kernel(f, cq_length(f))[0], [0.0])[0]) for f in mod_bands())
    report("C2", "worst DC gain [dB], 0.5-64 Hz half-octave bands, lengths rounded", db(worst))


# ===================================================================== C3
def c3():
    m, fm, fc = 0.5, 4.0, 1000.0
    x, _ = sam_tone(fc, fm, m, 4.0, on=(1.0, 3.0))
    cfs, env = band_envelopes(x)
    t = np.arange(env.shape[0]) / FE
    fr = mod_bands()
    frames = np.arange(0, env.shape[0], int(HOP * FE))
    P = np.zeros((len(frames), len(cfs), len(fr)))
    D = np.zeros_like(P)
    for j, f in enumerate(fr):
        h, w = kernel(f, cq_length(f))
        for b in range(len(cfs)):
            y, mm = sliding(env[:, b], h, w)
            P[:, b, j] = np.abs(2 * y[frames]) ** 2
            D[:, b, j] = depth(y[frames], mm[frames])
    tf = t[frames]
    mid = (tf > 1.5) & (tf < 2.5)
    b, j = np.unravel_index(np.argmax(P[mid].mean(0)), P.shape[1:])
    report("C3", "largest modulation power: audio band centre [Hz]", cfs[b])
    report("C3", "largest modulation power: modulation band [Hz]", fr[j])
    report("C3", "depth in that cell, 1.5-2.5 s [dB] (20 log10 0.5 = -6.02)", db(D[mid, b, j].mean()))
    report("C3", "  its spread over 1.5-2.5 s [dB, max - min]", np.ptp(db(D[mid, b, j])))
    for jj in (j - 1, j + 1):
        report("C3", f"  depth in the {fr[jj]:.3g} Hz band, 1.5-2.5 s [dB]", db(D[mid, b, jj].mean()))
    # the sound starts and ends abruptly; skip half a kernel at each end
    half = cq_length(fm) / FE / 2
    quiet = ((tf > half) & (tf < 1.0 - half)) | ((tf > 3.0 + half) & (tf < 4.0 - half))
    report("C3", "  depth outside the modulated part, beyond half a kernel [dB]", db(D[quiet, b, j].max()))
    d = np.where((tf > half) & (tf < 4.0 - half), D[:, b, j], 0.0)
    on = tf[np.argmax(d > m / 2)]
    off = tf[len(d) - 1 - np.argmax(d[::-1] > m / 2)]
    report("C3", "  depth first reaches half its value at [s] (modulation on at 1.0)", on)
    report("C3", "  and last at [s] (off at 3.0)", off)
    # the band below also passes the tone, through its skirt: same depth,
    # less power
    for bb in (b - 1,):
        report("C3", f"  audio band {cfs[bb]:.0f} Hz: depth [dB]", db(D[mid, bb, j].mean()))
        report(
            "C3",
            f"  audio band {cfs[bb]:.0f} Hz: power re peak cell [dB]",
            10 * np.log10(P[mid, bb, j].mean() / P[mid, b, j].mean()),
        )


# ===================================================================== C4
def c4():
    dur, r0, r1 = 6.0, 2.0, 32.0
    t = np.arange(int(dur * FS)) / FS
    # rate r0 * (r1/r0)**(t/dur): 4 octaves in 6 s; phase is its integral
    phase = 2 * np.pi * r0 * dur / np.log(r1 / r0) * ((r1 / r0) ** (t / dur) - 1)
    x = (1 + 0.5 * np.sin(phase)) * np.sin(2 * np.pi * 1000 * t)
    cfs, env = band_envelopes(x)
    b = np.argmin(np.abs(cfs - 1000))
    e = env[:, b]
    te = np.arange(len(e)) / FE
    true = r0 * (r1 / r0) ** (te / dur)
    frames = np.arange(0, len(e), int(HOP * FE))
    # quarter-octave bands for tracking; a fixed window only searches rates
    # above its own DC lobe (2 / T)
    for label, fr, lengths in (
        ("constant-Q, 3 cycles", mod_bands(1.0, 64.0, 4), None),
        ("fixed 1 s window", mod_bands(2.0, 64.0, 4), 1000),
        ("fixed 0.25 s window", mod_bands(8.0, 64.0, 4), 250),
    ):
        lengths = [cq_length(f) for f in fr] if lengths is None else [lengths] * len(fr)
        D = np.zeros((len(frames), len(fr)))
        for j, f in enumerate(fr):
            h, w = kernel(f, lengths[j])
            y, mm = sliding(e, h, w)
            D[:, j] = depth(y[frames], mm[frames]) + 1e-12
        k = np.argmax(D, axis=1)
        k = np.clip(k, 1, len(fr) - 2)
        y0, y1, y2 = (np.log(D[np.arange(len(k)), k + s]) for s in (-1, 0, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.nan_to_num(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
        est = fr[k] * 2.0 ** (off / 4)
        err = np.abs(np.log2(est / true[frames]))
        for lo, hi in ((1.0, 2.5), (2.5, 5.0)):
            sel = (te[frames] > lo) & (te[frames] < hi) & (true[frames] > fr[1]) & (true[frames] < fr[-2])
            if not sel.any():
                continue
            report(
                "C4",
                f"{label}: rate {true[frames][sel][0]:.1f}-{true[frames][sel][-1]:.1f} Hz, "
                "median error [oct]",
                np.median(err[sel]),
            )
            report("C4", f"{label}:   max error [oct]", err[sel].max())


# ===================================================================== C5
def c5():
    """Two modulators, 4 and 5 Hz, on one envelope. Resolved when the depth
    at 4.5 Hz dips below both peaks."""
    n = int(20 * FE)
    te = np.arange(n) / FE
    e = 1 + 0.3 * np.sin(2 * np.pi * 4 * te) + 0.3 * np.sin(2 * np.pi * 5 * te + 1.0)
    f = np.linspace(3, 6, 61)
    for T in (1.0, 2.0, 3.0):
        L = int(T * FE)
        dd = []
        for fm in f:
            h, w = kernel(fm, L)
            y, mm = sliding(e, h, w)
            sel = slice(L, n - L)
            dd.append(np.mean(depth(y[sel], mm[sel])))
        dd = np.array(dd)
        peak = max(dd[np.argmin(np.abs(f - 4))], dd[np.argmin(np.abs(f - 5))])
        dip = dd[np.argmin(np.abs(f - 4.5))]
        report("C5", f"4 + 5 Hz, fixed window {T:g} s: dip at 4.5 Hz re peaks [dB]", db(dip / peak))
    bw = 4.0 / (CYCLES / 1.4382)
    report("C5", "constant-Q band at 4 Hz: -3 dB bandwidth [Hz]", bw)


# ===================================================================== C6
def c6():
    """The envelope of a band W Hz wide holds no modulation above W."""
    n = int(20 * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    for W in (50.0, 200.0):
        X = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * (np.abs(f - 1000) < W / 2)
        full = np.zeros(n, complex)
        full[: len(f)] = 2 * X
        env = np.abs(np.fft.ifft(full))
        E = np.abs(np.fft.rfft(env - env.mean())) ** 2
        for frac, label in ((1.0, "W"), (0.5, "W/2")):
            share = 100 * E[f > frac * W].sum() / E.sum()
            report("C6", f"noise band {W:g} Hz wide: envelope AC power above {label} [%]", share)
    for cf in (125.0, 500.0, 1000.0, 4000.0):
        report("C6", f"ERB at {cf:g} Hz [Hz]", 24.7 * (4.37 * cf / 1000 + 1))


# ===================================================================== C7
def c7():
    """The time average of the sliding modulation power is the static band
    power of the envelope (Parseval), for circular correlation."""
    n = 8000
    e = np.abs(rng.standard_normal(n)).cumsum()
    e = 1 + (e - e.mean()) / e.std() * 0.1 + 0.2 * np.sin(2 * np.pi * 8 * np.arange(n) / FE)
    worst = 0.0
    for f in mod_bands():
        h, _ = kernel(f, cq_length(f))
        hh = np.zeros(n, complex)
        hh[: len(h)] = h
        y = np.fft.ifft(np.fft.fft(e) * np.conj(np.fft.fft(hh)))  # circular correlation
        lhs = np.mean(np.abs(y) ** 2)
        rhs = np.sum(np.abs(np.fft.fft(e)) ** 2 * np.abs(np.fft.fft(hh)) ** 2) / n**2
        worst = max(worst, abs(lhs - rhs) / rhs)
    report("C7", "time-averaged power vs static band power, worst relative difference", worst)


# ===================================================================== C8
def c8():
    """Block processing with a carried buffer equals the offline causal result,
    which is the centred result delayed by (L - 1) / 2 samples."""
    n = int(10 * FE)
    e = 1 + 0.4 * np.sin(2 * np.pi * 3 * np.arange(n) / FE) + 0.05 * rng.standard_normal(n)
    worst_block = worst_shift = 0.0
    hop = int(HOP * FE)
    for f in mod_bands():
        L = cq_length(f)
        h, w = kernel(f, L)
        y_off, _ = sliding(e, h, w, centered=False)
        buf = np.zeros(L - 1)
        out = []
        for start in range(0, n, 64):  # 64-sample blocks
            blk = e[start : start + 64]
            ext = np.concatenate([buf, blk])
            for i in range(len(blk)):
                k = start + i
                if k % hop == 0:
                    out.append(ext[i : i + L] @ np.conj(h))
            buf = ext[-(L - 1) :] if L > 1 else np.zeros(0)
        out = np.array(out)
        worst_block = max(worst_block, np.max(np.abs(out - y_off[::hop])))
        y_c, _ = sliding(e, h, w, centered=True)
        d = (L - 1) // 2
        worst_shift = max(worst_shift, np.max(np.abs(y_off[d:] - y_c[: n - d])))
    report("C8", "block-wise (64 samples) vs offline causal, worst difference", worst_block)
    report("C8", "causal vs centred shifted by (L-1)/2, worst difference", worst_shift)
    for f in (0.5, 4.0, 64.0):
        report("C8", f"causal latency (half the kernel) at {f:g} Hz [s]", (cq_length(f) - 1) / 2 / FE)
    # per frame and band: a complex kernel (2 real multiply-adds per tap) and
    # the window for the mean (1 per tap)
    ops = 3 * sum(cq_length(f) for f in mod_bands()) / HOP
    report("C8", "multiply-adds per second per audio band, direct, 10 ms hop [M]", ops / 1e6)


# ===================================================================== C9
def c9():
    """Summed squared responses of the half-octave constant-Q bank: how far
    from flat (a flat sum is what least-squares inversion would want)."""
    fr = mod_bands()
    f = np.geomspace(1.0, 32.0, 2001)
    S = sum(np.abs(response(kernel(fk, cq_length(fk))[0], f)) ** 2 for fk in fr)
    ripple = 10 * np.log10(S.max() / S.min())
    report("C9", "half-octave bank, 1-32 Hz: ripple of summed squared response [dB]", ripple)
    fr4 = mod_bands(per_octave=4)
    S4 = sum(np.abs(response(kernel(fk, cq_length(fk))[0], f)) ** 2 for fk in fr4)
    report("C9", "quarter-octave bank, 1-32 Hz: ripple [dB]", 10 * np.log10(S4.max() / S4.min()))
    # a fixed Hann window at hop T/4 is an STFT of the envelope: overlap-add of
    # squared windows is constant, so the complex coefficients invert exactly
    L = 1000
    w = hann(L) ** 2
    acc = np.zeros(4 * L)
    for s in range(0, 3 * L, L // 4):
        acc[s : s + L] += w
    report(
        "C9",
        "Hann^2 overlap-add at hop T/4, ripple in the middle [relative]",
        np.ptp(acc[L : 2 * L]) / acc[L : 2 * L].mean(),
    )


# ==================================================================== C10
FRONT_ENDS = (
    ("cosine ERB, linear", "cosine", 1.0),
    ("gammatone, linear", "gammatone", 1.0),
    ("cosine ERB, ^0.3", "cosine", 0.3),
    ("gammatone, ^0.3", "gammatone", 0.3),
)


def analyse(env, rates):
    """y and local mean for every band and rate, centred, at 10 ms frames:
    shapes (n_frames, n_bands, n_rates). Also a mask of frames at least half
    a kernel from either end."""
    frames = np.arange(0, env.shape[0], int(HOP * FE))
    Y = np.zeros((len(frames), env.shape[1], len(rates)), complex)
    M = np.zeros(Y.shape)
    ok = np.zeros((len(frames), len(rates)), bool)
    for j, f in enumerate(rates):
        L = cq_length(f)
        h, w = kernel(f, L)
        for b in range(env.shape[1]):
            y, m = sliding(env[:, b], h, w)
            Y[:, b, j], M[:, b, j] = y[frames], m[frames]
        ok[:, j] = (frames >= L // 2) & (frames < env.shape[0] - L // 2)
    return Y, M, ok


def pooled_depth(Y, M):
    """Depth pooled over bands, weighting bands by level: (n_frames, n_rates).
    Frames whose window sees only digital silence give NaN; they lie at the
    ends, outside the cells compared."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * np.sqrt(np.sum(np.abs(Y) ** 2, axis=1)) / np.sqrt(np.sum(M**2, axis=1))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def c10():
    """How much the front end (filter shape, compression) changes the picture."""
    # the 4 Hz AM tone of C3, steady part
    x, _ = sam_tone(1000.0, 4.0, 0.5, 3.0)
    for label, shape, pw in FRONT_ENDS:
        cfs, env = band_envelopes(x, shape=shape, power=pw)
        Y, M, ok = analyse(env, [4.0])
        sel = ok[:, 0]
        P = np.mean(np.abs(2 * Y[sel, :, 0]) ** 2, axis=0)
        b = int(np.argmax(P))
        d = np.mean(depth(Y[sel, b, 0], M[sel, b, 0]))
        report("C10", f"4 Hz AM, depth 0.5, {label}: depth in the loudest band [dB]", db(d))
        report("C10", f"  {label}: bands within 10 dB of its modulation power", np.sum(P > P[b] / 10))

    # the gallery sentence
    x, fs = sf.read(SPEECH / "bdl_arctic_a0131.flac")
    assert fs == FS
    rates = mod_bands(2.0, 32.0, 2)
    res = {}
    for label, shape, pw in FRONT_ENDS:
        _, env = band_envelopes(x, shape=shape, power=pw)
        res[label] = analyse(env, rates)
    ok = res[FRONT_ENDS[0][0]][2]
    for label, _, _ in FRONT_ENDS:
        Y, M, _ = res[label]
        D = pooled_depth(Y, M)
        avg = np.array([D[ok[:, j], j].mean() for j in range(len(rates))])
        j = int(np.argmax(avg))
        report("C10", f"sentence, {label}: rate of the largest mean pooled depth [Hz]", rates[j])
        report("C10", f"  {label}: mean pooled depth at 4 Hz [dB]", db(avg[rates == 4.0][0]))

    def image(label):
        Y, M, _ = res[label]
        return db(pooled_depth(Y, M))[ok]

    def band_image(label):
        Y, M, _ = res[label]
        j = int(np.flatnonzero(rates == 4.0)[0])
        P = np.abs(2 * Y[ok[:, j], :, j]) ** 2
        return 10 * np.log10(P + 1e-30 * P.max()).ravel()

    a, g, ac, gc = (lab for lab, _, _ in FRONT_ENDS)
    report(
        "C10",
        "sentence: correlation of rate x time depth [dB], cosine vs gammatone",
        corr(image(a), image(g)),
    )
    report("C10", "sentence: same, linear vs ^0.3 (cosine)", corr(image(a), image(ac)))
    report(
        "C10",
        "sentence: band x time power at 4 Hz [dB], cosine vs gammatone",
        corr(band_image(a), band_image(g)),
    )
    Ya, Ma, _ = res[a]
    Yc, Mc, _ = res[ac]
    ratio = pooled_depth(Yc, Mc)[ok] / pooled_depth(Ya, Ma)[ok]
    report("C10", "sentence: depth with ^0.3 / depth linear, median over cells", np.median(ratio))


if __name__ == "__main__":
    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
        fn()
