"""Numerical checks for the claims in docs/design/world.md (C1-C5).

Like the other claim checkers, this is independent of sonore: only NumPy and
SciPy, with every step written out from its formula. It holds ports of
WORLD's three steps, written from its C++ source (cheaptrick.cpp, d4c.cpp,
synthesis.cpp, common.cpp and matlabfunctions.cpp in github.com/mmorise/World,
commit d625e76), and a prototype of the harmonic-residual aperiodicity the
design proposes beside D4C, so the numbers in the document can be
reproduced; none of it is the library code.
tools/crosscheck_world_vocoder.py compares the ports with WORLD itself
(claim C10). Each line prints the claim number and the number that
supports it.

    python tools/check_world_claims.py

It runs in about half a minute (the aperiodicity fits are dense least
squares). The numbers come from NumPy 2.4.6 and SciPy 1.17.1.
"""

import numpy as np
from scipy.signal import freqz

FS = 16000.0
DUR = 1.0
HOP = 0.005
N_FFT = 1024  # WORLD's CheapTrick size at 16 kHz with its 71 Hz floor
FORMANTS = [(730, 60), (1090, 100), (2440, 120), (3400, 175), (4500, 250)]
F_MAX = 0.45 * FS
TINY = np.finfo(float).eps  # keeps empty bins finite in the log, like WORLD's added noise


def report(claim, text, value):
    print(f"{claim:4s} {text:<86s} {value:.4g}")


# ------------------------------------------------------------- test signals
def klatt_gain(f, freq, bw, fs=FS):
    """|H| of Klatt's resonator y = A x + B y1 + C y2 (unit gain at 0 Hz)."""
    T = 1.0 / fs
    c = -np.exp(-2 * np.pi * bw * T)
    b = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * freq * T)
    return np.abs(freqz([1 - b - c], [1, -b, -c], worN=np.atleast_1d(f), fs=fs)[1])


def envelope_amp(f):
    """Amplitude envelope of the test vowel: Klatt's glottal low-pass (RGP,
    F 0, BW 100 Hz), five /a/ formants, and the radiation difference."""
    f = np.asarray(f, float)
    g = klatt_gain(f, 0.0, 100.0)
    for fr, bw in FORMANTS:
        g = g * klatt_gain(f, fr, bw)
    return g * np.abs(2 * np.sin(np.pi * f / FS))


def contour(kind, n, f0=120.0):
    t = np.arange(n) / FS
    if kind == "steady":
        return np.full(n, f0)
    if kind == "vibrato":  # 5.5 Hz, +-3%
        return f0 * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * t))
    raise ValueError(kind)


def phase_of(f):
    """Running phase by the trapezoid rule (as so.harmonic_complex)."""
    return 2 * np.pi * np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2)]) / FS


def vowel(f, ap_db, rng, noise=True):
    """A vowel whose aperiodicity is known exactly at every frequency.

    Harmonic k has amplitude E(k f) sqrt(1 - A(k f)); the noise has power
    density A(f) E(f)^2 / (2 F0), so harmonic plus noise density is
    E^2 / (2 F0) everywhere and A(f) is, by construction, the share of the
    power at f that is not periodic. ap_db(f) gives A in dB.
    """
    n = len(f)
    phi = phase_of(f)
    x = np.zeros(n)
    for k in range(1, int(F_MAX / f.min()) + 1):
        fk = k * f
        a = envelope_amp(fk) * np.sqrt(1 - 10 ** (ap_db(fk) / 10))
        x += np.where(fk < F_MAX, a, 0.0) * np.cos(k * phi)
    if noise:
        fr = np.fft.rfftfreq(n, 1 / FS)
        g = envelope_amp(fr) * np.sqrt(10 ** (ap_db(fr) / 10) * FS / (4 * f.mean()))
        g[fr >= F_MAX] = 0
        x += np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n)
    return x


# ------------------------------------------------- CheapTrick-style envelope
def mround(v):
    """MATLAB's round (halves away from zero), as WORLD uses."""
    return int(v + 0.5) if v > 0 else int(v - 0.5)


def interp1q(x0, dx, y, xi):
    """WORLD's interp1Q: y sampled at x0 + k dx, read linearly at xi (the
    index truncated toward zero, the last step held flat)."""
    pos = (np.asarray(xi, float) - x0) / dx
    base = pos.astype(int)
    dy = np.append(np.diff(y), 0.0)
    return y[base] + dy[base] * (pos - base)


def boxcar(p, width, fs=FS):
    """Mean of p over [f - width/2, f + width/2], p piecewise constant per bin,
    mirrored at 0 Hz and Nyquist (WORLD's LinearSmoothing, step for step)."""
    half = len(p) - 1
    df = fs / (2 * half)
    b = int(width / df) + 1
    mirror = np.concatenate([p[b:0:-1], p[:half], p[half::-1][: b + 1]])
    segment = np.cumsum(mirror * df)
    f = np.arange(half + 1) * df - width / 2
    origin = -(b - 0.5) * df
    return (interp1q(origin, df, segment, f + width) - interp1q(origin, df, segment, f)) / width


def fold_below_f0(p, f0, fs=FS):
    """Add the mirror image about f0 / 2 below f0 (WORLD's DCCorrection)."""
    df = fs / (2 * (len(p) - 1))
    upper = 2 + int(f0 / df)
    f = np.arange(upper - 1) * df
    out = p.copy()
    out[: upper - 1] += interp1q(f0, -df, p[: upper + 1], f)
    return out


def windowed(x, t, f0, periods=3.0, fs=FS):
    """x under a Hann window `periods` F0 periods long, centred at t, scaled
    to unit energy, minus its weighted mean (as CheapTrick)."""
    h = mround(periods / 2 * fs / f0)
    i = np.arange(-h, h + 1)
    w = 0.5 + 0.5 * np.cos(np.pi * i / (periods / 2 * fs) * f0)
    w /= np.sqrt(np.sum(w**2))
    seg = x[np.clip(mround(t * fs + 0.001) + i, 0, len(x) - 1)] * w
    return seg - w * seg.sum() / w.sum(), w


def power_spectrum(x, t, f0, n_fft=N_FFT):
    seg, _ = windowed(x, t, f0)
    return np.abs(np.fft.rfft(seg, n_fft)) ** 2


def lifter_smooth(p, f0, q1, n_fft=N_FFT, fs=FS):
    """exp of the log spectrum liftered by sinc(f0 q) times CheapTrick's
    recovery lifter (1 - 2 q1) + 2 q1 cos(2 pi f0 q); q1 = 0 is no recovery."""
    c = np.fft.irfft(np.log(p + TINY), n_fft)
    q = np.minimum(np.arange(n_fft), n_fft - np.arange(n_fft)) / fs
    lift = np.sinc(f0 * q) * ((1 - 2 * q1) + 2 * q1 * np.cos(2 * np.pi * f0 * q))
    return np.exp(np.fft.rfft(c * lift).real)


def cheaptrick(x, t, f0, q1=-0.15):
    p = fold_below_f0(power_spectrum(x, t, f0), f0)
    p = boxcar(p, 2 * f0 / 3)
    return lifter_smooth(p, f0, q1)


def plain_lifter(x, t, f0):
    """The cepstrum's envelope: all quefrencies below half a period kept."""
    c = np.fft.irfft(np.log(power_spectrum(x, t, f0) + TINY), N_FFT)
    n = np.arange(N_FFT)
    c[(n >= 0.5 * FS / f0) & (n <= N_FFT - 0.5 * FS / f0)] = 0
    return np.exp(np.fft.rfft(c).real)


def peak_level(f0):
    """Power at a harmonic peak of the unit-energy Hann-windowed spectrum, per
    unit harmonic amplitude: (a/2)^2 (sum w)^2 / sum w^2."""
    _, w = windowed(np.zeros(10), 0.0, f0)
    return 0.25 * w.sum() ** 2


def rising(f):
    """A test aperiodicity [dB]: -30 dB at 0 Hz rising linearly to -5 dB at 8 kHz."""
    return -30 + 25 * np.asarray(f) / 8000


# ------------------------------------------------------ WORLD's D4C, ported
# A step-for-step port of d4c.cpp (WORLD commit d625e76), including the tiny
# safety noise WORLD adds to each windowed segment, from WORLD's own
# generator, drawn in the same order.
D4C_FLOOR_F0 = 47.0
D4C_BAND = 3000.0
D4C_UPPER = 15000.0
D4C_THRESHOLD = 0.85


class WorldRandn:
    """WORLD's randn: a sum of 12 xorshift draws (matlabfunctions.cpp)."""

    def __init__(self):
        self.state = [123456789, 362436069, 521288629, 88675123]

    def _step(self):
        x, y, z, w = self.state
        t = (x ^ (x << 11)) & 0xFFFFFFFF
        new = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        self.state = [y, z, w, new]
        return new

    def draw(self, n):
        out = np.empty(n)
        for k in range(n):
            total = 0
            for _ in range(12):
                total += self._step() >> 4
            out[k] = total / 268435456.0 - 6.0
        return out


def d4c_segment(x, t, f0, window, ratio, n_fft, rng, fs=FS):
    """x under an F0-adaptive Hann or Blackman window `ratio` periods long,
    minus its weighted mean, at the start of an n_fft buffer."""
    h = mround(ratio * fs / f0 / 2)
    i = np.arange(-h, h + 1)
    pos = 2 * i / ratio / fs
    if window == "hann":
        w = 0.5 * np.cos(np.pi * pos * f0) + 0.5
    else:
        w = 0.42 + 0.5 * np.cos(np.pi * pos * f0) + 0.08 * np.cos(2 * np.pi * pos * f0)
    seg = x[np.clip(mround(t * fs + 0.001) + i, 0, len(x) - 1)] * w + rng.draw(len(i)) * 1e-6
    buf = np.zeros(n_fft)
    buf[: 2 * h + 1] = seg - w * seg.sum() / w.sum()
    return buf


def d4c_centroid(x, t, f0, n_fft, rng):
    buf = d4c_segment(x, t, f0, "blackman", 4.0, n_fft, rng)
    buf /= np.sqrt(np.sum(buf[: mround(2 * FS / f0) * 2 + 1] ** 2))
    spec = np.fft.rfft(buf)
    spec2 = np.fft.rfft(buf * (np.arange(n_fft) + 1.0))
    return spec2.real * spec.real + spec.imag * spec2.imag


def d4c_band(x, t, f0, n_fft, nuttall, rng):
    """D4C's coarse aperiodicities [dB] at every multiple of 3 kHz."""
    first = d4c_centroid(x, t - 0.25 / f0, f0, n_fft, rng)
    centroid = fold_below_f0(first + d4c_centroid(x, t + 0.25 / f0, f0, n_fft, rng), f0)
    power = np.abs(np.fft.rfft(d4c_segment(x, t, f0, "hann", 4.0, n_fft, rng))) ** 2
    power = boxcar(fold_below_f0(power, f0), f0)
    delay = boxcar(centroid / power, f0 / 2)
    delay = delay - boxcar(delay, f0)
    n_bands = int(min(D4C_UPPER, FS / 2 - D4C_BAND) / D4C_BAND)
    wl = len(nuttall)
    boundary = mround(n_fft * 8.0 / wl)
    out = []
    for b in range(n_bands):
        centre = int(D4C_BAND * (b + 1) * n_fft / FS)
        buf = np.zeros(n_fft)
        buf[:wl] = delay[centre - wl // 2 : centre - wl // 2 + wl] * nuttall
        cum = np.cumsum(np.sort(np.abs(np.fft.rfft(buf)) ** 2))
        out.append(10 * np.log10(cum[n_fft // 2 - boundary - 1] / cum[n_fft // 2]))
    return np.minimum(0.0, np.array(out) + (f0 - 100) / 50)


def d4c_love_train(x, t, f0, rng):
    """The share of power (100 Hz to 7.9 kHz) below 4 kHz: D4C's voicing test."""
    f0 = max(f0, 40.0)
    n_fft = 2 ** (1 + int(np.log2(3 * FS / 40.0 + 1)))
    power = np.abs(np.fft.rfft(d4c_segment(x, t, f0, "blackman", 3.0, n_fft, rng))) ** 2
    b0, b1, b2 = (int(np.ceil(f * n_fft / FS)) for f in (100.0, 4000.0, 7900.0))
    power[: b0 + 1] = 0
    cum = np.cumsum(power)
    return cum[b1] / cum[b2]


def d4c(x, times, f0s, n_fft_out=N_FFT):
    """WORLD's aperiodicity (amplitude ratio, as WORLD stores it), one row
    per time window on n_fft_out // 2 + 1 bins."""
    n_fft = 2 ** (1 + int(np.log2(4 * FS / D4C_FLOOR_F0 + 1)))
    wl = int(D4C_BAND * n_fft / FS) * 2 + 1
    k = np.arange(wl) / (wl - 1.0)
    nuttall = (
        0.355768
        - 0.487396 * np.cos(2 * np.pi * k)
        + 0.144232 * np.cos(4 * np.pi * k)
        - 0.012604 * np.cos(6 * np.pi * k)
    )
    n_bands = int(min(D4C_UPPER, FS / 2 - D4C_BAND) / D4C_BAND)
    axis = np.append(np.arange(n_bands + 1) * D4C_BAND, FS / 2)
    freqs = np.arange(n_fft_out // 2 + 1) * FS / n_fft_out
    out = np.full((len(times), len(freqs)), 1 - 1e-12)
    rng = WorldRandn()
    voiced = [
        f0 != 0 and d4c_love_train(x, t, f0, rng) > D4C_THRESHOLD for t, f0 in zip(times, f0s, strict=True)
    ]
    for j, (t, f0) in enumerate(zip(times, f0s, strict=True)):
        if not voiced[j]:
            continue
        bands = d4c_band(x, t, max(D4C_FLOOR_F0, f0), n_fft, nuttall, rng)
        coarse = np.concatenate([[-60.0], bands, [-1e-12]])
        out[j] = 10 ** (np.interp(freqs, axis, coarse) / 20)
    return out


# ------------------------------------------------ WORLD's synthesis, ported
# A step-for-step port of synthesis.cpp (WORLD commit d625e76), with WORLD's
# noise generator drawn in the same order, so the output is WORLD's.
def minimum_phase(log_amp, n_fft):
    """WORLD's GetMinimumPhaseSpectrum: half-spectrum log amplitudes in,
    minimum-phase spectrum (n_fft // 2 + 1 bins) out."""
    full = np.concatenate([log_amp, log_amp[-2:0:-1]])
    c = np.fft.rfft(full)
    folded = np.zeros(n_fft, complex)
    folded[0] = np.conj(c[0])
    folded[1 : n_fft // 2] = 2 * np.conj(c[1 : n_fft // 2])
    folded[n_fft // 2] = np.conj(c[n_fft // 2])
    return np.exp(np.fft.fft(folded)[: n_fft // 2 + 1] / n_fft)


def c2r(spec, n_fft):
    """FFTW's unnormalized complex-to-real inverse."""
    return np.fft.irfft(spec, n_fft) * n_fft


def fftshift(v):
    half = len(v) // 2
    return np.concatenate([v[half:], v[:half]])


def world_synthesize(f0, sp, ap, hop_ms, n_out):
    """WORLD's Synthesis: pulses at the F0 track's phase crossings, each the
    minimum-phase response of S (1 - A^2) plus noise through that of S A^2."""
    n_fft = 2 * (sp.shape[1] - 1)
    rng = WorldRandn()
    fp = hop_ms / 1000.0
    lowest = FS / n_fft + 1.0
    n_windows = len(f0)
    coarse_t = np.arange(n_windows + 1) * fp
    cf0 = np.where(f0 < lowest, 0.0, f0)
    cvuv = (cf0 != 0).astype(float)
    cf0 = np.append(cf0, 2 * cf0[-1] - cf0[-2])
    cvuv = np.append(cvuv, 2 * cvuv[-1] - cvuv[-2])
    t = np.arange(n_out) / FS
    f_int = np.interp(t, coarse_t, cf0)
    vuv = (np.interp(t, coarse_t, cvuv) > 0.5).astype(float)
    f_int = np.where(vuv == 0, 500.0, f_int)
    total = np.cumsum(2 * np.pi * f_int / FS)
    wrap = np.fmod(total, 2 * np.pi)
    idx = np.nonzero(np.abs(np.diff(wrap)) > np.pi)[0]
    shift = -(wrap[idx] - 2 * np.pi) / (wrap[idx + 1] - (wrap[idx] - 2 * np.pi)) / FS
    k = np.arange(n_fft // 2)
    remover = 0.5 - 0.5 * np.cos(2 * np.pi * (k + 1.0) / (1.0 + n_fft))
    remover = np.concatenate([remover, remover[::-1]])
    remover /= 2 * remover[: n_fft // 2].sum()
    bins = np.arange(n_fft // 2 + 1)
    y = np.zeros(n_out)
    for p, i0 in enumerate(idx):
        noise_size = idx[min(len(idx) - 1, p + 1)] - i0
        now = t[i0]
        lo = min(n_windows - 1, int(np.floor(now / fp)))
        hi = min(n_windows - 1, int(np.ceil(now / fp)))
        frac = now / fp - lo
        safe = np.clip(ap, 0.001, 0.999999999999)
        if lo == hi:
            env, ratio = np.abs(sp[lo]), safe[lo] ** 2
        else:
            env = (1 - frac) * np.abs(sp[lo]) + frac * np.abs(sp[hi])
            ratio = ((1 - frac) * safe[lo] + frac * safe[hi]) ** 2
        # periodic part
        if vuv[i0] <= 0.5 or ratio[0] > 0.999:
            periodic = np.zeros(n_fft)
        else:
            spec = minimum_phase(np.log(env * (1 - ratio) + 1e-12) / 2, n_fft)
            re2 = np.cos(2 * np.pi * shift[p] * FS / n_fft * bins)
            im2 = np.sqrt(1 - re2**2)
            spec = (spec.real * re2 + spec.imag * im2) + 1j * (spec.imag * re2 - spec.real * im2)
            periodic = fftshift(c2r(spec, n_fft))
            dc = periodic[n_fft // 2 :].sum()
            # WORLD overwrites the first half (negative times) here rather
            # than subtracting from it
            periodic = np.concatenate([np.zeros(n_fft // 2), periodic[n_fft // 2 :]]) - dc * remover
        # aperiodic part
        noise = rng.draw(noise_size)
        buf = np.zeros(n_fft)
        if noise_size:
            buf[:noise_size] = noise - noise.mean()
        log_amp = np.log(env * ratio) / 2 if vuv[i0] != 0 else np.log(env) / 2
        aperiodic = fftshift(c2r(minimum_phase(log_amp, n_fft) * np.fft.rfft(buf), n_fft))
        response = (periodic * np.sqrt(noise_size) + aperiodic) / n_fft
        offset = i0 - n_fft // 2 + 1
        a, b = max(0, -offset), min(n_fft, n_out - offset)
        y[offset + a : offset + b] += response[a:b]
    return y


# --------------------------------------------- harmonic-residual aperiodicity
def harmonic_fit(x, t, f, phi, periods, linear=True, n_fft=1024):
    """Weighted least-squares fit of phase-locked harmonics k Phi(t) (and, if
    `linear`, their linear amplitude changes) over a Hann window `periods`
    long at t. Returns the power spectra of the windowed signal and of the
    windowed residual, and, per frequency, the share of white noise the
    residual keeps (the fit also absorbs some noise, near every harmonic)."""
    c = int(np.round(t * FS))
    h = int(np.round(periods / 2 * FS / f[c]))
    i = np.arange(-h, h + 1)
    w = 0.5 + 0.5 * np.cos(np.pi * i / (h + 1))
    k = np.arange(1, int(F_MAX / f[c]) + 1)
    kp = np.outer(phi[c + i], k)
    cols = [np.ones(len(i)), np.cos(kp), np.sin(kp)]
    if linear:
        tau = (i / h)[:, None]
        cols += [tau * np.cos(kp), tau * np.sin(kp)]
    s = np.sqrt(w)
    Q, _ = np.linalg.qr(np.column_stack(cols) * s[:, None])
    xs = x[c + i] * s
    r = (xs - Q @ (Q.T @ xs)) * s  # w times the residual
    # white noise e leaves w r = (W - S H S) e, with S = sqrt(W), H = Q Q'
    A = np.diag(w) - (s[:, None] * Q) @ (Q.T * s[None, :])
    kept = np.sum(np.abs(np.fft.rfft(A, n_fft, axis=0)) ** 2, axis=1) / np.sum(w**2)
    X = np.abs(np.fft.rfft(w * x[c + i], n_fft)) ** 2
    R = np.abs(np.fft.rfft(r, n_fft)) ** 2
    return X, R, kept


def band_aperiodicity(x, f, phi, times, bands, periods=4.0, linear=True, n_fft=1024):
    """Per band: residual power over total power, summed over time windows. The
    fit absorbs some noise near every harmonic, so the residual is divided
    by the share of white noise it keeps, cell by cell (cells two harmonic
    spacings wide, over which the noise spectrum is close to flat)."""
    fr = np.fft.rfftfreq(n_fft, 1 / FS)
    num = np.zeros(len(bands))
    den = np.zeros(len(bands))
    for t in times:
        X, R, kept = harmonic_fit(x, t, f, phi, periods, linear, n_fft)
        cell = (fr // (2 * f[int(np.round(t * FS))])).astype(int)
        r_cell = np.bincount(cell, R) / np.bincount(cell, kept)
        Rc = r_cell[cell]  # noise density estimate, per bin
        for b, (lo, hi) in enumerate(bands):
            m = (fr >= lo) & (fr < hi)
            num[b] += Rc[m].sum()
            den[b] += X[m].sum()
    return 10 * np.log10(num / den)


def true_band(ap_db, bands, f, times):
    """Each band's true aperiodicity, averaged over the time windows: noise power
    (the integral of A E^2 / (2 F0)) over noise plus harmonic power (the sum
    of a_k^2 / 2 over the harmonics in the band). The harmonics sample E at
    k F0, so near a narrow formant the band's harmonic power is not the
    integral of the density; the truth is computed from the harmonics."""
    fr = np.linspace(0, FS / 2, 16001)
    df = fr[1]
    noise_d = 10 ** (ap_db(fr) / 10) * envelope_amp(fr) ** 2
    out = np.zeros((len(times), len(bands)))
    for j, t in enumerate(times):
        f0 = f[int(np.round(t * FS))]
        fk = f0 * np.arange(1, int(F_MAX / f0) + 1)
        pk = envelope_amp(fk) ** 2 * (1 - 10 ** (ap_db(fk) / 10)) / 2
        for b, (lo, hi) in enumerate(bands):
            m = (fr >= lo) & (fr < min(hi, F_MAX))
            nb = noise_d[m].sum() * df / (2 * f.mean())
            out[j, b] = nb / (nb + pk[(fk >= lo) & (fk < hi)].sum())
    return 10 * np.log10(out.mean(0))


# ------------------------------------------------------------------- claims
def main():
    rng = np.random.default_rng(1)
    n = int(DUR * FS)

    # C1: the envelope at the harmonics, three F0s, against the plain lifter
    for f0 in (100.0, 200.0, 300.0):
        f = contour("steady", n, f0)
        x = vowel(f, lambda q: np.full_like(q, -200.0), rng, noise=False)
        k = np.arange(1, int(4000 / f0) + 1)
        true_db = 10 * np.log10(envelope_amp(k * f0) ** 2 * peak_level(f0))
        bins = np.round(k * f0 / FS * N_FFT).astype(int)
        times = 0.3 + np.arange(40) / f0 / 40  # 40 positions within a period
        for name, fn in (("CheapTrick", cheaptrick), ("lifter", plain_lifter)):
            est = np.array([10 * np.log10(fn(x, t, f0)[bins]) for t in times])
            err = est - true_db
            report("C1", f"{name}, F0 {f0:.0f} Hz: mean level re harmonic peaks [dB]", err.mean())
            report(
                "C1",
                f"{name}, F0 {f0:.0f} Hz: RMS shape error, offset removed [dB]",
                np.sqrt(np.mean((err - err.mean()) ** 2)),
            )
            report(
                "C2",
                f"{name}, F0 {f0:.0f} Hz: worst spread over positions in a period [dB]",
                (est.max(0) - est.min(0)).max(),
            )

    # C3: the recovery lifter: none, the CheapTrick paper's q1, WORLD's code
    for f0 in (100.0, 200.0, 300.0):
        f = contour("steady", n, f0)
        x = vowel(f, lambda q: np.full_like(q, -200.0), rng, noise=False)
        k = np.arange(1, int(4000 / f0) + 1)
        bins = np.round(k * f0 / FS * N_FFT).astype(int)
        true_db = 10 * np.log10(envelope_amp(k * f0) ** 2 * peak_level(f0))
        times = 0.3 + np.arange(10) / f0 / 10
        for q1 in (0.0, -0.09, -0.15):
            err = np.array([10 * np.log10(cheaptrick(x, t, f0, q1)[bins]) - true_db for t in times])
            report(
                "C3",
                f"F0 {f0:.0f} Hz, q1 = {q1:+.2f}: RMS shape error, offset removed [dB]",
                np.sqrt(np.mean((err - err.mean()) ** 2)),
            )
            if f0 == 200.0:
                f1 = np.argmin(np.abs(k * f0 - 730))
                report(
                    "C3",
                    f"F0 {f0:.0f} Hz, q1 = {q1:+.2f}: F1 harmonic re its peak, less the mean offset [dB]",
                    err[:, f1].mean() - err.mean(),
                )

    # C4: aperiodicity of the test vowel by harmonic residual
    bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]
    times = np.arange(0.1, 0.9, 2 * HOP)
    cases = {
        "flat -20 dB": lambda q: np.full_like(np.asarray(q, float), -20.0),
        "flat -6 dB": lambda q: np.full_like(np.asarray(q, float), -6.0),
        "rising -30 to -5 dB": rising,
    }
    for kind in ("steady", "vibrato"):
        f = contour(kind, n)
        phi = phase_of(f)
        for label, ap in cases.items():
            x = vowel(f, ap, rng)
            est = band_aperiodicity(x, f, phi, times, bands)
            err = est - true_band(ap, bands, f, times)
            report(
                "C4", f"{kind}, {label}: worst band error, 4 bands to 7 kHz [dB]", err[np.argmax(np.abs(err))]
            )
        x = vowel(f, lambda q: np.full_like(np.asarray(q, float), -200.0), rng, noise=False)
        est = band_aperiodicity(x, f, phi, times, bands)
        report("C4", f"{kind}, no noise: highest band estimate (the floor) [dB]", est.max())

    # C5: sensitivity to F0 error
    f = contour("vibrato", n)
    x = vowel(f, rising, rng)
    truth = true_band(rising, bands, f, times)
    for rel in (0.001, 0.003, 0.01):
        est = band_aperiodicity(x, f, phase_of(f * (1 + rel)), times, bands)
        err = est - truth
        report("C5", f"F0 {rel:.1%} high: error in the 2-4 kHz band [dB]", err[2])
        report("C5", f"F0 {rel:.1%} high: error in the 4-7 kHz band [dB]", err[3])


if __name__ == "__main__":
    main()
