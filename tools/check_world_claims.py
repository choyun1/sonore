"""Numerical checks for the claims in docs/design/world.md (C1-C5).

Like the other claim checkers, this is independent of sonore: only NumPy and
SciPy, with every step written out from its formula. It holds a small
prototype of the two estimators the design describes (a CheapTrick-style
envelope and a harmonic-residual aperiodicity) so
the numbers in the document can be reproduced; it is not the library code.
The CheapTrick prototype follows WORLD's C++ source (cheaptrick.cpp and
common.cpp in github.com/mmorise/World). Each line prints the claim number
and the number that supports it.

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
TINY = 1e-12  # keeps empty bins finite in the log, like WORLD's added noise


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
def boxcar(p, width, fs=FS):
    """Mean of p over [f - width/2, f + width/2], p piecewise constant per bin,
    mirrored at 0 Hz and Nyquist (WORLD's LinearSmoothing)."""
    df = fs / (2 * (len(p) - 1))
    b = int(width / df) + 1
    ext = np.concatenate([p[b:0:-1], p, p[-2 : -2 - b : -1]])
    edges = (np.arange(len(ext) + 1) - b - 0.5) * df
    cum = np.concatenate([[0.0], np.cumsum(ext) * df])
    f = np.arange(len(p)) * df
    return (np.interp(f + width / 2, edges, cum) - np.interp(f - width / 2, edges, cum)) / width


def fold_below_f0(p, f0, fs=FS):
    """Add the mirror image about f0 / 2 below f0 (WORLD's DCCorrection)."""
    df = fs / (2 * (len(p) - 1))
    f = np.arange(len(p)) * df
    low = f < f0 + df
    out = p.copy()
    out[low] += np.interp(f0 - f[low], f, p)
    return out


def windowed(x, t, f0, periods=3.0, fs=FS):
    """x under a Hann window `periods` F0 periods long, centred at t, scaled
    to unit energy, minus its weighted mean (as CheapTrick)."""
    h = int(np.round(periods / 2 * fs / f0))
    i = np.arange(-h, h + 1)
    w = 0.5 + 0.5 * np.cos(np.pi * i / (periods / 2 * fs) * f0)
    w /= np.sqrt(np.sum(w**2))
    seg = x[np.clip(int(np.round(t * fs)) + i, 0, len(x) - 1)] * w
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
    """Per band: residual power over total power, summed over frames. The
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
    """Each band's true aperiodicity, averaged over the frames: noise power
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

    # C3: the recovery lifter's share (q1 = 0 against WORLD's -0.15)
    f0 = 200.0
    f = contour("steady", n, f0)
    x = vowel(f, lambda q: np.full_like(q, -200.0), rng, noise=False)
    k = np.arange(1, int(4000 / f0) + 1)
    bins = np.round(k * f0 / FS * N_FFT).astype(int)
    true_db = 10 * np.log10(envelope_amp(k * f0) ** 2 * peak_level(f0))
    for q1 in (0.0, -0.15):
        e = 10 * np.log10(cheaptrick(x, 0.3, f0, q1)[bins]) - true_db
        f1 = np.argmin(np.abs(k * f0 - 730))
        report("C3", f"q1 = {q1:+.2f}: level at the harmonic nearest F1, re its peak [dB]", e[f1])

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
