"""Numerical checks for the claims in docs/design/klatt.md (C1-C6).

Like the other claim checkers, this is independent of sonore: only NumPy and
SciPy, with every step written out from its formula. It holds a small
prototype of the pieces of a Klatt-style formant synthesizer (Klatt, 1980)
that the design describes, so the numbers in the document can be
reproduced; it is not the library code. Each line prints the claim number
and the number that supports it.

    python tools/check_klatt_claims.py

It runs in about a second.
"""

import numpy as np
from scipy.signal import freqz, lfilter

FS = 16000.0
HOP = 0.005  # Klatt's parameter update interval [s]

# Vowel /a/ formants and bandwidths (rounded male averages, as in
# tools/check_cepstrum_claims.py), plus a fifth formant.
FORMANTS = [(730, 60), (1090, 100), (2440, 120), (3400, 175), (4500, 250)]


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


# ------------------------------------------------------------- the prototype
def resonator_coefs(f, bw, fs=FS):
    """Klatt (1980) digital resonator y[n] = A x[n] + B y[n-1] + C y[n-2]:
    C = -exp(-2 pi BW T), B = 2 exp(-pi BW T) cos(2 pi F T), A = 1 - B - C."""
    T = 1.0 / fs
    c = -np.exp(-2 * np.pi * bw * T)
    b = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * f * T)
    return 1 - b - c, b, c


def response(f, bw, freqs, fs=FS):
    a, b, c = resonator_coefs(f, bw, fs)
    return freqz([a], [1, -b, -c], worN=freqs, fs=fs)[1]


def resonate(x, f, bw, fs=FS):
    """A resonator whose F and BW are given per sample (arrays), state carried
    across every coefficient change, as in Klatt's update loop."""
    a, b, c = resonator_coefs(np.broadcast_to(f, x.shape), np.broadcast_to(bw, x.shape), fs)
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for n in range(len(x)):
        y0 = a[n] * x[n] + b[n] * y1 + c[n] * y2
        y[n], y2, y1 = y0, y1, y0
    return y


def frames_to_samples(track, n, hop=HOP, fs=FS, hold=True):
    """A parameter track given every `hop` seconds onto n samples: held for
    each 5 ms frame (Klatt's update) or interpolated linearly."""
    t = np.arange(n) / fs
    tf = np.arange(len(track)) * hop
    if hold:
        return np.asarray(track)[np.minimum((t / hop).astype(int), len(track) - 1)]
    return np.interp(t, tf, track)


def harmonic_source(f0, amps_of_freq, n, fs=FS, f_max=None):
    """Phase-locked harmonics of a per-sample F0 (trapezoid phase), harmonic k
    weighted by amps_of_freq(k f0(t)); the prototype of so.harmonic_source."""
    f_max = 0.45 * fs if f_max is None else f_max
    f = np.broadcast_to(f0, (n,)).astype(float)
    phi = 2 * np.pi * np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2)]) / fs
    x = np.zeros(n)
    for k in range(1, int(f_max / f.min()) + 1):
        fk = k * f
        x += np.where(fk < f_max, amps_of_freq(fk), 0.0) * np.cos(k * phi)
    return x


def rgp_gain(freq, fs=FS):
    """|H| of Klatt's glottal low-pass resonator RGP (F = 0, BW = 100 Hz)."""
    return np.abs(response(0.0, 100.0, np.atleast_1d(freq), fs))


# --------------------------------------------------------------------- claims
def c1_resonator():
    """The resonator: unity gain at 0 Hz, peak at F, -3 dB width BW."""
    freqs = np.linspace(0, FS / 2, 2**18)
    for f, bw in [(500, 60), (3500, 200)]:
        m = np.abs(response(f, bw, freqs))
        i = np.argmax(m)
        band = freqs[m >= m[i] / np.sqrt(2)]
        report("C1", f"F={f} BW={bw}: gain at 0 Hz", m[0])
        report("C1", f"F={f} BW={bw}: peak frequency error [Hz]", abs(freqs[i] - f))
        report("C1", f"F={f} BW={bw}: -3 dB width error [Hz]", abs(band.max() - band.min() - bw))


def c2_cascade_parallel():
    """Parallel formants match the cascade between peaks only with
    alternating signs."""
    w = np.linspace(50, 5000, 4000)
    hs = [response(f, bw, w) for f, bw in FORMANTS]
    inside = (w > 100) & (w < 4000)
    cascade = np.prod(hs, axis=0)
    for name, sign in [("same signs", lambda i: 1), ("alternating signs", lambda i: (-1) ** i)]:
        par = 0
        for i, ((f, _), h) in enumerate(zip(FORMANTS, hs, strict=True)):
            j = np.argmin(np.abs(w - f))
            par = par + sign(i) * np.abs(cascade[j]) / np.abs(h[j]) * h
        d = 20 * np.log10(np.abs(par) / np.abs(cascade))
        report(
            "C2",
            f"parallel, {name}, gains matched at peaks, vs cascade: max |dB| 100-4000 Hz",
            np.abs(d[inside]).max(),
        )


def c3_updates():
    """Holding coefficients for 5 ms frames vs interpolating them per sample,
    on a /da/-like F1 and F2 transition."""
    dur = 0.3
    n = int(dur * FS)
    nf = int(dur / HOP) + 1
    tf = np.arange(nf) * HOP
    f1 = np.interp(tf, [0, 0.04, dur], [300, 700, 700])
    f2 = np.interp(tf, [0, 0.04, dur], [1700, 1200, 1200])
    x = harmonic_source(120.0, lambda fk: rgp_gain(fk), n)
    hop = int(HOP * FS)
    for name, hold in [("held for 5 ms frames", True), ("interpolated per sample", False)]:
        y = x
        for track, bw in [(f1, 60), (f2, 100)]:
            y = resonate(y, frames_to_samples(track, n, hold=hold), bw)
        report(
            "C3",
            f"F1, F2 coefficients {name}, state carried: power above 5 kHz [dB re total]",
            _above(y, 5000),
        )
    # The same held coefficients, but each 5 ms frame filtered from zero
    # state, as if the filter were restarted at every update.
    y = x
    for track, bw in [(f1, 60), (f2, 100)]:
        held = frames_to_samples(track, n)
        z = np.zeros(n)
        for s in range(0, n, hop):
            a, b, c = resonator_coefs(held[s], bw)
            z[s : s + hop] = lfilter([a], [1, -b, -c], y[s : s + hop])
        y = z
    report(
        "C3", "the same, filter state restarted every 5 ms: power above 5 kHz [dB re total]", _above(y, 5000)
    )


def _above(y, lo, dur=0.06):
    """Share of the first `dur` seconds' power above `lo` Hz, in dB."""
    seg = y[: int(dur * FS)] * np.hanning(int(dur * FS))
    s = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / FS)
    return 10 * np.log10(s[f > lo].sum() / s.sum())


def c4_glottal():
    """RGP's slope, and the harmonic source carrying RGP's spectrum."""
    g = 20 * np.log10(rgp_gain(np.array([400.0, 3200.0])))
    report("C4", "RGP slope between 400 and 3200 Hz [dB/octave]", (g[1] - g[0]) / 3)
    # Klatt's source: an impulse every T0 samples through RGP. At a whole
    # number of samples per period its harmonics equal the harmonic source
    # weighted by |RGP|, up to RGP's phase.
    f0, n = 125.0, int(2 * FS)  # 128 samples per period
    imp = np.zeros(n)
    imp[:: int(FS / f0)] = 1.0
    klatt = lfilter(*_rgp_ba(), imp)
    spec_k = np.abs(np.fft.rfft(klatt[-int(FS) :]))  # 1 Hz bins
    spec_h = np.abs(np.fft.rfft(harmonic_source(f0, rgp_gain, n)[-int(FS) :]))
    k = (np.arange(1, 50) * f0).astype(int)
    r = 20 * np.log10(spec_k[k] / spec_k[k[0]]) - 20 * np.log10(spec_h[k] / spec_h[k[0]])
    report("C4", "impulses through RGP vs harmonic source x |RGP|: harmonics 1-49, max |dB|", np.abs(r).max())
    # At F0 = 230 Hz, 16 kHz gives 69.57 samples per period; whole-sample
    # pulses make the period 69 or 70 samples.
    p = FS / 230.0
    report(
        "C4",
        "F0 230 Hz at 16 kHz: whole-sample pulse period error, max [%]",
        100 * max(abs(np.floor(p) - p), abs(np.ceil(p) - p)) / p,
    )


def _rgp_ba():
    a, b, c = resonator_coefs(0.0, 100.0)
    return [a], [1, -b, -c]


def c5_noise():
    """Klatt's noise path: the integrator y[n] = x[n] + y[n-1] of white noise
    is -6 dB/octave; the radiation difference p[n] = u[n] - u[n-1] makes it
    flat again. A leaky integrator stands in so the level stays bounded."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2**18)
    u = lfilter([1.0], [1.0, -0.99], x)
    p = np.diff(u, prepend=0.0)
    f, sp = _psd(p)
    f, su = _psd(u)
    band = lambda s, lo, hi: 10 * np.log10(np.mean(s[(f > lo) & (f < hi)]))  # noqa: E731
    report(
        "C5",
        "integrated noise: 1-2 kHz vs 2-4 kHz octave bands [dB]",
        band(su, 1000, 2000) - band(su, 2000, 4000),
    )
    report(
        "C5",
        "after radiation: 1-2 kHz vs 2-4 kHz octave bands [dB]",
        band(sp, 1000, 2000) - band(sp, 2000, 4000),
    )
    # Klatt modulates noise by a square wave at F0, 50% depth, when voicing
    # is on: the noise envelope then carries the period.
    f0 = 125.0
    t = np.arange(len(x)) / FS
    m = 1 - 0.5 * (np.mod(t * f0, 1) >= 0.5)
    env = np.abs(np.fft.rfft((x * m) ** 2))
    fe = np.fft.rfftfreq(len(x), 1 / FS)
    k0 = np.argmin(np.abs(fe - f0))
    report(
        "C5",
        "50% F0-synchronous modulation: envelope line at F0 vs neighbours [dB]",
        20 * np.log10(env[k0] / np.median(env[k0 + 5 : k0 + 200])),
    )


def _psd(x):
    seg = x[: len(x) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024)
    return np.fft.rfftfreq(1024, 1 / FS), np.mean(np.abs(np.fft.rfft(seg, axis=1)) ** 2, axis=0)


def c6_vowel():
    """Cascade /a/ from the harmonic source: harmonic levels follow the
    all-pole envelope exactly, so the source and filter separate."""
    f0, n = 100.0, int(FS)
    x = harmonic_source(f0, rgp_gain, n)
    y = x
    for f, bw in FORMANTS:
        y = resonate(y, np.full(n, float(f)), bw)
    y = np.diff(y, prepend=0.0)  # radiation
    spec = np.abs(np.fft.rfft(y[-int(FS / 2) :]))
    k = np.arange(1, 40)
    bins = (k * f0 * 0.5).astype(int)
    pred = (
        rgp_gain(k * f0)
        * np.abs(np.prod([response(f, bw, k * f0) for f, bw in FORMANTS], axis=0))
        * np.abs(1 - np.exp(-2j * np.pi * k * f0 / FS))
    )
    r = 20 * np.log10(spec[bins] / spec[bins[0]]) - 20 * np.log10(pred / pred[0])
    report("C6", "/a/ harmonics 1-39 vs source x formants x radiation, max |dB|", np.abs(r).max())


if __name__ == "__main__":
    c1_resonator()
    c2_cascade_parallel()
    c3_updates()
    c4_glottal()
    c5_noise()
    c6_vowel()
