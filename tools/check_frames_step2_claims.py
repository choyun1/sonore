"""Numerical checks for the claims in docs/design/frames/frames.md, step 2 (C8-C14).

Like tools/check_frames_step1_claims.py, this is deliberately independent of sonore:
only NumPy and SciPy, with filter responses written out from their formulas
and operators built as explicit dense matrices on small signals, so the
checks share no code with the implementation they will later test. Each line
prints the claim number and the number that supports it.

    python tools/check_frames_step2_claims.py
"""

from math import factorial

import numpy as np
from scipy.signal import ShortTimeFFT, hilbert
from scipy.signal.windows import hann

RNG = np.random.default_rng(0)


def report(claim, text, value):
    print(f"{claim:4s} {text:<70s} {value:.4g}")


def dense(analysis, n):
    """Analysis operator as a dense (n_coefs, n) matrix: columns are T(e_i)."""
    return np.stack([np.ravel(analysis(e)) for e in np.eye(n)], axis=1)


def erb(f):
    return 24.7 * (4.37e-3 * f + 1)  # Glasberg & Moore (1990)


def erb_number(f):
    return 21.4 * np.log10(4.37e-3 * f + 1)


def erb_number_inv(e):
    return (10 ** (e / 21.4) - 1) / 4.37e-3


def gammatone(f, fc, order=4, causal=True):
    """Fourier transform of t^(n-1) exp(-2 pi b t) cos(2 pi fc t), t >= 0,
    b = 1.019 ERB(fc), normalized to unit gain at fc. Conjugate-symmetric in f
    (real impulse response). causal=False keeps only the magnitude."""
    b = 1.019 * erb(fc)
    k = factorial(order - 1) / (2 * np.pi) ** order / 2

    def H(f):
        return k * ((b + 1j * (f - fc)) ** -order + (b + 1j * (f + fc)) ** -order)

    out = H(np.asarray(f, float)) / abs(H(fc))
    return out if causal else np.abs(out)


def morlet(f, fc, cycles=6.0):
    """Frequency-domain Morlet with the standard DC correction (response 0 at
    f = 0), even in f, unit gain near fc. Width: sigma_f = fc / cycles."""
    s = fc / cycles
    f = np.abs(np.asarray(f, float))
    g = np.exp(-((f - fc) ** 2) / (2 * s**2)) - np.exp(-(f**2 + fc**2) / (2 * s**2))
    return g / (1 - np.exp(-(fc**2) / s**2))


def ringing(Hf, n, level_db=-60.0):
    """Samples until the (circular, first-half) impulse response stays below
    level_db re its peak: the same measure as Filterbank.ringing."""
    h = np.abs(np.fft.irfft(Hf, n=n))[: n // 2]
    above = np.nonzero(h > h.max() * 10 ** (level_db / 20))[0]
    return int(above[-1]) + 1


# ----------------------------------------------------------------- C8: Hilbert identity
# For a real signal and a conjugate-symmetric H, hilbert(real subband) equals
# filtering with the analytic filter H(f)(1 + sgn f), with DC and Nyquist at
# weight 1 (SciPy's convention). Checked for zero-phase and causal gammatone.
# On even grids H(fs/2) must be real for H to be conjugate-symmetric (C14).
fs = 16000.0
for n in (4001, 4000):
    x = RNG.standard_normal(n)
    f = np.fft.fftfreq(n, 1 / fs)
    step = np.where(f > 0, 2.0, 0.0)
    step[0] = 1.0
    if n % 2 == 0:
        step[n // 2] = 1.0
    X = np.fft.fft(x)
    worst = 0.0
    for causal in (False, True):
        H = gammatone(f, 1000.0, causal=causal)  # full grid; conj-symmetric
        if n % 2 == 0:
            H[n // 2] = H[n // 2].real  # the Nyquist rule of C14
        real_sb = np.fft.ifft(H * X).real
        analytic = np.fft.ifft(H * step * X)
        worst = max(worst, np.abs(hilbert(real_sb) - analytic).max() / np.abs(analytic).max())
    report("C8", f"N={n}: max |hilbert(real subband) - analytic filter| (relative)", worst)

# ------------------------------------------------ C9: analytic frame on real signals
# T_a x = ifft(H_k (1 + sgn f) X). Its real frame operator Re(T_a^H T_a) has
# eigenvalues 2 s(f) off DC/Nyquist and s(f) at DC/Nyquist, s = sum |H_k|^2.
n = 64
f = np.fft.fftfreq(n, 1.0)  # fs = 1
cfs = [0.08, 0.16, 0.3]
Hs = [morlet(f, c, cycles=3) for c in cfs]
step = np.where(f > 0, 2.0, 0.0)
step[0] = step[n // 2] = 1.0
Ta = np.vstack([dense(lambda e, H=H: np.fft.ifft(H * step * np.fft.fft(e)), n) for H in Hs])
Sa = (Ta.conj().T @ Ta).real
s = sum(np.abs(H) ** 2 for H in Hs)  # per full-grid bin
w = np.where((f == 0) | (np.abs(f) == 0.5), 1.0, 2.0)
pred = np.sort(np.concatenate([(w * s)[: n // 2 + 1], (w * s)[1 : n // 2]]))  # each +f/-f pair: 2 real dims
report(
    "C9",
    "max |eig(S_analytic) - {2s off DC/Nyq, s at DC/Nyq}| / max",
    np.abs(np.linalg.eigvalsh(Sa) - pred).max() / pred.max(),
)

# ------------------------------------------------------------ C10: gammatone formulas
# (a) The closed form is the FT of the sampled impulse response.
fs_hi, fc = 64000.0, 1000.0
b = 1.019 * erb(fc)
t = np.arange(int(0.25 * fs_hi)) / fs_hi
h = t**3 * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
Hn = np.fft.rfft(h) / fs_hi
fr = np.fft.rfftfreq(len(t), 1 / fs_hi)
band = fr < 8000
Hc = gammatone(fr, fc) * abs(np.fft.rfft(h)[np.argmin(abs(fr - fc))] / fs_hi)
report(
    "C10a",
    "gammatone closed form vs FFT of sampled IR, <8 kHz (relative)",
    np.abs(Hn - Hc)[band].max() / np.abs(Hc).max(),
)
# (b) group delay at CF ~ n/(2 pi b); envelope peak at (n-1)/(2 pi b).
for fc in (100.0, 4000.0):
    b = 1.019 * erb(fc)
    df = 0.01
    ph = np.unwrap(np.angle(gammatone(np.array([fc - df, fc + df]), fc)))
    gd = -(ph[1] - ph[0]) / (2 * np.pi * 2 * df)
    report(
        "C10b", f"fc={fc:g}: group delay at CF [ms] (4/(2 pi b) = {1e3 * 4 / (2 * np.pi * b):.3f})", 1e3 * gd
    )
    report("C10c", f"fc={fc:g}: envelope peak [ms] (3/(2 pi b))", 1e3 * 3 / (2 * np.pi * b))


# ------------------------------------------------------------ C11: edge coverage
# Bare banks leave s(f) small near DC and above the top filter. Two edge
# designs, with s_floor = min of s_bank over [cf_lo, cf_hi]:
#  - exact fill: |L|^2 = max(0, s_floor - s_bank) outside [cf_lo, cf_hi].
#    A = s_floor exactly, but the kinked response rings for hundreds of ms.
#  - raised cosine: magnitude sqrt(s_floor) with a raised-cosine transition
#    k filter spacings wide (in scale units) ending at cf_lo (lowpass) and
#    starting at cf_hi (highpass). Smooth, so it rings far less.
def taper(u):  # 1 -> 0 as u goes 0 -> 1
    return 0.5 * (1 + np.cos(np.pi * np.clip(u, 0, 1)))


fs, n = 16000.0, 16000
fr = np.fft.rfftfreq(n, 1 / fs)
banks = {
    "gammatone 1/ERB 50-7000": (
        erb_number_inv(np.arange(erb_number(50), erb_number(7000), 1.0)),
        gammatone,
        erb_number,
        erb_number_inv,
    ),
    "Morlet 6cyc 4/oct 50-7000": (
        50 * 2 ** np.arange(0, np.log2(7000 / 50), 0.25),
        morlet,
        np.log2,
        lambda e: 2.0**e,
    ),
}
for name, (cfs, fn, fwd, inv) in banks.items():
    Hk = np.array([fn(fr, c) for c in cfs])
    s_bank = np.sum(np.abs(Hk) ** 2, axis=0)
    lo, hi = cfs.min(), cfs.max()
    s_floor = s_bank[(fr >= lo) & (fr <= hi)].min()
    ring_bank = 1e3 * max(ringing(np.abs(H), n) for H in Hk) / fs
    report("C11", f"{name}: bare A/B (bank rings {ring_bank:.0f} ms)", s_bank.min() / s_bank.max())
    L = np.sqrt(np.where(fr < lo, np.clip(s_floor - s_bank, 0, None), 0))
    Hh = np.sqrt(np.where(fr > hi, np.clip(s_floor - s_bank, 0, None), 0))
    s_all = s_bank + L**2 + Hh**2
    ring = 1e3 * max(ringing(L, n), ringing(Hh, n)) / fs
    report(
        "C11",
        f"  exact fill: A/s_floor, B/B_bare = {s_all.max() / s_bank.max():.3g}; rings {ring:.0f} ms",
        s_all.min() / s_floor,
    )
    sp = np.diff(fwd(cfs)).mean()
    for k in (1, 2):
        f0, f1 = max(float(inv(fwd(lo) - k * sp)), 0.0), float(inv(fwd(hi) + k * sp))
        L = np.sqrt(s_floor) * taper((fr - f0) / (lo - f0))
        Hh = np.sqrt(s_floor) * (1 - taper((fr - hi) / (f1 - hi)))
        s_all = s_bank + L**2 + Hh**2
        ring = 1e3 * max(ringing(L, n), ringing(Hh, n)) / fs
        report(
            "C11",
            f"  raised cosine, {k} spacing(s): A/B; edges ring {ring:.0f} ms",
            s_all.min() / s_all.max(),
        )

# --------------------------------------------- C14: Nyquist bin of complex responses
# On an even grid, irfft keeps only Re of the Nyquist bin, so the filter
# actually applied there is Re H(fs/2). A dual that divides by |H(fs/2)|^2
# is then not exact; using Re H(fs/2) consistently in analysis, s and the
# dual restores exactness. Causal gammatones up to 7.5 kHz at fs = 16 kHz,
# plus the zero-phase raised-cosine edges, via the rfft/irfft path.
fs, n = 16000.0, 4000
fr = np.fft.rfftfreq(n, 1 / fs)
cfs = erb_number_inv(np.arange(erb_number(50), erb_number(7500), 1.0))
Hk = np.array([gammatone(fr, c) for c in cfs])
report(
    "C14", "max |Im H(fs/2)| / |H(fs/2)| over the bank", np.max(np.abs(Hk[:, -1].imag) / np.abs(Hk[:, -1]))
)
x = RNG.standard_normal(n)
X = np.fft.rfft(x)
for label, H in (("naive", Hk.copy()), ("Re at Nyquist", Hk.copy())):
    if label != "naive":
        H[:, -1] = H[:, -1].real
    s = np.sum(np.abs(H) ** 2, axis=0) + 1.0  # + 1: flat stand-in for the edge filters
    bands = np.fft.irfft(X * H, n=n)
    y = np.fft.irfft(np.sum(np.fft.rfft(bands) * np.conj(H), axis=0) / s, n=n)
    y += np.fft.irfft(X * 1.0 / s, n=n)  # the stand-in edge band, analyzed and dual-filtered
    report("C14", f"round trip, {label}: ||y - x|| / ||x||", np.linalg.norm(y - x) / np.linalg.norm(x))


# ----------------------------------------------- C12: time-varying Gabor, painless case
# Windows w_q of length L_q <= M at positions a_q, each FFT'd at length M (full
# complex FFT). S = T^H T is diagonal with s(t) = M sum_q |w_q(t - a_q)|^2.
# Once one window is longer than M, S is not diagonal.
def tv_gabor(n, positions, lengths, M):
    def analysis(x):
        cols = []
        for a, L in zip(positions, lengths, strict=True):
            seg = np.zeros(M, complex)
            idx = (a + np.arange(L)) % n
            seg[: min(L, M)] = (hann(L, sym=False) * x[idx])[:M]
            if L > M:  # time-aliased: fold the tail back in
                tail = hann(L, sym=False)[M:] * x[idx][M:]
                seg[: len(tail)] += tail
            cols.append(np.fft.fft(seg))
        return np.concatenate(cols)

    return analysis


n = 64
lengths = [8, 8, 12, 16, 16, 12, 8, 8, 12, 16]
positions = np.cumsum([0] + [L // 2 for L in lengths[:-1]])
positions = (positions * n // (positions[-1] + lengths[-1] // 2)).astype(int)  # spread over n
M = 16
T = dense(tv_gabor(n, positions, lengths, M), n)
S = (T.conj().T @ T).real
s_pred = np.zeros(n)
for a, L in zip(positions, lengths, strict=True):
    s_pred[(a + np.arange(L)) % n] += M * hann(L, sym=False) ** 2
off = S - np.diag(np.diag(S))
report("C12", "L_q <= M: max |diag S - s_pred| / max", np.abs(np.diag(S) - s_pred).max() / s_pred.max())
report("C12", "L_q <= M: max |off-diagonal S| / max", np.abs(off).max() / s_pred.max())
T2 = dense(tv_gabor(n, positions, [L if L != 16 else 20 for L in lengths], M), n)
S2 = (T2.conj().T @ T2).real
report(
    "C12",
    "one L_q > M: max |off-diagonal S| / max |S| (nonzero)",
    np.abs(S2 - np.diag(np.diag(S2))).max() / np.abs(S2).max(),
)

# ------------------------------------------ C13: ShortTimeFFT adjoint via dual_win=win
# istft of a ShortTimeFFT built with dual_win = win is the adjoint of stft for
# the C3-weighted inner product up to the factor mfft: T* = mfft * istft
# (weight 2 on bins strictly between DC and
# Nyquist, 1 on DC and Nyquist). Checked as <Tx, c>_w == <x, T* c>.
for m, hop, mfft in ((32, 10, 32), (32, 8, 48), (33, 10, 33)):
    win = hann(m, sym=False) ** 1.5
    sft = ShortTimeFFT(win, hop, 1.0, mfft=mfft, dual_win=win)
    n = 96
    x = RNG.standard_normal(n)
    X = sft.stft(x)
    c = RNG.standard_normal(X.shape) + 1j * RNG.standard_normal(X.shape)
    c[0].imag = 0
    if mfft % 2 == 0:
        c[-1].imag = 0  # coefficients of real signals have real DC/Nyquist
    wts = np.full(X.shape[0], 2.0)
    wts[0] = 1.0
    if mfft % 2 == 0:
        wts[-1] = 1.0
    lhs = np.sum(wts[:, None] * (X.conj() * c)).real
    y = sft.istft(c, k1=n)
    rhs = x @ y
    report("C13", f"win {m} hop {hop} mfft {mfft}: <Tx,c>_w / <x, istft(c)> (expect mfft)", lhs / rhs)
