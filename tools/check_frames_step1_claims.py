"""Numerical checks for the claims in docs/design/frames.md, step 1 (C1-C7).

Deliberately independent of sonore: only NumPy and SciPy, with every operator
built as an explicit dense matrix on a small signal length, so the checks
don't share code (or bugs) with the implementation they will later test.
Each check prints the claim number from the design doc and the number that
supports it. Runs in a few seconds.

    python tools/check_frames_step1_claims.py
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

RNG = np.random.default_rng(0)
N = 64  # signal length for all dense checks


def dense(analysis, n):
    """Analysis operator as a dense (n_coefs, n) matrix: columns are T(e_i)."""
    return np.stack([np.ravel(analysis(e)) for e in np.eye(n)], axis=1)


def real_lstsq(T, c, weights=None):
    """argmin over REAL x of sum_j w_j |(T x)_j - c_j|^2 (complex T, c allowed)."""
    w = np.ones(T.shape[0]) if weights is None else np.ravel(weights)
    r = np.sqrt(w)[:, None]
    A = np.vstack([(r * T).real, (r * T).imag])
    y = np.concatenate([(r[:, 0] * c).real, (r[:, 0] * c).imag])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def report(claim, text, value):
    print(f"{claim:4s} {text:<66s} {value:.3g}")


# ---------------------------------------------------------------- C1, C2, C7
T = RNG.standard_normal((3 * N, N))  # a generic frame: random tall matrix
S = T.T @ T
ev = np.linalg.eigvalsh(S)
sv = np.linalg.svd(T, compute_uv=False)
report(
    "C1",
    "extreme eig(S) vs extreme squared singular values (max abs diff)",
    max(abs(ev.min() - sv.min() ** 2), abs(ev.max() - sv.max() ** 2)) / ev.max(),
)
dual = np.linalg.solve(S, T.T)  # S^-1 T*
report("C2", "S^-1 T* vs pinv(T) (rel)", rel(dual, np.linalg.pinv(T)))
x = RNG.standard_normal(N)
report("C2", "exact reconstruction S^-1 T* T x (rel)", rel(dual @ T @ x, x))
c = RNG.standard_normal(3 * N)
report("C2", "S^-1 T* c vs least squares argmin ||Tx - c|| (rel)", rel(dual @ c, np.linalg.lstsq(T, c)[0]))
P = T @ dual
report(
    "C2",
    "T T^+ is an orthogonal projection: ||P^2 - P|| + ||P - P^T||",
    np.linalg.norm(P @ P - P) + np.linalg.norm(P - P.T),
)
ratio = max(np.linalg.norm(dual @ c) / np.linalg.norm(c) for c in RNG.standard_normal((200, 3 * N)))
report("C7", "max ||T^+ c|| / ||c|| minus 1/sqrt(A) (must be <= 0)", ratio - 1 / np.sqrt(ev.min()))


# -------------------------------------------------------- C4: filterbank, circular
def gaussian_bank(n, tight=False):
    f = np.fft.rfftfreq(n)
    H = np.exp(-(((f[:, None] - np.linspace(0, 0.5, 6)) / 0.06) ** 2))
    return H / np.sqrt((H**2).sum(1, keepdims=True)) if tight else H


def fb_analysis(H, n):
    return lambda x: np.fft.irfft(np.fft.rfft(x)[:, None] * H, n=n, axis=0)


def fb_dual(H, n):
    s = (H**2).sum(1)
    return lambda c: np.fft.irfft((np.fft.rfft(c, axis=0) * H / s[:, None]).sum(1), n=n)


H = gaussian_bank(N)
s = (H**2).sum(1)
T = dense(fb_analysis(H, N), N)
sv2 = np.linalg.svd(T, compute_uv=False) ** 2
report(
    "C4",
    "circular filterbank bounds: SVD vs (min s, max s), max rel diff",
    max(abs(sv2.min() - s.min()) / s.min(), abs(sv2.max() - s.max()) / s.max()),
)
c = fb_analysis(H, N)(x)
cm = c * (RNG.random(c.shape) > 0.5)
report(
    "C4",
    "dual filters H/s on modified coefs vs pinv (rel)",
    rel(fb_dual(H, N)(cm), np.linalg.pinv(T) @ cm.ravel()),
)

# ------------------------------------------------------ C6: filterbank with padding
p = 16
M = N + 2 * p
for tight in (False, True):
    H = gaussian_bank(M, tight)
    s = (H**2).sum(1)
    analysis = lambda x, H=H: fb_analysis(H, M)(np.pad(x, (p, p)))  # noqa: E731
    T = dense(analysis, N)
    sv2 = np.linalg.svd(T, compute_uv=False) ** 2
    c = analysis(x)
    cm = c * (RNG.random(c.shape) > 0.5)
    crop = fb_dual(H, M)(cm)[p : p + N]
    tag = "tight" if tight else "non-tight"
    if not tight:
        report("C6", f"{tag} padded bounds inside circular ones: A_pad - min s (>= 0)", sv2.min() - s.min())
        report("C6", f"{tag} padded bounds inside circular ones: max s - B_pad (>= 0)", s.max() - sv2.max())
    report(
        "C6",
        f"{tag}: crop(circular dual) exact on unmodified coefs (rel)",
        rel(fb_dual(H, M)(c)[p : p + N], x),
    )
    report(
        "C6",
        f"{tag}: crop(circular dual) vs canonical pinv, modified (rel)",
        rel(crop, np.linalg.pinv(T) @ cm.ravel()),
    )

# ------------------------------------------------------------ C3, C5, C6: Gabor
w = hann(16, sym=False) ** 1.5  # not tight at hop 5 (it would be at hop 4: sin^6 over 4 shifts)
hop = 5
for mode in ("twosided", "onesided"):
    sft = ShortTimeFFT(w, hop=hop, fs=1, fft_mode=mode)
    G = dense(sft.stft, N)
    weights = np.ones(sft.f_pts)
    if mode == "onesided":
        weights[1:] = 2.0
        if sft.mfft % 2 == 0:
            weights[-1] = 1.0  # Nyquist bin appears once
    wts = np.repeat(weights, G.shape[0] // sft.f_pts)
    S = np.real((np.conj(G) * wts[:, None]).T @ G)
    diag = np.zeros(N)
    for q in range(sft.p_min, sft.p_max(N)):
        for i, wi in enumerate(w):
            t = q * hop - sft.m_num_mid + i
            if 0 <= t < N:
                diag[t] += sft.mfft * wi**2
    ev = np.linalg.eigvalsh(S)
    report(
        "C5",
        f"{mode}: largest off-diagonal of S (frame operator diagonal in time)",
        np.abs(S - np.diag(np.diag(S))).max() / ev.max(),
    )
    report(
        "C5",
        f"{mode}: bounds eig(S) vs mfft * sum_q |w(t - q hop)|^2 (max rel)",
        max(abs(ev.min() - diag.min()) / diag.min(), abs(ev.max() - diag.max()) / diag.max()),
    )
    c = sft.stft(x)
    cm = c * (RNG.random(c.shape) > 0.5)
    istft = np.real(sft.istft(cm, k1=N))
    report(
        "C6",
        f"{mode}: istft vs canonical (weighted real LS), modified coefs (rel)",
        rel(istft, real_lstsq(G, cm.ravel(), wts)),
    )
    if mode == "onesided":
        report(
            "C3",
            "onesided: istft vs UNweighted real LS (should be large)",
            rel(istft, real_lstsq(G, cm.ravel())),
        )

# ---------------------------------------------------------- negative: coverage gap
for w_, hop_ in ((hann(16, sym=False), 16), (w, 20)):
    try:
        ShortTimeFFT(w_, hop=hop_, fs=1).stft(x)
        ShortTimeFFT(w_, hop=hop_, fs=1).istft(ShortTimeFFT(w_, hop=hop_, fs=1).stft(x), k1=N)
        print(f"NEG  hop {hop_}: SciPy did NOT refuse a non-frame")
    except ValueError as e:
        print(f"NEG  hop {hop_}: SciPy refuses: {e}")
