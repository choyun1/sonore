"""Numerical checks for the claims in docs/design/f0.md (C1-C8).

Like the other claim checkers, this is independent of sonore: only NumPy,
SciPy and soundfile (to read the gallery sentence), with every window,
filter and estimator written out from its formula. It contains a small
prototype of the tracker the design describes, so the numbers in the
document can be reproduced; it is not the library code. Each line prints
the claim number and the number that supports it.

    python tools/check_f0_claims.py

It takes about a minute. WORLD's own estimators are not used here (see
tools/crosscheck_f0_world.py for those).
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfiltfilt

FS = 16000.0
HOP = 0.005  # frame period [s], Harvest's default
F_LO, F_HI = 60.0, 500.0  # search range [Hz]
ROOT = Path(__file__).resolve().parent.parent
SPEECH = ROOT / "docs" / "speech"


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


# ------------------------------------------------------------ test signals

# Vowel /a/ formants (Peterson & Barney male averages, rounded) and bandwidths,
# as in check_cepstrum_claims.py.
FORMANTS = [(730, 60), (1090, 100), (2440, 120), (3400, 175)]


def vowel_gain(freqs):
    z = np.exp(-2j * np.pi * np.asarray(freqs) / FS)
    a = np.array([1.0])
    for f, bw in FORMANTS:
        r = np.exp(-np.pi * bw / FS)
        a = np.convolve(a, [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r])
    return np.abs(1 / np.polyval(a[::-1], z))


def vowel(f0):
    """Harmonics of a per-sample F0 contour below 0.95 Nyquist, each with the
    /a/ filter's gain at its instantaneous frequency. Unit RMS."""
    phase = 2 * np.pi * np.cumsum(f0) / FS
    y = np.zeros(len(f0))
    for k in range(1, int(0.95 * FS / 2 / f0.min()) + 1):
        fk = k * f0
        y += vowel_gain(fk) * (fk < 0.95 * FS / 2) * np.cos(k * phase)
    return y / np.sqrt(np.mean(y**2))


N1 = int(FS)  # one second
TT = np.arange(N1) / FS
CONTOURS = {
    "steady 80 Hz": np.full(N1, 80.0),
    "steady 120 Hz": np.full(N1, 120.0),
    "steady 200 Hz": np.full(N1, 200.0),
    "steady 350 Hz": np.full(N1, 350.0),
    "glide 100-200 Hz, 1 oct/s": 100 * 2**TT,
    "vibrato 150 Hz, +-6%, 5.5 Hz": 150 * (1 + 0.06 * np.sin(2 * np.pi * 5.5 * TT)),
}
VIBRATO = CONTOURS["vibrato 150 Hz, +-6%, 5.5 Hz"]
rng = np.random.default_rng(1)
NOISE = rng.standard_normal(N1)  # one draw, reused at every SNR


def with_noise(x, snr_db):
    return x + NOISE[: len(x)] * np.sqrt(np.mean(x**2)) * 10 ** (-snr_db / 20)


HP300 = butter(8, 300, "highpass", fs=FS, output="sos")


def frame_times(n):
    return np.arange(0, n / FS, HOP)


# -------------------------------------------------- candidates: filter bank


def nuttall(n):
    a = [0.355768, 0.487396, 0.144232, 0.012604]
    x = 2 * np.pi * np.arange(n) / (n - 1)
    return a[0] - a[1] * np.cos(x) + a[2] * np.cos(2 * x) - a[3] * np.cos(3 * x)


def crossings(y, rising):
    """Sub-sample positions where y crosses zero upward (or downward)."""
    i = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0) if rising else (y[:-1] > 0) & (y[1:] <= 0))
    return i + y[i] / (y[i] - y[i + 1])


def interval_rate(pos, t):
    """1 / interval between consecutive events, placed at each interval's
    middle and interpolated to the times t. NaN outside the events."""
    if len(pos) < 3:
        return np.full(len(t), np.nan)
    p = pos / FS
    return np.interp(t, 0.5 * (p[1:] + p[:-1]), 1 / np.diff(p), left=np.nan, right=np.nan)


def bank_candidates(x, t, per_oct=24, tol=0.1):
    """Fundamental-component candidates. Each channel is a Nuttall-windowed
    cosine at fc spanning 4 periods; its output's four event series (upward
    and downward zero crossings, peaks, dips) each give a rate, and their
    mean is a candidate where it lies within tol of fc. Candidates within 3%
    of each other are merged. Returns one array of candidates per frame."""
    n_ch = int(np.ceil(np.log2(F_HI / F_LO) * per_oct)) + 1
    rates = np.full((n_ch, len(t)), np.nan)
    for i, fc in enumerate(F_LO * 2 ** (np.arange(n_ch) / per_oct)):
        half = int(round(2 * FS / fc))
        n = np.arange(-half, half + 1)
        y = fftconvolve(x, nuttall(len(n)) * np.cos(2 * np.pi * fc * n / FS), mode="same")
        dy = np.diff(y)
        four = np.array(
            [
                interval_rate(crossings(y, True), t),
                interval_rate(crossings(y, False), t),
                interval_rate(crossings(dy, True) + 0.5, t),
                interval_rate(crossings(dy, False) + 0.5, t),
            ]
        )
        m = four.mean(0)
        ok = np.abs(m / fc - 1) < tol
        rates[i, ok] = m[ok]
    out = []
    for col in rates.T:
        c = np.sort(col[np.isfinite(col)])
        groups = np.split(c, np.flatnonzero(c[1:] / c[:-1] > 1.03) + 1) if len(c) else []
        out.append(np.array([np.median(g) for g in groups]))
    return out


# ---------------------------------------------- candidates: difference function


def yin_candidates(x, t, win=0.025, max_cand=4, d_max=0.5):
    """Local minima of the cumulative-mean-normalized difference function
    (de Cheveigne & Kawahara, 2002) below d_max, refined by a parabola, best
    max_cand per frame. The integration window is win seconds."""
    W = int(win * FS)
    tau_max = int(np.ceil(FS / F_LO)) + 1
    tau_min = max(int(np.floor(FS / F_HI)), 1)
    xp = np.concatenate([np.zeros(W), x, np.zeros(W + tau_max)])
    starts = np.round(t * FS).astype(int) + W - W // 2
    seg = np.lib.stride_tricks.sliding_window_view(xp, W + tau_max)[starts]
    a = seg[:, :W]
    # d(tau) = sum a^2 + sum b_tau^2 - 2 sum a b_tau, the cross term by FFT.
    n_fft = 1 << int(np.ceil(np.log2(2 * (W + tau_max))))
    cross = np.fft.irfft(np.conj(np.fft.rfft(a, n_fft)) * np.fft.rfft(seg, n_fft), n_fft)[:, : tau_max + 1]
    c2 = np.concatenate([np.zeros((len(t), 1)), np.cumsum(seg**2, axis=1)], axis=1)
    energy_b = c2[:, W : W + tau_max + 1] - c2[:, : tau_max + 1]
    d = np.maximum(np.sum(a * a, axis=1, keepdims=True) + energy_b - 2 * cross, 0)
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * np.arange(1, tau_max + 1) / np.maximum(cm, 1e-300)
    out = []
    for row in dn:
        tau = np.arange(tau_min, tau_max)
        is_min = (row[tau] < row[tau - 1]) & (row[tau] <= row[tau + 1]) & (row[tau] < d_max)
        cands = []
        for k in tau[is_min]:
            y0, y1, y2 = row[k - 1], row[k], row[k + 1]
            den = y0 - 2 * y1 + y2
            cands.append((y1, FS / (k + (0.5 * (y0 - y2) / den if den > 0 else 0.0))))
        cands.sort()
        out.append(np.array([f for _, f in cands[:max_cand]]))
    return out


# ------------------------------------------------- refinement and scoring


def blackman(n):
    x = 2 * np.pi * np.arange(n) / (n - 1)
    return 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)


def padded(x, pad):
    return np.concatenate([np.zeros(pad), x, np.zeros(pad)])


def refine(x, tc, f, periods=3.0, K=6, iters=2):
    """Instantaneous-frequency refinement. A Blackman window `periods` periods
    of f long, centred at tc; the DTFT at the harmonics k f (k = 1..K) of the
    segment and of the segment one sample later; each harmonic's
    instantaneous frequency is its phase advance per sample. The new F0 is
    the power-weighted mean of IF_k / k. Repeated `iters` times."""
    pad = int(4 * periods * FS / F_LO)
    xp = padded(x, pad)
    center = int(round(tc * FS)) + pad
    k = np.arange(1, K + 1)
    for _ in range(iters):
        L = int(round(periods * FS / f)) | 1
        n = np.arange(L) - L // 2
        w = blackman(L)
        e = np.exp(-2j * np.pi * np.outer(k, f * n / FS))
        X0 = e @ (xp[center + n] * w)
        X1 = e @ (xp[center + n + 1] * w)
        power = np.abs(X0) ** 2
        if power.sum() <= 0:
            return f
        f_new = np.sum(power * np.angle(X1 * np.conj(X0)) * FS / (2 * np.pi) / k) / power.sum()
        if not (np.isfinite(f_new) and f_new > 0):
            return f
        f = f_new
    return f


def periodicity(x, tc, f, periods=3.0):
    """Normalized correlation between a stretch `periods` periods long centred
    half a period before tc and the same stretch one period later (fractional
    shifts by linear interpolation)."""
    T = FS / f
    L = int(round(periods * T))
    pad = int(2 * periods * FS / F_LO) + 64
    xp = padded(x, pad)
    n = np.arange(L) + int(round(tc * FS - L / 2 - T / 2)) + pad
    a = xp[n]
    pos = n + T
    i = np.floor(pos).astype(int)
    b = (1 - (pos - i)) * xp[i] + (pos - i) * xp[i + 1]
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.sum(a * b) / den if den > 0 else 0.0


def scored(x, t, cands, do_refine=True):
    """Refine and score every candidate: one (f, score) pair of arrays per frame."""
    out = []
    for tc, cs in zip(t, cands, strict=True):
        f = np.array([refine(x, tc, c) if do_refine else c for c in cs])
        f = f[(f > 0.9 * F_LO) & (f < 1.1 * F_HI)]
        out.append((f, np.array([periodicity(x, tc, fi) for fi in f])))
    return out


# ------------------------------------------------------------- tracking


def viterbi(frames, theta=0.5, jump=2.0, switch=0.5):
    """Cheapest path through (unvoiced, candidates...) per frame. Local cost
    1 - score for a candidate, 1 - theta for unvoiced; moving between
    candidates costs jump * |log2 ratio|, and a voicing change costs switch."""
    prev_f, prev_cost, states, back = np.zeros(1), np.zeros(1), [], []
    for j, (f, s) in enumerate(frames):
        fj = np.concatenate([[0.0], f])
        local = np.concatenate([[1 - theta], 1 - s])
        va, vb = fj[:, None] > 0, prev_f[None, :] > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tr = np.where(va & vb, jump * np.abs(np.log2(fj[:, None] / prev_f[None, :])), 0.0)
        tr = np.where(va != vb, switch if j else 0.0, tr)
        tot = tr + prev_cost[None, :]
        bi = np.argmin(tot, axis=1)
        prev_cost = tot[np.arange(len(fj)), bi] + local
        prev_f = fj
        states.append(fj)
        back.append(bi)
    path = np.zeros(len(frames))
    k = int(np.argmin(prev_cost))
    for j in range(len(frames) - 1, -1, -1):
        path[j] = states[j][k]
        k = back[j][k]
    return path


def best_per_frame(frames, theta=0.5):
    """No tracking: the highest-scoring candidate if its score exceeds theta."""
    return np.array([f[np.argmax(s)] if len(s) and s.max() > theta else 0.0 for f, s in frames])


def track(x, gen="diff", **kw):
    t = frame_times(len(x))
    cands = yin_candidates(x, t) if gen == "diff" else bank_candidates(x, t)
    return t, viterbi(scored(x, t, cands), **kw)


# ---------------------------------------------------- cepstral baseline


def cepstral_track(x, win=0.050, threshold=0.1):
    """Classic cepstral F0 as in so.Cepstrum.f0: a Hann window three periods of
    F_LO long, the largest real-cepstrum peak in 1/F_HI..1/F_LO refined by a
    parabola, unvoiced where the peak is below threshold."""
    t = frame_times(len(x))
    L = int(win * FS)
    n_fft = 1 << int(np.ceil(np.log2(2 * L)))
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
    xp = padded(x, L)
    q_lo, q_hi = int(np.floor(FS / F_HI)), int(np.ceil(FS / F_LO))
    est = np.zeros(len(t))
    for j, tc in enumerate(t):
        s = int(round(tc * FS)) - L // 2 + L
        mag = np.abs(np.fft.rfft(xp[s : s + L] * w, n_fft))
        if mag.max() == 0:
            continue
        c = np.fft.irfft(np.log(np.maximum(mag, 1e-10 * mag.max())), n_fft)
        k = q_lo + int(np.argmax(c[q_lo : q_hi + 1]))
        if c[k] < threshold:
            continue
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        est[j] = FS / (k + (0.5 * (y0 - y2) / den if den < 0 else 0.0))
    return t, est


# ------------------------------------------------------------- scoring


def errors(t, est, f0):
    """Gross error rate (unvoiced, or off by more than 20%, as in Morise 2017)
    and the median and largest fine error [%] over the other frames, on frames
    at least 50 ms from either end."""
    truth = np.interp(t, np.arange(len(f0)) / FS, f0)
    inner = (t > 0.05) & (t < len(f0) / FS - 0.05)
    rel = np.where(est > 0, np.abs(est / truth - 1), np.inf)[inner]
    fine = rel[rel <= 0.2] * 100
    return (
        np.mean(rel > 0.2),
        (np.median(fine) if len(fine) else np.nan),
        (fine.max() if len(fine) else np.nan),
    )


def candidate_hit(t, cands, f0):
    truth = np.interp(t, np.arange(len(f0)) / FS, f0)
    inner = (t > 0.05) & (t < len(f0) / FS - 0.05)
    hit = np.array([np.any(np.abs(c / v - 1) < 0.05) for c, v in zip(cands, truth, strict=True)])
    return np.mean(hit[inner])


def main():
    # ================================================================== C1
    # The bank finds the fundamental only when it is there; the difference
    # function finds the period either way.
    for label, x in [
        ("clean vibrato", vowel(VIBRATO)),
        ("vibrato, white noise 10 dB SNR", with_noise(vowel(VIBRATO), 10)),
        ("vibrato, high-passed at 300 Hz", sosfiltfilt(HP300, vowel(VIBRATO))),
    ]:
        t = frame_times(len(x))
        report(
            "C1",
            f"{label}: frames with a bank candidate within 5% of F0",
            candidate_hit(t, bank_candidates(x, t), VIBRATO),
        )
        report(
            "C1",
            f"{label}: frames with a difference-function candidate within 5%",
            candidate_hit(t, yin_candidates(x, t), VIBRATO),
        )
    report(
        "C1",
        "vibrato's /a/: gain at 150 Hz relative to 750 Hz (near F1), dB",
        20 * np.log10(vowel_gain([150])[0] / vowel_gain([750])[0]),
    )
    x = vowel(CONTOURS["steady 120 Hz"])
    ch = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    t = frame_times(len(x))
    for r in ch:
        fc = 120 * r
        half = int(round(2 * FS / fc))
        n = np.arange(-half, half + 1)
        y = fftconvolve(x, nuttall(len(n)) * np.cos(2 * np.pi * fc * n / FS), mode="same")
        m = np.nanmedian(interval_rate(crossings(y, True), t))
        report("C1", f"steady 120 Hz: channel at {fc:5.0f} Hz, median rate of its output, Hz", m)

    # ================================================================== C2
    # Refinement by instantaneous frequency.
    for label, f0 in CONTOURS.items():
        x = vowel(f0)
        t = frame_times(len(x))
        cands = yin_candidates(x, t)
        raw = viterbi(scored(x, t, cands, do_refine=False))
        ref = viterbi(scored(x, t, cands))
        g0, m0, x0 = errors(t, raw, f0)
        g1, m1, x1 = errors(t, ref, f0)
        report("C2", f"{label}: median error before refinement, %", m0)
        report("C2", f"{label}: median error after refinement, %", m1)
        report("C2", f"{label}: largest error after refinement, %", x1)
        report("C2", f"{label}: gross errors after refinement, fraction", g1)

    # ================================================================== C3
    # The periodicity score of a periodic sound in independent noise.
    # Ideally the score is s / (1 + s) for a power SNR s. Linear interpolation
    # by a fraction phi of a sample scales white noise power by
    # g = (1 - phi)^2 + phi^2 in the shifted copy only, which raises the score
    # to s / sqrt((1 + s) (s + g)). At 120 Hz the period is 133 1/3 samples.
    x = vowel(CONTOURS["steady 120 Hz"])
    phi = FS / 120 % 1
    g = (1 - phi) ** 2 + phi**2
    for snr in (20, 10, 0, -5):
        y = with_noise(x, snr)
        s = 10 ** (snr / 10)
        r = np.median([periodicity(y, tc, 120.0) for tc in np.arange(0.1, 0.9, 0.01)])
        report("C3", f"steady 120 Hz, white noise {snr:3d} dB: score at 1/F0, median", r)
        report(
            "C3",
            f"    ideal s / (1 + s); with linear interpolation, {s / (1 + s):.3f}",
            s / np.sqrt((1 + s) * (s + g)),
        )
    t = frame_times(N1)
    best = [s.max() if len(s) else 0.0 for _, s in scored(NOISE, t, yin_candidates(NOISE, t))]
    report("C3", "white noise alone: largest score of any frame", np.max(best))

    # ================================================================== C4
    # Tracking against picking the best candidate in each frame.
    snd, fs = sf.read(SPEECH / "bdl_arctic_a0131.flac")
    assert fs == FS
    harvest = np.loadtxt(SPEECH / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    t_sent = frame_times(len(snd))
    h = np.interp(t_sent, harvest[:, 0], harvest[:, 1])
    hv = np.interp(t_sent, harvest[:, 0], (harvest[:, 1] > 0).astype(float)) > 0.5
    frames_sent = scored(snd, t_sent, yin_candidates(snd, t_sent))
    noisy = with_noise(vowel(VIBRATO), 5)
    t5 = frame_times(N1)
    frames_noisy = scored(noisy, t5, yin_candidates(noisy, t5))
    truth5 = np.interp(t5, TT, VIBRATO)
    for label, est_fn in [("best candidate per frame", best_per_frame), ("tracked (Viterbi)", viterbi)]:
        est = est_fn(frames_sent)
        both = (est > 0) & hv
        ratio = est[both] / h[both]
        report(
            "C4",
            f"sentence, {label}: frames voiced by both, off from Harvest by >20%",
            np.sum(np.abs(ratio - 1) > 0.2),
        )
        report("C4", f"sentence, {label}: voicing switches", np.sum(np.diff(est > 0)))
        est = est_fn(frames_noisy)
        v = est > 0
        report(
            "C4",
            f"vibrato at 5 dB SNR, {label}: voiced frames off by >20%, fraction",
            np.mean(np.abs(est[v] / truth5[v] - 1) > 0.2),
        )
        report("C4", f"vibrato at 5 dB SNR, {label}: frames voiced, fraction", np.mean(v))

    # ================================================================== C5
    # Synthetic set: tracker (both candidate stages) against the cepstral baseline.
    cases = dict(CONTOURS)
    for snr in (20, 10, 0):
        cases[f"vibrato, white noise {snr} dB SNR"] = (VIBRATO, snr)
    cases["vibrato, high-passed at 300 Hz"] = (VIBRATO, "hp")
    for label, spec in cases.items():
        f0, mod = spec if isinstance(spec, tuple) else (spec, None)
        x = vowel(f0)
        if mod == "hp":
            x = sosfiltfilt(HP300, x)
        elif mod is not None:
            x = with_noise(x, mod)
        for name, (t, est) in [
            ("tracker", track(x)),
            ("tracker, bank candidates", track(x, gen="bank")),
            ("cepstral baseline", cepstral_track(x)),
        ]:
            g, med, mx = errors(t, est, f0)
            report("C5", f"{label} | {name}: gross error rate", g)
            report("C5", f"{label} | {name}: median fine error, %", med)
    for theta in (0.5, 0.3):
        x = with_noise(vowel(VIBRATO), 0)
        t, est = track(x, theta=theta)
        report(
            "C5",
            f"vibrato at 0 dB SNR, tracker with voicing threshold {theta}: gross error rate",
            errors(t, est, VIBRATO)[0],
        )

    # ================================================================== C6
    # Noise alone.
    for name, (_, est) in [("tracker", track(NOISE)), ("cepstral baseline", cepstral_track(NOISE))]:
        report("C6", f"one second of white noise, {name}: frames called voiced, fraction", np.mean(est > 0))

    # ================================================================== C7, C8
    # The sentence against Harvest and against the corpus pitch marks.
    pm = np.loadtxt(SPEECH / "bdl_arctic_a0131.pm", skiprows=8)[:, 0]
    d = np.diff(pm)
    # The marks run on through unvoiced stretches as runs of identical intervals
    # slower than 110 Hz; frames in those runs have no mark-based F0.
    filler = np.zeros(len(d), bool)
    i = 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and abs(d[j + 1] - d[j]) < 2e-6:
            j += 1
        if j - i + 1 >= 3 and 1 / d[i] < 110:
            filler[i : j + 1] = True
        i = j + 1
    k = np.searchsorted(pm, t_sent) - 1
    inside = (k >= 0) & (k < len(d))
    pm_f0 = np.where(inside, 1 / d[np.clip(k, 0, len(d) - 1)], 0.0)
    pm_ok = inside & ~filler[np.clip(k, 0, len(d) - 1)] & (d[np.clip(k, 0, len(d) - 1)] < 0.015)
    report("C7", "sentence: frames Harvest calls voiced", hv.sum())
    report("C7", "sentence: frames with a pitch-mark F0 (marks not in an unvoiced run)", pm_ok.sum())
    report(
        "C7",
        "sentence: Harvest-voiced frames inside the marks' unvoiced runs",
        np.sum(hv & inside & filler[np.clip(k, 0, len(d) - 1)]),
    )
    for name, est in [
        ("tracker", viterbi(frames_sent)),
        ("tracker, bank candidates", track(snd, gen="bank")[1]),
        ("cepstral baseline", cepstral_track(snd)[1]),
        ("Harvest", h * hv),
    ]:
        v = est > 0
        both = v & hv
        report("C7", f"{name}: frames voiced", v.sum())
        if name != "Harvest":
            report("C7", f"{name}: Harvest-voiced frames it calls voiced, fraction", np.mean(v[hv]))
            report("C7", f"{name}: Harvest-unvoiced frames it calls voiced, fraction", np.mean(v[~hv]))
            report(
                "C7",
                f"{name}: frames both voiced, within 5% of Harvest, fraction",
                np.mean(np.abs(est[both] / h[both] - 1) < 0.05),
            )
        report("C8", f"{name}: frames with a pitch-mark F0 that it calls voiced, fraction", np.mean(v[pm_ok]))
        m = v & pm_ok
        report(
            "C8",
            f"{name}: voiced frames with a pitch-mark F0, within 5% of the marks, fraction",
            np.mean(np.abs(est[m] / pm_f0[m] - 1) < 0.05),
        )
        report(
            "C8",
            f"{name}: of those, median deviation from the marks, %",
            np.median(np.abs(est[m] / pm_f0[m] - 1)) * 100,
        )


if __name__ == "__main__":
    main()
