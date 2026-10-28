"""Numerical checks for the claims in docs/design/frames/decimated-filterbanks.md (C1-C9).

The decimated transform is written out here from its definition (Velasco et
al., 2011; Balazs et al., 2011), with NumPy only, so these checks share no
code with an implementation that does not exist yet. sonore is imported only
for the filters themselves (to show the transform can take any bank's
responses unchanged) and for the timing comparison with today's undecimated
Filterbank. Each line prints the claim number and the number that supports
it.

    python tools/check_constant_q_claims.py
"""

import time

import numpy as np
import scipy.fft as sp_fft

import sonore as so

FS = 44100.0
DURATION = 3.0
N = int(FS * DURATION)  # 132300 samples
RNG = np.random.default_rng(0)


def report(claim, text, value):
    print(f"{claim:4s} {text:<78s} {value:.4g}")


# ------------------------------------------------------------- the transform
def full_responses(bank, n, fs):
    """Every filter on the two-sided DFT grid of n samples, one column per
    filter: the lowpass and highpass as they are (they cover both signs of
    frequency), and each bandpass split into its positive-frequency half and
    its mirror. Responses satisfy H(-f) = conj(H(f)), as Filterbank assumes."""
    half = bank.rfft_response(n, fs)  # (n // 2 + 1, n_filters), f >= 0
    signed = np.round(np.fft.fftfreq(n) * n).astype(int)  # signed bin index
    index = np.abs(signed).astype(int)
    full = half[index]  # |f| lookup
    negative = signed < 0
    full[negative] = np.conj(full[negative])
    low, bands, high = full[:, :1], full[:, 1:-1], full[:, -1:]
    positive = np.where(negative[:, None], 0, bands)
    mirror = np.where(negative[:, None], bands, 0)
    if n % 2 == 0:  # the Nyquist bin belongs to the highpass only
        positive[n // 2] = mirror[n // 2] = 0
    return np.concatenate([low, positive, mirror, high], axis=1), bands.shape[1]


def support(column):
    """The circular arc of bins where a response is nonzero: (first bin, length)."""
    nonzero = np.flatnonzero(column)
    n = len(column)
    if len(nonzero) == 0:
        return 0, 0
    gaps = np.diff(np.concatenate([nonzero, [nonzero[0] + n]]))
    last_gap = int(np.argmax(gaps))  # the arc starts after the largest gap
    start = int(nonzero[(last_gap + 1) % len(nonzero)])
    length = n - int(gaps[last_gap]) + 1
    return start, length


def plan(responses, rounding="minimal", threshold_db=None):
    """Per filter: its support and the number of coefficients M (at least the
    support's length, so no two bins of the support share a coefficient bin)."""
    columns = []
    for k in range(responses.shape[1]):
        column = responses[:, k].copy()
        if threshold_db is not None:  # truncate a filter that never reaches zero
            peak = np.abs(column).max()
            column[np.abs(column) < peak * 10 ** (threshold_db / 20)] = 0
        start, length = support(column)
        m = {"minimal": length, "fast": sp_fft.next_fast_len(length)}[rounding]
        columns.append((column, start, length, m))
    return columns


def analyze(x, columns):
    """c_k = sqrt(M_k / N) ifft_M(V_k), with V_k[(start + j) mod M] = X g_k on the support."""
    n = len(x)
    spectrum = np.fft.fft(x)
    coefs = []
    for column, start, length, m in columns:
        bins = (start + np.arange(length)) % n
        folded = np.zeros(m, complex)
        folded[(start + np.arange(length)) % m] = spectrum[bins] * column[bins]
        coefs.append(np.sqrt(m / n) * np.fft.ifft(folded))
    return coefs


def frame_power(columns, n):
    return sum(np.abs(column) ** 2 for column, *_ in columns)


def synthesize(coefs, columns, n, dual=True):
    """Canonical dual: X = sum_k (conj(g_k) / s) * unfolded fft_M(c_k) * sqrt(N / M_k)."""
    s = frame_power(columns, n) if dual else np.ones(n)
    spectrum = np.zeros(n, complex)
    for c, (column, start, length, m) in zip(coefs, columns, strict=True):
        bins = (start + np.arange(length)) % n
        folded = np.fft.fft(c) * np.sqrt(n / m)
        spectrum[bins] += np.conj(column[bins]) * folded[(start + np.arange(length)) % m] / s[bins]
    return np.fft.ifft(spectrum)


def real_count(columns, n_bands):
    """Real numbers stored for a real signal: the mirror bands are the complex
    conjugates of the positive ones and are not kept; every kept band is
    complex (2 reals per coefficient)."""
    kept = [columns[0], *columns[1 : 1 + n_bands], columns[-1]]
    return sum(2 * m for *_, m in kept)


x = RNG.standard_normal(N)

# The semitone bank of the chroma page: 55 Hz to 7040 Hz (seven octaves), 83 bandpass filters.
semitones = so.cosine_filterbank(f_lo=55, f_hi=7040, scale="cents", spacing=100)
responses, n_bands = full_responses(semitones, N, FS)
columns = plan(responses)

# C1. The filters are today's: the decimated transform takes Filterbank responses unchanged.
report("C1", "semitone cosine bank: bandpass filters", n_bands)
report(
    "C1",
    "max |s - 1| on the two-sided grid (tight before decimation)",
    np.abs(frame_power(columns, N) - 1).max(),
)

# C2. Decimated to the support of each filter, it is still exact and still tight (Parseval).
coefs = analyze(x, columns)
y = synthesize(coefs, columns, N)
report(
    "C2",
    "reconstruction error, max |x - y| / max |x| (white noise, 3 s, 44.1 kHz)",
    np.abs(y - x).max() / np.abs(x).max(),
)
report("C2", "largest imaginary part of y (the signal comes back real)", np.abs(y.imag).max())
energy = sum(np.sum(np.abs(c) ** 2) for c in coefs)
report(
    "C2",
    "|coefficient energy / signal energy - 1| (0 means tight, A = B = 1)",
    abs(energy / np.sum(x**2) - 1),
)
y_adjoint = synthesize(coefs, columns, N, dual=False)
report(
    "C2",
    "max |dual - adjoint| / max |x| (tight: synthesis is the adjoint)",
    np.abs(y_adjoint - y).max() / np.abs(x).max(),
)


# C3. A real signal needs only the edges and the positive half: the mirror bands add the conjugate.
def keep(indices):
    return [c if i in indices else np.zeros_like(c) for i, c in enumerate(coefs)]


edges_only = synthesize(keep({0, len(coefs) - 1}), columns, N).real
positive_only = synthesize(keep(set(range(1, 1 + n_bands))), columns, N)
report(
    "C3",
    "error rebuilding from the edges and positive bands only (edges + 2 Re)",
    np.abs(edges_only + 2 * positive_only.real - x).max() / np.abs(x).max(),
)

# C4. How many numbers it keeps, against the undecimated bank's N per filter.
undecimated = N * semitones.n_filters
decimated = real_count(columns, n_bands)
low_m, high_m = columns[0][3], columns[-1][3]
bands_m = sum(m for *_, m in columns[1 : 1 + n_bands])
report("C4", "undecimated: real numbers / N (one real band per filter)", undecimated / N)
report("C4", "decimated, minimal M: real numbers / N", decimated / N)
report("C4", "  of which the 83 bandpass bands", 2 * bands_m / N)
report("C4", "  of which the highpass (7 kHz to 22 kHz)", 2 * high_m / N)
report("C4", "  of which the lowpass (0 to 58 Hz)", 2 * low_m / N)
full_range = so.cosine_filterbank(f_lo=55, f_hi=20000, scale="cents", spacing=100)
r_full, nb_full = full_responses(full_range, N, FS)
report(
    "C4", "decimated, bank up to 20 kHz (101 bands): real numbers / N", real_count(plan(r_full), nb_full) / N
)
fine = so.cosine_filterbank(f_lo=55, f_hi=7040, scale="cents", spacing=25)
r_fine, nb_fine = full_responses(fine, N, FS)
report("C4", "48 per octave: undecimated real numbers / N", N * fine.n_filters / N)
report("C4", "48 per octave: decimated real numbers / N", real_count(plan(r_fine), nb_fine) / N)

# C5. Hops: band k's coefficients are N / M_k samples apart; low bands slow, high bands fast.
hops_ms = [1e3 * (N / m) / FS for *_, m in columns[1 : 1 + n_bands]]
report("C5", "hop of the 58 Hz band [ms]", hops_ms[0])
report("C5", "hop of the 6.6 kHz band [ms]", hops_ms[-1])
fast = plan(responses, rounding="fast")
report(
    "C5",
    "M rounded up to a fast FFT length: extra numbers (fraction)",
    real_count(fast, n_bands) / decimated - 1,
)
y_fast = synthesize(analyze(x, fast), fast, N)
report("C5", "  and its reconstruction error", np.abs(y_fast - x).max() / np.abs(x).max())
single_m = max(m for *_, m in columns[1 : 1 + n_bands])
one_hop = [columns[0]] + [(c, st, ln, single_m) for c, st, ln, _ in columns[1:-1]] + [columns[-1]]
report(
    "C5",
    "one hop for every bandpass (a matrix, like a spectrogram): numbers / N",
    real_count(one_hop, n_bands) / N,
)

# C6. A decimated band is the undecimated complex band sampled every hop, when the hop is whole.
divisors = [m for m in range(1, N + 1) if N % m == 0]
k = 40  # a band near 600 Hz
column, start, length, _ = columns[1 + k]
m = min(d for d in divisors if d >= length)
(c,) = analyze(x, [(column, start, length, m)])
band = np.fft.ifft(np.fft.fft(x) * column)  # the undecimated positive-frequency band
sampled = band[:: N // m] * np.sqrt(N / m)
report(
    "C6",
    f"band {k} ({semitones.band_cfs[k]:.0f} Hz), hop {N // m} samples: max |c - sampled band| / max |c|",
    np.abs(c - sampled).max() / np.abs(c).max(),
)

# C7. Filters without compact support (gammatone, Morlet) are cut at a level; the cut bank is still exact.
for name, bank in [
    ("gammatone, 60 bands, 70 Hz-7 kHz", so.gammatone_filterbank(60, 70, 7000)),
    ("Morlet, 54 bands, 70 Hz-7 kHz", so.morlet_filterbank(54, 70, 7000, cycles=6)),
]:
    r_bank, nb_bank = full_responses(bank, N, FS)
    for level in (-40, -60, -100):
        cut = plan(r_bank, threshold_db=level)
        y_cut = synthesize(analyze(x, cut), cut, N)
        report("C7", f"{name}, cut at {level} dB: real numbers / N", real_count(cut, nb_bank) / N)
        report("C7", "  reconstruction error with the cut filters", np.abs(y_cut - x).max() / np.abs(x).max())
    report("C7", f"{name}: undecimated real numbers / N", bank.n_filters)

cut_cosine = plan(responses, threshold_db=-100)
report(
    "C7",
    "semitone cosine bank cut at -100 dB: response bins changed",
    sum(np.count_nonzero(a[0] != b[0]) for a, b in zip(cut_cosine, columns, strict=True)),
)
gammatone = so.gammatone_filterbank(60, 70, 7000)
r_gammatone, _ = full_responses(gammatone, N, FS)
report(
    "C7",
    "gammatone positive band, uncut: support length / (N / 2)",
    min(support(r_gammatone[:, k])[1] for k in range(1, 61)) / (N / 2),
)

# C8. Modified coefficients: synthesis is the least-squares signal (analysis after synthesis is a projection).
masked = [c * (RNG.uniform(size=c.shape) > 0.5) for c in coefs]
projected = analyze(synthesize(masked, columns, N).real, columns)
twice = analyze(synthesize(projected, columns, N).real, columns)
report(
    "C8",
    "projection: max |P(P c) - P c| / max |P c| on half-masked coefficients",
    max(np.abs(a - b).max() for a, b in zip(twice, projected, strict=True))
    / max(np.abs(p).max() for p in projected),
)


# C9. Time: analysis plus synthesis, decimated (this unoptimized loop) against today's Filterbank.
def timed(function, repeats=3):
    best = np.inf
    for _ in range(repeats):
        start_time = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start_time)
    return best


snd = so.Sound(x, FS)
t_undecimated = timed(lambda: semitones.analyze(snd, pad=0).to_sound())
t_decimated = timed(lambda: synthesize(analyze(x, columns), columns, N))
report("C9", "undecimated Filterbank, analyze + to_sound, pad=0 [s]", t_undecimated)
report("C9", "decimated loop, analyze + synthesize [s]", t_decimated)
