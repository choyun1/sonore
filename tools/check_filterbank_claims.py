"""Numerical checks for the claims in docs/design/frames/filterbanks.md (C1-C10).

Like the frames checkers, this is independent of sonore: only NumPy and
SciPy, with every scale and filter written out from its formula, so the
checks share no code with the implementation they will later test. Each line
prints the claim number and the number that supports it.

    python tools/check_filterbank_claims.py
"""

import time

import numpy as np

FS = 16000.0
N_SAMPLES = 16000


def report(claim, text, value):
    print(f"{claim:4s} {text:<74s} {value:.4g}")


# Frequency scales: (to_scale, from_scale), written out from their formulas.
SCALES = {
    "ERB": (lambda f: 9.265 * np.log1p(f / (24.7 * 9.265)), lambda e: 24.7 * 9.265 * np.expm1(e / 9.265)),
    "octave": (np.log2, np.exp2),
    "mel": (lambda f: 2595 * np.log10(1 + f / 700), lambda m: 700 * (10 ** (m / 2595) - 1)),
    "linear": (lambda f: f, lambda f: f),
}


def cosine_uniform(freqs, to_scale, knots):
    """Today's cosine bank: distance from each knot in units of the (uniform)
    spacing, half-cycle cosine within one spacing, flat lowpass and highpass
    outside the outermost knots."""
    with np.errstate(divide="ignore"):
        position = to_scale(freqs)[:, None]
    spacing = knots[1] - knots[0]
    distance = (position - knots[None, :]) / spacing
    transfer = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
    transfer[:, 0] = np.where(position[:, 0] <= knots[0], 1.0, transfer[:, 0])
    transfer[:, -1] = np.where(position[:, 0] >= knots[-1], 1.0, transfer[:, -1])
    return transfer


def cosine_gaps(freqs, to_scale, knots):
    """The same bank for any increasing knots: between knots k and k+1, with
    u the fractional position in that gap, filter k is cos(pi u / 2) and
    filter k+1 is sin(pi u / 2) = cos(pi (1 - u) / 2)."""
    with np.errstate(divide="ignore"):
        position = to_scale(freqs)
    transfer = np.zeros((len(freqs), len(knots)))
    gap = np.clip(np.searchsorted(knots, position, side="right") - 1, 0, len(knots) - 2)
    u = np.clip((position - knots[gap]) / (knots[gap + 1] - knots[gap]), 0, 1)
    rows = np.arange(len(freqs))
    transfer[rows, gap] = np.cos(np.pi / 2 * u)
    transfer[rows, gap + 1] = np.cos(np.pi / 2 * (1 - u))
    transfer[position <= knots[0], 0] = 1.0
    transfer[position >= knots[-1], -1] = 1.0
    return transfer


def power_sum(transfer):
    return np.sum(np.abs(transfer) ** 2, axis=1)


rfft_freqs = np.fft.rfftfreq(N_SAMPLES, 1 / FS)

# C1. Cosine banks are tight on any increasing scale, up to rounding.
for name, (to_scale, _) in SCALES.items():
    f_lo = 125.0 if name == "octave" else 50.0
    knots = np.linspace(to_scale(f_lo), to_scale(7600.0), 32)
    s = power_sum(cosine_uniform(rfft_freqs, to_scale, knots))
    report("C1", f"{name}: max |s - 1|, 30 bands, 16 kHz, 1 s grid", np.abs(s - 1).max())
    report("C1", f"{name}: fraction of bins where s == 1 exactly", np.mean(s == 1))

# C2. Tightness survives arbitrary (non-uniform) centers with the gap formula.
erb_to, erb_from = SCALES["ERB"]
rng = np.random.default_rng(0)
random_knots = np.sort(rng.uniform(erb_to(50.0), erb_to(7600.0), 32))
s = power_sum(cosine_gaps(rfft_freqs, erb_to, random_knots))
report("C2", "random ERB centers, gap formula: max |s - 1|", np.abs(s - 1).max())

# C3. On uniform knots the gap formula is not bit-identical to today's.
uniform_knots = np.linspace(erb_to(50.0), erb_to(7600.0), 32)
today = cosine_uniform(rfft_freqs, erb_to, uniform_knots)
gaps = cosine_gaps(rfft_freqs, erb_to, uniform_knots)
report("C3", "uniform ERB knots: fraction of responses that differ in any bit", np.mean(today != gaps))
report("C3", "uniform ERB knots: largest difference", np.abs(today - gaps).max())

# C4. Storing centers in Hz and mapping back to the scale changes the knots.
round_trip = erb_to(erb_from(uniform_knots))
knots_differ = np.mean(round_trip != uniform_knots)
report("C4", "ERB knots -> Hz -> ERB: fraction of knots that differ in any bit", knots_differ)
report(
    "C4",
    "  and fraction of responses that then differ",
    np.mean(cosine_uniform(rfft_freqs, erb_to, round_trip) != today),
)

# C5. Other widths: s stays constant (= width) only when twice the width, in
# spacings, is a whole number of at least 2. Checked only where every cosine
# that reaches a point is in the bank (ceil(width) knots in from each end).
for width in (0.75, 1.25, 1.5, 2.0, 2.5):
    position = erb_to(rfft_freqs)[:, None]
    distance = (position - uniform_knots[None, :]) / ((uniform_knots[1] - uniform_knots[0]) * width)
    transfer = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
    reach = int(np.ceil(width))
    inner = (position[:, 0] > uniform_knots[reach]) & (position[:, 0] < uniform_knots[-1 - reach])
    s = power_sum(transfer)[inner]
    report("C5", f"width {width} spacings: min s between the inner centers", s.min())
    report("C5", f"width {width} spacings: max s between the inner centers", s.max())


# C6. The envelope's peak delay, measured from the impulse response, against
# (order - 1) / (2 pi b) for a causal 4th-order gammatone.
def gammatone_response(freqs, fc, b, order=4):
    def transform(f):
        return (b + 1j * (f - fc)) ** -order + (b + 1j * (f + fc)) ** -order

    return transform(freqs) / np.abs(transform(fc))


for fs in (16000.0, 44100.0):
    n_grid = 1 << int(np.ceil(np.log2(4 * fs)))
    freqs = np.fft.rfftfreq(n_grid, 1 / fs)
    for fc in (100.0, 500.0, 2000.0, 6000.0):
        b = 1.019 * 24.7 * (4.37e-3 * fc + 1)
        transfer = gammatone_response(freqs, fc, b)
        analytic = np.fft.ifft(np.concatenate([transfer, np.zeros(n_grid - len(freqs))]) * 2, n=n_grid)
        measured = np.argmax(np.abs(analytic[: n_grid // 2])) / fs
        formula = 3 / (2 * np.pi * b)
        report("C6", f"{fs:g} Hz, fc {fc:g} Hz: formula [ms]", 1000 * formula)
        difference_ms = 1000 * (measured - formula)
        report("C6", f"{fs:g} Hz, fc {fc:g} Hz: measured envelope peak - formula [ms]", difference_ms)

# C7. The octave modulation bank is a cosine bank on log2 frequency, spacing
# one octave, without the flat edges.
mod_freqs = np.fft.rfftfreq(4000, 1 / 400.0)
mod_cfs = 100.0 / 2.0 ** np.arange(6, -1, -1)
with np.errstate(divide="ignore"):
    octave_offset = (np.log2(mod_freqs)[:, None] - np.log2(mod_cfs)[None, :]) / 2
    texture_bank = np.where(
        np.abs(octave_offset) < 0.5, np.cos(np.pi * np.clip(octave_offset, -0.5, 0.5)), 0.0
    )
    distance = (np.log2(mod_freqs)[:, None] - np.log2(mod_cfs)[None, :]) / 1.0
    cosine_bank = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
largest = np.abs(texture_bank - cosine_bank).max()
report("C7", "octave modulation bank vs cosine bank on log2: largest difference", largest)
report("C7", "  fraction of responses that differ in any bit", np.mean(texture_bank != cosine_bank))

# C8. Computing s in synthesize costs little next to the transforms.
transfer = cosine_uniform(np.fft.rfftfreq(5 * 44100, 1 / 44100), erb_to, uniform_knots)
bands = rng.standard_normal((5 * 44100, 32))
repeats = 5
start = time.perf_counter()
for _ in range(repeats):
    np.fft.irfft((np.fft.rfft(bands, axis=0) * transfer).sum(axis=1), n=5 * 44100)
transforms = (time.perf_counter() - start) / repeats
start = time.perf_counter()
for _ in range(repeats):
    np.abs(np.sum(np.abs(transfer) ** 2, axis=1) - 1).max()
measuring = (time.perf_counter() - start) / repeats
report("C8", "5 s at 44.1 kHz, 32 filters: time to measure s / time of synthesis", measuring / transforms)


# C9. Taking the tight path (re-filter, no division by s) on a bank whose s is
# not constant loses exactly the ripple: per frequency the output is s / c
# times the input, so the error is bounded by max |s / c - 1|.
signal = rng.standard_normal(N_SAMPLES)
spectrum = np.fft.rfft(signal)
for ripple in (1e-14, 1e-12, 1e-6, 1e-2):
    s_rippled = 1 + ripple * np.cos(np.linspace(0, 40 * np.pi, len(rfft_freqs)))
    rebuilt = np.fft.irfft(spectrum * s_rippled, n=N_SAMPLES)
    error = np.abs(rebuilt - signal).max() / np.abs(signal).max()
    report("C9", f"ripple {ripple:g} in s, tight path: largest error / peak", error)


# C10. A frame with a small lower bound inverts exactly in theory but loses
# accuracy in floating point: the canonical dual divides by s, amplifying
# round-off by up to B / A.
for condition in (1e2, 1e6, 1e10, 1e12):
    s_bounds = np.where(rfft_freqs < 1000, 1.0, 1.0 / condition)
    responses = np.sqrt(s_bounds)
    bands = np.fft.irfft(spectrum * responses, n=N_SAMPLES)
    bands = bands + np.finfo(float).eps * np.abs(bands).max() * rng.standard_normal(N_SAMPLES)
    rebuilt = np.fft.irfft(np.fft.rfft(bands) * responses / s_bounds, n=N_SAMPLES)
    error = np.abs(rebuilt - signal).max() / np.abs(signal).max()
    report("C10", f"B/A = {condition:g}, one rounding step in the bands: error / peak", error)
