"""Numerical checks for the claims in docs/design/views/lpc.md (C1-C12).

Like the other claim checkers, this is independent of sonore: only NumPy,
SciPy and soundfile (to read the gallery sentences), with the vowels, windows,
autocorrelation, Levinson-Durbin recursion and root picking written out from
their formulas. Each line prints the claim number and the number that
supports it.

    python tools/check_lpc_claims.py
"""

from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter, resample_poly

FS = 16000
ROOT = Path(__file__).resolve().parent.parent
SPEECH = ROOT / "docs" / "speech"

# Peterson and Barney's (1952) averages, F1 F2 F3 (Hz), as on the Formant synthesis page.
MALE = {
    "heed": (270, 2290, 3010),
    "head": (530, 1840, 2480),
    "had": (660, 1720, 2410),
    "hod": (730, 1090, 2440),
    "hawed": (570, 840, 2410),
    "who'd": (300, 870, 2240),
}
FEMALE = {
    "heed": (310, 2790, 3310),
    "head": (610, 2330, 2990),
    "had": (860, 2050, 2850),
    "hod": (850, 1220, 2810),
    "hawed": (590, 920, 2710),
    "who'd": (370, 950, 2670),
}
# Klatt's default bandwidths of F1 to F5 and the higher formants of each set.
BANDWIDTHS = (60, 90, 150, 200, 250)
HIGHER = {"male": (3500, 4500), "female": (4100, 4900)}
# Formant ceilings, Praat's advice for a male and a female voice.
CEILINGS = {"male": 5000, "female": 5500}


def report(claim, text, value):
    print(f"{claim:4s} {text:<78s} {value:.4g}")


def hann(n):
    """The periodic Hann window, sonore's STFT default."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def hamming(n):
    """The symmetric Hamming window."""
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def resonator_coefficients(freq, bandwidth):
    """Klatt's resonator y[n] = A x[n] + B y[n-1] + C y[n-2], as lfilter (b, a)."""
    c = -np.exp(-2 * np.pi * bandwidth / FS)
    b = 2 * np.exp(-np.pi * bandwidth / FS) * np.cos(2 * np.pi * freq / FS)
    return [1 - b - c], [1, -b, -c]


def vowel(formants, f0, duration=0.5):
    """An impulse train at f0 through Klatt's glottal low-pass (0 Hz, 100 Hz wide), a first
    difference for radiation, and five formant resonators in cascade. f0 is rounded so the
    period is a whole number of samples."""
    period = round(FS / f0)
    source = np.zeros(int(duration * FS))
    source[::period] = 1.0
    signal = lfilter(*resonator_coefficients(0, 100), source)
    signal = np.diff(signal, prepend=0.0)
    for freq, bandwidth in zip(formants, BANDWIDTHS, strict=True):
        signal = lfilter(*resonator_coefficients(freq, bandwidth), signal)
    return signal, FS / period


def autocorrelation(segment, order):
    """r[0..order] of a windowed segment, straight from the sum."""
    return np.array([segment[: len(segment) - lag] @ segment[lag:] for lag in range(order + 1)])


def levinson(r, order):
    """The Levinson-Durbin recursion: predictor a (a[0] = 1), reflection coefficients, and the
    prediction error power after each order."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    error = r[0]
    reflections = np.zeros(order)
    errors = [error]
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1 : 0 : -1]) / error
        a[1 : i + 1] = a[1 : i + 1] + k * np.r_[a[i - 1 : 0 : -1], 1.0]
        reflections[i - 1] = k
        error *= 1 - k * k
        errors.append(error)
    return a, reflections, np.array(errors)


def lpc(segment, order):
    a, _, errors = levinson(autocorrelation(segment, order), order)
    return a, errors[-1]


def burg(segment, order):
    """Burg's method: the predictor whose reflection coefficients minimize the summed forward and
    backward prediction errors, one order at a time."""
    forward = segment.astype(float).copy()
    backward = segment.astype(float).copy()
    a = np.array([1.0])
    for m in range(order):
        f, b = forward[m + 1 :], backward[m:-1]
        k = -2 * (f @ b) / (f @ f + b @ b)
        a = np.r_[a, 0.0] + k * np.r_[a, 0.0][::-1]
        forward[m + 1 :], backward[m + 1 :] = f + k * b, b + k * f
    return a


def formants_from_roots(a, max_bandwidth=400.0, min_freq=90.0):
    """Frequencies and bandwidths [Hz] of the roots of A(z) in the upper half plane, narrower than
    max_bandwidth and above min_freq, sorted by frequency."""
    roots = np.roots(a)
    roots = roots[roots.imag > 0]
    freqs = np.angle(roots) * FS / (2 * np.pi)
    bandwidths = -np.log(np.abs(roots)) * FS / np.pi
    keep = (bandwidths < max_bandwidth) & (freqs > min_freq)
    order = np.argsort(freqs[keep])
    return freqs[keep][order], bandwidths[keep][order]


def preemphasize(signal, coefficient=0.97):
    return np.r_[signal[0], signal[1:] - coefficient * signal[:-1]]


def recovery_errors(
    formants, f0, order, win_dur=0.025, preemphasis=0.97, positions=20, window_function=hamming, method=lpc
):
    """For one vowel: the error [Hz] of the root nearest each of F1-F3, and of its bandwidth, at
    `positions` window offsets spread over one period (the estimate depends on where the window
    falls against the pulses). Two arrays of shape (positions, 3)."""
    signal, f0 = vowel(formants, f0)
    if preemphasis:
        signal = preemphasize(signal, preemphasis)
    length = round(win_dur * FS)
    window = window_function(length)
    period = FS / f0
    start0 = len(signal) // 2
    errors = np.full((positions, 3), np.nan)
    bandwidth_errors = np.full((positions, 3), np.nan)
    for position in range(positions):
        start = start0 + round(position * period / positions)
        a = method(signal[start : start + length] * window, order)
        if isinstance(a, tuple):
            a = a[0]
        freqs, bandwidths = formants_from_roots(a)
        for index, (target, target_bandwidth) in enumerate(zip(formants[:3], BANDWIDTHS[:3], strict=True)):
            if len(freqs):
                nearest = np.argmin(np.abs(freqs - target))
                errors[position, index] = freqs[nearest] - target
                bandwidth_errors[position, index] = bandwidths[nearest] - target_bandwidth
    return errors, bandwidth_errors


def vowel_sets():
    for label, vowels, f0 in (("male", MALE, 120), ("female", FEMALE, 220)):
        for word, f123 in vowels.items():
            yield label, word, (*f123, *HIGHER[label]), f0


def voiced_windows(speaker, win_dur=0.025, hop=0.005):
    """Pre-emphasized, Hamming-windowed segments of a gallery sentence, centered on the times the
    Harvest track calls voiced."""
    signal, fs = sf.read(SPEECH / f"{speaker}_arctic_a0131.flac")
    assert fs == FS
    signal = preemphasize(signal)
    track = np.loadtxt(SPEECH / f"{speaker}_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    length = round(win_dur * FS)
    window = hamming(length)
    segments = []
    for time, f0 in track:
        start = round(time * FS) - length // 2
        if f0 > 0 and start >= 0 and start + length <= len(signal):
            segments.append(signal[start : start + length] * window)
    return segments


# Formant tracking: an utterance with moving formants, and a tracker over LPC candidates.

STEADY = 0.20  # seconds each vowel is held
TRANSITION = 0.06  # seconds of linear glide between vowels
NOMINAL = (500.0, 1500.0, 2500.0)  # a uniform 17.5 cm tube closed at one end


def formant_tracks(vowels, n_samples):
    """F1-F5 per sample for the vowels in turn, held STEADY seconds each and joined by linear
    TRANSITION-second glides, plus the label's higher formants. Shape (5, n_samples)."""
    times = np.arange(n_samples) / FS
    knots_t, knots_f = [], []
    for index, formants in enumerate(vowels):
        start = index * (STEADY + TRANSITION)
        knots_t += [start, start + STEADY]
        knots_f += [formants, formants]
    knots_f = np.array(knots_f, dtype=float)
    return np.array([np.interp(times, knots_t, knots_f[:, k]) for k in range(knots_f.shape[1])])


def moving_resonator(signal, freqs, bandwidth):
    """Klatt's resonator with its frequency changing every sample."""
    output = np.zeros_like(signal)
    previous, before = 0.0, 0.0
    for n, (x, freq) in enumerate(zip(signal, freqs, strict=True)):
        (gain,), (_, minus_b, minus_c) = resonator_coefficients(freq, bandwidth)
        y = gain * x - minus_b * previous - minus_c * before
        output[n], before, previous = y, previous, y
    return output


def utterance(label):
    """Six Peterson and Barney vowels in a row with no silence ("heed head had hod hawed who'd"),
    the pitch falling linearly (130 to 100 Hz male, 240 to 200 Hz female). Returns the signal and
    its true F1-F3 per sample."""
    vowels, f0_range = (MALE, (130, 100)) if label == "male" else (FEMALE, (240, 200))
    vowels = [(*f123, *HIGHER[label]) for f123 in vowels.values()]
    n_samples = round((len(vowels) * (STEADY + TRANSITION) - TRANSITION) * FS)
    f0 = np.linspace(*f0_range, n_samples)
    phase = np.cumsum(f0) / FS
    source = np.diff(np.floor(phase), prepend=0.0)  # one impulse per period
    signal = np.diff(lfilter(*resonator_coefficients(0, 100), source), prepend=0.0)
    tracks = formant_tracks(vowels, n_samples)
    for k, bandwidth in enumerate(BANDWIDTHS):
        signal = moving_resonator(signal, tracks[k], bandwidth)
    return signal, tracks[:3]


def candidates(signal, times, order=18, win_dur=0.025, fmax=5000.0):
    """LPC candidates (frequencies, bandwidths) below fmax at each time, pre-emphasized,
    Hamming window."""
    emphasized = preemphasize(signal)
    length = round(win_dur * FS)
    window = hamming(length)
    found = []
    for time in times:
        start = round(time * FS) - length // 2
        freqs, bandwidths = formants_from_roots(lpc(emphasized[start : start + length] * window, order)[0])
        keep = freqs < fmax
        found.append((freqs[keep], bandwidths[keep]))
    return found


def ceiling_candidates(signal, times, ceiling, n_formants=5, win_dur=0.025):
    """Praat's recipe for formants: resample to twice the ceiling, fit 2 n_formants poles, and keep
    every root between 50 Hz and 50 Hz below the ceiling, whatever its bandwidth. Returns
    (frequencies, bandwidths) at each time, at the original times."""
    ratio = Fraction(int(2 * ceiling), FS)
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    fs = 2 * ceiling
    emphasized = preemphasize(resampled)
    length = round(win_dur * fs)
    window = hamming(length)
    found = []
    for time in times:
        start = round(time * fs) - length // 2
        a, _ = lpc(emphasized[start : start + length] * window, 2 * n_formants)
        roots = np.roots(a)
        roots = roots[roots.imag > 0]
        freqs = np.angle(roots) * fs / (2 * np.pi)
        bandwidths = -np.log(np.abs(roots)) * fs / np.pi
        keep = (freqs > 50) & (freqs < ceiling - 50)
        order = np.argsort(freqs[keep])
        found.append((freqs[keep][order], bandwidths[keep][order]))
    return found


def by_count(found, n_formants=3):
    """The k-th candidate as Fk, NaN where there are fewer than n_formants."""
    out = np.full((len(found), n_formants), np.nan)
    for index, (freqs, _) in enumerate(found):
        out[index, : min(n_formants, len(freqs))] = freqs[:n_formants]
    return out


def track(
    found,
    nominal=NOMINAL,
    nominal_weight=1.0,
    bandwidth_weight=1.0,
    transition_weight=5.0,
    missing_cost=3.0,
    skip_cost=2.0,
):
    """F1-F3 for every time window by dynamic programming (Viterbi) over assignments of candidates
    to formants. A state is an increasing choice of three candidates; a formant may be missing only
    when there are fewer candidates than formants.
    Its cost in one window is, per formant, nominal_weight |ln(f / nominal)| plus
    bandwidth_weight * bandwidth / f, or missing_cost if missing; moving from one window to the
    next costs transition_weight |ln(f_now / f_before)| per formant present in both. Returns
    (n_windows, 3) frequencies, NaN where missing."""
    n_formants = len(nominal)
    nominal = np.asarray(nominal)
    state_freqs, state_costs = [], []
    for freqs, bandwidths in found:
        options = []
        for n_present in range(min(n_formants, len(freqs)), min(n_formants, len(freqs)) + 1):
            for chosen in combinations(range(len(freqs)), n_present):
                for slots in combinations(range(n_formants), n_present):
                    values = np.full(n_formants, np.nan)
                    values[list(slots)] = freqs[list(chosen)]
                    widths = np.full(n_formants, np.nan)
                    widths[list(slots)] = bandwidths[list(chosen)]
                    top = values[~np.isnan(values)].max() if n_present else 0.0
                    skipped = sum(1 for c in range(len(freqs)) if c not in chosen and freqs[c] < top)
                    options.append((values, widths, skipped))
        values = np.array([v for v, _, _ in options])
        widths = np.array([w for _, w, _ in options])
        skipped = np.array([k for _, _, k in options])
        present = ~np.isnan(values)
        local = (
            np.where(
                present,
                nominal_weight * np.abs(np.log(np.where(present, values, 1) / nominal))
                + bandwidth_weight * np.where(present, widths, 0) / np.where(present, values, 1),
                missing_cost,
            ).sum(axis=1)
            + skip_cost * skipped
        )
        state_freqs.append(values)
        state_costs.append(local)
    total = state_costs[0]
    back = []
    for index in range(1, len(found)):
        before, now = state_freqs[index - 1], state_freqs[index]
        jump = np.abs(np.log(now[:, None, :] / before[None, :, :]))
        jump = np.nan_to_num(jump, nan=0.0).sum(axis=2)  # (now, before)
        candidates_total = total[None, :] + transition_weight * jump
        best = np.argmin(candidates_total, axis=1)
        back.append(best)
        total = candidates_total[np.arange(len(now)), best] + state_costs[index]
    path = [int(np.argmin(total))]
    for best in reversed(back):
        path.append(int(best[path[-1]]))
    path.reverse()
    return np.array([state_freqs[index][state] for index, state in enumerate(path)])


def main():
    rng = np.random.default_rng(0)
    order = 18  # fs / 1000 + 2 at 16 kHz

    # C1: Levinson-Durbin solves the normal equations.
    segment = rng.standard_normal(400) * hamming(400)
    segment = lfilter([1], [1, -1.3, 0.8], segment)
    r = autocorrelation(segment, order)
    a, reflections, errors = levinson(r, order)
    direct = solve_toeplitz(r[:order], -r[1 : order + 1])
    report(
        "C1", "Levinson vs scipy solve_toeplitz, largest coefficient difference", np.abs(a[1:] - direct).max()
    )
    residual_power = r[0] + a[1:] @ r[1:]
    report(
        "C1",
        "final error power vs r[0] + a . r[1:], relative difference",
        abs(errors[-1] - residual_power) / residual_power,
    )

    # C2: stability. Every reflection coefficient is inside (-1, 1) and every root inside the
    # unit circle, for every voiced window of both talkers.
    largest_reflection, largest_root = 0.0, 0.0
    count = 0
    for speaker in ("bdl", "slt"):
        for segment in voiced_windows(speaker):
            a, reflections, _ = levinson(autocorrelation(segment, order), order)
            largest_reflection = max(largest_reflection, np.abs(reflections).max())
            largest_root = max(largest_root, np.abs(np.roots(a)).max())
            count += 1
    report("C2", "voiced windows of both talkers analyzed", count)
    report("C2", "largest |reflection coefficient| over them", largest_reflection)
    report("C2", "largest root magnitude of A(z) over them", largest_root)

    # C3: the autocorrelation from the power spectrum is exact when n_fft >= L + order, and
    # circular (aliased) otherwise.
    length = 400
    segment = vowel(MALE["hod"] + HIGHER["male"], 120)[0][4000 : 4000 + length]
    segment = preemphasize(segment) * hamming(length)
    direct_r = autocorrelation(segment, order)
    for n_fft in (length, length + order, 1024):
        from_spectrum = np.fft.irfft(np.abs(np.fft.rfft(segment, n_fft)) ** 2, n_fft)[: order + 1]
        a_direct, _ = lpc(segment, order)
        a_spectrum, _, _ = levinson(from_spectrum, order)
        f_direct, _ = formants_from_roots(a_direct)
        f_spectrum, _ = formants_from_roots(a_spectrum)
        shift = np.abs(f_direct[:3] - f_spectrum[:3]).max() if len(f_spectrum) >= 3 else np.nan
        report(
            "C3",
            f"n_fft {n_fft}: largest error in r, relative to r[0]",
            np.abs(from_spectrum - direct_r).max() / direct_r[0],
        )
        report("C3", f"n_fft {n_fft}: largest F1-F3 shift against the direct autocorrelation [Hz]", shift)

    # C4: recovery of the formants put into synthesized vowels, order 18, 25 ms Hamming,
    # pre-emphasis 0.97, 20 window positions per vowel.
    for label in ("male", "female"):
        all_errors, all_bandwidth, worst = [], [], {}
        for set_label, word, formants, f0 in vowel_sets():
            if set_label != label:
                continue
            errors, bandwidth_errors = recovery_errors(formants, f0, order)
            all_errors.append(errors)
            all_bandwidth.append(bandwidth_errors)
            worst[word] = np.abs(errors).max()
        word = max(worst, key=worst.get)
        report("C4", f"{label}: worst vowel is '{word}', its largest |F1-F3 error| [Hz]", worst[word])
        all_errors = np.concatenate(all_errors)
        all_bandwidth = np.concatenate(all_bandwidth)
        for index in range(3):
            column = np.abs(all_errors[:, index])
            relative = column / np.array([f[index] for s, w, f, _ in vowel_sets() if s == label]).repeat(20)
            report("C4", f"{label} F{index + 1}: median |error| [Hz]", np.median(column))
            report(
                "C4", f"{label} F{index + 1}: largest |error| as a fraction of the formant", relative.max()
            )
        report("C4", f"{label}: median |bandwidth error| of F1-F3 [Hz]", np.nanmedian(np.abs(all_bandwidth)))

    # C5: the error grows with F0. "hod" (male formants) at F0 from 100 to 300 Hz.
    for f0 in (100, 150, 200, 250, 300):
        errors, _ = recovery_errors(MALE["hod"] + HIGHER["male"], f0, order)
        report(
            "C5",
            f"hod at F0 {f0} Hz: largest |F1 error| over window positions [Hz]",
            np.nanmax(np.abs(errors[:, 0])),
        )
    for f0 in (100, 150, 200, 250, 300):
        errors, _ = recovery_errors(MALE["heed"] + HIGHER["male"], f0, order)
        report(
            "C5",
            f"heed at F0 {f0} Hz: largest |F1 error| over window positions [Hz]",
            np.nanmax(np.abs(errors[:, 0])),
        )

    # C6: order. Too low merges formants, too high splits them; the median F1-F3 error and the
    # number of extra narrow roots below 5 kHz, over the male and female sets.
    for test_order in (8, 12, 14, 16, 18, 20, 24, 30):
        medians = []
        for _, _, formants, f0 in vowel_sets():
            errors, _ = recovery_errors(formants, f0, test_order, positions=5)
            medians.append(np.nanmedian(np.abs(errors)))
        report(
            "C6",
            f"order {test_order}: median over vowels of the median |F1-F3 error| [Hz]",
            np.median(medians),
        )

    # C7: pre-emphasis. Without it the glottal tilt takes poles and F1 to F3 move.
    for preemphasis in (0.0, 0.97):
        medians = []
        for _, _, formants, f0 in vowel_sets():
            errors, _ = recovery_errors(formants, f0, order, preemphasis=preemphasis, positions=5)
            medians.append(np.nanmedian(np.abs(errors)))
        report(
            "C7",
            f"pre-emphasis {preemphasis}: median over vowels of the median |F1-F3 error| [Hz]",
            np.median(medians),
        )

    # C8: the window's shape and length matter little. Median and worst vowel of the median
    # |F1-F3 error| for a Hamming and a Hann window, 25 and 40 ms long.
    for name, window_function in (("Hamming", hamming), ("Hann", hann)):
        for win_dur in (0.025, 0.040):
            medians = [
                np.nanmedian(
                    np.abs(
                        recovery_errors(
                            formants, f0, order, win_dur, positions=5, window_function=window_function
                        )[0]
                    )
                )
                for _, _, formants, f0 in vowel_sets()
            ]
            report(
                "C8",
                f"{name} {win_dur * 1e3:.0f} ms: median over vowels of the median |F1-F3 error| [Hz]",
                np.median(medians),
            )
            report(
                "C8",
                f"{name} {win_dur * 1e3:.0f} ms: worst vowel's median |F1-F3 error| [Hz]",
                np.max(medians),
            )

    # C9: the all-pole envelope hugs the harmonic peaks. Male "hod" at 120 Hz, 40 ms Hamming:
    # mean of (harmonic level - envelope) in dB over harmonics below 4 kHz, for LPC and for a
    # cepstral lifter at half a period. Also the matching condition mean(P / P_model) = 1.
    signal, f0 = vowel(MALE["hod"] + HIGHER["male"], 120)
    signal = preemphasize(signal)
    length = round(0.040 * FS)
    segment = signal[4000 : 4000 + length] * hamming(length)
    n_fft = 4096
    power = np.abs(np.fft.rfft(segment, n_fft)) ** 2
    a, error_power = lpc(segment, order)
    model = error_power / np.abs(np.fft.rfft(a, n_fft)) ** 2
    full_power = np.r_[power, power[-2:0:-1]]
    full_model = np.r_[model, model[-2:0:-1]]
    report(
        "C9", "mean over all bins of P / P_model (the matching condition)", np.mean(full_power / full_model)
    )
    cepstrum = np.fft.irfft(np.log(power), n_fft)
    cutoff = int(0.5 / f0 * FS)
    lifter = np.zeros(n_fft)
    lifter[: cutoff + 1] = 1
    lifter[n_fft - cutoff :] = 1
    cepstral = np.exp(np.fft.rfft(cepstrum * lifter, n_fft).real)
    freqs = np.arange(n_fft // 2 + 1) * FS / n_fft
    harmonic_bins = [np.argmin(np.abs(freqs - k * f0)) for k in range(1, int(4000 / f0) + 1)]
    peaks = [b - 3 + np.argmax(power[b - 3 : b + 4]) for b in harmonic_bins]
    for name, envelope in (("LPC", model), ("cepstral lifter", cepstral)):
        gap = 10 * np.log10(power[peaks] / envelope[peaks])
        report("C9", f"{name}: mean harmonic level above the envelope [dB]", gap.mean())

    # C10: Praat's recipe (resample to twice a ceiling, 2 n_formants poles, every root kept) on the
    # static vowels: ceiling 5000 Hz male, 5500 Hz female, 20 window positions per vowel.
    for label, vowels, f0 in (("male", MALE, 120), ("female", FEMALE, 220)):
        errors = []
        for f123 in vowels.values():
            signal, true_f0 = vowel((*f123, *HIGHER[label]), f0, duration=0.6)
            times = 0.3 + np.arange(20) / 20 / true_f0
            for freqs, _ in ceiling_candidates(signal, times, CEILINGS[label]):
                errors.append([np.min(np.abs(freqs - target)) for target in f123])
        errors = np.array(errors)
        for index in range(3):
            report("C10", f"{label} F{index + 1}: median |error| [Hz]", np.median(errors[:, index]))
        report("C10", f"{label}: largest |F1-F3 error| [Hz]", errors.max())

    # C11: tracking the utterance with moving formants, F1-F3 every 5 ms, against the formants
    # given to the synthesizer: labeling the k-th candidate as Fk, and the tracker. Gross: missing
    # or more than 10% off. Clean, and with white noise 30 dB below the signal.
    noise_rng = np.random.default_rng(1)
    for label in ("male", "female"):
        signal, truth = utterance(label)
        times = np.arange(0.03, len(signal) / FS - 0.03, 0.005)
        true = truth[:, np.round(times * FS).astype(int)].T
        for noise_name, noise_level in (("clean", 0.0), ("noise -30 dB", 10 ** (-30 / 20))):
            noisy = signal + noise_rng.standard_normal(len(signal)) * np.std(signal) * noise_level
            found = ceiling_candidates(noisy, times, CEILINGS[label])
            for method, estimate in (("by count", by_count(found)), ("tracker", track(found))):
                errors = np.abs(estimate - true)
                gross = np.isnan(errors) | (errors > 0.1 * true)
                report(
                    "C11", f"{label}, {noise_name}, {method}: median |F1-F3 error| [Hz]", np.nanmedian(errors)
                )
                report("C11", f"{label}, {noise_name}, {method}: gross errors, fraction", gross.mean())

    # C12: Burg's method recovers the same formants as the autocorrelation method on the static
    # vowels (root nearest each given formant, order 18, 25 ms Hamming, pre-emphasis).
    for name, method in (("autocorrelation", lpc), ("Burg", burg)):
        for label in ("male", "female"):
            errors = np.concatenate(
                [
                    recovery_errors(formants, f0, order, method=method)[0]
                    for s, _, formants, f0 in vowel_sets()
                    if s == label
                ]
            )
            report("C12", f"{name}, {label}: median |F1-F3 error| [Hz]", np.median(np.abs(errors)))
            report("C12", f"{name}, {label}: largest |F1-F3 error| [Hz]", np.abs(errors).max())


if __name__ == "__main__":
    main()
