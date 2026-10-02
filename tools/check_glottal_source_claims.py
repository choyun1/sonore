"""Numerical checks for the claims in docs/design/glottal-source.md (C1-C9).

Like the other claim checkers, this is independent of sonore: only NumPy and
SciPy, with every step written out from its formula. It holds a small
prototype of the Liljencrants-Fant (LF) glottal flow derivative (Fant,
Liljencrants & Lin, 1985), its Rd parametrization (Fant, 1995), the
KLGLOTT88-style polynomial pulse, and Klatt's (1980) RGP source, so the
numbers in the document can be reproduced; it is not library code. Each line
prints the claim number and the number that supports it.

    python tools/check_glottal_source_claims.py

It runs in a few seconds.

Time is in fractions of a period throughout: one glottal cycle runs from
x = 0 (the glottis starts to open) to x = 1, so a shape is independent of
F0. The LF symbols tp, te, ta, alpha, epsilon, omega_g, E0, Ee and Fant's
Ra, Rg, Rk, Rd are the literature's names, kept so the code can be read
against the papers.
"""

from functools import partial

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.signal import lfilter

FS = 16000.0

# Vowel /a/ formants and bandwidths, as in tools/check_klatt_claims.py.
FORMANTS = [(730, 60), (1090, 100), (2440, 120), (3400, 175), (4500, 250)]


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


# ------------------------------------------------------------- the LF model
class LFShape:
    """One period of the LF flow derivative E(x), with Ee = 1 (the negative
    peak at x = te is -1):

        open phase    E(x) = E0 exp(alpha x) sin(omega_g x),         0 <= x <= te
        return phase  E(x) = -(exp(-epsilon (x - te)) - exp(-epsilon (1 - te)))
                             / (epsilon ta),                         te < x <= 1

    with omega_g = pi / tp (the flow peaks at tp), epsilon fixed by
    epsilon ta = 1 - exp(-epsilon (1 - te)) (the return phase starts with
    slope 1/ta), E0 by continuity at te, and alpha by zero net flow over the
    cycle (the flow starts and ends at zero). This is the form VOICEBOX's
    v_glotlf implements, which notes that the return-phase equation printed
    in Fig. 2 of Fant, Liljencrants & Lin (1985) has an error.
    """

    def __init__(self, tp, te, ta):
        if not 0 < tp < te < 1 or not 0 < ta < 1 - te:
            raise ValueError(f"infeasible LF timing tp={tp}, te={te}, ta={ta}")
        self.tp, self.te, self.ta = tp, te, ta
        self.omega_g = np.pi / tp
        self.epsilon = self._solve_epsilon()
        self.alpha = self._solve_alpha()
        self.E0 = -1.0 / (np.exp(self.alpha * te) * np.sin(self.omega_g * te))

    def _solve_epsilon(self):
        closed_length = 1 - self.te

        def mismatch(epsilon):
            return epsilon * self.ta - 1 + np.exp(-epsilon * closed_length)

        # mismatch is 0 at epsilon = 0, dips below 0, and is positive at 2/ta.
        lower = 1e-3 * (closed_length - self.ta) / closed_length**2
        return brentq(mismatch, lower, 2 / self.ta, xtol=1e-14, rtol=1e-15)

    def return_area(self):
        """Integral of the return phase (negative)."""
        epsilon, closed_length = self.epsilon, 1 - self.te
        tail = np.exp(-epsilon * closed_length)
        return -((1 - tail) / epsilon - closed_length * tail) / (epsilon * self.ta)

    def open_area(self, alpha):
        """Integral of the open phase for a given alpha, with E0 set so E(te) = -1."""
        omega_g, te = self.omega_g, self.te
        integral = (
            np.exp(alpha * te) * (alpha * np.sin(omega_g * te) - omega_g * np.cos(omega_g * te)) + omega_g
        ) / (alpha**2 + omega_g**2)
        return -integral / (np.exp(alpha * te) * np.sin(omega_g * te))

    def _solve_alpha(self):
        target = -self.return_area()

        def mismatch(alpha):
            return self.open_area(alpha) - target

        lower, upper = -1.0, 1.0
        while mismatch(lower) * mismatch(upper) > 0:
            lower, upper = 2 * lower, 2 * upper
            if upper > 1e4:
                raise ValueError("no alpha gives zero net flow")
        return brentq(mismatch, lower, upper, xtol=1e-14, rtol=1e-15)

    def derivative(self, x):
        """E(x), the flow derivative, for x in [0, 1) (wrapped to one period)."""
        x = np.mod(np.asarray(x, float), 1.0)
        open_part = self.E0 * np.exp(self.alpha * x) * np.sin(self.omega_g * x)
        return_part = -(np.exp(-self.epsilon * (x - self.te)) - np.exp(-self.epsilon * (1 - self.te))) / (
            self.epsilon * self.ta
        )
        return np.where(x <= self.te, open_part, return_part)

    def harmonics(self, numbers):
        """Complex Fourier coefficients c_k = integral over one period of
        E(x) exp(-2 pi i k x) dx, in closed form. The periodic E is
        sum_k c_k exp(2 pi i k x), so harmonic k has amplitude 2|c_k| and
        phase angle(c_k) in a cos(k Phi + phase) sum."""
        beta = 2 * np.pi * np.asarray(numbers, float)
        te, epsilon, omega_g, alpha = self.te, self.epsilon, self.omega_g, self.alpha

        def exp_integral(rate, start, stop):  # integral of exp(rate x) from start to stop
            return (np.exp(rate * stop) - np.exp(rate * start)) / rate

        rising = exp_integral(alpha + 1j * (omega_g - beta), 0, te)
        falling = exp_integral(alpha - 1j * (omega_g + beta), 0, te)
        open_part = self.E0 * (rising - falling) / 2j
        # integral of exp(-epsilon (x - te)) exp(-i beta x) from te to 1, kept finite for large epsilon
        rate = epsilon + 1j * beta
        decay = np.exp(-1j * beta * te) * (1 - np.exp(-rate * (1 - te))) / rate
        floor = np.exp(-epsilon * (1 - te)) * exp_integral(-1j * beta, te, 1)
        return open_part - (decay - floor) / (epsilon * self.ta)

    def flow_peak(self):
        """U0, the peak of the flow (the integral of E), at x = tp."""
        return quad(self.derivative, 0, self.tp, epsabs=1e-13, epsrel=1e-13, limit=200)[0]


def rd_to_r_parameters(rd):
    """Fant's (1995) prediction of Ra, Rk, Rg from Rd. The Ra and Rk lines
    and the Rd equation they are solved against are as widely quoted
    (e.g. Degottex's implementations); they could not be read in the paper's
    extracted text, see the design document."""
    ra = (-1 + 4.8 * rd) / 100
    rk = (22.4 + 11.8 * rd) / 100
    # Rd = (1 / 0.11) (0.5 + 1.2 Rk) (Rk / (4 Rg) + Ra), solved for Rg
    rg = rk * (0.5 + 1.2 * rk) / (4 * (0.11 * rd - ra * (0.5 + 1.2 * rk)))
    return ra, rg, rk


def lf_from_rd(rd):
    """Ra = ta/T0, Rg = T0/(2 tp), Rk = (te - tp)/tp, in fractions of a period."""
    ra, rg, rk = rd_to_r_parameters(rd)
    tp = 1 / (2 * rg)
    return LFShape(tp, tp * (1 + rk), ra)


# ------------------------------------------------------ KLGLOTT88-style pulse
def polynomial_pulse_harmonics(open_quotient, numbers):
    """Flow U(x) = (x/OQ)^2 - (x/OQ)^3 while open (0 <= x <= OQ), 0 when
    closed: the shape Praat's manual attributes to Rosenberg (1971) and to
    the KLSYN88 source (Klatt & Klatt, 1990). Returns the closed-form Fourier
    coefficients of its derivative E(x) = 2x/OQ^2 - 3x^2/OQ^3, which jumps
    from -1/OQ to 0 at closure."""
    beta = 2 * np.pi * np.asarray(numbers, float)
    length = open_quotient
    # moments J_n = integral_0^L x^n exp(-i beta x) dx, by recursion
    edge = np.exp(-1j * beta * length)
    j0 = (1 - edge) / (1j * beta)
    j1 = (-length * edge + j0) / (1j * beta)
    j2 = (-(length**2) * edge + 2 * j1) / (1j * beta)
    return 2 / open_quotient**2 * j1 - 3 / open_quotient**3 * j2


def polynomial_pulse_derivative(open_quotient, x):
    x = np.mod(x, 1.0)
    return np.where(x < open_quotient, 2 * x / open_quotient**2 - 3 * x**2 / open_quotient**3, 0.0)


def tilt_lowpass_gain(tilt_db, f, fs=FS):
    """The one-pole low-pass Praat's KlattGrid uses for spectral tilt:
    y[n] = a x[n] + b y[n-1], unit gain at 0 Hz, tilt_db down at 3 kHz."""
    attenuation = 10 ** (-tilt_db / 10)
    q = (1 - attenuation * np.cos(2 * np.pi * 3000 / fs)) / (1 - attenuation)
    b = q - np.sqrt(q * q - 1)
    a = 1 - b
    return a / np.sqrt(1 - 2 * b * np.cos(2 * np.pi * np.asarray(f) / fs) + b * b)


# ------------------------------------------------------ Klatt's RGP source
def resonator_gain(f_res, bw, f, fs=FS):
    """|H(f)| of Klatt's (1980) resonator, A = 1 - B - C."""
    period = 1 / fs
    c = -np.exp(-2 * np.pi * bw * period)
    b = 2 * np.exp(-np.pi * bw * period) * np.cos(2 * np.pi * f_res * period)
    z_inv = np.exp(-2j * np.pi * np.asarray(f, float) / fs)
    return np.abs((1 - b - c) / (1 - b * z_inv - c * z_inv**2))


def first_difference_gain(f, fs=FS):
    return np.abs(1 - np.exp(-2j * np.pi * np.asarray(f, float) / fs))


def rgp_at_lips(f0, numbers, fs=FS):
    """Harmonic levels of sonore's current voiced source: impulses through
    RGP (0 Hz, 100 Hz wide), then the radiation difference."""
    f = f0 * np.asarray(numbers, float)
    return resonator_gain(0.0, 100.0, f, fs) * first_difference_gain(f, fs)


def lf_at_lips(shape, f0, numbers, fs=FS):
    """LF harmonic levels at the lips, built the way klatt_synthesize builds
    its source: the flow U (coefficients c_k / (2 pi i k)) through the same
    radiation difference."""
    numbers = np.asarray(numbers, float)
    flow = np.abs(shape.harmonics(numbers) / (2j * np.pi * numbers))
    return flow * first_difference_gain(f0 * numbers, fs)


def db(x):
    return 20 * np.log10(np.abs(x))


def octave_slope(levels_db, numbers, low, high):
    """Least-squares slope in dB per octave of harmonic levels between harmonic numbers low and high."""
    keep = (numbers >= low) & (numbers <= high)
    return np.polyfit(np.log2(numbers[keep]), levels_db[keep], 1)[0]


# ---------------------------------------------------------------- the claims
def fourier_integral(function, number, start, stop, breakpoint=None):
    """integral of function(x) exp(-2 pi i number x) dx from start to stop, numerically."""
    points = None if breakpoint is None else [breakpoint]

    def real_part(x):
        return function(x) * np.cos(2 * np.pi * number * x)

    def imaginary_part(x):
        return -function(x) * np.sin(2 * np.pi * number * x)

    options = {"points": points, "limit": 400, "epsabs": 1e-13}
    return quad(real_part, start, stop, **options)[0] + 1j * quad(imaginary_part, start, stop, **options)[0]


def claim_closed_form():
    """C1: the closed-form harmonics equal numerical integration, and the
    flow returns to zero."""
    worst = 0.0
    for rd in (0.3, 1.0, 2.7):
        shape = lf_from_rd(rd)
        numbers = np.array([1, 2, 3, 10, 37])
        closed = shape.harmonics(numbers)
        for number, value in zip(numbers, closed, strict=True):
            numeric = fourier_integral(shape.derivative, number, 0, 1, breakpoint=shape.te)
            worst = max(worst, abs(value - numeric) / abs(value))
        net_flow = abs(shape.open_area(shape.alpha) + shape.return_area())
    report("C1", "LF harmonics: closed form vs numerical integral, worst relative error", worst)
    report("C1", "LF net flow over one cycle (Rd 2.7; should be 0)", net_flow)
    worst = 0.0
    for open_quotient in (0.4, 0.7):
        numbers = np.array([1, 2, 5, 23])
        closed = polynomial_pulse_harmonics(open_quotient, numbers)
        pulse = partial(polynomial_pulse_derivative, open_quotient)
        numeric = np.array([fourier_integral(pulse, number, 0, open_quotient) for number in numbers])
        worst = max(worst, np.max(np.abs(closed - numeric) / np.abs(closed)))
    report("C1", "KLGLOTT88-style pulse: closed form vs numerical integral, worst relative error", worst)


def claim_rd_mapping():
    """C2: the Rd prediction gives feasible pulses whose true Rd is close to the request."""
    worst = 0.0
    for rd in np.arange(0.3, 2.71, 0.1):
        shape = lf_from_rd(rd)
        true_rd = shape.flow_peak() / 0.11  # U0 / (Ee T0) / 0.11, with Ee = 1 and T0 = 1
        worst = max(worst, abs(true_rd - rd) / rd)
    report("C2", "Rd 0.3-2.7: worst relative error of the pulse's own Rd = U0 F0/(0.11 Ee)", worst)
    for rd in (0.5, 1.0, 2.5):
        report("C2", f"Rd {rd}: the pulse's own Rd", lf_from_rd(rd).flow_peak() / 0.11)
    for rd in (0.3, 1.0, 2.7):
        shape = lf_from_rd(rd)
        report("C2", f"Rd {rd}: open quotient te + ta (approx.), share of the period", shape.te + shape.ta)


def claim_aliasing():
    """C3: sampling the LF waveform aliases; the harmonic sum does not."""
    samples_per_period = 80  # 200 Hz at 16 kHz, a whole number to isolate aliasing
    x = np.arange(samples_per_period) / samples_per_period
    numbers = np.arange(1, samples_per_period // 2)  # below Nyquist
    for rd in (0.3, 1.0, 2.7):
        shape = lf_from_rd(rd)
        sampled = shape.derivative(x)
        coefficients = shape.harmonics(numbers)
        band_limited = 2 * np.real(coefficients[None, :] * np.exp(2j * np.pi * np.outer(x, numbers))).sum(1)
        error = 10 * np.log10(np.sum((sampled - band_limited) ** 2) / np.sum(band_limited**2))
        report("C3", f"Rd {rd}, 200 Hz at 16 kHz: sampled LF minus harmonic sum, dB re signal", error)
    sampled = polynomial_pulse_derivative(0.6, x)
    coefficients = polynomial_pulse_harmonics(0.6, numbers)
    band_limited = 2 * np.real(coefficients[None, :] * np.exp(2j * np.pi * np.outer(x, numbers))).sum(1)
    error = 10 * np.log10(np.sum((sampled - band_limited) ** 2) / np.sum(band_limited**2))
    report("C3", "KLGLOTT88-style pulse, OQ 0.6, same test, dB re signal", error)


def claim_slopes():
    """C4: high-frequency slopes of the flow derivative."""
    numbers = np.arange(1, 400, dtype=float)
    abrupt = LFShape(0.45, 0.6, 0.0005)  # Fa = F0/(2 pi Ra) at harmonic 318
    levels = db(abrupt.harmonics(numbers))
    report(
        "C4",
        "LF Ra 0.0005 (near-abrupt closure): slope, harmonics 8-64, dB/octave",
        octave_slope(levels, numbers, 8, 64),
    )
    for ra in (0.01, 0.05):
        shape = LFShape(0.45, 0.6, ra)
        levels = db(shape.harmonics(numbers))
        corner = 1 / (2 * np.pi * ra)
        report(
            "C4",
            f"LF Ra {ra} (Fa at harmonic {corner:.1f}): slope, harmonics 64-399, dB/octave",
            octave_slope(levels, numbers, 64, 399),
        )
    # Fant (1986): the return phase acts as a first-order low-pass,
    # delta L = -10 log10(1 + (2 pi ta f)^2). T0 = 10 ms, tp 4 ms, te 5 ms.
    period_ms = 10.0
    reference = LFShape(0.4, 0.5, 1e-5)
    low_numbers = np.arange(5, 41, dtype=float)  # 500 Hz to 4 kHz at 100 Hz
    for ta_ms in (0.15, 0.6):
        shape = LFShape(0.4, 0.5, ta_ms / period_ms)
        change = db(shape.harmonics(low_numbers)) - db(reference.harmonics(low_numbers))
        frequencies = low_numbers * 1000 / period_ms
        predicted = -10 * np.log10(1 + (2 * np.pi * ta_ms / 1000 * frequencies) ** 2)
        mismatch = change - predicted
        report(
            "C4",
            f"ta {ta_ms} ms vs abrupt, 0.5-4 kHz: spread of (change - Fant's low-pass), dB",
            np.ptp(mismatch),
        )
        report("C4", f"ta {ta_ms} ms: mean offset of that difference (overall level), dB", np.mean(mismatch))
    levels = db(polynomial_pulse_harmonics(0.6, numbers))
    report(
        "C4",
        "KLGLOTT88-style pulse OQ 0.6: slope, harmonics 50-399, dB/octave",
        octave_slope(levels, numbers, 50, 399),
    )
    rgp_levels = db(rgp_at_lips(100.0, numbers[:79]))
    report(
        "C4",
        "current RGP source at the lips, F0 100 Hz: slope 400-3200 Hz, dB/octave",
        octave_slope(rgp_levels, numbers[:79], 4, 32),
    )
    for rd in (0.5, 1.0, 2.5):
        lips = db(lf_at_lips(lf_from_rd(rd), 100.0, numbers[:79]))
        report(
            "C4",
            f"LF Rd {rd} at the lips, F0 100 Hz: slope 400-3200 Hz, dB/octave",
            octave_slope(lips, numbers[:79], 4, 32),
        )


def claim_h1_h2():
    """C5: H1-H2 rises with Rd; the current source's depends on F0 instead."""
    rds = np.arange(0.3, 2.71, 0.1)
    h1_h2 = []
    for rd in rds:
        shape = lf_from_rd(rd)
        first, second = np.abs(shape.harmonics([1, 2]))
        h1_h2.append(db(first / second))
    h1_h2 = np.array(h1_h2)
    for rd in (0.3, 1.0, 2.7):
        report("C5", f"LF Rd {rd}: H1-H2 of the flow derivative, dB", h1_h2[np.argmin(abs(rds - rd))])
    for rd in (0.3, 1.0, 2.7):
        report("C5", f"Fant's (1995) reported line -7.6 + 11.1 Rd at Rd {rd}, dB", -7.6 + 11.1 * rd)
    slope, intercept = np.polyfit(rds, h1_h2, 1)
    report("C5", "LF: fitted line H1-H2 = a + b Rd over Rd 0.3-2.7, intercept a (dB)", intercept)
    report("C5", "LF: fitted line, slope b (dB per unit Rd)", slope)
    report("C5", "LF: worst deviation from that line, dB", np.max(np.abs(h1_h2 - (intercept + slope * rds))))
    for f0 in (100.0, 200.0):
        first, second = rgp_at_lips(f0, [1, 2])
        report("C5", f"current RGP source at the lips, F0 {f0:.0f} Hz: H1-H2, dB", db(first / second))
    for open_quotient in (0.4, 0.7):
        first, second = np.abs(polynomial_pulse_harmonics(open_quotient, [1, 2]))
        report("C5", f"KLGLOTT88-style pulse OQ {open_quotient}: H1-H2, dB", db(first / second))


def claim_levels():
    """C6: where the current source sits among LF pulses: harmonic levels re H1 at the lips."""
    f0 = 100.0
    numbers = np.array([1, 2, 5, 10, 20, 30])
    sources = {"current RGP": rgp_at_lips(f0, numbers)}
    for rd in (0.5, 1.0, 2.5):
        sources[f"LF Rd {rd}"] = lf_at_lips(lf_from_rd(rd), f0, numbers)
    for name, levels in sources.items():
        relative = db(levels / levels[0])
        for number, level in zip(numbers[1:], relative[1:], strict=True):
            report("C6", f"F0 100 Hz, {name}: harmonic {number} ({number * f0:.0f} Hz) re H1, dB", level)


def claim_harmonic_source():
    """C7: the LF pulse train is a harmonic sum with fixed amplitudes and
    phases, so a fixed voice quality needs no change to harmonic_complex."""
    f0 = 125.0
    samples_per_period = int(FS / f0)
    length = 4 * samples_per_period
    t = np.arange(length) / FS
    shape = lf_from_rd(1.0)
    numbers = np.arange(1, int(0.45 * FS / f0) + 1)
    coefficients = shape.harmonics(numbers)
    amplitudes, phases = 2 * np.abs(coefficients), np.angle(coefficients)
    harmonic_sum = np.sum(amplitudes * np.cos(2 * np.pi * f0 * np.outer(t, numbers) + phases), axis=1)
    complex_sum = 2 * np.real(np.exp(2j * np.pi * f0 * np.outer(t, numbers)) @ coefficients)
    report(
        "C7",
        "Rd 1, 125 Hz: amplitude/phase cosine sum vs complex Fourier sum, max difference",
        np.max(np.abs(harmonic_sum - complex_sum)),
    )
    # Radiation: klatt_synthesize differences the flow; the LF E is the true derivative.
    f_max = 0.45 * FS
    report(
        "C7",
        "first difference re the derivative 2 pi f/fs at 0.45 fs, dB",
        db(first_difference_gain(f_max) / (2 * np.pi * f_max / FS)),
    )


def claim_interpolation():
    """C8: a time-varying Rd: exact harmonics per Rd are cheap, and a table
    interpolated in level and unwrapped phase is close."""
    import time

    numbers = np.arange(1, 81)
    start_time = time.perf_counter()
    for rd in np.linspace(0.3, 2.7, 200):
        lf_from_rd(rd).harmonics(numbers)
    report(
        "C8",
        "exact LF shape and 80 harmonics for one Rd value, milliseconds",
        (time.perf_counter() - start_time) / 200 * 1e3,
    )
    for step in (0.01, 0.002):
        grid = np.arange(0.3, 2.7 + step / 2, step)
        table = np.array([lf_from_rd(rd).harmonics(numbers) for rd in grid])
        levels = db(table)
        phases = np.unwrap(np.angle(table), axis=0)
        worst_db = worst_phase = 0.0
        for rd in np.arange(0.3 + step / 2, 2.7 - step, 0.0373):
            index = int((rd - grid[0]) / step)
            weight = (rd - grid[index]) / step
            level = (1 - weight) * levels[index] + weight * levels[index + 1]
            phase = (1 - weight) * phases[index] + weight * phases[index + 1]
            exact = lf_from_rd(rd).harmonics(numbers)
            strong = db(exact) > db(np.abs(exact).max()) - 40
            worst_db = max(worst_db, np.max(np.abs(level[strong] - db(exact[strong]))))
            worst_phase = max(
                worst_phase, np.max(np.abs(np.angle(np.exp(1j * phase[strong]) / exact[strong])))
            )
        report(
            "C8",
            f"Rd table at {step} steps, harmonics 1-80 within 40 dB of the strongest: worst level error, dB",
            worst_db,
        )
        report("C8", f"Rd table at {step} steps: worst phase error, radians", worst_phase)


def claim_phase():
    """C9: LF phases make a peakier source than the same levels in cosine phase? Crest factors."""
    f0 = 100.0
    length = int(FS * 0.2)
    t = np.arange(length) / FS
    numbers = np.arange(1, int(0.45 * FS / f0) + 1)
    shape = lf_from_rd(1.0)
    coefficients = shape.harmonics(numbers)
    amplitudes = 2 * np.abs(coefficients)

    def crest_db(signal):
        return db(np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)))

    def vowel(signal):
        for f, bw in FORMANTS:
            period = 1 / FS
            c = -np.exp(-2 * np.pi * bw * period)
            b = 2 * np.exp(-np.pi * bw * period) * np.cos(2 * np.pi * f * period)
            signal = lfilter([1 - b - c], [1, -b, -c], signal)
        return signal[length // 2 :]

    lf_phase = np.sum(
        amplitudes * np.cos(2 * np.pi * f0 * np.outer(t, numbers) + np.angle(coefficients)), axis=1
    )
    cosine_phase = np.sum(amplitudes * np.cos(2 * np.pi * f0 * np.outer(t, numbers)), axis=1)
    report("C9", "Rd 1, 100 Hz source: crest factor with LF phases, dB", crest_db(lf_phase))
    report("C9", "same levels, all cosine phase (as the current source), dB", crest_db(cosine_phase))
    report("C9", "through /a/ formants: crest factor with LF phases, dB", crest_db(vowel(lf_phase)))
    report("C9", "through /a/ formants: crest factor, cosine phase, dB", crest_db(vowel(cosine_phase)))


def claim_tilt_filter():
    """Praat's spectral-tilt low-pass is down by exactly TL at 3 kHz (used in C4's discussion)."""
    report(
        "C4",
        "KlattGrid tilt low-pass, TL 20 dB at 16 kHz: gain at 3 kHz, dB",
        db(tilt_lowpass_gain(20.0, 3000.0)),
    )


if __name__ == "__main__":
    claim_closed_form()
    claim_rd_mapping()
    claim_aliasing()
    claim_slopes()
    claim_tilt_filter()
    claim_h1_h2()
    claim_levels()
    claim_harmonic_source()
    claim_interpolation()
    claim_phase()
