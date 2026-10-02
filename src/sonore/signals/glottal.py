"""The Liljencrants-Fant (LF) glottal pulse, and a voiced source made from it.

The LF model (Fant, Liljencrants & Lin, 1985) describes one period of the
glottal flow derivative: an exponentially growing sinusoid while the glottis
opens, a sharp negative peak where it closes, and an exponential return
phase. Fant's (1995) single shape parameter Rd sets the whole pulse, from
tense, pressed voice (small Rd: short open phase, abrupt closure, strong high
harmonics) to lax, breathy voice (large Rd: long open phase, gradual closure,
a dominant fundamental). The design, and the checks behind it, are in
``docs/design/glottal-source.md``.

Names follow the papers, an exception to sonore's descriptive names: ``tp``,
``te``, ``ta`` (times of peak flow, of the main excitation and of the return
phase's time constant, in fractions of a period), ``alpha``, ``epsilon``,
``omega_g`` and ``e0`` (the waveform's constants), and Fant's normalized
parameters ``rd``, ``ra``, ``rg``, ``rk``.
"""

from __future__ import annotations

import numbers
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from sonore.core.sound import Sound
from sonore.signals.generators import F0Contour, _n_max, harmonic_complex

__all__ = ["lf_harmonics", "lf_pulse", "glottal_source"]

#: Fant's (1995) main range of Rd, from tight, adducted to breathy phonation.
RD_RANGE = (0.3, 2.7)

#: Spacing of the Rd table used when Rd changes over time: fine enough that
#: interpolated phases of the first 80 harmonics stay within 0.001 rad.
_RD_STEP = 0.002


def _r_parameters(rd: float) -> tuple[float, float, float]:
    """Fant's (1995) prediction of (Ra, Rg, Rk) from Rd: his Eqs. 2 and 3 for
    Ra and Rk, and Rg from his Eq. 4 given those."""
    ra = (-1 + 4.8 * rd) / 100
    rk = (22.4 + 11.8 * rd) / 100
    denominator = 4 * (0.11 * rd - ra * (0.5 + 1.2 * rk))
    if ra <= 0 or denominator <= 0:
        raise ValueError(f"Rd = {rd:g} gives no LF pulse; Fant's main range is {RD_RANGE[0]}-{RD_RANGE[1]}")
    return ra, rk * (0.5 + 1.2 * rk) / denominator, rk


class _LFShape:
    """One period of the LF flow derivative E(x), x in fractions of the
    period, scaled so that E(te) = -1 (Ee = 1):

    - open phase, 0 <= x <= te: E(x) = e0 exp(alpha x) sin(omega_g x),
      omega_g = pi / tp;
    - return phase, te < x <= 1: E(x) = -(exp(-epsilon (x - te))
      - exp(-epsilon (1 - te))) / (epsilon ta), with
      epsilon ta = 1 - exp(-epsilon (1 - te));
    - alpha set so E integrates to zero over the period, e0 so E is
      continuous at te.

    These are Eqs. 1, 11 and 12 of Fant, Liljencrants & Lin (1985), with the
    period's end tc at x = 1.
    """

    def __init__(self, tp: float, te: float, ta: float):
        if not (0 < tp < te < 1 and 0 < ta < 1 - te):
            raise ValueError(
                f"no LF pulse with tp = {tp:g}, te = {te:g}, ta = {ta:g} "
                "(it needs 0 < tp < te < 1 and 0 < ta < 1 - te)"
            )
        self.tp, self.te, self.ta = tp, te, ta
        self.omega_g = np.pi / tp
        self.epsilon = self._solve_epsilon()
        self.alpha = self._solve_alpha()
        self.e0 = -1.0 / (np.exp(self.alpha * te) * np.sin(self.omega_g * te))

    def _solve_epsilon(self) -> float:
        closed_length = 1 - self.te

        def mismatch(epsilon):
            return epsilon * self.ta - 1 + np.exp(-epsilon * closed_length)

        # 0 at epsilon = 0, negative just above it, positive at 2 / ta
        lower = 1e-3 * (closed_length - self.ta) / closed_length**2
        return brentq(mismatch, lower, 2 / self.ta, xtol=1e-14, rtol=1e-15)

    def _return_area(self) -> float:
        epsilon, closed_length = self.epsilon, 1 - self.te
        tail = np.exp(-epsilon * closed_length)
        return -((1 - tail) / epsilon - closed_length * tail) / (epsilon * self.ta)

    def _open_area(self, alpha: float) -> float:
        omega_g, te = self.omega_g, self.te
        integral = (
            np.exp(alpha * te) * (alpha * np.sin(omega_g * te) - omega_g * np.cos(omega_g * te)) + omega_g
        ) / (alpha**2 + omega_g**2)
        return -integral / (np.exp(alpha * te) * np.sin(omega_g * te))

    def _solve_alpha(self) -> float:
        target = -self._return_area()

        def mismatch(alpha):
            return self._open_area(alpha) - target

        lower, upper = -1.0, 1.0
        while mismatch(lower) * mismatch(upper) > 0:
            lower, upper = 2 * lower, 2 * upper
            if upper > 1e4:
                raise ValueError("no LF pulse with these parameters has zero net flow")
        return brentq(mismatch, lower, upper, xtol=1e-14, rtol=1e-15)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.mod(x, 1.0)
        open_part = self.e0 * np.exp(self.alpha * x) * np.sin(self.omega_g * x)
        return_part = -(np.exp(-self.epsilon * (x - self.te)) - np.exp(-self.epsilon * (1 - self.te))) / (
            self.epsilon * self.ta
        )
        return np.where(x <= self.te, open_part, return_part)

    def flow(self, x: np.ndarray) -> np.ndarray:
        x = np.mod(x, 1.0)
        alpha, omega_g, te, epsilon = self.alpha, self.omega_g, self.te, self.epsilon

        def open_flow(position):
            oscillation = alpha * np.sin(omega_g * position) - omega_g * np.cos(omega_g * position)
            return self.e0 * (np.exp(alpha * position) * oscillation + omega_g) / (alpha**2 + omega_g**2)

        since_te = np.maximum(x - te, 0.0)
        return_flow = open_flow(te) - (
            (1 - np.exp(-epsilon * since_te)) / epsilon - since_te * np.exp(-epsilon * (1 - te))
        ) / (epsilon * self.ta)
        return np.where(x <= te, open_flow(np.minimum(x, te)), return_flow)

    def harmonics(self, numbers: np.ndarray) -> np.ndarray:
        """c_k = integral over one period of E(x) exp(-2 pi i k x) dx, in closed form."""
        beta = 2 * np.pi * np.asarray(numbers, float)
        te, epsilon, omega_g, alpha = self.te, self.epsilon, self.omega_g, self.alpha

        def exp_integral(rate, stop):  # integral of exp(rate x) from 0 to stop
            return (np.exp(rate * stop) - 1) / rate

        rising = exp_integral(alpha + 1j * (omega_g - beta), te)
        falling = exp_integral(alpha - 1j * (omega_g + beta), te)
        open_part = self.e0 * (rising - falling) / 2j
        rate = epsilon + 1j * beta
        decay = np.exp(-1j * beta * te) * (1 - np.exp(-rate * (1 - te))) / rate
        floor = np.exp(-epsilon * (1 - te)) * (np.exp(-1j * beta * te) - np.exp(-1j * beta)) / (1j * beta)
        return open_part - (decay - floor) / (epsilon * self.ta)


@lru_cache(maxsize=4096)
def _shape(rd: float | None, ra: float | None, rg: float | None, rk: float | None) -> _LFShape:
    if rd is not None:
        ra, rg, rk = _r_parameters(rd)
    tp = 1 / (2 * rg)
    return _LFShape(tp, tp * (1 + rk), ra)


def _shape_from(rd, ra, rg, rk) -> _LFShape:
    given = [value is not None for value in (ra, rg, rk)]
    if any(given):
        if not all(given):
            raise ValueError("give all of ra, rg and rk, or rd alone")
        return _shape(None, float(ra), float(rg), float(rk))
    return _shape(float(rd), None, None, None)


def lf_harmonics(
    harmonics: ArrayLike,
    rd: float = 0.7,
    *,
    ra: float | None = None,
    rg: float | None = None,
    rk: float | None = None,
    flow: bool = False,
) -> np.ndarray:
    """Complex Fourier coefficients of the LF pulse, one per harmonic number.

    Harmonic ``k`` of a train of LF pulses is ``2 |c_k| cos(k Phi + angle(c_k))``
    where ``Phi`` is the running phase of F0, so ``c_k`` depends only on ``k``
    and the pulse's shape, not on F0. The coefficients are those of the flow
    derivative scaled to -1 at the main excitation (``Ee = 1``); with ``flow=True``,
    those of the flow itself, ``c_k / (2 pi i k)``, in units of ``Ee * T0``.
    They are exact (a closed form), so a source built from them does not
    alias.

    The shape is set by Fant's (1995) ``rd`` (main range 0.3-2.7; 0.7 is
    close to his typical adult male values), through his prediction of
    ``ra``, ``rg`` and ``rk`` from it; or directly by ``ra = ta/T0``,
    ``rg = T0/(2 tp)`` and ``rk = (te - tp)/tp``, given together.
    """
    numbers = np.asarray(harmonics)
    if np.any(numbers < 1) or np.any(numbers != np.round(numbers)):
        raise ValueError("harmonic numbers must be whole numbers >= 1")
    coefficients = _shape_from(rd, ra, rg, rk).harmonics(numbers)
    return coefficients / (2j * np.pi * numbers) if flow else coefficients


def lf_pulse(
    x: ArrayLike,
    rd: float = 0.7,
    *,
    ra: float | None = None,
    rg: float | None = None,
    rk: float | None = None,
    flow: bool = False,
) -> np.ndarray:
    """The LF flow derivative (or with ``flow=True``, the flow) at times ``x``
    in fractions of a period (wrapped into [0, 1)), scaled to -1 at the main
    excitation ``te`` (``Ee = 1``; for the laxest voices, near Rd 2.7, the
    open phase dips slightly below that just before it).

    Evaluating the formula at sample times aliases, most for tense voices
    with an abrupt closure; use it to draw a period, and
    :func:`glottal_source` to make a sound. The shape is set as in
    :func:`lf_harmonics`.
    """
    shape = _shape_from(rd, ra, rg, rk)
    x = np.asarray(x, float)
    return shape.flow(x) if flow else shape.derivative(x)


def _rd_track_gains(rd_times: np.ndarray, rd_values: np.ndarray, harmonic_numbers: np.ndarray, flow: bool):
    """A gain function ``(t, f, n)`` giving harmonic ``n``'s complex
    coefficient at Rd(t): a table over Rd at ``_RD_STEP`` spacing,
    interpolated in log level and unwrapped phase."""
    low, high = float(rd_values.min()), float(rd_values.max())
    grid = low + _RD_STEP * np.arange(int(np.ceil((high - low) / _RD_STEP)) + 1)
    table = np.array([lf_harmonics(harmonic_numbers, rd, flow=flow) for rd in grid])
    log_levels = np.log(np.abs(table))
    phases = np.unwrap(np.angle(table), axis=0)
    grid_index = np.arange(len(grid))
    column = {int(number): position for position, number in enumerate(harmonic_numbers)}

    def gain(t, freq, number):
        position = (np.interp(t, rd_times, rd_values) - low) / _RD_STEP
        index = column[int(number)]
        level = np.exp(np.interp(position, grid_index, log_levels[:, index]))
        return 2 * level * np.exp(1j * np.interp(position, grid_index, phases[:, index]))

    return gain


def glottal_source(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    rd: float | tuple[ArrayLike, ArrayLike] = 0.7,
    *,
    flow: bool = False,
    f_max: float | None = None,
    ramp: float = 0.005,
) -> Sound:
    """A voiced source of LF glottal pulses on a fixed F0 or an F0 contour.

    The pulses are built from their harmonics (:func:`lf_harmonics`) by
    :func:`harmonic_complex`, so they do not alias, each period takes its
    own length on a moving F0, and ``f0``, ``f_max`` and ``ramp`` work as
    there (unvoiced time windows are silent). The result is the flow derivative,
    the source as it excites the vocal tract with radiation folded in, or
    with ``flow=True`` the flow itself; normalized to RMS 1.

    ``rd`` is Fant's (1995) shape parameter (main range 0.3-2.7, default 0.7,
    close to his typical adult male values): a number, or a ``(times,
    values)`` track for a voice quality that changes, interpolated to every
    sample. LF models the periodic pulse only: breathy voice also needs
    aspiration noise, as in :func:`~sonore.klatt_synthesize`.
    """
    harmonic_numbers = np.arange(1, _n_max(f0, fs, f_max) + 1)
    if harmonic_numbers.size == 0:
        raise ValueError("no harmonic of this F0 fits below f_max")
    contour_options = {"f_max": f_max} if isinstance(f0, numbers.Real) else {"f_max": f_max, "ramp": ramp}
    if isinstance(rd, numbers.Real):
        coefficients = lf_harmonics(harmonic_numbers, rd, flow=flow)
        amplitudes, phases = 2 * np.abs(coefficients), np.angle(coefficients)
        return harmonic_complex(duration, fs, f0, harmonic_numbers, amplitudes, phases, **contour_options)
    try:
        rd_times, rd_values = (np.asarray(part, float) for part in rd)
    except (TypeError, ValueError):
        raise ValueError("rd must be a number or a (times, values) pair") from None
    if rd_times.ndim != 1 or rd_times.shape != rd_values.shape or len(rd_times) == 0:
        raise ValueError("an rd track needs one value per time")
    gains = _rd_track_gains(rd_times, rd_values, harmonic_numbers, flow)
    return harmonic_complex(duration, fs, f0, harmonic_numbers, gains, **contour_options)
