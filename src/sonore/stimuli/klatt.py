"""A Klatt-style cascade/parallel formant synthesizer.

Speech from named acoustic parameters, after Klatt (1980): a voiced source
and two noise sources, formant resonators in cascade (voicing and
aspiration) and in parallel (frication), and radiation from the lips. Every
control is something visible on a spectrogram, so the source-filter model
can be taken apart one parameter at a time, and stimuli such as vowel and
/ba/-/da/-/ga/ continua are set exactly. The design, and the checks behind
it, are in ``docs/design/klatt.md``.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import as_rng, n_samples, time_axis
from sonore.signals.generators import RNG, harmonic_complex
from sonore.signals.processing import (
    Track,
    _check_resonance,
    _resonator_coefs,
    _track,
    antiresonator,
    resonator,
)

__all__ = ["KLATT_DEFAULTS", "klatt_synthesize", "klatt_continuum"]

#: Every parameter :func:`klatt_synthesize` takes, with its default: a
#: neutral adult male vowel, voiced at 100 Hz, with no noise. Frequencies and
#: bandwidths are in Hz, amplitudes in dB (60 = the source at unit level, 0
#: or less = off).
KLATT_DEFAULTS = MappingProxyType(
    {
        "F0": 100.0,  # fundamental frequency; 0 = unvoiced
        "AV": 60.0,  # voicing amplitude
        "AH": 0.0,  # aspiration amplitude (through the cascade)
        "AF": 0.0,  # frication amplitude (through the parallel branch)
        "AB": 0.0,  # frication bypassing the formants (flat)
        "F1": 500.0,
        "B1": 60.0,
        "F2": 1500.0,
        "B2": 90.0,
        "F3": 2500.0,
        "B3": 150.0,
        "F4": 3500.0,
        "B4": 200.0,
        "F5": 4500.0,
        "B5": 250.0,
        "F6": 4900.0,  # parallel branch only
        "B6": 1000.0,
        "A1": 0.0,  # parallel formant amplitudes
        "A2": 0.0,
        "A3": 0.0,
        "A4": 0.0,
        "A5": 0.0,
        "A6": 0.0,
        "FNP": 250.0,  # nasal pole and zero; equal values cancel (no nasality)
        "BNP": 100.0,
        "FNZ": 250.0,
        "BNZ": 100.0,
    }
)

_CASCADE = (1, 2, 3, 4, 5)
_PARALLEL = (1, 2, 3, 4, 5, 6)


def _gain(level_db):
    """dB (60 = unit) to a linear gain; 0 dB or less is silence."""
    level_db = np.asarray(level_db, float)
    return np.where(level_db > 0, 10 ** ((level_db - 60) / 20), 0.0)


def _resonator_gain(f_res, bw, f, fs: float):
    """|H(f)| of Klatt's resonator at ``f_res`` with bandwidth ``bw``."""
    a, b, c = _resonator_coefs(f_res, bw, fs)
    z = np.exp(-2j * np.pi * np.asarray(f, float) / fs)
    return np.abs(a / (1 - b * z - c * z**2))


def _peak_gain(f, bw, fs: float, difference: bool):
    """A resonator's gain at its own frequency, times the first difference's
    gain there when its input is differenced."""
    g = _resonator_gain(f, bw, f, fs)
    return g * np.abs(1 - np.exp(-2j * np.pi * np.asarray(f, float) / fs)) if difference else g


def _rgp(f, fs: float):
    """Klatt's glottal low-pass RGP (a resonator at 0 Hz, 100 Hz wide): the
    harmonic amplitudes of an impulse train shaped by it, falling about
    12 dB per octave above 50 Hz."""
    return _resonator_gain(0.0, 100.0, f, fs)


def klatt_synthesize(
    duration: float,
    fs: float,
    params: Mapping[str, Track] | None = None,
    *,
    rng: RNG = None,
    **kwargs: Track,
) -> Sound:
    """Speech from Klatt's (1980) parameters, as a cascade/parallel formant synthesizer.

    Parameters are given by Klatt's names, in ``params`` (a mapping, such as
    one made by :func:`klatt_continuum`) or as keyword arguments, which take
    precedence; anything not given takes its value in :data:`KLATT_DEFAULTS`.
    Each is a number, or a ``(times, values)`` pair: times in seconds,
    interpolated linearly to every sample and held beyond the ends. A table
    of values every 5 ms, as Klatt used, is the pair
    ``(0.005 * np.arange(len(values)), values)``. ``F0`` may also be an
    :class:`~sonore.F0Track`, to resynthesize a measured pitch contour.

    - **Sources.** Voicing: harmonics of ``F0`` (:func:`harmonic_complex`, so
      the pitch can glide with no phase jumps or aliasing) with the spectrum
      of Klatt's glottal low-pass, falling about 12 dB per octave, at level
      ``AV``. Where ``F0`` is 0 the harmonics are switched off. Aspiration
      (``AH``) and frication (``AF``, ``AB``): white Gaussian noise. While
      voicing is on, both noises are amplitude modulated by a square wave at
      ``F0``, 50% deep, as in Klatt's synthesizer.
    - **Cascade branch.** Voicing and aspiration pass through the nasal pole
      and zero (``FNP``, ``BNP``, ``FNZ``, ``BNZ``) and formants 1-5
      (``F1``-``F5``, bandwidths ``B1``-``B5``) in series: formant levels
      follow from the frequencies, as in a vowel. A formant default at or
      above Nyquist is left out (``F5`` at ``fs`` below 9 kHz).
    - **Parallel branch.** Frication passes through formants 1-6 side by
      side, each at its own level ``A1``-``A6`` (dB at the formant's peak),
      added with alternating signs so that they sum like the cascade between
      peaks; formants 2-6 get a differenced input, as in Klatt's, to keep
      low frequencies out. ``AB`` adds the frication with no formants.
    - **Radiation.** The voiced source is differenced (+6 dB per octave), the
      radiation from the lips; the noises are white as they leave the lips.

    Amplitudes are in dB: at 60 a source has RMS 1 as it leaves the lips
    (before the formants), so equal values mean equal source levels; each
    20 dB is a factor of 10, and 0 or less is off. The result is normalized
    to RMS 1, like every generator; relative levels within one call are
    kept. The voiced source
    is Klatt's impulses through his glottal low-pass, made from harmonics;
    his quasi-sinusoidal voicing (``AVS``) and later glottal pulse models are
    not included.

    ``rng`` seeds the noise, so a call can be repeated exactly.
    """
    p = dict(KLATT_DEFAULTS)
    given = dict(params or {}) | kwargs
    unknown = sorted(set(given) - set(p))
    if unknown:
        raise ValueError(f"unknown Klatt parameter(s) {unknown}; the names are {sorted(p)}")
    p.update(given)
    if hasattr(p["F0"], "f0"):  # an F0Track or any F0 contour object
        p["F0"] = (np.asarray(p["F0"].t, float), np.ravel(p["F0"].f0))
    n = n_samples(duration, fs)
    t = time_axis(n, fs)
    rng = as_rng(rng)

    def track(name):
        return np.broadcast_to(_track(p[name], t, name), t.shape)

    def formant(k):
        f, bw = _track(p[f"F{k}"], t, f"F{k}"), _track(p[f"B{k}"], t, f"B{k}")
        if f"F{k}" not in given and np.any(np.asarray(f) >= fs / 2):
            return None  # a default formant that doesn't fit at this rate
        _check_resonance(f, bw, fs)
        return p[f"F{k}"], p[f"B{k}"]

    # --- sources
    f0 = track("F0")
    if np.any(f0 < 0):
        raise ValueError("F0 must be >= 0 (0 where unvoiced)")
    av = _gain(track("AV"))
    contour = ([0.0], [p["F0"]]) if isinstance(p["F0"], numbers.Real) else p["F0"]
    voice = np.zeros(n)
    if np.any(av > 0) and np.any(f0 > 0):
        h = harmonic_complex(duration, fs, contour, amplitudes=lambda _, f: _rgp(f, fs)).data[:, 0]
        # Radiation from the lips, a first difference; then RMS 1 where voiced.
        lips = np.diff(h, prepend=0.0)
        voice = av * lips / np.sqrt(np.mean(lips[h != 0] ** 2))
    # Noise modulated at F0 while voiced: full level in the first half of
    # each period, half in the second.
    phase = np.concatenate([[0.0], np.cumsum((f0[1:] + f0[:-1]) / 2)]) / fs
    voiced = (f0 > 0) & (av > 0)
    modulation = np.where(voiced & (np.mod(phase, 1.0) >= 0.5), 0.5, 1.0)
    noise = rng.standard_normal(n) * modulation

    # --- cascade: voicing and aspiration
    x = Sound(voice + _gain(track("AH")) * noise, fs)
    x = resonator(x, p["FNP"], p["BNP"])
    x = antiresonator(x, p["FNZ"], p["BNZ"])
    for k in _CASCADE:
        fb = formant(k)
        if fb is not None:
            x = resonator(x, *fb)
    out = x.data[:, 0]

    # --- parallel: frication
    fric = _gain(track("AF")) * noise
    if np.any(fric):
        differenced = np.diff(fric, prepend=0.0)
        for k in _PARALLEL:
            a_k = _gain(track(f"A{k}"))
            if not np.any(a_k > 0):
                continue
            fb = formant(k)
            if fb is None:
                raise ValueError(f"A{k} is on, but F{k} is at or above Nyquist ({fs / 2:g} Hz)")
            src = fric if k == 1 else differenced
            y = resonator(Sound(src, fs), *fb).data[:, 0]
            peak = _peak_gain(track(f"F{k}"), track(f"B{k}"), fs, difference=k > 1)
            out = out + (-1) ** (k + 1) * a_k / peak * y
        out = out + _gain(track("AB")) * fric

    s = Sound(out, fs)
    return s.normalize() if s.rms > 0 else s


def klatt_continuum(
    start: Mapping[str, Track], end: Mapping[str, Track], steps: int
) -> list[dict[str, Track]]:
    """``steps`` parameter sets for :func:`klatt_synthesize`, evenly spaced from
    ``start`` to ``end`` (both included), such as a /ba/-/da/ continuum.

    A parameter given in only one of them takes its :data:`KLATT_DEFAULTS`
    value in the other. Numbers are interpolated directly; ``(times,
    values)`` pairs are interpolated value by value, so they must share their
    times (a number paired with a track is held at every one of its times).
    """
    if steps < 2:
        raise ValueError("a continuum needs at least 2 steps")
    out = [{} for _ in range(steps)]
    for name in sorted(set(start) | set(end)):
        if name not in KLATT_DEFAULTS:
            raise ValueError(f"unknown Klatt parameter {name!r}")
        a = start.get(name, KLATT_DEFAULTS[name])
        b = end.get(name, KLATT_DEFAULTS[name])
        ta, va = _as_points(a)
        tb, vb = _as_points(b)
        if ta is None and tb is None:
            for i, w in enumerate(np.linspace(0, 1, steps)):
                out[i][name] = float((1 - w) * va + w * vb)
            continue
        times = ta if ta is not None else tb
        if ta is not None and tb is not None and not np.array_equal(ta, tb):
            raise ValueError(f"{name}: the two tracks must share their times to be interpolated")
        va, vb = np.broadcast_to(va, times.shape), np.broadcast_to(vb, times.shape)
        for i, w in enumerate(np.linspace(0, 1, steps)):
            out[i][name] = (times.copy(), (1 - w) * va + w * vb)
    return out


def _as_points(value: Track) -> tuple[np.ndarray | None, np.ndarray | float]:
    if isinstance(value, numbers.Real):
        return None, float(value)
    try:
        times, values = (np.asarray(v, float) for v in value)
    except (TypeError, ValueError):
        times = values = None
    if times is not None and times.ndim == 1 and times.shape == values.shape:
        return times, values
    raise ValueError("a Klatt parameter must be a number or a (times, values) pair")
