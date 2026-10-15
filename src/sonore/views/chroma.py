"""Chroma: the power in each of the 12 pitch classes, the octaves folded together.

The sound is split into semitone bands by the cosine filterbank on the cents
scale (an exact constant-Q filter bank), each band's power is averaged over
time windows, and every band of the same pitch class is added together: C2,
C3, C4 and so on become one row for C. The way back plays each pitch class
as a Shepard tone, whose partials are all octaves of that one class, so its
chroma comes back unchanged. The design and its numbers are in
docs/design/views/chroma.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import cents_scale, note_to_freq
from sonore.frames.filterbank import cosine_filterbank
from sonore.sources.waveforms import harmonic_complex
from sonore.views.view import View

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["PITCH_CLASSES", "Chroma", "chroma"]


PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

SHEPARD_WIDTH = 1.0  # standard deviation [octaves] of the bell with no height
NARROW_WIDTH = 0.25  # and of the bell around a chosen note: an octave away is 70 dB down


@dataclass(frozen=True)
class Chroma(View):
    """The power in each pitch class, C to B, in each time window, for each
    channel. A view: it adds the octaves of each pitch class together.

    ``power`` has shape ``(n_windows, 12, n_channels)``: the mean power of the
    semitone bands of each pitch class over each time window, which start at
    ``t - window / 2``. ``reference`` is the frequency of A4 that placed the
    bands; ``f_lo`` and ``f_hi`` are the centers of the lowest and highest
    band [Hz].
    """

    discards = (
        "Chroma adds the semitone bands of each pitch class together, so it discards octave height "
        "(how each pitch class's power spreads over octaves) and everything within a band: many "
        "sounds share one chroma."
    )
    back_to_sound = (
        "Chroma.to_sound plays each pitch class as a Shepard tone, with octaves= saying how its power "
        "spreads over octaves."
    )

    t: np.ndarray
    power: np.ndarray
    window: float
    reference: float
    f_lo: float
    f_hi: float

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.power.shape[2]

    @property
    def duration(self) -> float:
        """Length [s] the time windows cover."""
        return len(self.t) * self.window

    def __repr__(self) -> str:
        return f"Chroma({len(self.t)} time windows of {self.window:g} s, {self.n_channels} ch)"

    def to_sound(
        self, fs: float, *, octaves: str | float | Callable[[np.ndarray], np.ndarray] | None = None
    ) -> Sound:
        """A sound with this chroma: each pitch class a Shepard tone (Shepard,
        1964), its level following that pitch class's row.

        ``octaves`` is what the view dropped, how each pitch class's power
        spreads over octaves, as the level of each partial at its frequency:

        - ``None``: a bell over log frequency centered between ``f_lo`` and
          ``f_hi``, one octave wide (standard deviation), so no octave stands
          out;
        - a note such as ``"C4"``, or a frequency [Hz]: a narrow bell there,
          so each pitch class sounds in the octave nearest it;
        - a function of frequency [Hz], for any other spread.

        The partials are the octaves of each pitch class from ``f_lo`` to
        ``f_hi``, the range the chroma was measured on, so the chroma of the
        result is this chroma, up to the smoothing of the time windows. The
        sound is ``duration`` long, at the level the chroma measured."""
        level = _octave_level(octaves, self.reference, self.f_lo, self.f_hi)
        n_out = round(self.duration * fs)
        t = np.arange(n_out) / fs
        channels = np.zeros((n_out, self.n_channels))
        for pitch_class in range(12):
            lowest = _lowest_octave(self.reference, pitch_class, self.f_lo)
            top = min(self.f_hi, np.nextafter(fs / 2, 0))
            harmonics = 2 ** np.arange(int(np.floor(np.log2(top / lowest))) + 1)
            for channel in range(self.n_channels):
                row = self.power[:, pitch_class, channel]
                if not np.any(row > 0):
                    continue
                gains = _gains(level, np.sqrt(np.interp(t, self.t, row)), fs)
                tone = harmonic_complex(self.duration, fs, lowest, harmonics=harmonics, amplitudes=gains)
                # harmonic_complex returns RMS 1; put back this pitch class's power
                channels[:, channel] += tone.data[:n_out, 0] * np.sqrt(np.mean(row))
        return Sound(channels, fs)

    def plot(self, ax: Axes | None = None, *, channel: int = 0, db_range: float = 40.0, **kwargs) -> Axes:
        """The 12 rows against time, in dB (see :func:`~sonore.plotting.plot_chroma`)."""
        from sonore.plotting import plot_chroma

        return plot_chroma(self, ax=ax, channel=channel, db_range=db_range, **kwargs)


def _gains(level: Callable, amplitude: np.ndarray, fs: float) -> Callable:
    """``harmonic_complex``'s ``amplitudes(t, f)``: the octave level at each
    partial's frequency times the pitch class's amplitude at each sample."""

    def gains(times: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        samples = np.rint(times * fs).astype(int).clip(0, len(amplitude) - 1)
        return level(freqs) * amplitude[samples]

    return gains


def _lowest_octave(reference: float, pitch_class: int, f_lo: float) -> float:
    """The frequency of ``pitch_class`` (0 = C) in the first octave at or above
    ``f_lo``, with A4 at ``reference``."""
    freq = reference * 2 ** ((pitch_class - 9) / 12)
    return float(freq * 2 ** np.ceil(np.log2(f_lo / freq) - 1e-9))


def _bell(center: float, width: float) -> Callable[[np.ndarray], np.ndarray]:
    def level(freqs: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (np.log2(np.asarray(freqs) / center) / width) ** 2)

    return level


def _octave_level(octaves, reference: float, f_lo: float, f_hi: float) -> Callable[[np.ndarray], np.ndarray]:
    match octaves:
        case None:
            return _bell(np.sqrt(f_lo * f_hi), SHEPARD_WIDTH)
        case str():
            return _bell(note_to_freq(octaves, a4=reference), NARROW_WIDTH)
        case int() | float() if octaves > 0:
            return _bell(float(octaves), NARROW_WIDTH)
        case _ if callable(octaves):
            return octaves
    raise ValueError(f"octaves must be None, a note, a frequency above 0 Hz or a function, not {octaves!r}")


def chroma(
    sound: Sound,
    *,
    window: float = 0.05,
    reference: float = 440.0,
    f_lo: float = 55.0,
    f_hi: float = 7040.0,
) -> Chroma:
    """The power in each pitch class in each time window (Fujishima, 1999,
    who called it a pitch class profile).

    The sound goes through a cosine filterbank on the cents scale, one band
    per equal-tempered semitone with A4 at ``reference`` Hz (by default
    440 Hz, the standard pitch of ISO 16), centered from ``f_lo`` to
    ``f_hi`` (A1 to A8 by default; both are moved to the nearest semitone).
    Each band's power is averaged over time windows ``window`` seconds long
    (a last, shorter stretch is dropped), and the bands of each pitch class
    are added together. Power is kept as measured: a window is not scaled to
    its own maximum. Overtones land on other pitch classes (the third
    harmonic of C is a G), so chroma summarizes partials, not notes."""
    if not window > 0:
        raise ValueError(f"window must be above 0 s, not {window!r}")
    scale = cents_scale(reference)
    lowest, highest = (round(float(scale.to_scale(freq)) / 100) for freq in (f_lo, f_hi))
    if not lowest < highest:
        raise ValueError(f"need f_lo < f_hi by at least a semitone, not {f_lo!r} and {f_hi!r}")
    knots = 100.0 * np.arange(lowest - 1, highest + 2)  # band centers, one more at each end
    bank = cosine_filterbank(scale=scale, centers=scale.from_scale(knots))
    band_power = bank.analyze(sound).data[:, 1:-1, :] ** 2  # the bandpass filters, not the edges
    n_window = round(window * sound.fs)
    n_windows = len(band_power) // n_window
    if n_windows < 1:
        raise ValueError(f"the sound is shorter than one time window ({window:g} s)")
    trimmed = band_power[: n_windows * n_window]
    window_power = trimmed.reshape(n_windows, n_window, *trimmed.shape[1:]).mean(axis=1)
    pitch_classes = (np.arange(lowest, highest + 1) + 9) % 12  # semitones from A4; A is class 9
    folding = np.zeros((len(pitch_classes), 12))  # band by pitch class: 1 where the band belongs
    folding[np.arange(len(pitch_classes)), pitch_classes] = 1.0
    power = np.einsum("wbc,bp->wpc", window_power, folding)
    t = (np.arange(n_windows) + 0.5) * n_window / sound.fs
    return Chroma(
        t,
        power,
        n_window / sound.fs,
        reference,
        float(scale.from_scale(100.0 * lowest)),
        float(scale.from_scale(100.0 * highest)),
    )
