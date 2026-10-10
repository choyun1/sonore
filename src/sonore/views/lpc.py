"""Linear prediction: an all-pole model of each time window, its spectral
envelope, and formant tracks read from the roots of its polynomial."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.signal import lfilter

from sonore.core.sound import Sound
from sonore.frames.gabor import STFT, TVSTFT, GaborFrame
from sonore.views.mfcc import symmetric_hamming
from sonore.views.spectral_envelope import GridEnvelope
from sonore.views.view import View

__all__ = ["levinson", "LPC", "FormantTrack", "formant_track"]


def levinson(autocorrelation: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the normal equations of linear prediction by the Levinson-Durbin
    recursion (Levinson, 1947; Durbin, 1960).

    ``autocorrelation`` holds ``r[0] .. r[order]`` along its last axis; any
    leading axes are solved at once. Returns the predictor ``a`` (shape
    ``(..., order + 1)``, ``a[..., 0] = 1``) minimizing the error of
    ``x[n] + a[1] x[n-1] + ... + a[p] x[n-p]``, the reflection coefficients
    ``k[1] .. k[p]`` (shape ``(..., order)``), and the error power left,
    ``E = r[0] prod(1 - k[i]**2)`` (shape ``(...)``). Where ``r[0]`` is 0 (a
    silent time window) the predictor is ``a = [1, 0, ..., 0]`` with no error.
    """
    r = np.asarray(autocorrelation, dtype=float)
    leading = r.shape[:-1]
    a = np.zeros((*leading, order + 1))
    a[..., 0] = 1.0
    reflections = np.zeros((*leading, order))
    error = r[..., 0].copy()
    silent = error <= 0
    error_safe = np.where(silent, 1.0, error)
    for i in range(1, order + 1):
        # k = -(r[i] + a[1] r[i-1] + ... + a[i-1] r[1]) / E
        k = -np.einsum("...j,...j->...", a[..., :i], r[..., i:0:-1]) / error_safe
        k = np.where(silent, 0.0, k)
        a[..., 1 : i + 1] = a[..., 1 : i + 1] + k[..., None] * a[..., i - 1 :: -1][..., :i]
        reflections[..., i - 1] = k
        error_safe = error_safe * (1 - k * k)
        error = np.where(silent, 0.0, error_safe)
    return a, reflections, error


def _roots(predictors: np.ndarray) -> np.ndarray:
    """Roots of each polynomial ``1 + a[1] z^-1 + ... + a[p] z^-p`` (rows of
    ``predictors``, shape ``(n, p + 1)``), as eigenvalues of companion
    matrices, shape ``(n, p)``."""
    n, order = predictors.shape[0], predictors.shape[1] - 1
    companion = np.zeros((n, order, order))
    companion[:, 0, :] = -predictors[:, 1:]
    companion[:, np.arange(1, order), np.arange(order - 1)] = 1.0
    return np.linalg.eigvals(companion)


def _longest_window(coefs: STFT | TVSTFT) -> int:
    if isinstance(coefs, TVSTFT):
        return int(coefs.frame.layout(coefs.fs).lengths.max())
    return coefs.shortest_window  # every window of an STFT has the same length


class LPC(View):
    """Linear prediction of each time window, by the autocorrelation method.

    For a windowed segment with autocorrelation ``r[k]``, the predictor
    ``A(z) = 1 + a[1] z^-1 + ... + a[p] z^-p`` minimizes the power of the
    prediction error, and solves ``sum_j a[j] r[|i - j|] = -r[i]`` for
    ``i = 1 .. p`` (:func:`levinson`). The model of the segment is the all-pole
    filter ``1 / A(z)`` driven by white noise of the error power ``E``, with
    power spectrum ``E / |A|^2`` (:meth:`envelope`). Because the segment is
    zero outside its window, the autocorrelation is positive definite and
    every root of ``A(z)`` lies inside the unit circle: the model is always
    stable. A pair of roots ``|z| e^(+-j theta)`` is a resonance at
    ``theta fs / 2 pi`` Hz, ``-ln|z| fs / pi`` Hz wide (:meth:`candidates`).

    ``source`` is a :class:`~sonore.core.sound.Sound` or the coefficients of
    an analysis:

    - a ``Sound`` is analyzed with the speech recipe: a symmetric Hamming
      window ``win_dur`` long (25 ms), a hop of ``hop_dur`` (10 ms), and an
      FFT length rounded up to a power of two at least ``order`` samples
      longer than the window;
    - an :class:`~sonore.frames.gabor.STFT` or
      :class:`~sonore.frames.gabor.TVSTFT` is used as it is. The
      autocorrelation of each time window is the inverse FFT of its power,
      which is the circular autocorrelation; it equals the true one at lags
      ``0 .. order`` only when the FFT is at least ``order`` samples longer
      than the window, so a shorter FFT raises.

    Pre-emphasis (``y[n] = x[n] - 0.97 x[n-1]``) is a change to the sound,
    so it is applied to the sound first, for example ``so.Sound(
    scipy.signal.lfilter([1, -0.97], 1, snd.data, axis=0), snd.fs)``. For
    formants it matters: without it, two of the poles go to the falling tilt
    of the voice source, and on Peterson and Barney's vowels the formants
    come back about twice as far off. :func:`formant_track` does it itself.

    This is a view that discards information: what the predictor cannot
    predict (the residual, in speech mostly the excitation: the pulses and
    the noise), the phase beyond the minimum phase of ``1 / A(z)``, and the
    zeros of the spectrum, which an all-pole model can only imitate with
    poles.

    Parameters
    ----------
    source
        A sound, or the STFT or TVSTFT to model.
    order
        The number of predictor coefficients ``p``. Defaults to
        ``round(fs / 1000) + 2``, the rule of thumb of one resonance per
        kilohertz plus two poles for the tilt (18 at 16 kHz).
    win_dur, hop_dur
        Window length and hop [s] used when ``source`` is a sound.

    Attributes
    ----------
    data
        The predictors ``a[0] .. a[p]``, shape ``(n_channels, order + 1,
        n_windows)``, with ``a[0] = 1``.
    error_power
        The prediction error power ``E`` of each time window, shape
        ``(n_channels, n_windows)``.
    reflection
        The reflection coefficients, shape ``(n_channels, order, n_windows)``;
        all inside ``(-1, 1)``.
    source
        The STFT or TVSTFT analyzed.
    """

    discards = (
        "LPC keeps p coefficients and one error power per time window: the residual (the excitation), "
        "the phase and the zeros of the spectrum are gone."
    )
    back_to_sound = (
        "so.formant_track reads formant tracks from it, and so.klatt_synthesize makes a voice from those "
        "and an F0 track; envelope_view() gives an envelope for so.world_synthesize."
    )

    def __init__(
        self,
        source: Sound | STFT | TVSTFT,
        order: int | None = None,
        *,
        win_dur: float | None = None,
        hop_dur: float | None = None,
    ):
        match source:
            case Sound():
                order = round(source.fs / 1000) + 2 if order is None else order
                win_dur = 0.025 if win_dur is None else win_dur
                hop_dur = 0.010 if hop_dur is None else hop_dur
                window_length = round(win_dur * source.fs)
                n_fft = 1 << (window_length + order - 1).bit_length()
                frame = GaborFrame(win_dur, hop_dur, window=symmetric_hamming, n_fft=n_fft)
                coefs = STFT(source, frame=frame)
            case STFT() | TVSTFT():
                if win_dur is not None or hop_dur is not None:
                    raise TypeError("win_dur and hop_dur apply only when analyzing a Sound")
                coefs = source
                order = round(coefs.fs / 1000) + 2 if order is None else order
                longest = _longest_window(coefs)
                if coefs.n_fft < longest + order:
                    raise ValueError(
                        f"n_fft ({coefs.n_fft}) must be at least the longest window ({longest}) plus the "
                        f"order ({order}), or the autocorrelation wraps around; analyze with "
                        f"n_fft={longest + order} or more"
                    )
            case _:
                raise TypeError(f"expected a Sound, STFT or TVSTFT, not {type(source).__name__}")
        if order < 1:
            raise ValueError(f"order must be at least 1, not {order}")
        self.source = coefs
        self.fs = coefs.fs
        self.n_fft = coefs.n_fft
        self.order = order
        power = np.abs(coefs.data) ** 2
        autocorrelation = np.fft.irfft(power, n=self.n_fft, axis=1)[:, : order + 1]
        predictors, reflections, error = levinson(np.moveaxis(autocorrelation, 1, -1), order)
        self.data = np.moveaxis(predictors, -1, 1)
        self.reflection = np.moveaxis(reflections, -1, 1)
        self.error_power = error

    def __repr__(self) -> str:
        n_channels, _, n_windows = self.data.shape
        return f"LPC(order {self.order} x {n_windows} time windows, {n_channels} ch, fs {self.fs:g} Hz)"

    @property
    def t(self) -> np.ndarray:
        """Window center times [s], those of :attr:`source`."""
        return self.source.t

    def envelope(self, f) -> np.ndarray:
        """The model's power spectrum ``E / |A|^2`` at frequencies ``f`` [Hz],
        shape ``(n_channels, len(f), n_windows)``, on the scale of the time
        window's power spectrum ``|X|^2``: averaged over all ``n_fft`` bins,
        ``|X|^2`` divided by it is exactly 1. That is why it runs close to
        the harmonic peaks and bridges the dips between them, where a
        cepstral envelope runs through the middle."""
        f = np.asarray(f, dtype=float)
        lags = np.arange(self.order + 1)
        phasors = np.exp(-2j * np.pi * np.outer(f, lags) / self.fs)  # (len(f), order + 1)
        response = np.einsum("fk,ckw->cfw", phasors, self.data)
        return self.error_power[:, None, :] / np.abs(response) ** 2

    def envelope_view(self, f=None) -> GridEnvelope:
        """:meth:`envelope` on this analysis's time windows, at frequencies
        ``f`` (by default its FFT bins), read as ``env(t, f)``, so it goes
        wherever a spectral envelope is taken
        (:func:`~sonore.views.spectral_envelope.warp_frequency`,
        :func:`~sonore.views.world.world_synthesize`,
        :func:`~sonore.sources.waveforms.harmonic_complex`)."""
        freqs = np.arange(self.n_fft // 2 + 1) * self.fs / self.n_fft if f is None else np.asarray(f, float)
        return GridEnvelope(self.envelope(freqs), self.t, freqs)

    def candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """Every resonance of the model: the frequencies and bandwidths [Hz]
        of the roots of ``A(z)`` in the upper half plane, each shape
        ``(n_channels, order // 2, n_windows)``, sorted by frequency and
        padded with NaN. Real roots, which shape the tilt rather than a
        resonance, are left out. No root is dropped for being wide: on
        speech, a fixed bandwidth limit drops a formant now and then, and
        every formant above it is then mislabeled."""
        n_channels, _, n_windows = self.data.shape
        predictors = np.moveaxis(self.data, 1, -1).reshape(-1, self.order + 1)
        roots = _roots(predictors)
        upper = roots.imag > 1e-12
        freqs = np.where(upper, np.angle(roots) * self.fs / (2 * np.pi), np.nan)
        radius = np.where(upper, np.abs(roots), 1.0)  # a silent time window's roots are all at 0
        bandwidths = np.where(upper, -np.log(radius) * self.fs / np.pi, np.nan)
        order = np.argsort(freqs, axis=1)  # NaN sorts last
        n_kept = self.order // 2
        freqs = np.take_along_axis(freqs, order, axis=1)[:, :n_kept]
        bandwidths = np.take_along_axis(bandwidths, order, axis=1)[:, :n_kept]
        shape = (n_channels, n_windows, n_kept)
        return (
            np.moveaxis(freqs.reshape(shape), -1, 1),
            np.moveaxis(bandwidths.reshape(shape), -1, 1),
        )

    def plot(self, ax=None, channel: int = 0, db_range: float = 60.0, fmax: float | None = None, **kwargs):
        """The envelope as a time-frequency image with the candidates as dots
        (see :func:`~sonore.plotting.plot_lpc`)."""
        from sonore.plotting import plot_lpc

        return plot_lpc(self, ax=ax, channel=channel, db_range=db_range, fmax=fmax, **kwargs)


@dataclass(frozen=True)
class FormantTrack(View):
    """Formant tracks: F1, F2 and F3 of each channel every ``hop`` seconds, from
    :func:`formant_track`. A view: it keeps only the formants.

    ``frequencies`` and ``bandwidths`` have shape ``(n_channels, 3,
    n_windows)``, NaN where a time window had fewer resonances than formants.
    ``candidates`` and ``candidate_bandwidths`` have shape ``(n_channels,
    n_formants, n_windows)`` and hold every resonance the tracker chose from,
    NaN-padded. Every time window gets formants, silent and unvoiced ones too:
    mask them with an F0 track's voicing (:attr:`F0Track.voiced
    <sonore.views.f0.F0Track.voiced>`) before reading them as formants.
    """

    discards = "FormantTrack keeps only the frequencies and bandwidths of three formants in each time window."
    back_to_sound = (
        "so.klatt_synthesize makes a voice from formant tracks, an F0 track and a voicing amplitude: "
        "copy synthesis, which keeps neither the source nor the higher formants of the original."
    )

    t: np.ndarray
    frequencies: np.ndarray
    bandwidths: np.ndarray
    candidates: np.ndarray
    candidate_bandwidths: np.ndarray
    ceiling: float

    def __repr__(self) -> str:
        n_channels, n_tracks, n_windows = self.frequencies.shape
        medians = ", ".join(f"F{k + 1} {np.nanmedian(self.frequencies[:, k]):.0f}" for k in range(n_tracks))
        return (
            f"FormantTrack({n_windows} time windows, {n_channels} ch, ceiling {self.ceiling:g} Hz, "
            f"medians {medians} Hz)"
        )

    def track(self, number: int, channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Formant ``number`` (1, 2 or 3) of one channel as a ``(times,
        values)`` pair, the form :func:`~sonore.sources.klatt.klatt_synthesize`
        takes for ``F1``, ``F2`` and ``F3``; time windows where it is missing
        are left out."""
        values = self.frequencies[channel, number - 1]
        present = np.isfinite(values)
        return self.t[present], values[present]

    def plot(self, ax=None, channel: int = 0, candidates: bool = False, voiced=None, **kwargs):
        """F1 to F3 against time (see :func:`~sonore.plotting.plot_formant_track`)."""
        from sonore.plotting import plot_formant_track

        return plot_formant_track(
            self, ax=ax, channel=channel, candidates=candidates, voiced=voiced, **kwargs
        )


def _assignments(freqs: np.ndarray, bandwidths: np.ndarray, n_tracks: int):
    """Every way of giving one window's resonances (sorted, no NaN) to the
    formants in increasing order: a formant goes missing only when there are
    fewer resonances than formants, and then the lowest formants are filled.
    Returns the frequencies and bandwidths, shape ``(n_states, n_tracks)``,
    and how many resonances each state skips below its highest choice."""
    n_present = min(n_tracks, len(freqs))
    if n_present == 0:
        chosen = np.zeros((1, 0), dtype=int)  # one state, with every formant missing
    else:
        chosen = np.array(list(combinations(range(len(freqs)), n_present)), dtype=int)
    values = np.full((len(chosen), n_tracks), np.nan)
    widths = np.full((len(chosen), n_tracks), np.nan)
    values[:, :n_present] = freqs[chosen]
    widths[:, :n_present] = bandwidths[chosen]
    highest = chosen[:, -1] if n_present else np.full(len(chosen), -1)
    skipped = highest + 1 - n_present  # resonances below the highest chosen that were not chosen
    return values, widths, skipped


def _track(freqs, bandwidths, nominal, nominal_weight, bandwidth_weight, skip_cost, transition_weight):
    """Viterbi search for one channel: ``freqs`` and ``bandwidths`` have shape
    ``(n_candidates, n_windows)``. Returns ``(n_tracks, n_windows)`` frequencies
    and bandwidths."""
    nominal = np.asarray(nominal, dtype=float)
    n_tracks, n_windows = len(nominal), freqs.shape[1]
    states, costs = [], []
    for window in range(n_windows):
        present = np.isfinite(freqs[:, window])
        values, widths, skipped = _assignments(freqs[present, window], bandwidths[present, window], n_tracks)
        filled = np.isfinite(values)
        safe = np.where(filled, values, 1.0)
        local = np.where(
            filled,
            nominal_weight * np.abs(np.log(safe / nominal))
            + bandwidth_weight * np.where(filled, widths, 0) / safe,
            0.0,
        ).sum(axis=1)
        states.append((values, widths))
        costs.append(local + skip_cost * skipped)
    total = costs[0]
    best_previous = []
    for window in range(1, n_windows):
        before, now = states[window - 1][0], states[window][0]
        jump = np.nan_to_num(np.abs(np.log(now[:, None, :] / before[None, :, :])), nan=0.0).sum(axis=2)
        through = total[None, :] + transition_weight * jump
        best = np.argmin(through, axis=1)
        best_previous.append(best)
        total = through[np.arange(len(now)), best] + costs[window]
    path = [int(np.argmin(total))]
    for best in reversed(best_previous):
        path.append(int(best[path[-1]]))
    path.reverse()
    frequencies = np.array([states[window][0][state] for window, state in enumerate(path)]).T
    widths = np.array([states[window][1][state] for window, state in enumerate(path)]).T
    return frequencies, widths


def formant_track(
    sound: Sound,
    ceiling: float = 5000.0,
    n_formants: int = 5,
    *,
    win_dur: float = 0.025,
    hop: float = 0.005,
    nominal: tuple[float, float, float] = (500.0, 1500.0, 2500.0),
    nominal_weight: float = 1.0,
    bandwidth_weight: float = 1.0,
    skip_cost: float = 2.0,
    transition_weight: float = 5.0,
) -> FormantTrack:
    """Track the first three formants of each channel of a sound.

    Two stages, every ``hop`` seconds from time 0:

    1. **Candidates**, by Praat's recipe for formants: the sound is resampled
       to twice ``ceiling``, pre-emphasized (``y[n] = x[n] - 0.97 x[n-1]``),
       and modeled by :class:`LPC` with ``2 n_formants`` coefficients on a
       ``win_dur`` symmetric Hamming window; every resonance between 50 Hz
       and ``ceiling - 50`` Hz is a candidate, however wide. Limiting the
       band to the ceiling, rather than fitting the whole band with a
       bandwidth limit, is what keeps the k-th candidate the k-th formant in
       most time windows. Praat's advice for the ceiling is 5000 Hz for a
       male voice and 5500 Hz for a female one. On the two gallery sentences
       these candidates match Praat's ``To Formant (burg)`` to within a
       median of 1 to 7 Hz, though Praat uses Burg's method on a Gaussian
       window.
    2. **Tracking.** A Viterbi pass over the ways of giving each window's
       candidates to F1, F2 and F3 in increasing order. In one window a
       choice costs, per formant, ``nominal_weight |ln(f / nominal)|`` plus
       ``bandwidth_weight * bandwidth / f``, plus ``skip_cost`` for every
       candidate skipped below the highest one chosen; moving from one window
       to the next costs ``transition_weight |ln(f_now / f_before)|`` per
       formant. The nominal frequencies are those of a uniform tube 17.5 cm
       long, closed at one end. The weights were set by trying a few values
       on two sentences and a synthesized utterance, not optimized. On
       synthesized vowels the tracks equal the k-th candidates; on the
       female gallery sentence the tracker halves how often F3 jumps by more
       than 20% between windows.

    Unvoiced and silent time windows get formants too; mask them with an F0
    track's voicing.

    Parameters
    ----------
    sound
        The sound.
    ceiling
        The highest frequency a formant can have [Hz]; the sound is
        resampled to twice it.
    n_formants
        How many resonances to look for below the ceiling; the LPC order is
        twice this.
    win_dur, hop
        Window length and hop [s].
    nominal, nominal_weight, bandwidth_weight, skip_cost, transition_weight
        The tracker's costs (above).
    """
    if not 0 < ceiling < sound.fs / 2:
        raise ValueError(
            f"ceiling must be between 0 and half the sampling rate ({sound.fs / 2:g} Hz), not {ceiling:g}"
        )
    resampled = sound.resample(2 * ceiling)
    emphasized = Sound(lfilter([1.0, -0.97], 1.0, resampled.data, axis=0), resampled.fs)
    model = LPC(emphasized, 2 * n_formants, win_dur=win_dur, hop_dur=hop)
    freqs, bandwidths = model.candidates()
    inside = (freqs > 50.0) & (freqs < ceiling - 50.0)
    freqs = np.where(inside, freqs, np.nan)
    bandwidths = np.where(inside, bandwidths, np.nan)
    order = np.argsort(freqs, axis=1)  # NaN sorts last
    freqs = np.take_along_axis(freqs, order, axis=1)
    bandwidths = np.take_along_axis(bandwidths, order, axis=1)
    tracks = [
        _track(
            freqs[channel],
            bandwidths[channel],
            nominal,
            nominal_weight,
            bandwidth_weight,
            skip_cost,
            transition_weight,
        )
        for channel in range(sound.n_channels)
    ]
    return FormantTrack(
        t=model.t,
        frequencies=np.array([frequencies for frequencies, _ in tracks]),
        bandwidths=np.array([widths for _, widths in tracks]),
        candidates=freqs,
        candidate_bandwidths=bandwidths,
        ceiling=float(ceiling),
    )
