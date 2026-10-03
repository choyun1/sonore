"""Mel-frequency cepstral coefficients (MFCCs): the log power in triangular
bands on a mel scale, summarised by a discrete cosine transform."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct
from scipy.signal import savgol_filter

from sonore.core.sound import Sound
from sonore.core.utils import db_to_power, freq_to_mel, mel_to_freq
from sonore.frames.gabor import STFT, TVSTFT, GaborFrame
from sonore.views.spectral_envelope import GridEnvelope
from sonore.views.view import View

__all__ = ["mel_filterbank", "symmetric_hamming", "delta_features", "MFCC"]


DB_PER_NEPER = 10 / np.log(10)  # 10 log10(x) = DB_PER_NEPER * ln(x)


def mel_filterbank(
    n_mels: int,
    freqs: np.ndarray,
    f_lo: float,
    f_hi: float,
    scale: str = "htk",
    triangles: str = "height",
    triangle_axis: str = "mel",
) -> tuple[np.ndarray, np.ndarray]:
    """Triangular mel weights on the frequencies ``freqs`` [Hz].

    The ``n_mels + 2`` band edges are equally spaced in mel from ``f_lo`` to
    ``f_hi``; band ``m`` rises linearly from edge ``m`` to edge ``m + 1``
    and falls to edge ``m + 2``. "Linearly" is in mel with
    ``triangle_axis="mel"``, as HTK and Kaldi build them, and in Hz with
    ``"hz"``, as librosa does; the two differ inside each band, most in the
    wide high bands. With ``triangles="height"`` each peak is 1, and between
    the first and last centres the weights sum to 1.
    With ``"area"`` band ``m`` is scaled by ``2 / (edge[m+2] - edge[m])``
    (Slaney's normalisation), which after the log only adds a constant to
    each band. The triangles are evaluated at the exact frequencies, not
    rounded to FFT bins.

    Returns the weights, shape ``(n_mels, len(freqs))``, and the edges [Hz].
    """
    if triangles not in ("height", "area"):
        raise ValueError(f"triangles must be 'height' or 'area', not {triangles!r}")
    if triangle_axis not in ("mel", "hz"):
        raise ValueError(f"triangle_axis must be 'mel' or 'hz', not {triangle_axis!r}")
    edge_mels = np.linspace(freq_to_mel(f_lo, scale), freq_to_mel(f_hi, scale), n_mels + 2)
    edges = mel_to_freq(edge_mels, scale)
    freqs = np.asarray(freqs, dtype=float)
    if triangle_axis == "mel":
        positions, edge_positions = freq_to_mel(freqs, scale), edge_mels
    else:
        positions, edge_positions = freqs, edges
    weights = np.zeros((n_mels, len(freqs)))
    for band in range(n_mels):
        left, centre, right = edge_positions[band : band + 3]
        rising = (positions - left) / (centre - left)
        falling = (right - positions) / (right - centre)
        weights[band] = np.maximum(0, np.minimum(rising, falling))
        if triangles == "area":
            weights[band] *= 2 / (edges[band + 2] - edges[band])
    return weights, edges


def symmetric_hamming(n_samples: int) -> np.ndarray:
    """The Hamming window as HTK and Kaldi define it,
    ``0.54 - 0.46 cos(2 pi i / (n - 1))``: symmetric, with both ends at 0.08
    (SciPy's ``"hamming"`` spec, as a ``GaborFrame`` samples it, is the
    periodic one)."""
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_samples) / (n_samples - 1))


def delta_features(values: np.ndarray, order: int = 1, width: int = 5, axis: int = -1) -> np.ndarray:
    """Time derivatives of a feature track, computed as librosa's
    ``feature.delta`` does.

    Along ``axis``, a polynomial of degree ``order`` is fitted by least
    squares to each run of ``width`` (odd) values and its ``order``-th
    derivative taken: SciPy's ``savgol_filter`` with ``mode="interp"``, so
    the first and last ``width // 2`` values use the fit to the first or
    last ``width``. For ``order=1`` and ``width=5`` this is HTK's regression
    ``sum(n * c[t+n]) / sum(n**2)`` over ``n = -2..2`` away from the ends.
    ``order=2`` is the second derivative of a local quadratic, not the delta
    of the delta. Units are per time window."""
    if width < 3 or width % 2 == 0:
        raise ValueError(f"width must be an odd number of at least 3, not {width}")
    if order < 1 or order >= width:
        raise ValueError(f"order must be at least 1 and less than width, not {order}")
    n_values = np.shape(values)[axis]
    if width > n_values:
        raise ValueError(f"width {width} is longer than the {n_values} values along the axis")
    return savgol_filter(values, width, polyorder=order, deriv=order, axis=axis, mode="interp")


class MFCC(View):
    """Mel-frequency cepstral coefficients of each time window (Davis &
    Mermelstein, 1980).

    For each time window with power spectrum ``P[k]``: band powers
    ``E[m] = sum_k W[m, k] P[k]`` through ``n_mels`` triangular mel bands
    (:func:`mel_filterbank`), their natural log, and the orthonormal
    DCT-II of that, of which the first ``n_mfcc`` coefficients are kept.
    The DCT of a log spectrum is a cepstrum, so the low coefficients are the
    slow ripples of the log mel spectrum: its envelope, without the pitch.

    ``source`` is a :class:`~sonore.core.sound.Sound` or the coefficients of
    an analysis:

    - a ``Sound`` is analysed with the usual speech settings: a symmetric
      Hamming window ``win_dur`` long (25 ms), as HTK and Kaldi use, a hop of
      ``hop_dur`` (10 ms), and an FFT length of the next power of two;
    - an :class:`~sonore.frames.gabor.STFT` or
      :class:`~sonore.frames.gabor.TVSTFT` is used as it is, so
      any window, hop or pitch-adaptive analysis can be summarised.

    This is a view that discards information: the phase, everything inside
    each mel band, the faster ripples of the log mel spectrum beyond
    ``n_mfcc``, and the level apart from ``c0``. Many spectra have the same
    coefficients, and no sound is recovered from them; :meth:`envelope`
    shows the smoothed spectrum they stand for.

    The mel power is floored at ``floor_db`` below each channel's largest
    band power over all time windows before the log, so scaling a sound
    changes only ``c0`` and digital silence gives finite coefficients.
    Because the floor is set by the loudest moment, a deep floor (the
    default) keeps each time window independent of the rest of the sound.

    The defaults follow Kaldi and HTK: HTK's mel scale, height-1 triangles
    straight in mel, and, for a sound, a symmetric Hamming window. With no
    pre-emphasis, no DC removal and ``use_energy=false``, Kaldi's MFCCs
    (``compute-mfcc-feats``, here through kaldi-native-fbank) are reproduced
    to float32 precision; Kaldi's own lifter, ``low-freq`` of 20 Hz, 23 bins
    and "povey" window (a Hann window to the power 0.85, as a callable
    ``window`` of the ``GaborFrame``) are all available. Kaldi's time windows
    start at the first sample (``snip-edges``) where sonore's are centred
    on multiples of the hop, so the grids line up only after trimming the
    sound by half a window modulo the hop.

    To reproduce librosa's ``feature.mfcc`` (Slaney mel, area-normalised
    triangles straight in Hz, power in dB floored 80 dB below the loudest
    cell), analyse with ``so.GaborFrame(n_fft / fs, hop / fs, window="hann",
    n_fft=n_fft)`` and use ``mel_scale="slaney"``, ``triangles="area"``,
    ``triangle_axis="hz"``, ``n_mels=128``,
    ``n_mfcc=20``, ``floor_db=-80`` and :attr:`db`. sonore's grid has one
    more time window, centred one hop before the first sample, and one or two
    more at the end; the others are librosa's. librosa also floors the
    power at an absolute 1e-10, which matters only for sounds far quieter than
    any recording, and takes the loudest cell over all channels rather than
    per channel.

    Pre-emphasis (``y[n] = x[n] - 0.97 x[n-1]``, as in HTK and
    python_speech_features) is a change to the sound, so it is applied to the
    sound first, for example ``so.Sound(scipy.signal.lfilter([1, -0.97], 1,
    snd.data, axis=0), snd.fs)``. python_speech_features also rounds the
    triangle feet to FFT bins and replaces ``c0`` with the log energy of the
    time window; neither is done here.

    Parameters
    ----------
    source
        A sound, or the STFT or TVSTFT to summarise.
    n_mfcc
        Coefficients kept, ``c0`` included.
    n_mels
        Mel bands.
    f_lo, f_hi
        The outer feet of the first and last bands [Hz]; ``f_hi`` defaults to
        half the sampling rate.
    mel_scale
        ``"htk"`` or ``"slaney"`` (see :func:`freq_to_mel`).
    triangles
        ``"height"`` (peaks of 1, as HTK and Kaldi) or ``"area"`` (Slaney's and
        librosa's area normalisation).
    triangle_axis
        ``"mel"``: the triangles are straight lines in mel (HTK, Kaldi);
        ``"hz"``: straight in Hz (librosa).
    floor_db
        Floor of the band powers, in dB below each channel's largest.
    lifter
        ``L`` of HTK's sinusoidal lifter, which multiplies ``c[n]`` by
        ``1 + (L / 2) sin(pi n / L)``; 0 for none. It is a fixed gain per
        coefficient: it changes Euclidean distances and nothing else.
    win_dur, hop_dur
        Window length and hop [s] used when ``source`` is a sound.

    Attributes
    ----------
    data
        The coefficients, shape ``(n_channels, n_mfcc, n_windows)``, in natural
        log units (:attr:`db` for dB).
    mel_power
        Band powers before the floor, shape ``(n_channels, n_mels, n_windows)``:
        the mel spectrogram.
    edges
        Band edges [Hz], ``n_mels + 2`` of them; :attr:`cfs` are the centres.
    source
        The STFT or TVSTFT analysed.
    """

    discards = (
        "MFCC discards the phase, the detail inside each mel band, and every coefficient past the last kept, "
        "so no sound has these MFCCs alone."
    )
    back_to_sound = (
        "For an approximate voice, read the envelope with mfcc.envelope_view() and pass it to "
        "so.world_synthesize with an F0 track and an aperiodicity."
    )

    def __init__(
        self,
        source: Sound | STFT | TVSTFT,
        *,
        n_mfcc: int = 13,
        n_mels: int = 26,
        f_lo: float = 0.0,
        f_hi: float | None = None,
        mel_scale: str = "htk",
        triangles: str = "height",
        triangle_axis: str = "mel",
        floor_db: float = -200.0,
        lifter: float = 0.0,
        win_dur: float | None = None,
        hop_dur: float | None = None,
    ):
        match source:
            case Sound():
                win_dur = 0.025 if win_dur is None else win_dur
                hop_dur = 0.010 if hop_dur is None else hop_dur
                window_length = round(win_dur * source.fs)
                n_fft = 1 << (window_length - 1).bit_length()
                frame = GaborFrame(win_dur, hop_dur, window=symmetric_hamming, n_fft=n_fft)
                coefs = STFT(source, frame=frame)
            case STFT() | TVSTFT():
                if win_dur is not None or hop_dur is not None:
                    raise TypeError("win_dur and hop_dur apply only when analysing a Sound")
                coefs = source
            case _:
                raise TypeError(f"expected a Sound, STFT or TVSTFT, not {type(source).__name__}")
        f_hi = coefs.fs / 2 if f_hi is None else f_hi
        if not 0 <= f_lo < f_hi <= coefs.fs / 2:
            raise ValueError(
                f"need 0 <= f_lo < f_hi <= fs / 2 ({coefs.fs / 2:g} Hz), got {f_lo:g} and {f_hi:g}"
            )
        if not 1 <= n_mfcc <= n_mels:
            raise ValueError(f"n_mfcc must be between 1 and n_mels ({n_mels}), not {n_mfcc}")
        self.source = coefs
        self.fs = coefs.fs
        self.n_fft = coefs.n_fft
        self.mel_scale = mel_scale
        self.floor_db = floor_db
        self.lifter = lifter
        bin_freqs = np.arange(self.n_fft // 2 + 1) * self.fs / self.n_fft
        self.weights, self.edges = mel_filterbank(
            n_mels, bin_freqs, f_lo, f_hi, mel_scale, triangles, triangle_axis
        )
        power = np.abs(coefs.data) ** 2
        self.mel_power = np.einsum("mk,ckw->cmw", self.weights, power)
        log_power = np.log(self._floored(self.mel_power))
        self.data = (
            dct(log_power, type=2, norm="ortho", axis=1)[:, :n_mfcc] * self._lifter_gains(n_mfcc)[:, None]
        )

    def __repr__(self) -> str:
        n_channels, n_mfcc, n_windows = self.data.shape
        return (
            f"MFCC({n_mfcc} coefficients x {n_windows} time windows, {n_channels} ch, "
            f"{len(self.cfs)} {self.mel_scale} mel bands to {self.edges[-1]:g} Hz)"
        )

    def _floored(self, mel_power: np.ndarray) -> np.ndarray:
        peak = mel_power.max(axis=(1, 2), keepdims=True)
        floor = np.where(peak > 0, peak * db_to_power(self.floor_db), np.finfo(float).tiny)
        return np.maximum(mel_power, floor)

    def _lifter_gains(self, n_mfcc: int) -> np.ndarray:
        if self.lifter == 0:
            return np.ones(n_mfcc)
        return 1 + self.lifter / 2 * np.sin(np.pi * np.arange(n_mfcc) / self.lifter)

    @property
    def t(self) -> np.ndarray:
        """Window center times [s], those of :attr:`source`."""
        return self.source.t

    @property
    def cfs(self) -> np.ndarray:
        """Band centres [Hz]."""
        return self.edges[1:-1]

    @property
    def db(self) -> np.ndarray:
        """The coefficients in dB units: :attr:`data` times ``10 / ln 10``, as
        if the band powers had been taken in dB (``10 log10``) before the DCT."""
        return self.data * DB_PER_NEPER

    @property
    def mel_db(self) -> np.ndarray:
        """The floored mel spectrogram in dB, shape ``(n_channels, n_mels, n_windows)``."""
        return 10 * np.log10(self._floored(self.mel_power))

    def deltas(self, order: int = 1, width: int = 5) -> np.ndarray:
        """Time derivatives of the coefficients, per time window, computed as
        librosa's ``feature.delta`` (see :func:`delta_features`); ``width=9``
        is librosa's default. Same shape as :attr:`data`.

        With a 10 ms hop, the first-order deltas of width 5 act as a
        band-pass filter on each coefficient's track, strongest near 14 Hz:
        they pick out changes at the rate of syllables and phonemes."""
        return delta_features(self.data, order=order, width=width, axis=-1)

    def envelope(self, f) -> np.ndarray:
        """The smoothed power spectrum the kept coefficients stand for, at
        frequencies ``f`` [Hz], shape ``(n_channels, len(f), n_windows)``.

        The lifter is undone, the missing coefficients are taken as zero, and
        the inverse DCT gives a smoothed log band power at each band centre;
        between centres it is interpolated linearly in mel, and held constant
        beyond the first and last. It is a display of what the coefficients
        keep, for plotting against a spectrum or envelope, not an estimate of
        the spectrum: with as many coefficients as bands it returns the
        floored band powers at the centres, which are sums over bands, not
        spectral densities."""
        n_channels, n_mfcc, n_windows = self.data.shape
        n_mels = len(self.cfs)
        padded = np.zeros((n_channels, n_mels, n_windows))
        padded[:, :n_mfcc] = self.data / self._lifter_gains(n_mfcc)[:, None]
        smoothed_log_power = idct(padded, type=2, norm="ortho", axis=1)
        centre_mels = freq_to_mel(self.cfs, self.mel_scale)
        query_mels = freq_to_mel(np.asarray(f, dtype=float), self.mel_scale)
        # Linear interpolation is linear in the values, so it is a matrix:
        # column m interpolates the m-th unit vector.
        interpolation = np.stack(
            [np.interp(query_mels, centre_mels, unit) for unit in np.eye(n_mels)], axis=1
        )
        log_power = np.einsum("qm,cmw->cqw", interpolation, smoothed_log_power)
        return np.exp(log_power)

    def envelope_view(self, f=None) -> GridEnvelope:
        """:meth:`envelope` on this analysis's time windows, at frequencies
        ``f`` (by default its FFT bins), read as ``env(t, f)``, so it goes
        wherever a spectral envelope is taken
        (:func:`~sonore.views.spectral_envelope.warp_frequency`,
        :func:`~sonore.views.world.world_synthesize`,
        :func:`~sonore.sources.waveforms.harmonic_complex`). It holds band
        powers, sums over triangles that widen with frequency, so with
        ``triangles="height"`` it tilts upward against a spectral density;
        ``triangles="area"`` removes most of that tilt."""
        freqs = np.arange(self.n_fft // 2 + 1) * self.fs / self.n_fft if f is None else np.asarray(f, float)
        return GridEnvelope(self.envelope(freqs), self.t, freqs)

    def plot(self, ax=None, channel: int = 0, kind: str = "mfcc", **kwargs):
        """The coefficients against time (``kind="mfcc"``) or the mel
        spectrogram in dB (``kind="mel"``); see :func:`~sonore.plotting.plot_mfcc`."""
        from sonore.plotting import plot_mfcc

        return plot_mfcc(self, ax=ax, channel=channel, kind=kind, **kwargs)
