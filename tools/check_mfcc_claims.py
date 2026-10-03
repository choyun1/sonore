"""Numerical checks for the claims in docs/design/views/mfcc.md (C1-C7).

Like the other claim checkers, this is deliberately independent of sonore:
only NumPy, SciPy and soundfile (to read the gallery sentence), with every
scale, filter and transform written out from its formula. Each line prints
the claim number and the number that supports it.

    python tools/check_mfcc_claims.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent.parent
SPEECH = ROOT / "docs" / "speech" / "bdl_arctic_a0131.flac"
DB_PER_NEPER = 10 / np.log(10)  # 10 log10(e): natural-log units to dB of power
rng = np.random.default_rng(0)


def report(claim, text, value):
    print(f"{claim:4s} {text:<78s} {value:.4g}")


# ---------------------------------------------------------------- scales


def htk_mel(freq):
    """O'Shaughnessy's mel formula as used by HTK: 2595 log10(1 + f / 700)."""
    return 2595 * np.log10(1 + np.asarray(freq, float) / 700)


def htk_mel_to_hz(mel):
    return 700 * (10 ** (np.asarray(mel, float) / 2595) - 1)


SLANEY_HZ_PER_MEL = 200 / 3  # linear part: 15 mel at 1000 Hz
SLANEY_LOG_STEP = np.log(6.4) / 27  # log part: 27 mel per factor 6.4


def slaney_mel(freq):
    """Slaney's Auditory Toolbox mel: linear below 1 kHz, logarithmic above."""
    freq = np.asarray(freq, float)
    linear = freq / SLANEY_HZ_PER_MEL
    logarithmic = 15 + np.log(np.maximum(freq, 1e-12) / 1000) / SLANEY_LOG_STEP
    return np.where(freq < 1000, linear, logarithmic)


def slaney_mel_to_hz(mel):
    mel = np.asarray(mel, float)
    return np.where(mel < 15, mel * SLANEY_HZ_PER_MEL, 1000 * np.exp(SLANEY_LOG_STEP * (mel - 15)))


def erb_number(freq):
    """Glasberg & Moore (1990): 21.4 log10(1 + 0.00437 f)."""
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(freq, float))


def erb_bandwidth(freq):
    return 24.7 * (4.37 * np.asarray(freq, float) / 1000 + 1)


# ---------------------------------------------------------------- filterbank


def mel_edges(n_mels, f_lo, f_hi, scale="htk"):
    to_mel, to_hz = (htk_mel, htk_mel_to_hz) if scale == "htk" else (slaney_mel, slaney_mel_to_hz)
    return to_hz(np.linspace(to_mel(f_lo), to_mel(f_hi), n_mels + 2))


def mel_matrix(n_mels, n_fft, fs, f_lo=0.0, f_hi=None, scale="htk", area_normalized=False):
    """Triangles on the exact bin frequencies: filter i rises from edge i to
    edge i+1 and falls to edge i+2, height 1 (HTK) or area-normalized
    (Slaney's 2 / (edge[i+2] - edge[i]))."""
    f_hi = fs / 2 if f_hi is None else f_hi
    edges = mel_edges(n_mels, f_lo, f_hi, scale)
    bin_freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    weights = np.zeros((n_mels, len(bin_freqs)))
    for band in range(n_mels):
        rising = (bin_freqs - edges[band]) / (edges[band + 1] - edges[band])
        falling = (edges[band + 2] - bin_freqs) / (edges[band + 2] - edges[band + 1])
        weights[band] = np.maximum(0, np.minimum(rising, falling))
        if area_normalized:
            weights[band] *= 2 / (edges[band + 2] - edges[band])
    return weights, edges


# ---------------------------------------------------------------- analysis


def time_windows(signal, fs, win_dur=0.025, hop_dur=0.010):
    win_len, hop_len = round(win_dur * fs), round(hop_dur * fs)
    n_windows = 1 + (len(signal) - win_len) // hop_len
    starts = hop_len * np.arange(n_windows)
    return signal[starts[:, None] + np.arange(win_len)[None, :]]


def power_spectra(segments, n_fft=512, window="hamming"):
    win_len = segments.shape[1]
    taper = np.hamming(win_len) if window == "hamming" else np.ones(win_len)
    return np.abs(np.fft.rfft(segments * taper, n_fft)) ** 2


def log_mel(power, weights):
    return np.log(np.maximum(power @ weights.T, 1e-300))


def mfcc_from_log_mel(log_mel_energies, n_mfcc=13):
    return dct(log_mel_energies, type=2, norm="ortho", axis=-1)[..., :n_mfcc]


# ---------------------------------------------------------------- vowels

# Formant frequencies are rounded Peterson & Barney (1952) male averages; the
# bandwidths are the checker's choice.
VOWELS = {
    "a": [(730, 60), (1090, 100), (2440, 120), (3400, 175)],
    "i": [(270, 60), (2290, 100), (3010, 120), (3400, 175)],
}


def vowel_gain(freqs, vowel, fs):
    """|H(f)| of an all-pole cascade of second-order resonators."""
    z = np.exp(-2j * np.pi * np.asarray(freqs, float) / fs)
    denominator = np.ones_like(z)
    for centre, bandwidth in VOWELS[vowel]:
        radius = np.exp(-np.pi * bandwidth / fs)
        denominator *= 1 - 2 * radius * np.cos(2 * np.pi * centre / fs) * z + radius**2 * z**2
    return 1 / np.abs(denominator)


def harmonic_vowel(f0, vowel, fs, duration=0.2):
    times = np.arange(round(duration * fs)) / fs
    harmonics = np.arange(1, int(0.5 * fs / f0))
    harmonics = harmonics[harmonics * f0 < 0.45 * fs]
    amplitudes = vowel_gain(harmonics * f0, vowel, fs)
    return (amplitudes[:, None] * np.cos(2 * np.pi * f0 * harmonics[:, None] * times[None, :])).sum(0)


def main():
    fs = 16000
    n_fft = 512
    n_mels, n_mfcc = 26, 13

    # C1. The DCT-II is the DFT of the log mel spectrum mirrored to even symmetry.
    values = rng.standard_normal(n_mels)
    mirrored = np.concatenate([values, values[::-1]])
    via_dft = np.fft.fft(mirrored)[:n_mels] * np.exp(-1j * np.pi * np.arange(n_mels) / (2 * n_mels))
    direct = dct(values, type=2)
    report(
        "C1",
        "DCT-II vs half-sample-shifted DFT of the mirrored sequence, max abs diff",
        np.abs(via_dft - direct).max(),
    )
    report("C1", "  imaginary part left after the shift, max", np.abs(via_dft.imag).max())
    basis = dct(np.eye(n_mels), type=2, norm="ortho", axis=0)
    report(
        "C1",
        "orthonormal DCT-II: max |B B^T - I| (Parseval holds)",
        np.abs(basis @ basis.T - np.eye(n_mels)).max(),
    )

    # C2. Mel and ERB-number scales, normalized over 0-8 kHz.
    freqs = np.linspace(0, 8000, 8001)

    def normalized(scale):
        return scale(freqs) / scale(8000.0)

    report(
        "C2",
        "max |HTK mel - ERB number| on 0-8 kHz, both scaled to 0..1",
        np.abs(normalized(htk_mel) - normalized(erb_number)).max(),
    )
    report(
        "C2",
        "max |HTK mel - Slaney mel| on 0-8 kHz, both scaled to 0..1",
        np.abs(normalized(htk_mel) - normalized(slaney_mel)).max(),
    )
    htk_edges = mel_edges(n_mels, 0, 8000, "htk")
    slaney_edges = mel_edges(n_mels, 0, 8000, "slaney")
    report(
        "C2",
        "26 bands 0-8 kHz: max |HTK centre - Slaney centre| [Hz]",
        np.abs(htk_edges[1:-1] - slaney_edges[1:-1]).max(),
    )
    centres = htk_edges[1:-1]
    triangle_widths = (
        htk_edges[2:] - htk_edges[:-2]
    ) / 2  # equivalent rectangular width of a unit-height triangle
    for target in (250, 500, 1000, 2000, 4000):
        band = np.argmin(np.abs(centres - target))
        report(
            "C2",
            f"  HTK band at {centres[band]:6.0f} Hz: triangle width / ERB",
            triangle_widths[band] / erb_bandwidth(centres[band]),
        )
    report("C2", "  lowest HTK band: triangle base [Hz]", htk_edges[2] - htk_edges[0])

    # C3. Unit-height triangles sum to 1 between the first and last centre;
    # Slaney's area normalization adds a per-band constant to the log, so a
    # constant vector to the MFCCs, the same for every time window.
    weights, edges = mel_matrix(n_mels, n_fft, fs, scale="htk")
    bin_freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    inside = (bin_freqs >= edges[1]) & (bin_freqs <= edges[-2])
    report(
        "C3",
        "unit-height triangles: max |sum - 1| between first and last centre",
        np.abs(weights.sum(0)[inside] - 1).max(),
    )

    speech, speech_fs = sf.read(SPEECH)
    assert speech_fs == fs
    power = power_spectra(time_windows(speech, fs), n_fft)
    loud = power.sum(1) > 1e-6 * power.sum(1).max()
    power = power[loud]
    weights_area, _ = mel_matrix(n_mels, n_fft, fs, scale="htk", area_normalized=True)
    coefficient_shift = mfcc_from_log_mel(log_mel(power, weights_area)) - mfcc_from_log_mel(
        log_mel(power, weights)
    )
    report(
        "C3",
        "area vs unit-height MFCCs on the sentence: std of shift over time windows (max)",
        coefficient_shift.std(0).max(),
    )
    report(
        "C3", "  size of that constant shift, max over coefficients", np.abs(coefficient_shift.mean(0)).max()
    )

    # C4. Many sounds give the same MFCCs: the mel matrix has a null space.
    report("C4", "mel matrix 26 x 257: rank", np.linalg.matrix_rank(weights))
    voiced = np.argmax(power.sum(1))
    spectrum = power[voiced]
    # A relative change u of each bin, |u| <= 0.9, that no mel band sees:
    # u lies in the null space of (weights times the spectrum).
    _, _, right_vectors = np.linalg.svd(weights * spectrum[None, :])
    null_space = right_vectors[n_mels:]
    relative_change = null_space.T @ (null_space @ rng.choice([-1.0, 1.0], spectrum.size))
    relative_change *= 0.9 / np.abs(relative_change).max()
    altered = spectrum * (1 + relative_change)
    mfcc_change = np.abs(
        mfcc_from_log_mel(log_mel(altered, weights)) - mfcc_from_log_mel(log_mel(spectrum, weights))
    ).max()
    db_change = np.abs(10 * np.log10(altered / spectrum))
    report("C4", "loudest time window, spectrum moved in the null space: max |MFCC change|", mfcc_change)
    report("C4", "  median |change| of the power spectrum over bins [dB]", np.median(db_change))
    report("C4", "  bins changed by more than 3 dB [%]", 100 * np.mean(db_change > 3))
    log_mel_speech = log_mel(power, weights)
    truncated = dct(
        np.pad(mfcc_from_log_mel(log_mel_speech, n_mfcc), ((0, 0), (0, n_mels - n_mfcc))),
        type=3,
        norm="ortho",
        axis=-1,
    )
    residual_db = DB_PER_NEPER * np.sqrt(np.mean((truncated - log_mel_speech) ** 2, axis=1))
    report(
        "C4", "13 of 26 coefficients: rms error of the log mel spectrum, median [dB]", np.median(residual_db)
    )
    report("C4", "  90th percentile over time windows [dB]", np.percentile(residual_db, 90))
    report(
        "C4", "  numbers kept per time window vs power-spectrum bins (13 / 257)", n_mfcc / (n_fft // 2 + 1)
    )

    # C5. MFCCs of a vowel change with F0, because the low mel bands are
    # narrower than the harmonic spacing. Distance: rms over the 26 bands of
    # the 13-coefficient smoothed log mel spectra, c0 (level) excluded, in dB.
    def smoothed_shape(log_mel_energies):
        coefficients = mfcc_from_log_mel(log_mel_energies, n_mfcc)
        coefficients[..., 0] = 0
        return coefficients

    def distance_db(first, second):
        return DB_PER_NEPER * np.sqrt(np.sum((first - second) ** 2) / n_mels)

    dense_freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    true_shape = {
        vowel: smoothed_shape(log_mel(vowel_gain(dense_freqs, vowel, fs) ** 2, weights)) for vowel in VOWELS
    }
    report(
        "C5",
        "/a/ vs /i/ envelopes (flat excitation): distance [dB]",
        distance_db(true_shape["a"], true_shape["i"]),
    )
    shapes = {}
    for vowel in VOWELS:
        for f0 in (100, 150, 200, 250, 300):
            signal = harmonic_vowel(f0, vowel, fs)
            middle = len(signal) // 2 - 200
            spectrum = power_spectra(signal[None, middle : middle + 400], n_fft)[0]
            shapes[vowel, f0] = smoothed_shape(log_mel(spectrum, weights))
    for f0 in (100, 200, 300):
        report(
            "C5",
            f"/a/ vs /i/, both at F0 {f0} Hz: distance [dB]",
            distance_db(shapes["a", f0], shapes["i", f0]),
        )
    f0s = (100, 150, 200, 250, 300)
    same_vowel = [
        distance_db(shapes[vowel, low], shapes[vowel, high])
        for vowel in VOWELS
        for low in f0s
        for high in f0s
        if low < high
    ]
    report("C5", "same vowel at two F0s in 100-300 Hz (/a/ and /i/): largest distance [dB]", max(same_vowel))
    report("C5", "  median distance [dB]", np.median(same_vowel))
    lowest_spacing = htk_edges[1] - htk_edges[0]
    report("C5", "spacing of the lowest HTK band centres [Hz]", lowest_spacing)
    report(
        "C5",
        "band centres are closer than 200 Hz up to [Hz]",
        centres[np.nonzero(np.diff(htk_edges)[:-1] < 200)[0].max()],
    )

    # C6. Deltas: the regression slope over +-2 time windows is a bandpass
    # modulation filter.
    half_width = 2
    offsets = np.arange(-half_width, half_width + 1)
    taps = offsets / np.sum(offsets**2)
    track = rng.standard_normal(50)
    regression = np.convolve(track, taps[::-1], mode="valid")
    slopes = np.array(
        [np.polyfit(offsets, track[i : i + 2 * half_width + 1], 1)[0] for i in range(len(regression))]
    )
    report(
        "C6",
        "delta formula vs least-squares slope over 5 time windows, max abs diff",
        np.abs(regression - slopes).max(),
    )
    hop = 0.010
    rates = np.linspace(0.01, 50, 50000)
    response = np.abs(np.exp(-2j * np.pi * rates[:, None] * hop * offsets[None, :]) @ taps)
    peak = np.argmax(response)
    report("C6", "delta response peak, at a 10 ms hop [Hz]", rates[peak])
    report(
        "C6",
        "  gain there, relative to an ideal differentiator at 1 Hz",
        response[peak] / response[np.argmin(np.abs(rates - 1))],
    )
    report(
        "C6",
        "  ratio |delta| / ideal derivative per time window at 1 Hz",
        response[np.argmin(np.abs(rates - 1))] / (2 * np.pi * 1 * hop),
    )
    report("C6", "  response at 50 Hz (Nyquist of a 10 ms hop)", response[-1])
    above_half = rates[response >= response[peak] / np.sqrt(2)]
    report("C6", "  -3 dB band, low edge [Hz]", above_half.min())
    report("C6", "  -3 dB band, high edge [Hz]", above_half.max())

    # C7. Log units: dB is the natural log times 10 / ln 10, so dB MFCCs are
    # the natural-log ones scaled; a floor relative to the loudest cell clips.
    natural = mfcc_from_log_mel(log_mel(power, weights))
    decibel = mfcc_from_log_mel(10 * np.log10(power @ weights.T))
    report(
        "C7",
        "dB MFCCs / natural-log MFCCs, max |ratio - 10/ln10|",
        np.abs(decibel / natural - DB_PER_NEPER).max(),
    )
    all_power = power_spectra(time_windows(speech, fs), n_fft)
    mel_db = 10 * np.log10(np.maximum(all_power @ weights.T, 1e-10))
    report(
        "C7",
        "sentence, all time windows: mel cells more than 80 dB below the max [%]",
        100 * np.mean(mel_db < mel_db.max() - 80),
    )


if __name__ == "__main__":
    main()
