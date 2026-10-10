"""Cross-checks for docs/design/views/lpc.md (C13, C14).

C13 compares LPC formants of the gallery sentences with Praat's
``To Formant (burg)`` (through parselmouth): Burg's algorithm on a Gaussian
window, after resampling to twice the formant ceiling (5000 Hz for the male
talker, 5500 Hz for the female, Praat's advice), 5 formants. Neither is the
truth for a recording, so this measures agreement, and how often a track
jumps by more than 20% between neighboring time windows, which is a symptom
of mislabeled formants. The LPC side is the formula-level code of
``tools/check_lpc_claims.py``.

C14 compares ways of reading formants off a spectral envelope, on the
synthesized vowels whose formants are known: the LPC roots (Praat's recipe),
and the peaks of sonore's cepstral envelope, CheapTrick envelope and MFCC
envelope.

Praat is a development-time dependency only:

    pip install praat-parselmouth
    PYTHONPATH=src python tools/crosscheck_lpc.py
"""

import numpy as np
import parselmouth
import soundfile as sf
from check_lpc_claims import (
    CEILINGS,
    FEMALE,
    FS,
    HIGHER,
    MALE,
    SPEECH,
    by_count,
    candidates,
    ceiling_candidates,
    preemphasize,
    track,
    vowel,
)

import sonore as so

SPEAKERS = {"bdl": "male", "slt": "female"}
FREQS = np.arange(0, 5001, 2.0)  # where the envelopes are read for peak picking


def report(claim, text, value):
    print(f"{claim:4s} {text:<78s} {value:.4g}")


def jump_fraction(estimate, voiced):
    """Fraction of neighboring voiced time windows where a formant moves by more than 20% or goes
    missing, per formant."""
    ratio = np.abs(np.log(estimate[1:] / estimate[:-1]))[voiced[1:] & voiced[:-1]]
    return (np.isnan(ratio) | (ratio > np.log(1.2))).mean(axis=0)


def praat_comparison():
    for speaker, label in SPEAKERS.items():
        signal, fs = sf.read(SPEECH / f"{speaker}_arctic_a0131.flac")
        assert fs == FS
        praat = parselmouth.Sound(signal, fs).to_formant_burg(
            time_step=0.005, max_number_of_formants=5, maximum_formant=CEILINGS[label], window_length=0.025
        )
        harvest = np.loadtxt(SPEECH / f"{speaker}_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
        inside = (harvest[:, 0] > 0.03) & (harvest[:, 0] < len(signal) / FS - 0.03)
        times, voiced = harvest[inside, 0], harvest[inside, 1] > 0
        reference = np.array(
            [[praat.get_value_at_time(number, time) for number in (1, 2, 3)] for time in times]
        )
        found_16k = candidates(signal, times)
        found_ceiling = ceiling_candidates(signal, times, CEILINGS[label])
        estimates = {
            "16 kHz, order 18, bandwidth < 400 Hz, by count": by_count(found_16k),
            "16 kHz, order 18, bandwidth < 400 Hz, tracker": track(found_16k),
            "Praat's recipe, by count": by_count(found_ceiling),
            "Praat's recipe, tracker": track(found_ceiling),
        }
        report("C13", f"{speaker}: voiced time windows compared", voiced.sum())
        for name, estimate in estimates.items():
            difference = np.abs(estimate[voiced] - reference[voiced])
            for index in range(3):
                report(
                    "C13",
                    f"{speaker}, {name}: F{index + 1} median |diff|",
                    np.nanmedian(difference[:, index]),
                )
            jumps = jump_fraction(estimate, voiced)
            for index in range(3):
                report("C13", f"{speaker}, {name}: F{index + 1} jumps", jumps[index])
        jumps = jump_fraction(reference, voiced)
        for index in range(3):
            report("C13", f"{speaker}, Praat: F{index + 1} jumps", jumps[index])


def envelope_peaks(freqs, log_envelope, low=90.0, high=5000.0):
    """Local maxima of a log envelope sampled on an even grid, refined by a parabola."""
    middle = log_envelope[1:-1]
    index = np.flatnonzero((middle > log_envelope[:-2]) & (middle >= log_envelope[2:])) + 1
    before, at, after = log_envelope[index - 1], log_envelope[index], log_envelope[index + 1]
    peaks = freqs[index] + 0.5 * (before - after) / (before - 2 * at + after) * (freqs[1] - freqs[0])
    return peaks[(peaks > low) & (peaks < high)]


def first_three(values):
    out = np.full(3, np.nan)
    out[: min(3, len(values))] = values[:3]
    return out


def nearest_to(peaks, truth):
    """The peak nearest each true formant: the best any labeling could do."""
    return np.array([peaks[np.argmin(np.abs(peaks - target))] if len(peaks) else np.nan for target in truth])


def method_comparison():
    """F1-F3 of each synthesized vowel at 40 time windows 2.5 ms apart (several positions within
    a period), by each method, labeled by count and, as the best case, by the peak nearest each
    true formant. The sound is pre-emphasized for every method
    (inside Praat's recipe for LPC) so that the glottal tilt does not move the peaks."""
    errors = {}
    for label, vowels, f0 in (("male", MALE, 120), ("female", FEMALE, 220)):
        for f123 in vowels.values():
            signal, true_f0 = vowel((*f123, *HIGHER[label]), f0, duration=0.6)
            sound = so.Sound(preemphasize(signal), FS)
            times = 0.25 + np.arange(40) * 0.0025
            found = {"LPC roots": [freqs for freqs, _ in ceiling_candidates(signal, times, CEILINGS[label])]}

            stft = so.STFT(sound, win_dur=0.040, hop_dur=0.0025)
            cepstrum = so.Cepstrum(stft).lifter(0.5 / true_f0)
            log_envelope = np.log(cepstrum.envelope()[0])
            bin_freqs = np.arange(log_envelope.shape[0]) * FS / cepstrum.n_fft
            columns = [np.argmin(np.abs(cepstrum.t - time)) for time in times]
            found["cepstral envelope"] = [envelope_peaks(bin_freqs, log_envelope[:, i]) for i in columns]

            cheaptrick = so.cheaptrick(sound, (times, np.full(len(times), true_f0)))
            log_envelope = np.log(cheaptrick(times, FREQS)[0])
            found["CheapTrick"] = [envelope_peaks(FREQS, log_envelope[:, i]) for i in range(len(times))]

            mfcc = so.MFCC(sound, triangles="area", hop_dur=0.0025)
            log_envelope = np.log(mfcc.envelope(FREQS)[0])
            columns = [np.argmin(np.abs(mfcc.t - time)) for time in times]
            found["MFCC envelope"] = [envelope_peaks(FREQS, log_envelope[:, i]) for i in columns]

            for method, peaks_per_window in found.items():
                by_order = np.array([first_three(peaks) for peaks in peaks_per_window])
                nearest = np.array([nearest_to(peaks, f123) for peaks in peaks_per_window])
                for labeling, estimate in (("by count", by_order), ("nearest to the truth", nearest)):
                    errors.setdefault((method, labeling, label), []).append(
                        (np.abs(estimate - f123), np.array(f123))
                    )
    for (method, labeling, label), pairs in errors.items():
        error = np.concatenate([e for e, _ in pairs])
        relative = np.concatenate([e / truth for e, truth in pairs])
        gross = np.isnan(relative) | (relative > 0.1)
        name = f"{method}, {labeling}, {label}"
        report("C14", f"{name}: F1-F3 median |error| [Hz]", np.nanmedian(error))
        report("C14", f"{name}: gross errors (missing or > 10%), fraction", gross.mean())


if __name__ == "__main__":
    praat_comparison()
    method_comparison()
