"""Checks so.source_filter_synthesize, which rebuilds a voice from an F0 track, a spectral
envelope and an aperiodicity in place of WORLD's synthesis.

The recipe, for an envelope S (power), a noise share a and an F0 track:

- harmonics: so.harmonic_complex on the track, harmonic n carrying the power S (1 - a)
  over one harmonic spacing. The envelope reads unit white noise as 1 (its window has
  unit energy), so a power density S with a harmonic every F0 is a_n^2 / 2 = 2 F0 S / fs.
- noise: white noise whose STFT is multiplied by sqrt(S a), then inverted. Unvoiced time
  windows have a = 1, so there the noise carries all of S.
- the two added.

Part 1, the calibration alone: the recipe given a known envelope and a known noise share
(no analysis), with the output measured by its long-term spectrum (the level) and by
so.harmonic_aperiodicity (the share). This is the recipe's own error.

Part 2, end to end on synthetic vowels whose envelope and share are known: analyzed with
so.pitch_adaptive_envelope and so.band_aperiodicity on the true F0, rebuilt by the recipe,
and the rebuilt sound's share measured as in part 1. The analyses' errors add to the
recipe's here.

Part 3, the bdl and slt recordings rebuilt by the recipe and by WORLD's synthesis
(pyworld.synthesize, development only; part 3 is skipped without it) from the same
analyses (F0 from so.f0_track): the level-free RMS dB between the log mel spectra (40
bands) of the original and the rebuilt sound, median over voiced and over unvoiced time
windows, as in tools/compare_voice_methods.py.

    python tools/check_voice_recipe_claims.py
"""

import sys

import numpy as np
from scipy.signal import freqz, lfilter

import sonore as so

try:
    import pyworld
except ImportError:  # WORLD's synthesis is only a comparison
    pyworld = None

FS = 16000
HOP = 0.005
results = []


def report(name, value, passed, detail=""):
    results.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {value}" + (f"  ({detail})" if detail else ""))


# ------------------------------------------------------------ known vowels
def vowel_filter(fs):
    """An all-pole filter with four formants of a vowel like /a/."""
    poles = []
    for frequency, bandwidth in [(700, 130), (1220, 70), (2600, 160), (3300, 250)]:
        radius = np.exp(-np.pi * bandwidth / fs)
        poles += [radius * np.exp(2j * np.pi * frequency / fs), radius * np.exp(-2j * np.pi * frequency / fs)]
    return np.real(np.poly(poles))


def true_views(f0, noise_db, duration):
    """The envelope (the filter's power response, the excitation having unit density) and
    a share that is the same at every frequency, as views on a 5 ms grid."""
    times = np.arange(int(duration / HOP)) * HOP
    freqs = np.arange(513) * FS / 1024
    _, response = freqz([1.0], vowel_filter(FS), worN=freqs, fs=FS)
    power = np.repeat((np.abs(response) ** 2)[None, :, None], len(times), axis=2)
    share = 10 ** (noise_db / 10) / (1 + 10 ** (noise_db / 10))
    envelope = so.SpectralEnvelope(power, times, freqs, fs=FS)
    aperiodicity = so.Aperiodicity(np.full_like(power, np.sqrt(share)), times, FS, "truth")
    return (times, np.full(len(times), f0)), envelope, aperiodicity, share


def vowel(f0, noise_db, duration, rng):
    """Harmonics of f0 at unit density plus white noise noise_db re the harmonics, through
    the filter: the share of noise is the same at every frequency."""
    sample_times = np.arange(int(duration * FS)) / FS
    numbers = np.arange(1, int(0.45 * FS / f0))
    phases = rng.uniform(0, 2 * np.pi, len(numbers))
    excitation = (
        2 * np.sqrt(f0 / FS) * np.cos(2 * np.pi * f0 * np.outer(sample_times, numbers) + phases).sum(axis=1)
    )
    excitation += rng.standard_normal(len(sample_times)) * np.sqrt(0.9 * 10 ** (noise_db / 10))
    return so.Sound(lfilter([1.0], vowel_filter(FS), excitation), FS)


def measured(sound, track, freqs_band=(300, 6000)):
    """The rebuilt sound's share (so.harmonic_aperiodicity, power-weighted over the band)
    and its level against the filter (long-term spectrum, in dB)."""
    inside = (track[0] > 0.1) & (track[0] < track[0][-1] - 0.1)
    share = so.harmonic_aperiodicity(sound, track)
    envelope = so.pitch_adaptive_envelope(sound, track)
    band_share = share.bands(list(freqs_band), envelope=envelope)[0, 0, inside]
    return 10 * np.log10(np.mean(band_share)), envelope


def part1():
    print("Part 1: the recipe given a known envelope and share (16 kHz, /a/ filter)")
    rng = np.random.default_rng(1)
    for f0 in (120.0, 220.0):
        for noise_db in (-20.0, -10.0, -3.0):
            track, envelope, aperiodicity, share = true_views(f0, noise_db, 1.0)
            out = so.source_filter_synthesize(track, envelope, aperiodicity, duration=1.0, rng=rng)
            share_db, rebuilt = measured(out, track)
            inside = (track[0] > 0.1) & (track[0] < 0.9)
            band = (rebuilt.f > 300) & (rebuilt.f < 6000)
            level = np.mean(rebuilt.db[0][band][:, inside] - envelope.db[0][band][:, inside])
            truth_db = 10 * np.log10(share)
            # the share within 2 dB and the level within 1 dB: the measurement itself adds
            # error (harmonic_aperiodicity reads a known share within about 1.5 dB in
            # tests/views/test_aperiodicity.py, the envelope's level within 0.3 dB in
            # tools/check_paper_vocoder_claims.py), and the noise is one random draw
            report(
                f"F0 {f0:.0f} Hz, share {truth_db:5.1f} dB",
                f"rebuilt share {share_db:5.1f} dB, level {level:+.2f} dB",
                abs(share_db - truth_db) < 2.0 and abs(level) < 1.0,
            )


def part2():
    print("Part 2: analyzed, then rebuilt (16 kHz, /a/ filter, true F0)")
    rng = np.random.default_rng(2)
    for f0 in (120.0, 220.0):
        for noise_db in (-20.0, -10.0):
            sound = vowel(f0, noise_db, 1.0, rng)
            track = (np.arange(int(1.0 / HOP)) * HOP, np.full(int(1.0 / HOP), f0))
            original_db, _ = measured(sound, track)
            envelope = so.pitch_adaptive_envelope(sound, track)
            for name, aperiodicity in (
                ("D4C", so.band_aperiodicity(sound, track)),
                ("harmonic residual", so.harmonic_aperiodicity(sound, track)),
            ):
                out = so.source_filter_synthesize(track, envelope, aperiodicity, duration=1.0, rng=rng)
                share_db, _ = measured(out, track)
                label = f"F0 {f0:.0f} Hz, noise {noise_db:5.1f} dB, {name}"
                value = f"original {original_db:5.1f} dB, rebuilt {share_db:5.1f} dB"
                if name == "harmonic residual":
                    # this measure reads the share at every frequency, so the rebuilt share
                    # should come back within the 2 dB of part 1
                    report(label, value, abs(share_db - original_db) < 2.0)
                else:
                    # D4C at 16 kHz measures one band, at 3 kHz, and runs down to -60 dB at
                    # 0 Hz, so below 3 kHz, where most of a vowel's power is, it reads far
                    # too periodic; reported, not checked
                    print(f"    {label}: {value}")


def log_mel_spectrogram(samples, n_mels=40):
    win, hop, n_fft = 400, 160, 512
    starts = np.arange(0, len(samples) - win + 1, hop)
    segments = np.stack([samples[s : s + win] for s in starts]) * np.hamming(win)
    power = np.abs(np.fft.rfft(segments, n_fft, axis=1)) ** 2
    bin_freqs = np.fft.rfftfreq(n_fft, 1 / FS)
    # HTK mel edges, evenly spaced from 0 Hz to Nyquist
    highest_mel = 2595 * np.log10(1 + FS / 2 / 700)
    edges = 700 * (10 ** (np.linspace(0, highest_mel, n_mels + 2) / 2595) - 1)
    weights = np.zeros((n_mels, len(bin_freqs)))
    for band in range(n_mels):
        lower, center, upper = edges[band : band + 3]
        weights[band] = np.maximum(
            0, np.minimum((bin_freqs - lower) / (center - lower), (upper - bin_freqs) / (upper - center))
        )
    return (starts + win / 2) / FS, 10 * np.log10(np.maximum(power @ weights.T, 1e-20))


def level_free_rms(a, b):
    difference = a - b
    return np.sqrt(np.mean((difference - difference.mean()) ** 2))


def part3():
    print("Part 3: bdl and slt rebuilt; level-free RMS dB between log mel spectra, median")
    for speaker in ("bdl", "slt"):
        sound = so.load(f"docs/speech/{speaker}_arctic_a0131.flac")
        times = np.arange(int(sound.duration / HOP)) * HOP
        tracked = so.f0_track(sound)
        values = np.where(tracked.voiced[0], tracked.f0[0], 0.0)
        track = (times, values[np.clip(np.searchsorted(tracked.t, times), 0, len(values) - 1)])
        envelope = so.pitch_adaptive_envelope(sound, track)
        aperiodicity = so.band_aperiodicity(sound, track)
        mel_t, original = log_mel_spectrogram(sound.data[:, 0])
        voiced = track[1][np.clip(np.round(mel_t / HOP).astype(int), 0, len(times) - 1)] > 0
        scores = {}
        rebuilt = {
            "WORLD's synthesis": pyworld.synthesize(
                np.ascontiguousarray(track[1]),
                np.ascontiguousarray(envelope.data[0].T),
                np.ascontiguousarray(aperiodicity.data[0].T),
                FS,
                HOP * 1000,
            ),
            "recipe": so.source_filter_synthesize(
                track, envelope, aperiodicity, duration=sound.duration, rng=3
            ).data[:, 0],
        }
        for name, samples in rebuilt.items():
            _, spectrum = log_mel_spectrogram(samples)
            count = min(len(original), len(spectrum))
            errors = np.array([level_free_rms(original[w], spectrum[w]) for w in range(count)])
            scores[name] = np.median(errors[voiced[:count]]), np.median(errors[~voiced[:count]])
            voiced_score, unvoiced_score = scores[name]
            print(f"    {speaker} {name:17s}: voiced {voiced_score:.2f} dB, unvoiced {unvoiced_score:.2f} dB")
        # the recipe drops WORLD's pulse shape and its noise, so it may score a little
        # differently; it should not be worse than WORLD's synthesis by more than 0.5 dB
        worst = max(np.subtract(scores["recipe"], scores["WORLD's synthesis"]))
        report(
            f"{speaker}: recipe against WORLD's synthesis, worse by at most", f"{worst:+.2f} dB", worst < 0.5
        )


part1()
part2()
if pyworld is None:
    print("Part 3 skipped: pyworld is not installed")
else:
    part3()
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
