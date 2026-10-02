"""Which F0 tracker and which spectral envelope resynthesize a voice best,
and which carry a voice change best? A comparison for
docs/design/voice-change.md ("Comparing methods").

Every F0 source is mixed with every envelope source and put back together,
unchanged and changed (pitch x 1.5, formants x 1.2), by the operations the
design proposes (prototyped here, as in tools/check_voice_change_claims.py).

Part 1, synthetic vowels, where the truth is exact. Three vowels with male
and with female average formants (Hillenbrand et al. 1995), at two F0s each,
built as tools/check_female_voices.py builds them (envelope and noise share
known at every frequency). The score does not use any of the methods being
compared: the output's harmonics are measured by least squares at their
known frequencies over 0.2-0.6 s and compared, level removed, with the true
envelope over 100-5000 Hz (for the change: the true envelope read at
f / 1.2, at harmonics of 1.5 F0). The F0 error is scored separately.

Part 2, the bdl and slt recordings of the same sentence, resynthesized
unchanged. There is no truth, so the score is the distance between the
original's and the output's log mel spectra (40 HTK bands, 25 ms Hamming
time windows, 10 ms hop, level removed per time window, median over windows
the stored or tracked F0 calls voiced). A mel spectrum smooths in the way
the MFCC envelope does, so it favours that envelope; it is a sanity check,
not a ranking.

Sources compared:
  F0:       the truth (synthetic only), so.f0_track, Harvest (pyworld, a
            development-only tool, skipped if missing), Cepstrum.f0 (40 ms Hann)
  envelope: the truth (synthetic only), so.cheaptrick, the cepstral lifter
            (40 ms Hann STFT, lifter at half the median period), and
            so.MFCC's envelope (13 coefficients, its speech defaults), with
            height-1 triangles (the default) and with area-normalized ones
  synthesis: so.world_synthesize (with so.d4c measured on the same F0) and
            so.harmonic_complex (envelope read point by point, no noise)

    python tools/compare_voice_methods.py

Takes a few minutes. Nothing here is a listening test.
"""

import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import freqz

import sonore as so
from sonore.analysis.vocoder import Aperiodicity, SpectralEnvelope, world_fft_size

try:
    import pyworld
except ImportError:  # Harvest is optional
    pyworld = None

FS = 16000.0
HOP = 0.005
SCORED = (100.0, 5000.0)
PITCH, FORMANT = 1.5, 1.2
HILLENBRAND = {
    "male": {"i": (342, 2322, 3000), "a": (768, 1333, 2522), "u": (378, 997, 2343)},
    "female": {"i": (437, 2761, 3372), "a": (936, 1551, 2815), "u": (459, 1105, 2735)},
}
UPPER = {"male": (3500, 4500), "female": (4100, 4900)}
F0S = {"male": (110.0, 150.0), "female": (200.0, 250.0)}
BANDWIDTHS = (60, 100, 120, 175, 250)


# ------------------------------------------------------------ synthetic vowels
def resonator_gain(f, freq, bw):
    T = 1 / FS
    c = -np.exp(-2 * np.pi * bw * T)
    b = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * freq * T)
    return np.abs(freqz([1 - b - c], [1, -b, -c], worN=np.atleast_1d(f), fs=FS)[1])


def envelope_amplitude(f, formants):
    f = np.asarray(f, float)
    gain = resonator_gain(f, 0.0, 100.0)
    for freq, bw in zip(formants, BANDWIDTHS, strict=True):
        gain = gain * resonator_gain(f, freq, bw)
    return gain * np.abs(2 * np.sin(np.pi * f / FS))


def noise_share_db(f):
    return -30 + 25 * np.asarray(f) / 8000


def vowel(f0, formants, rng, duration=0.8):
    f_max = 0.45 * FS
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for k in range(1, int(f_max / f0) + 1):
        fk = k * f0
        amplitude = envelope_amplitude(fk, formants) * np.sqrt(1 - 10 ** (noise_share_db(fk) / 10))
        x += amplitude * np.cos(2 * np.pi * fk * t)
    freqs = np.fft.rfftfreq(n, 1 / FS)
    gain = envelope_amplitude(freqs, formants) * np.sqrt(10 ** (noise_share_db(freqs) / 10) * FS / (4 * f0))
    gain[freqs >= f_max] = 0
    x += np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gain, n)
    return so.Sound(x / np.std(x) * 0.1, FS)


def harmonic_levels_db(sound, f0, start=0.2, stop=0.6):
    """Amplitudes [dB] of the harmonics of f0 below SCORED[1], fitted by least
    squares over start-stop s, and their frequencies."""
    samples = sound.data[int(start * FS) : int(stop * FS), 0]
    t = np.arange(len(samples)) / FS
    freqs = f0 * np.arange(1, int(SCORED[1] / f0) + 1)
    basis = np.hstack([np.cos(2 * np.pi * np.outer(t, freqs)), np.sin(2 * np.pi * np.outer(t, freqs))])
    weights = np.linalg.lstsq(basis, samples, rcond=None)[0]
    amplitudes = np.hypot(weights[: len(freqs)], weights[len(freqs) :])
    return freqs, 20 * np.log10(np.maximum(amplitudes, 1e-12))


def level_free_rms(a_db, b_db):
    difference = np.asarray(a_db) - np.asarray(b_db)
    return float(np.sqrt(np.mean((difference - difference.mean()) ** 2)))


# ------------------------------------------------------------ the interface
class GridEnvelope:
    """Power on any (frequency, time) grid read as env(t, f), shape
    (n_channels, len(f), len(t)): linear in time and in dB over frequency."""

    def __init__(self, power, t, f):
        self.log_power = np.log(np.maximum(power, 1e-30))
        self.t, self.f = np.asarray(t, float), np.asarray(f, float)

    def __call__(self, t, f):
        times, freqs = np.atleast_1d(t), np.atleast_1d(f)
        out = np.empty((self.log_power.shape[0], len(freqs), len(times)))
        for channel, log_power in enumerate(self.log_power):
            over_f = np.array([np.interp(freqs, self.f, column) for column in log_power.T]).T
            out[channel] = np.array([np.interp(times, self.t, row) for row in over_f])
        return np.exp(out)


class TrueEnvelope:
    """The synthetic vowel's own envelope, the same at every time."""

    def __init__(self, formants):
        self.formants = formants

    def __call__(self, t, f):
        power = np.maximum(envelope_amplitude(np.atleast_1d(f), self.formants) ** 2, 1e-30)
        return np.repeat(power[None, :, None], len(np.atleast_1d(t)), axis=2)


class Warped:
    def __init__(self, envelope, ratio):
        self.envelope, self.ratio = envelope, ratio

    def __call__(self, t, f):
        return self.envelope(t, np.asarray(f) / self.ratio)


def scale_f0(track, ratio):
    return track[0], np.asarray(track[1], float) * ratio


def contour_on_grid(times, f0_values, grid):
    times, f0_values = np.asarray(times, float), np.asarray(f0_values, float)
    voiced = f0_values > 0
    if not voiced.any():
        return np.zeros(len(grid))
    nearest = np.abs(times[None, :] - grid[:, None]).argmin(axis=1)
    return np.where(voiced[nearest], np.interp(grid, times[voiced], f0_values[voiced]), 0.0)


def pointwise_amplitude(envelope, times, freqs):
    log_amplitude = 0.5 * np.log(np.maximum(envelope(times, freqs)[0], 1e-30)).T
    interpolator = RegularGridInterpolator((times, freqs), log_amplitude, bounds_error=False, fill_value=None)
    return lambda t, f: np.exp(interpolator(np.column_stack([np.ravel(t), np.ravel(f)]))).reshape(np.shape(t))


# ------------------------------------------------------------ the sources
def f0_sources(sound, grid, true_f0=None):
    sources = {}
    if true_f0 is not None:
        sources["truth"] = np.full(len(grid), true_f0)
    tracked = so.f0_track(sound)
    sources["so.f0_track"] = contour_on_grid(tracked.t, tracked.f0[0], grid)
    if pyworld is not None:
        samples = np.ascontiguousarray(sound.data[:, 0], dtype=float)
        harvest_f0, harvest_t = pyworld.harvest(samples, int(FS), frame_period=HOP * 1000)
        sources["Harvest"] = contour_on_grid(harvest_t, harvest_f0, grid)
    cep_t, cep_f0, _ = so.Cepstrum(so.STFT(sound, win_dur=0.040, hop_dur=HOP)).f0()
    sources["Cepstrum.f0"] = contour_on_grid(cep_t, cep_f0[0], grid)
    return sources


def htk_mel(freq):
    return 2595 * np.log10(1 + np.asarray(freq) / 700)


def htk_mel_to_hz(mel):
    return 700 * (10 ** (np.asarray(mel) / 2595) - 1)


def mfcc_envelope(sound, triangles="height"):
    """so.MFCC's smoothed envelope (13 coefficients, 26 HTK bands, 25 ms
    symmetric Hamming, 10 ms hop) as an env(t, f) view. MFCC.envelope gives
    band powers, sums over triangles that widen with frequency, unless the
    triangles are area-normalized."""
    mfcc = so.MFCC(sound, triangles=triangles)
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    return GridEnvelope(mfcc.envelope(freqs), mfcc.t, freqs)


def envelope_sources(sound, track, formants=None):
    sources = {}
    if formants is not None:
        sources["truth"] = TrueEnvelope(formants)
    sources["CheapTrick"] = so.cheaptrick(sound, track)
    voiced = track[1][track[1] > 0]
    median_f0 = np.median(voiced) if len(voiced) else 100.0
    cepstrum = so.Cepstrum(so.STFT(sound, win_dur=0.040, hop_dur=HOP)).lifter(0.5 / median_f0)
    magnitude = cepstrum.envelope()
    stft_freqs = np.fft.rfftfreq(2 * (magnitude.shape[1] - 1), 1 / FS)
    sources["cepstral"] = GridEnvelope(magnitude**2, cepstrum.t, stft_freqs)
    sources["MFCC (13)"] = mfcc_envelope(sound)
    sources["MFCC, area"] = mfcc_envelope(sound, triangles="area")
    return sources


def synthesize(synth, sound, track, envelope, aperiodicity, grid):
    if synth == "world_synthesize":
        n_fft = world_fft_size(FS)
        freqs = np.arange(n_fft // 2 + 1) * FS / n_fft
        world_envelope = SpectralEnvelope(envelope(grid, freqs), grid, FS, np.nan)
        return so.world_synthesize(track, world_envelope, aperiodicity)
    amplitude = pointwise_amplitude(envelope, grid, np.arange(0, FS / 2 + 1, 10.0))
    return so.harmonic_complex(sound.duration, FS, track, amplitudes=amplitude)


# ------------------------------------------------------------ part 1
def part1():
    print("Part 1: synthetic vowels (3 vowels x male and female formants x 2 F0s)")
    print("  envelope error: level-free RMS dB of the output's harmonics against the truth, 100-5000 Hz")
    print("  F0 error: median |F0 source / true F0 - 1| on the vowel's voiced time windows")
    rng = np.random.default_rng(1)
    results = {}  # (f0 name, env name, synth) -> list of (plain, changed, f0 error)
    labels = {}
    for voice, vowels in HILLENBRAND.items():
        for vowel_name, f123 in vowels.items():
            formants = (*f123, *UPPER[voice])
            for f0 in F0S[voice]:
                sound = vowel(f0, formants, rng)
                grid = np.arange(int(sound.duration / HOP)) * HOP
                for f0_name, values in f0_sources(sound, grid, true_f0=f0).items():
                    track = (grid, values)
                    inner = (grid > 0.2) & (grid < 0.6)
                    voiced_inner = values[inner][values[inner] > 0]
                    f0_error = np.median(np.abs(voiced_inner / f0 - 1)) if len(voiced_inner) else np.nan
                    used_f0 = np.median(voiced_inner) if len(voiced_inner) else f0
                    aperiodicity = so.d4c(sound, track)
                    for env_name, envelope in envelope_sources(sound, track, formants).items():
                        synths = ["world_synthesize"]
                        if f0_name == "truth":
                            synths.append("harmonic_complex")
                        for synth in synths:
                            scores = []
                            for pitch, formant in ((1.0, 1.0), (PITCH, FORMANT)):
                                view = envelope if formant == 1.0 else Warped(envelope, formant)
                                out = synthesize(
                                    synth, sound, scale_f0(track, pitch), view, aperiodicity, grid
                                )
                                freqs, out_db = harmonic_levels_db(out, used_f0 * pitch)
                                true_db = 20 * np.log10(envelope_amplitude(freqs / formant, formants))
                                keep = freqs >= SCORED[0]
                                scores.append(level_free_rms(out_db[keep], true_db[keep]))
                            results.setdefault((f0_name, env_name, synth), []).append((*scores, f0_error))
                            labels.setdefault((f0_name, env_name, synth), []).append(
                                f"{voice} {vowel_name} {f0:.0f} Hz"
                            )
    print(
        f"  {'F0 from':12s} {'envelope':11s} {'synthesizer':17s} | {'F0 error':>8s} | "
        f"{'resynthesis':>11s} {'worst':>6s} | {'x1.5, x1.2':>10s} {'worst':>6s}"
    )
    for (f0_name, env_name, synth), rows in results.items():
        rows = np.array(rows)
        print(
            f"  {f0_name:12s} {env_name:11s} {synth:17s} | {np.nanmedian(rows[:, 2]):8.2%} | "
            f"{np.median(rows[:, 0]):8.2f} dB {rows[:, 0].max():6.2f} | "
            f"{np.median(rows[:, 1]):7.2f} dB {rows[:, 1].max():6.2f} | worst: "
            f"{labels[(f0_name, env_name, synth)][int(np.argmax(rows[:, 0]))]},"
            f" max F0 error {np.nanmax(rows[:, 2]):.1%}"
        )
    print("  (median and worst over the 12 vowels; F0 error is the source's, before synthesis)")
    by_voice = {}
    for (f0_name, env_name, synth), rows in results.items():
        if f0_name == "truth" and synth == "world_synthesize":
            rows = np.array(rows)
            by_voice[env_name] = (np.median(rows[:6, 0]), np.median(rows[6:, 0]))
    print("  with the true F0 and world_synthesize, resynthesis error by voice (male / female vowels):")
    for env_name, (male, female) in by_voice.items():
        print(f"    {env_name:11s} {male:5.2f} / {female:5.2f} dB")


# ------------------------------------------------------------ part 2
def log_mel_spectrogram(sound, n_mels=40):
    samples = sound.data[:, 0]
    win, hop, n_fft = 400, 160, 512
    starts = np.arange(0, len(samples) - win + 1, hop)
    segments = np.stack([samples[s : s + win] for s in starts]) * np.hamming(win)
    power = np.abs(np.fft.rfft(segments, n_fft, axis=1)) ** 2
    bin_freqs = np.fft.rfftfreq(n_fft, 1 / FS)
    edges = htk_mel_to_hz(np.linspace(0, htk_mel(FS / 2), n_mels + 2))
    weights = np.zeros((n_mels, len(bin_freqs)))
    for band in range(n_mels):
        lower, centre, upper = edges[band : band + 3]
        weights[band] = np.maximum(
            0, np.minimum((bin_freqs - lower) / (centre - lower), (upper - bin_freqs) / (upper - centre))
        )
    return (starts + win / 2) / FS, 10 * np.log10(np.maximum(power @ weights.T, 1e-20))


def part2():
    print("Part 2: the bdl and slt sentence, resynthesized unchanged")
    print("  score: level-free RMS dB between log mel spectra (40 bands), median over voiced time windows")
    print(
        f"  {'speaker':7s} {'F0 from':12s} | "
        + " ".join(f"{name:>10s}" for name in ("CheapTrick", "cepstral", "MFCC (13)", "MFCC, area"))
    )
    for speaker in ("bdl", "slt"):
        sound = so.load(f"docs/speech/{speaker}_arctic_a0131.flac")
        grid = np.arange(int(sound.duration / HOP)) * HOP
        mel_t, original = log_mel_spectrogram(sound)
        sources = f0_sources(sound, grid)
        reference_voicing = sources["so.f0_track"]
        voiced_at_mel = contour_on_grid(grid, reference_voicing, mel_t) > 0
        for f0_name, values in sources.items():
            track = (grid, values)
            aperiodicity = so.d4c(sound, track)
            scores = []
            for envelope in envelope_sources(sound, track).values():
                out = synthesize("world_synthesize", sound, track, envelope, aperiodicity, grid)
                _, resynthesized = log_mel_spectrogram(out)
                count = min(len(original), len(resynthesized))
                errors = [
                    level_free_rms(original[w], resynthesized[w])
                    for w in np.flatnonzero(voiced_at_mel[:count])
                ]
                scores.append(np.median(errors))
            print(f"  {speaker:7s} {f0_name:12s} | " + " ".join(f"{score:8.2f} dB" for score in scores))


# ------------------------------------------------------------ part 3
def part3():
    print("Part 3: world_synthesize given the true envelope, by FFT size (male who'd, F0 150 Hz, no noise)")
    rng = np.random.default_rng(1)
    formants = (*HILLENBRAND["male"]["u"], *UPPER["male"])
    sound = vowel(150.0, formants, rng)
    grid = np.arange(int(sound.duration / HOP)) * HOP
    track = (grid, np.full(len(grid), 150.0))
    truth = TrueEnvelope(formants)
    for n_fft in (world_fft_size(FS), 4096, 16384):
        freqs = np.arange(n_fft // 2 + 1) * FS / n_fft
        envelope = SpectralEnvelope(truth(grid, freqs), grid, FS, np.nan)
        quiet = Aperiodicity(np.full_like(envelope.data, 1e-3), grid, FS, "none")
        out = so.world_synthesize(track, envelope, quiet)
        harmonic_freqs, out_db = harmonic_levels_db(out, 150.0)
        keep = harmonic_freqs >= SCORED[0]
        true_db = 20 * np.log10(envelope_amplitude(harmonic_freqs[keep], formants))
        print(f"  n_fft {n_fft:5d}: {level_free_rms(out_db[keep], true_db):.2f} dB")
    estimate = so.cheaptrick(sound, track)
    harmonic_freqs = 150.0 * np.arange(1, int(SCORED[1] / 150.0) + 1)
    keep = harmonic_freqs >= SCORED[0]
    estimate_db = 10 * np.log10(estimate(np.array([0.4]), harmonic_freqs[keep])[0, :, 0])
    truth_db = 20 * np.log10(envelope_amplitude(harmonic_freqs[keep], formants))
    full_range = np.ptp(20 * np.log10(envelope_amplitude(np.arange(50, 5001, 10.0), formants)))
    estimate_error = level_free_rms(estimate_db, truth_db)
    print(f"  CheapTrick's estimate against the truth at the harmonics: {estimate_error:.2f} dB")
    print(f"  the true envelope spans {full_range:.0f} dB over 50-5000 Hz")


def main():
    start = time.time()
    if pyworld is None:
        print("(pyworld is not installed, so Harvest is left out)")
    part1()
    part2()
    part3()
    print(f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
