"""Claims for docs/design/voice-change.md: changing a voice's pitch and its
formants with the WORLD pieces already in sonore (so.f0_track, so.cheaptrick,
so.d4c, so.harmonic_aperiodicity, so.world_synthesize).

The operations themselves are prototyped here, not in the library:

- a pitch change multiplies the voiced values of the F0 track by a ratio;
- a formant shift warps the spectral envelope along frequency, so the new
  envelope at f is the old one at f / ratio (read with the envelope's own
  interpolation, linear in dB between bins, held above fs / 2);
- the aperiodicity is either left as measured or warped the same way.

Everything is measured with the library's own analyses, so the numbers say
whether the operations do what they claim as seen by those analyses; they
are not listening tests.

    python tools/check_voice_change_claims.py

Runs in about a minute.
"""

import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks, freqz

import sonore as so
from sonore.views.aperiodicity import Aperiodicity
from sonore.views.spectral_envelope import SpectralEnvelope
from sonore.views.world import world_fft_size

FS = 16000.0
HOP = 0.005
BDL = "docs/speech/bdl_arctic_a0131.flac"
SLT = "docs/speech/slt_arctic_a0131.flac"
FITTED_RANGE = (100.0, 5000.0)  # the warp is fitted to envelopes over this range [Hz]
# Hillenbrand, Getty, Clark & Wheeler (1995), Table V, as quoted in
# tools/check_female_voices.py: average F1-F3 [Hz] of heed, hod and who'd.
HILLENBRAND = {
    "male": {"i": (342, 2322, 3000), "a": (768, 1333, 2522), "u": (378, 997, 2343)},
    "female": {"i": (437, 2761, 3372), "a": (936, 1551, 2815), "u": (459, 1105, 2735)},
}
BANDWIDTHS = (60, 100, 120, 175, 250)


# ------------------------------------------------------------ the prototypes
def change_pitch(track, ratio):
    """The F0 track with every voiced value multiplied by ratio (0 stays 0)."""
    times, f0_values = track
    return times, np.asarray(f0_values, float) * ratio


def warp_envelope(envelope, ratio, *, exact_identity=True):
    """The envelope moved along frequency: new(f) = old(f / ratio)."""
    if ratio == 1 and exact_identity:
        return envelope
    warped = envelope(envelope.t, envelope.f / ratio)
    return SpectralEnvelope(warped, envelope.t, envelope.fs, envelope.q1)


def warp_aperiodicity(aperiodicity, ratio):
    warped = aperiodicity(aperiodicity.t, aperiodicity.f / ratio)
    return Aperiodicity(np.minimum(warped, 1.0), aperiodicity.t, aperiodicity.fs, aperiodicity.method)


# ------------------------------------------------------------ measures
def mean_db(envelope, voiced):
    """The envelope in dB averaged over the voiced time windows, channel 0."""
    return envelope.db[0][:, voiced].mean(axis=1)


def fitted_warp(reference_db, changed_db, freqs, *, remove_tilt=False):
    """The ratio a that best explains changed(f) = reference(f / a) + level,
    over FITTED_RANGE, searched on a grid of 0.1% steps from 0.45 to 2.25.
    With remove_tilt, a straight line in dB against log frequency is also
    allowed (a different spectral slope), not only a level."""
    inside = (freqs >= FITTED_RANGE[0]) & (freqs <= FITTED_RANGE[1])
    log_f = np.log(freqs[inside])
    best_ratio, best_error = 1.0, np.inf
    for ratio in np.exp(np.arange(np.log(0.45), np.log(2.25), 0.001)):
        predicted = np.interp(freqs[inside] / ratio, freqs, reference_db)
        difference = changed_db[inside] - predicted
        if remove_tilt:
            difference = difference - np.polyval(np.polyfit(log_f, difference, 1), log_f)
        error = np.sqrt(np.mean((difference - difference.mean()) ** 2))
        if error < best_error:
            best_ratio, best_error = ratio, error
    return best_ratio, best_error


def level_free_rms(a_db, b_db, freqs):
    inside = (freqs >= FITTED_RANGE[0]) & (freqs <= FITTED_RANGE[1])
    difference = a_db[inside] - b_db[inside]
    return float(np.sqrt(np.mean((difference - difference.mean()) ** 2)))


def voiced_ratio(track_in, track_out):
    """Median of out / in over time windows voiced in both, and the share
    of those within 5% of the median."""
    f0_in, f0_out = np.asarray(track_in[1], float), np.asarray(track_out[1], float)
    count = min(len(f0_in), len(f0_out))
    f0_in, f0_out = f0_in[:count], f0_out[:count]
    both = (f0_in > 0) & (f0_out > 0)
    ratios = f0_out[both] / f0_in[both]
    median = np.median(ratios)
    return median, np.mean(np.abs(ratios / median - 1) < 0.05), np.mean(both) / np.mean(f0_in > 0)


def detrended_correlation(a, b, freqs):
    """Correlation across frequency of a and b once a straight line in
    frequency is taken out of each, so a shared tilt does not count."""
    a = a - np.polyval(np.polyfit(freqs, a, 1), freqs)
    b = b - np.polyval(np.polyfit(freqs, b, 1), freqs)
    return np.corrcoef(a, b)[0, 1]


def rms_db(a, b):
    return 20 * np.log10(np.sqrt(np.mean(a.data**2)) / np.sqrt(np.mean(b.data**2)))


# ------------------------------------------------------------ a synthetic vowel
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
    """The vowel's aperiodicity, a property of its source: -30 dB at 0 Hz
    rising to -5 dB at 8 kHz."""
    return -30 + 25 * np.asarray(f) / 8000


def vowel(f0, formants, rng, duration=0.8):
    """A steady vowel with known envelope and noise share, built as in
    tools/check_female_voices.py."""
    f_max = 0.45 * FS
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for k in range(1, int(f_max / f0) + 1):
        fk = k * f0
        x += (
            envelope_amplitude(fk, formants)
            * np.sqrt(1 - 10 ** (noise_share_db(fk) / 10))
            * np.cos(2 * np.pi * fk * t)
        )
    freqs = np.fft.rfftfreq(n, 1 / FS)
    gain = envelope_amplitude(freqs, formants) * np.sqrt(10 ** (noise_share_db(freqs) / 10) * FS / (4 * f0))
    gain[freqs >= f_max] = 0
    x += np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gain, n)
    return so.Sound(x / np.std(x) * 0.1, FS)


def formant_peaks(envelope_db, freqs, expected):
    """For each expected formant, the highest peak of envelope_db within 15%
    of it (peaks found with 1 dB prominence)."""
    peaks, _ = find_peaks(envelope_db, prominence=1.0)
    found = []
    for formant in expected:
        near = peaks[np.abs(freqs[peaks] / formant - 1) < 0.15]
        found.append(freqs[near[np.argmax(envelope_db[near])]] if len(near) else np.nan)
    return np.array(found)


# ------------------------------------------------------------ the claims
def c1_identity(sentence, track, envelope, aperiodicity):
    print("C1. Identity settings against so.world_synthesize")
    reference = so.world_synthesize(track, envelope, aperiodicity)
    through_interp = warp_envelope(envelope, 1.0, exact_identity=False)
    relative = np.max(np.abs(through_interp.data / envelope.data - 1))
    out_interp = so.world_synthesize(change_pitch(track, 1.0), through_interp, aperiodicity)
    out_exact = so.world_synthesize(change_pitch(track, 1.0), warp_envelope(envelope, 1.0), aperiodicity)
    print(f"  envelope read back through its interpolation at ratio 1: max relative change {relative:.1e}")
    print(
        f"  output, interpolated envelope: max |difference|"
        f" {np.max(np.abs(out_interp.data - reference.data)):.1e}"
        f" (output peak {np.max(np.abs(reference.data)):.2f})"
    )
    print(
        f"  output, ratio 1 returns the envelope itself: identical ="
        f" {np.array_equal(out_exact.data, reference.data)}"
    )
    print(
        f"  F0 times 1.0 is the same array:"
        f" {np.array_equal(change_pitch(track, 1.0)[1], np.asarray(track[1]))}"
    )
    return reference


def c2_pitch(sentence, track, envelope, aperiodicity, reference_out):
    print("C2. Pitch change on the bdl sentence: measured F0 ratio, and the formants")
    print("  ratio | median F0 out/in  within 5%  voiced kept | envelope warp  so.pitch_shift's warp")
    voiced = np.asarray(track[1]) > 0
    in_db = mean_db(envelope, voiced)
    for ratio in (0.5, 2 ** (-5 / 12), 2 ** (7 / 12), 2.0):
        changed = change_pitch(track, ratio)
        out = so.world_synthesize(changed, envelope, aperiodicity)
        tracked = so.f0_track(out)
        median, within, kept = voiced_ratio(track, (tracked.t, tracked.f0[0]))
        # the formants of the output, analysed with the imposed track
        out_env = so.cheaptrick(out, changed)
        warp, _ = fitted_warp(in_db, mean_db(out_env, voiced), envelope.f)
        # the phase vocoder's pitch shift, for contrast
        shifted = so.pitch_shift(sentence, 12 * np.log2(ratio))
        pv_env = so.cheaptrick(shifted, changed)
        pv_warp, _ = fitted_warp(in_db, mean_db(pv_env, voiced), envelope.f)
        print(f"  {ratio:5.3f} | {median:8.3f}  {within:13.0%}  {kept:9.0%} | {warp:10.3f}  {pv_warp:14.3f}")


def c3_vowel_formants(rng):
    print("C3. Formant shift on a synthetic vowel (male 'hod', F0 120 Hz): peaks and fitted warp")
    formants = (*HILLENBRAND["male"]["a"], 3500, 4500)
    snd = vowel(120.0, formants, rng)
    times = np.arange(int(snd.duration / HOP)) * HOP
    track = (times, np.full(len(times), 120.0))
    envelope = so.cheaptrick(snd, track)
    aperiodicity = so.d4c(snd, track)
    inner = (times > 0.1) & (times < snd.duration - 0.1)
    in_db = mean_db(envelope, inner)
    in_peaks = formant_peaks(in_db, envelope.f, formants[:3])
    print(f"  input F1-F3 as CheapTrick sees them: {np.round(in_peaks)} Hz (true {formants[:3]})")
    print("  ratio | F1, F2, F3 out / in       | fitted warp")
    for ratio in (0.8, 0.9, 1.1, 1.2, 1.25):
        out = so.world_synthesize(track, warp_envelope(envelope, ratio), aperiodicity)
        out_db = mean_db(so.cheaptrick(out, track), inner)
        out_peaks = formant_peaks(out_db, envelope.f, np.array(formants[:3]) * ratio)
        warp, residual = fitted_warp(in_db, out_db, envelope.f)
        print(
            f"  {ratio:5.2f} | {np.array2string(out_peaks / in_peaks, precision=3):24s} |"
            f" {warp:.3f} (residual {residual:.2f} dB)"
        )


def c4_sentence_formants(track, envelope, aperiodicity, reference_out):
    print("C4. Formant shift on the bdl sentence: fitted warp, and the level")
    print("  ratio | fitted warp (residual) | output RMS vs identity | envelope power vs identity (median)")
    voiced = np.asarray(track[1]) > 0
    in_db = mean_db(envelope, voiced)
    for ratio in (0.8, 0.9, 1.1, 1.2):
        warped = warp_envelope(envelope, ratio)
        out = so.world_synthesize(track, warped, aperiodicity)
        warp, residual = fitted_warp(in_db, mean_db(so.cheaptrick(out, track), voiced), envelope.f)
        power_change = 10 * np.log10(warped.data[0].sum(axis=0) / envelope.data[0].sum(axis=0))
        print(
            f"  {ratio:5.2f} | {warp:.3f} ({residual:.2f} dB) | {rms_db(out, reference_out):+6.2f} dB"
            f" | {np.median(power_change[voiced]):+6.2f} dB"
        )
    top_read = envelope.fs / 2 / 0.8
    print(
        f"  at ratio 0.8 the top bins read up to {top_read:.0f} Hz, above fs/2 = {envelope.fs / 2:.0f} Hz;"
        f" the last {1 - 0.8:.0%} of the band holds the value at fs/2"
    )


def c5_aperiodicity(sentence, track, envelope, rng):
    print("C5. Aperiodicity under a formant shift")
    voiced = np.asarray(track[1]) > 0
    inside = (envelope.f >= 100) & (envelope.f <= 6000)
    measures = {
        "D4C": so.d4c(sentence, track),
        "harmonic residual": so.harmonic_aperiodicity(sentence, track),
    }
    for name, aperiodicity in measures.items():
        correlations = [
            detrended_correlation(
                envelope.db[0][inside, w], aperiodicity.db[0][inside, w], envelope.f[inside]
            )
            for w in np.flatnonzero(voiced)
        ]
        warped_env = warp_envelope(envelope, 1.2)
        kept = so.world_synthesize(track, warped_env, aperiodicity)
        moved = so.world_synthesize(track, warped_env, warp_aperiodicity(aperiodicity, 1.2))
        difference = rms_db(so.Sound(moved.data - kept.data, FS), kept)
        spread = np.ptp(aperiodicity.db[0][inside][:, voiced].mean(axis=1))
        print(
            f"  {name}: across 100-6000 Hz in voiced time windows, detrended correlation of"
            f" envelope dB and aperiodicity dB, median {np.median(correlations):+.2f};"
        )
        print(
            f"    mean voiced aperiodicity spans {spread:.1f} dB over that range;"
            f" at ratio 1.2, warped minus kept aperiodicity output is {difference:.1f} dB re the output"
        )
    # A vowel whose noise share is a source property: shift by 1.2 and ask
    # which choice gives the output the noise share of the shifted vowel.
    formants = (*HILLENBRAND["male"]["a"], 3500, 4500)
    snd = vowel(120.0, formants, rng)
    times = np.arange(int(snd.duration / HOP)) * HOP
    vowel_track = (times, np.full(len(times), 120.0))
    inner = (times > 0.1) & (times < snd.duration - 0.1)
    vowel_env = so.cheaptrick(snd, vowel_track)
    vowel_ap = so.harmonic_aperiodicity(snd, vowel_track)
    band = (vowel_env.f >= 200) & (vowel_env.f <= 6000)
    for name, ap in (("D4C", so.d4c(snd, vowel_track)), ("harmonic residual", vowel_ap)):
        correlations = [
            detrended_correlation(vowel_env.db[0][inside, w], ap.db[0][inside, w], vowel_env.f[inside])
            for w in np.flatnonzero(inner)
        ]
        print(
            f"  synthetic vowel (noise share a straight line in frequency, so the truth's detrended"
            f" correlation is 0): {name} median {np.median(correlations):+.2f}"
        )
    truth = noise_share_db(vowel_env.f[band])
    for label, ap in (("kept", vowel_ap), ("warped", warp_aperiodicity(vowel_ap, 1.2))):
        out = so.world_synthesize(vowel_track, warp_envelope(vowel_env, 1.2), ap)
        measured = so.harmonic_aperiodicity(out, vowel_track)
        measured_db = 10 * np.log10(np.maximum(measured.share[0][band][:, inner].mean(axis=1), 1e-12))
        print(
            f"  synthetic vowel shifted by 1.2, aperiodicity {label}: median |measured - source truth|"
            f" {np.median(np.abs(measured_db - truth)):.1f} dB over 200-6000 Hz"
        )


def c6_bdl_to_slt(bdl, bdl_track, bdl_env, bdl_ap):
    print("C6. The same sentence by bdl and slt: how far apart, and how far a change moves bdl")
    slt = so.load(SLT)
    slt_tracked = so.f0_track(slt)
    slt_track = (slt_tracked.t, slt_tracked.f0[0])
    slt_env = so.cheaptrick(slt, slt_track)
    bdl_f0 = np.asarray(bdl_track[1])
    f0_ratio = np.median(slt_track[1][slt_track[1] > 0]) / np.median(bdl_f0[bdl_f0 > 0])
    bdl_db, slt_db = mean_db(bdl_env, bdl_f0 > 0), mean_db(slt_env, slt_track[1] > 0)
    warp, residual = fitted_warp(bdl_db, slt_db, bdl_env.f)
    tilt_warp, tilt_residual = fitted_warp(bdl_db, slt_db, bdl_env.f, remove_tilt=True)
    print(
        f"  median voiced F0: bdl {np.median(bdl_f0[bdl_f0 > 0]):.0f} Hz,"
        f" slt {np.median(slt_track[1][slt_track[1] > 0]):.0f} Hz,"
        f" ratio {f0_ratio:.2f}"
    )
    print(
        f"  warp best fitting bdl's mean voiced envelope to slt's: {warp:.3f} (residual {residual:.2f} dB,"
        f" {level_free_rms(bdl_db, slt_db, bdl_env.f):.2f} dB unwarped);"
        f" allowing a spectral slope too: {tilt_warp:.3f} (residual {tilt_residual:.2f} dB)"
    )
    female_over_male = [
        np.array(HILLENBRAND["female"][v]) / np.array(HILLENBRAND["male"][v]) for v in ("i", "a", "u")
    ]
    print(
        f"  Hillenbrand et al. (1995) female/male, F1-F3 of three vowels:"
        f" {np.round(np.ravel(female_over_male), 2)},"
        f" geometric mean {np.exp(np.mean(np.log(female_over_male))):.3f}"
    )
    choices = {
        "F0 only": (f0_ratio, 1.0),
        "F0 and fitted warp": (f0_ratio, warp),
        "F0 and the slope-free warp": (f0_ratio, tilt_warp),
        "F0 and Hillenbrand's 1.174": (f0_ratio, 1.174),
    }
    for label, (pitch, formant) in choices.items():
        changed = change_pitch(bdl_track, pitch)
        out = so.world_synthesize(changed, warp_envelope(bdl_env, formant), bdl_ap)
        out_db = mean_db(so.cheaptrick(out, changed), bdl_f0 > 0)
        print(
            f"  bdl, {label} ({pitch:.2f}, {formant:.3f}): mean voiced envelope"
            f" {level_free_rms(out_db, slt_db, bdl_env.f):.2f} dB"
            f" from slt's (level removed, 100-5000 Hz)"
        )


# ------------------------------------------------------------ any source, any synthesizer
class GridEnvelope:
    """Any envelope held as power on a (frequency, time) grid, read as
    env(t, f) like SpectralEnvelope: shape (n_channels, len(f), len(t)),
    linear in time and in dB over frequency, held beyond the ends."""

    def __init__(self, power, t, f):
        self.log_power, self.t, self.f = np.log(np.maximum(power, 1e-30)), np.asarray(t), np.asarray(f)

    def __call__(self, t, f):
        times, freqs = np.atleast_1d(t), np.atleast_1d(f)
        out = np.empty((self.log_power.shape[0], len(freqs), len(times)))
        for channel, log_power in enumerate(self.log_power):
            over_f = np.array([np.interp(freqs, self.f, column) for column in log_power.T]).T
            out[channel] = np.array([np.interp(times, self.t, row) for row in over_f])
        return np.exp(out)


class Warped:
    """Any envelope moved along frequency: warped(t, f) = env(t, f / ratio)."""

    def __init__(self, envelope, ratio):
        self.envelope, self.ratio = envelope, ratio

    def __call__(self, t, f):
        return self.envelope(t, np.asarray(f) / self.ratio)


def on_world_grid(envelope, times, fs, n_fft):
    """Any envelope sampled at WORLD's time windows and frequencies."""
    freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    return SpectralEnvelope(envelope(times, freqs), np.asarray(times), fs, np.nan)


def pointwise_amplitude(envelope, times, freqs):
    """Any envelope as harmonic_complex reads amplitudes(t, f): one value per
    point, t and f the same shape, not a grid. Sampled once on a grid and
    interpolated in dB, as the Voices from harmonics gallery page does."""
    log_amplitude = 0.5 * np.log(envelope(times, freqs)[0]).T  # (times, freqs)
    interpolator = RegularGridInterpolator((times, freqs), log_amplitude, bounds_error=False, fill_value=None)
    return lambda t, f: np.exp(interpolator(np.column_stack([np.ravel(t), np.ravel(f)]))).reshape(np.shape(t))


def contour_on_grid(times, f0_values, grid):
    """Any F0 contour read at the grid's times: linear between voiced
    neighbours, 0 where the nearest time window is unvoiced."""
    f0_values = np.asarray(f0_values, float)
    voiced = f0_values > 0
    nearest = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    previous = np.clip(nearest - 1, 0, len(times) - 1)
    nearest = np.where(np.abs(times[previous] - grid) < np.abs(times[nearest] - grid), previous, nearest)
    filled = np.interp(grid, times[voiced], f0_values[voiced])
    return np.where(voiced[nearest], filled, 0.0)


def c7_any_source(sentence, tracked):
    print("C7. Mixed sources: F0 from three trackers, envelope from two estimators, two synthesizers")
    print("    pitch x 1.5 and formants x 1.2, measured against the same sources unchanged")
    fs = sentence.fs
    grid = tracked.t
    harvest_times, harvest_f0 = np.loadtxt("docs/speech/bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2).T
    cepstrum = so.Cepstrum(so.STFT(sentence, win_dur=0.040, hop_dur=HOP))
    cep_times, cep_f0, _ = cepstrum.f0()
    f0_sources = {
        "so.f0_track": (tracked.t, tracked.f0[0]),
        "Harvest": (harvest_times, harvest_f0),
        "Cepstrum.f0": (cep_times, cep_f0[0]),
    }
    tracks = {name: (grid, contour_on_grid(*source, grid)) for name, source in f0_sources.items()}
    voiced_median = np.median(tracked.f0[0][tracked.f0[0] > 0])
    lifted = cepstrum.lifter(0.5 / voiced_median)
    stft_freqs = np.fft.rfftfreq(2 * (lifted.envelope().shape[1] - 1), 1 / fs)
    cepstral = GridEnvelope(lifted.envelope() ** 2, cep_times, stft_freqs)
    n_fft = world_fft_size(fs)
    print("  F0 source    envelope    synthesizer        | F0 out/unchanged  within 5% | fitted warp")
    for f0_name, track in tracks.items():
        aperiodicity = so.d4c(sentence, track)
        envelopes = {"CheapTrick": so.cheaptrick(sentence, track), "cepstral": cepstral}
        for env_name, envelope in envelopes.items():
            for synth_name in ("world_synthesize", "harmonic_complex"):
                if synth_name == "harmonic_complex" and f0_name != "Harvest":
                    continue
                outputs = []
                for pitch, formant in ((1.0, 1.0), (1.5, 1.2)):
                    changed = change_pitch(track, pitch)
                    view = envelope if formant == 1.0 else Warped(envelope, formant)
                    if synth_name == "world_synthesize":
                        world_envelope = on_world_grid(view, grid, fs, n_fft)
                        outputs.append(so.world_synthesize(changed, world_envelope, aperiodicity))
                    else:
                        amplitude = pointwise_amplitude(view, grid, np.arange(0, fs / 2 + 1, 10.0))
                        outputs.append(
                            so.harmonic_complex(sentence.duration, fs, changed, amplitudes=amplitude)
                        )
                plain, moved = outputs
                plain_track, moved_track = so.f0_track(plain), so.f0_track(moved)
                median, within, _ = voiced_ratio(
                    (plain_track.t, plain_track.f0[0]), (moved_track.t, moved_track.f0[0])
                )
                both_voiced = np.asarray(track[1]) > 0
                plain_db = mean_db(so.cheaptrick(plain, track), both_voiced)
                moved_db = mean_db(so.cheaptrick(moved, change_pitch(track, 1.5)), both_voiced)
                warp, residual = fitted_warp(plain_db, moved_db, np.arange(n_fft // 2 + 1) * fs / n_fft)
                print(
                    f"  {f0_name:12s} {env_name:11s} {synth_name:18s} | {median:8.3f} {within:16.0%} |"
                    f" {warp:.3f} ({residual:.2f} dB)"
                )


def main():
    start = time.time()
    rng = np.random.default_rng(1)
    sentence = so.load(BDL)
    tracked = so.f0_track(sentence)
    track = (tracked.t, tracked.f0[0])
    envelope = so.cheaptrick(sentence, track)
    aperiodicity = so.d4c(sentence, track)
    reference_out = c1_identity(sentence, track, envelope, aperiodicity)
    c2_pitch(sentence, track, envelope, aperiodicity, reference_out)
    c3_vowel_formants(rng)
    c4_sentence_formants(track, envelope, aperiodicity, reference_out)
    c5_aperiodicity(sentence, track, envelope, rng)
    c6_bdl_to_slt(sentence, track, envelope, aperiodicity)
    c7_any_source(sentence, tracked)
    print(f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
