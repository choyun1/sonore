"""Numerical checks for the claims in docs/design/views/chroma.md (K1 to K6, K8).

Like the other design checkers, this shares no code with sonore: NumPy only, every formula
written out here. Each line prints the claim number and the number that supports it.

    python tools/check_chroma_claims.py
"""

import numpy as np

FS = 22050
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def report(claim, text, value):
    print(f"{claim:4s} {text:<78s} {value:12.6g}")


def pitch_class(freq, reference=440.0):
    """Nearest equal-tempered pitch class, 0 = C, with A at ``reference``."""
    semitones_from_a = 12 * np.log2(np.asarray(freq) / reference)
    return (np.round(semitones_from_a).astype(int) + 9) % 12


def cosine_bank(freqs, centers_oct, spacing_oct):
    """Half-cycle cosines in log2 frequency, zero at the neighboring centers, with a flat
    lowpass below the first center and a flat highpass above the last (the sum of squares is
    then 1 everywhere)."""
    with np.errstate(divide="ignore"):
        position = np.log2(freqs)
    distance = (position[:, None] - centers_oct[None, :]) / spacing_oct
    bank = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
    bank[:, 0] = np.where(position <= centers_oct[0], 1.0, bank[:, 0])
    bank[:, -1] = np.where(position >= centers_oct[-1], 1.0, bank[:, -1])
    return bank


def k1_q():
    # cos^2(pi d / 2) = 1/2 at d = +-1/2 spacing, so the half-power band runs from
    # 2^(-s/2) to 2^(s/2) times the center frequency.
    for spacing, label in [(1 / 12, "one band per semitone"), (1 / 36, "three per semitone")]:
        q = 1 / (2 ** (spacing / 2) - 2 ** (-spacing / 2))
        report("K1", f"Q (center / half-power bandwidth), {label}", q)


def k2_exact():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(2 * FS)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / FS)
    centers = np.log2(55.0) + np.arange(0, 7 * 12 + 1) / 12  # A1 to A8 in semitones
    bank = cosine_bank(freqs, centers, 1 / 12)
    report(
        "K2", "largest |sum of squared responses - 1| on the DFT grid", np.max(np.abs((bank**2).sum(1) - 1))
    )
    bands = np.fft.irfft(spectrum[:, None] * bank, n=len(signal), axis=0)  # analysis
    back = np.fft.irfft(np.fft.rfft(bands, axis=0) * bank, n=len(signal), axis=0).sum(1)  # synthesis
    report(
        "K2",
        "relative error of analysis then synthesis, 2 s of white noise",
        np.linalg.norm(back - signal) / np.linalg.norm(signal),
    )
    report("K2", "real values kept per input sample (bands, undecimated)", bank.shape[1])


def k3_triad():
    # C major triad C4 E4 G4, each a harmonic tone with amplitudes 1/n, 10 harmonics.
    power = np.zeros(12)
    for root in [261.63, 329.63, 392.00]:
        harmonics = np.arange(1, 11)
        np.add.at(power, pitch_class(root * harmonics), (1 / harmonics) ** 2)
    share = power / power.sum()
    for index in np.argsort(share)[::-1][:5]:
        report("K3", f"share of power in pitch class {NAMES[index]}", share[index])
    report("K3", "share outside C, E, G (from overtones)", 1 - share[[0, 4, 7]].sum())


def shepard_partials(lowest, center=500.0, sd_oct=1.0, f_min=20.0):
    """Octave-spaced partials from ``lowest`` up to Nyquist (those at or above ``f_min``),
    with a Gaussian level over log2 frequency."""
    freqs = lowest * 2.0 ** np.arange(0, 20)
    freqs = freqs[(freqs >= f_min) & (freqs < FS / 2)]
    return freqs, np.exp(-0.5 * (np.log2(freqs / center) / sd_oct) ** 2)


def k4_octave():
    # Start the lower tone at C1 and keep every partial up to Nyquist, so the two tones differ
    # only by partials at the ends of the range.
    c = 261.63 / 8  # C1
    low_freqs, low_amps = shepard_partials(c, f_min=0)
    high_freqs, high_amps = shepard_partials(2 * c, f_min=0)
    shared = np.intersect1d(low_freqs, high_freqs)
    differing = np.setxor1d(low_freqs, high_freqs)
    every = np.concatenate([low_freqs, high_freqs])
    every_amps = np.concatenate([low_amps, high_amps])
    differing_power = sum(every_amps[every == f][0] ** 2 for f in differing)
    report("K4", "partials shared by a Shepard tone and the one an octave up", len(shared))
    report(
        "K4",
        "power in partials not shared, as a fraction of one tone's power",
        differing_power / (low_amps**2).sum(),
    )


def centroid_oct(freqs, amps):
    return np.sum(amps**2 * np.log2(freqs)) / np.sum(amps**2)


def k5_tritone():
    c_freqs, c_amps = shepard_partials(261.63 / 16)
    fs_freqs, fs_amps = shepard_partials(261.63 / 16 * 2 ** (6 / 12))
    report(
        "K5",
        "power-weighted mean log2 frequency, C minus F# Shepard tone [oct]",
        centroid_oct(c_freqs, c_amps) - centroid_oct(fs_freqs, fs_amps),
    )
    shifts = np.arange(12)
    centroids = [centroid_oct(*shepard_partials(261.63 / 16 * 2 ** (k / 12))) for k in shifts]
    report("K5", "largest spread of that mean over the 12 pitch classes [oct]", np.ptp(centroids))


def k6_glide():
    # Risset glide: octave partials of an F0 rising one octave every 2 s, starting far below
    # hearing so that new partials keep entering from below.
    duration, rate = 8.0, 0.5  # seconds, octaves per second
    t = np.arange(int(duration * FS)) / FS
    lowest = 20.0 / 2**5 * 2 ** (rate * t)
    signal = np.zeros_like(t)
    log_centroid = np.zeros_like(t)
    weight = np.zeros_like(t)
    phase = 2 * np.pi * np.cumsum(lowest) / FS
    for k in range(25):
        freq = lowest * 2**k
        amp = np.where(freq < FS / 2, np.exp(-0.5 * np.log2(freq / 500.0) ** 2), 0.0)
        signal += amp * np.cos(2**k * phase)
        log_centroid += amp**2 * np.log2(freq)
        weight += amp**2
    windows = signal[: int(duration * FS) // FS * FS].reshape(-1, FS)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    report("K6", "RMS per second, largest / smallest, over 8 s of glide", rms.max() / rms.min())
    report(
        "K6",
        "spread of the power-weighted mean log2 frequency over the glide [oct]",
        np.ptp(log_centroid / weight),
    )


def k8_round_trip():
    # Steady tones through the semitone bank of K2, folded to pitch classes: the share of power
    # that lands in the tone's own pitch class.
    centers = np.log2(55.0) + np.arange(0, 7 * 12 + 1) / 12  # A1 to A8 in semitones
    classes = (np.arange(len(centers)) + 9) % 12  # A1 is pitch class 9

    def own_class_share(freqs, amps, own):
        band_power = (cosine_bank(freqs, centers, 1 / 12) ** 2 * (amps**2)[:, None]).sum(0)
        return band_power[classes == own].sum() / band_power.sum()

    c_freqs, c_amps = shepard_partials(261.63 / 16)
    report("K8", "Shepard tone on C: share of its chroma in C", own_class_share(c_freqs, c_amps, 0))
    harmonics = np.arange(1, 11)
    report(
        "K8",
        "harmonic tone C4 (amplitudes 1/n, 10 harmonics): share of its chroma in C",
        own_class_share(261.63 * harmonics, 1 / harmonics, 0),
    )


if __name__ == "__main__":
    k1_q()
    k2_exact()
    k3_triad()
    k4_octave()
    k5_tritone()
    k6_glide()
    k8_round_trip()
