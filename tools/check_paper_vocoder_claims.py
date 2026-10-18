"""Checks so.pitch_adaptive_envelope (CheapTrick) and so.band_aperiodicity (D4C),
both written from their papers (Morise, 2015 and 2016), where the truth is known.

Part 1, the envelope: harmonics of 100, 150 and 250 Hz through a known all-pole
filter (four formants of an /a/) at 16 kHz, compared with the filter's power
response; the same filter at two F0s gives the same level; white noise reads its
level within the log-domain bias of averaging noise.

Part 2, the aperiodicity: harmonics plus white noise at 48 kHz, as in the D4C
paper's experiment 1 (its Eq. 29), at SNRs from 0 to 40 dB and F0s of 125 and
250 Hz, compared with the true noise share in each band. It prints the bias of the
paper's values (correction_db=0), which is where the default correction of 8 dB
comes from, and checks that the corrected values follow the truth.

Part 3, edges: unvoiced time windows are all noise, a time window at t = 0 and
one past the end work, the 0 Hz anchor is -60 dB, and the output is the same on
every run.

Part 4, only if pyworld is installed (a development tool, never a dependency): the
distance to WORLD's own CheapTrick and D4C on the bdl and slt sentences, with
F0 from so.f0_track. These are written from the papers, not from WORLD's code,
so this is a distance, not a match.

    python tools/check_paper_vocoder_claims.py
"""

import sys

import numpy as np
from scipy.signal import freqz, lfilter

import sonore as so

results = []


def report(name, value, passed, detail=""):
    results.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {value}" + (f"  ({detail})" if detail else ""))


def db(values):
    return 10.0 * np.log10(values)


def vowel_filter(fs):
    """An all-pole filter with four formants (Hz, bandwidth Hz) of a vowel like /a/."""
    poles = []
    for frequency, bandwidth in [(700, 130), (1220, 70), (2600, 160), (3300, 250)]:
        radius = np.exp(-np.pi * bandwidth / fs)
        poles += [radius * np.exp(2j * np.pi * frequency / fs), radius * np.exp(-2j * np.pi * frequency / fs)]
    return np.real(np.poly(poles))


def harmonics(f0, fs, duration, rng, *, from_zero=False):
    """Harmonics of f0 below Nyquist with random phases. Each carries the power
    2 f0 / fs, so the excitation has the power density of unit white noise and,
    with the envelope's unit-energy window, an envelope of 1 at any F0. With
    ``from_zero``, unit amplitudes from k = 0 as in the D4C paper's Eq. 29."""
    sample_times = np.arange(int(duration * fs)) / fs
    numbers = np.arange(0 if from_zero else 1, int(np.ceil(fs / 2 / f0)))
    numbers = numbers[numbers * f0 < fs / 2]
    amplitude = 1.0 if from_zero else 2.0 * np.sqrt(f0 / fs)
    phases = rng.uniform(0, 2 * np.pi, len(numbers))
    return amplitude * np.cos(2 * np.pi * f0 * np.outer(sample_times, numbers) + phases).sum(axis=1)


def part1():
    print("Part 1: the envelope at 16 kHz, through an all-pole filter (formants 700/1220/2600/3300 Hz)")
    fs = 16000
    rng = np.random.default_rng(2015)
    coefficients = vowel_filter(fs)
    times = np.arange(0.3, 0.7, 0.005)
    envelopes_db = {}
    for f0 in (100.0, 150.0, 250.0):
        sound = so.Sound(lfilter([1.0], coefficients, harmonics(f0, fs, 1.0, rng)), fs)
        envelope = so.pitch_adaptive_envelope(sound, (times, np.full(len(times), f0)))
        _, response = freqz([1.0], coefficients, worN=envelope.f, fs=fs)
        envelopes_db[f0] = envelope.db[0]
        error = envelope.db[0] - db(np.abs(response) ** 2)[:, None]
        # between the first and the last harmonic, where the envelope has harmonics to rest on
        band = (envelope.f >= f0) & (envelope.f <= fs / 2 - f0)
        mean_abs, level = np.mean(np.abs(error[band])), np.mean(error[band])
        # the paper reports tenths of a dB on flat spectra and more on steep ones; formant
        # peaks fall between harmonics, so the error grows with F0
        report(
            f"F0 {f0:.0f} Hz, mean |error| dB",
            f"{mean_abs:.2f}",
            mean_abs < 1.0 and abs(level) < 0.5,
            f"level {level:+.2f} dB, largest {np.max(np.abs(error[band])):.1f} dB",
        )
    band = (envelope.f >= 250.0) & (envelope.f <= fs / 2 - 250.0)
    for low, high in ((100.0, 150.0), (100.0, 250.0)):
        difference = np.mean(envelopes_db[low] - envelopes_db[high], axis=1)[band]
        # the unit-energy window promises the same level at every F0
        report(
            f"level at F0 {low:.0f} vs {high:.0f} Hz, dB",
            f"{np.mean(difference):+.2f}",
            abs(np.mean(difference)) < 0.5,
        )
    noise = so.Sound(rng.standard_normal(fs), fs)
    noise_times = np.arange(0.1, 0.9, 0.01)
    envelope = so.pitch_adaptive_envelope(noise, (noise_times, np.zeros(len(noise_times))))
    level = np.mean(envelope.db[0][(envelope.f > 500) & (envelope.f < fs / 2 - 500)])
    # unit white noise, analyzed as unvoiced: averaging its ragged spectrum in dB puts the
    # mean up to 10 log10(exp(-Euler's gamma)) = -2.51 dB below its level
    report("unit white noise, mean dB", f"{level:+.2f}", -2.6 <= level <= 0.0, "expected between -2.51 and 0")


def part2():
    print("Part 2: the aperiodicity at 48 kHz, harmonics plus white noise (the D4C paper's Eq. 29)")
    fs = 48000
    rng = np.random.default_rng(2016)
    times = np.arange(0.05, 0.95, 0.01)
    snrs = np.arange(0, 41, 5)
    centers = np.arange(3000, 15001, 3000)
    biases = []
    for f0 in (125.0, 250.0):
        periodic = harmonics(f0, fs, 1.0, rng, from_zero=True)
        white = rng.standard_normal(len(periodic))
        raw, corrected, truth = [], [], []
        for snr in snrs:
            noise = white * np.sqrt(np.mean(periodic**2) / 10 ** (snr / 10))
            sound = so.Sound(periodic + noise, fs)
            track = (times, np.full(len(times), f0))
            paper = so.band_aperiodicity(sound, track, correction_db=0.0)
            default = so.band_aperiodicity(sound, track)
            at_centers = np.searchsorted(paper.f, centers)
            raw.append(paper.db[0][at_centers].mean())
            corrected.append(default.db[0][at_centers].mean())
            # the true share in each band, from the noise's and the mix's power spectra
            freqs = np.fft.rfftfreq(len(periodic), 1 / fs)
            noise_power, mix_power = (
                np.abs(np.fft.rfft(noise)) ** 2,
                np.abs(np.fft.rfft(periodic + noise)) ** 2,
            )
            shares = [
                noise_power[abs(freqs - c) <= 3000].sum() / mix_power[abs(freqs - c) <= 3000].sum()
                for c in centers
            ]
            truth.append(db(np.mean(shares)))
        raw, corrected, truth = np.array(raw), np.array(corrected), np.array(truth)
        above_5 = snrs > 5
        bias = np.mean(raw[above_5] - truth[above_5])
        biases.append(bias)
        print(f"    F0 {f0:.0f} Hz | SNR " + " ".join(f"{snr:6d}" for snr in snrs))
        print("    true           | " + " ".join(f"{value:6.1f}" for value in truth))
        print("    paper's values | " + " ".join(f"{value:6.1f}" for value in raw))
        print("    corrected      | " + " ".join(f"{value:6.1f}" for value in corrected))
        slope = np.polyfit(truth[1:], raw[1:], 1)[0]
        # the paper reports a steady bias of about 6 dB above 5 dB SNR, too periodic
        report(
            f"F0 {f0:.0f} Hz, bias of the paper's values above 5 dB SNR",
            f"{bias:+.1f} dB",
            -10.0 <= bias <= -3.0,
        )
        report(f"F0 {f0:.0f} Hz, slope against the truth", f"{slope:.2f}", 0.8 <= slope <= 1.2)
        remaining = np.mean(corrected[above_5] - truth[above_5])
        # the 8 dB correction leaves the F0-dependent part, about 1 dB either way
        report(f"F0 {f0:.0f} Hz, corrected bias above 5 dB SNR", f"{remaining:+.1f} dB", abs(remaining) < 2.0)
    print(f"    mean bias over both F0s: {np.mean(biases):+.1f} dB (the default correction_db is 8)")


def part3():
    print("Part 3: unvoiced time windows, edges, and repeatability")
    fs = 44100
    rng = np.random.default_rng(7)
    samples = lfilter([1.0], vowel_filter(fs), harmonics(120.0, fs, 0.5, rng))
    samples = 0.5 * samples / np.max(np.abs(samples))
    samples[int(0.3 * fs) : int(0.35 * fs)] = 0.0  # digital silence
    sound = so.Sound(samples, fs)
    track = (np.array([0.0, 0.1, 0.2, 0.32, 0.5, 0.55]), np.array([120.0, 0.0, 120.0, 120.0, 120.0, 50.0]))
    envelope, aperiodicity = so.pitch_adaptive_envelope(sound, track), so.band_aperiodicity(sound, track)
    report(
        "unvoiced is all noise",
        f"{aperiodicity.share[0][:, 1].min():.3f}",
        np.all(aperiodicity.share[0][:, 1] == 1.0),
    )
    report(
        "finite everywhere (t = 0, past the end, silence)",
        "",
        bool(
            np.all(np.isfinite(envelope.data))
            and np.all(envelope.data > 0)
            and np.all(np.isfinite(aperiodicity.data))
        ),
    )
    report("0 Hz anchor", f"{aperiodicity.db[0][0, 0]:.1f} dB", abs(aperiodicity.db[0][0, 0] + 60.0) < 1e-9)
    again = so.pitch_adaptive_envelope(sound, track), so.band_aperiodicity(sound, track)
    report(
        "the same on every run",
        "",
        np.array_equal(again[0].data, envelope.data) and np.array_equal(again[1].data, aperiodicity.data),
    )


def part4():
    try:
        import pyworld
    except ImportError:
        print("Part 4 skipped: pyworld is not installed (pip install pyworld, development only)")
        return
    print(f"Part 4: distance to WORLD (pyworld {pyworld.__version__}), F0 from so.f0_track, 5 ms windows")
    for speaker in ("bdl", "slt"):
        sound = so.load(f"docs/speech/{speaker}_arctic_a0131.flac")
        samples, fs = np.ascontiguousarray(sound.data[:, 0], dtype=float), int(sound.fs)
        times = np.arange(int(sound.duration / 0.005)) * 0.005
        tracked = so.f0_track(sound)
        values = np.where(tracked.voiced[0], tracked.f0[0], 0.0)
        f0 = values[np.clip(np.searchsorted(tracked.t, times), 0, len(values) - 1)]
        voiced = f0 > 0
        envelope = so.pitch_adaptive_envelope(sound, (times, f0)).data[0]
        difference = db(envelope / pyworld.cheaptrick(samples, f0, times, fs, q1=-0.09).T)
        voiced_gap, unvoiced_gap = abs(difference[:, voiced]), abs(difference[:, ~voiced])
        print(f"    {speaker} envelope vs WORLD at q1 = -0.09, |dB|:")
        for name, gap in (("voiced", voiced_gap), ("unvoiced", unvoiced_gap)):
            print(f"      {name}: median {np.median(gap):.2f}, 95th percentile {np.percentile(gap, 95):.2f}")
        aperiodicity = so.band_aperiodicity(sound, (times, f0))
        at_3k = np.searchsorted(aperiodicity.f, 3000)
        world = pyworld.d4c(samples, f0, times, fs, threshold=0.0).T
        gap = 20 * np.log10(aperiodicity.data[0][at_3k, voiced] / world[at_3k, voiced])
        print(f"    {speaker} D4C at 3 kHz vs WORLD, voiced: median {np.median(gap):+.1f} dB")


part1()
part2()
part3()
part4()
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
