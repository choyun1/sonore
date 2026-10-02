"""How sonore's speech analyses hold up on female voices.

Female voices have F0 roughly an octave above male ones, so their harmonics
sample the vocal tract's response about half as densely. This checker runs
the library's own analyses (so.f0_track, Cepstrum.f0 and the cepstral
envelope, so.cheaptrick, so.d4c and so.harmonic_aperiodicity) where the
truth is known, and prints the errors side by side for male and female
voices. Nothing is tuned here: every analysis runs with its defaults, and
the gallery's 40 ms window for the cepstrum.

Part 1, synthetic vowels: three vowels with men's and with women's average
formant frequencies (Hillenbrand et al. 1995), each at steady F0 from 100 to
350 Hz, built so that the envelope and the share of noise at every frequency
are known exactly. Prints the envelope error of CheapTrick and of the
cepstral lifter, and the aperiodicity error of D4C and the harmonic residual.

Part 2, the FDA evaluation database (Bagshaw, CSTR, University of
Edinburgh): 50 sentences from a male (rl) and a female (sb) speaker with a
laryngograph reference F0. It has no stated licence, so it is not in this
repository; pass the folder that holds its rl/ and sb/ subfolders. Prints F0
accuracy by reference F0, and, where there is no ground truth, how the
envelope and aperiodicity behave on each voice.

Part 3, CMU ARCTIC: male (bdl) and female (slt) sentences, which have no
EGG channel in their single-channel release, so the only reference is WORLD's
Harvest (through pyworld, a development-only tool, not a dependency). Harvest
is not ground truth; agreement with it says how often two independent
methods concur, and where they don't. Pass a folder of files named like
bdl_arctic_a0131.wav or slt_arctic_a0131.flac.

    python tools/check_female_voices.py                      # part 1 only
    python tools/check_female_voices.py --fda path/to/fda_eval --arctic path/to/arctic

Part 1 takes under a minute, part 2 several minutes (the harmonic-residual
aperiodicity fits dense least squares at every time window).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import freqz, resample_poly

import sonore as so

FS = 16000.0
HOP = 0.005
F_MAX = 0.45 * FS  # synthetic harmonics and noise stop here
F0S = [100, 150, 200, 250, 300, 350]
# Hillenbrand, Getty, Clark & Wheeler (1995), Table V: average F1-F3 [Hz] of
# heed, hod and who'd; F4 and F5 are fixed. Bandwidths as the WORLD checker's.
VOWELS = {
    "men": {"i": (342, 2322, 3000), "a": (768, 1333, 2522), "u": (378, 997, 2343)},
    "women": {"i": (437, 2761, 3372), "a": (936, 1551, 2815), "u": (459, 1105, 2735)},
}
UPPER = {"men": (3500, 4500), "women": (4100, 4900)}
BANDWIDTHS = (60, 100, 120, 175, 250)
SCORED = (100.0, 5000.0)  # envelope error is scored over this range [Hz]


def resonator_gain(f, freq, bw):
    """|H| of Klatt's resonator, unit gain at 0 Hz."""
    T = 1 / FS
    c = -np.exp(-2 * np.pi * bw * T)
    b = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * freq * T)
    return np.abs(freqz([1 - b - c], [1, -b, -c], worN=np.atleast_1d(f), fs=FS)[1])


def envelope_amplitude(f, formants):
    """The test vowel's amplitude envelope: a glottal low-pass (100 Hz
    bandwidth at 0 Hz), five formants, and the radiation difference."""
    f = np.asarray(f, float)
    gain = resonator_gain(f, 0.0, 100.0)
    for freq, bw in zip(formants, BANDWIDTHS, strict=True):
        gain = gain * resonator_gain(f, freq, bw)
    return gain * np.abs(2 * np.sin(np.pi * f / FS))


def noise_share_db(f):
    """The test vowels' aperiodicity: -30 dB at 0 Hz rising to -5 dB at 8 kHz."""
    return -30 + 25 * np.asarray(f) / 8000


def vowel(f0, formants, rng, duration=0.6):
    """A steady vowel whose envelope and noise share are known at every
    frequency: harmonic k has amplitude E(k f0) sqrt(1 - A), and the noise
    power density is A E^2 / (2 f0), so A is the share of the power that is
    noise (as tools/check_world_claims.py builds its test vowel)."""
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for k in range(1, int(F_MAX / f0) + 1):
        fk = k * f0
        amplitude = envelope_amplitude(fk, formants) * np.sqrt(1 - 10 ** (noise_share_db(fk) / 10))
        x += amplitude * np.cos(2 * np.pi * fk * t)
    freqs = np.fft.rfftfreq(n, 1 / FS)
    gain = envelope_amplitude(freqs, formants) * np.sqrt(10 ** (noise_share_db(freqs) / 10) * FS / (4 * f0))
    gain[freqs >= F_MAX] = 0
    x += np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gain, n)
    return so.Sound(x / np.std(x) * 0.1, FS)


def offset_free_rms(est_db, true_db):
    """RMS dB error once the constant offset (overall level) is removed."""
    diff = est_db - true_db
    return float(np.sqrt(np.mean((diff - diff.mean()) ** 2)))


def envelope_errors(est_db, true_db, f1_bin):
    """Median over time windows (columns of est_db) of the level-free RMS dB
    error, and of the level-free error at the F1 bin."""
    rms, at_f1 = [], []
    for column in est_db.T:
        diff = column - true_db
        rms.append(offset_free_rms(column, true_db))
        at_f1.append((diff - diff.mean())[f1_bin])
    return np.median(rms), np.median(at_f1)


def part1():
    print("Part 1: synthetic vowels, steady F0, known envelope and noise share")
    print("  envelope: RMS dB error over 100-5000 Hz, level removed (median over time windows)")
    print("  F1: estimated minus true level at F1 [dB], with the same level removed")
    print("  noise share: median |error| [dB] over 200-6000 Hz")
    rng = np.random.default_rng(1)
    print(
        f"  {'voice':6s} {'vowel':5s} {'F0':>4s} | {'CheapTrick':>10s} {'F1':>6s} | "
        f"{'cepstral':>8s} {'F1':>6s} | {'D4C':>5s} {'harm.':>5s} | {'f0_track':>8s} {'Cep.f0':>7s}"
    )
    summary = {}
    for voice, vowels in VOWELS.items():
        for name, f123 in vowels.items():
            formants = (*f123, *UPPER[voice])
            for f0 in F0S:
                snd = vowel(f0, formants, rng)
                t = np.arange(0.1, snd.duration - 0.1 + 1e-9, HOP)
                track = (t, np.full(len(t), float(f0)))
                # CheapTrick, given the true F0
                env = so.cheaptrick(snd, track)
                in_range = (env.f >= SCORED[0]) & (env.f <= SCORED[1])
                true_db = 20 * np.log10(envelope_amplitude(env.f[in_range], formants))
                est_db = env.db[0][in_range]  # (freqs, windows)
                ct_err, ct_f1 = envelope_errors(est_db, true_db, np.argmin(np.abs(env.f[in_range] - f123[0])))
                # The cepstral envelope: 40 ms Hann time windows, lifter at half a period
                cep = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=HOP))
                keep = (cep.t > 0.1) & (cep.t < snd.duration - 0.1)
                cep_env = cep.lifter(0.5 / f0).envelope()[0][:, keep]
                stft_f = np.fft.rfftfreq(2 * (cep_env.shape[0] - 1), 1 / FS)
                cep_in = (stft_f >= SCORED[0]) & (stft_f <= SCORED[1])
                cep_true = 20 * np.log10(envelope_amplitude(stft_f[cep_in], formants))
                cep_db = 20 * np.log10(cep_env[cep_in])
                cep_err, cep_f1 = envelope_errors(
                    cep_db, cep_true, np.argmin(np.abs(stft_f[cep_in] - f123[0]))
                )
                # Aperiodicity, given the true F0
                ap_in = (env.f >= 200) & (env.f <= 6000)
                true_ap = noise_share_db(env.f[ap_in])
                ap_err = []
                for measure in (so.d4c, so.harmonic_aperiodicity):
                    ap = measure(snd, track)
                    est = 10 * np.log10(np.maximum(ap.share[0][ap_in], 1e-12))
                    ap_err.append(np.median(np.abs(est - true_ap[:, None])))
                # F0, estimated
                f0_tr = so.f0_track(snd)
                inner = (f0_tr.t > 0.1) & (f0_tr.t < snd.duration - 0.1)
                tracked = f0_tr.f0[0][inner]
                tr_err = np.median(np.abs(tracked / f0 - 1)) if np.all(tracked > 0) else np.nan
                ct, cf, _ = cep.f0()
                cf = cf[0][(ct > 0.1) & (ct < snd.duration - 0.1)]
                cep_f0_gross = np.mean((cf == 0) | (np.abs(cf / f0 - 1) > 0.2))
                print(
                    f"  {voice:6s} {name:5s} {f0:4d} | {ct_err:10.2f} {ct_f1:+6.1f} | "
                    f"{cep_err:8.2f} {cep_f1:+6.1f} | {ap_err[0]:5.1f} {ap_err[1]:5.1f} | "
                    f"{tr_err:8.2%} {cep_f0_gross:6.0%}"
                )
                summary.setdefault(f0, []).append((ct_err, cep_err, *ap_err))
    print(
        "  Mean over the six vowels, by F0: CheapTrick / cepstral envelope dB, D4C / harmonic noise share dB"
    )
    for f0, rows in summary.items():
        m = np.mean(rows, axis=0)
        print(f"    {f0:3d} Hz: {m[0]:5.2f} / {m[1]:5.2f}    {m[2]:4.1f} / {m[3]:4.1f}")
    print("  (Cep.f0 column: share of time windows unvoiced or more than 20% off; f0_track: median error)")


# ----------------------------------------------------------------- part 2: FDA
GUARD = 0.010  # time windows this close to a reference voicing boundary are not scored for voicing [s]
BANDS = [(0, 150), (150, 200), (200, 250), (250, 300), (300, 1000)]
AP_BANDS = [0, 1000, 2000, 4000, 7000]


def read_fx(path):
    """Voiced segments of an XMG .fx file as (time [s], F0 [Hz]) arrays (see
    tools/check_f0_fda.py)."""
    data = path.read_bytes()
    segments, current = [], []
    for line in data[data.find(b"\x0c") + 1 :].decode().splitlines():
        line = line.strip()
        if line == "=":
            if current:
                segments.append(np.array(current))
            current = []
        elif line:
            t_ms, f = line.split()[:2]
            current.append((float(t_ms) / 1000, float(f)))
    if current:
        segments.append(np.array(current))
    return segments


def reference(segments, t):
    """Reference F0 at times t (0 outside every segment), and which time windows
    are far enough from a voicing boundary to score voicing."""
    f0 = np.zeros(len(t))
    near_edge = np.zeros(len(t), bool)
    for s in segments:
        inside = (t >= s[0, 0]) & (t <= s[-1, 0])
        f0[inside] = np.interp(t[inside], s[:, 0], s[:, 1])
        for boundary in (s[0, 0], s[-1, 0]):
            near_edge |= np.abs(t - boundary) < GUARD
    return f0, ~near_edge


def f0_scores(rows):
    """Voicing and pitch errors from stacked (reference, estimate, scorable) rows."""
    ref = np.concatenate([r[0] for r in rows])
    est = np.concatenate([r[1] for r in rows])
    ok = np.concatenate([r[2] for r in rows])
    voiced, est_voiced = ref > 0, est > 0
    both = voiced & est_voiced
    ratio = est[both] / ref[both]
    out = {
        "voicing error": np.sum(ok & (voiced != est_voiced)) / np.sum(ok),
        "misses voiced": np.sum(ok & voiced & ~est_voiced) / np.sum(ok & voiced),
        "gross >20%": np.mean(np.abs(ratio - 1) > 0.2),
        "octave down": np.mean(np.abs(ratio - 0.5) < 0.1),
        "octave up": np.mean(np.abs(ratio - 2) < 0.2),
        "within 5%": np.mean(np.abs(ratio - 1) <= 0.05),
    }
    by_band = []
    for lo, hi in BANDS:
        in_band = both & (ref >= lo) & (ref < hi)
        r = est[in_band] / ref[in_band]
        by_band.append((int(in_band.sum()), np.mean(np.abs(r - 1) > 0.2) if len(r) else np.nan))
    return out, by_band


def band_summary(by_band):
    return ", ".join(
        f"{lo}-{hi if hi < 1000 else ''} Hz {error:.1%} (n={n})"
        for (lo, hi), (n, error) in zip(BANDS, by_band, strict=True)
        if n
    )


def part2(root):
    print()
    print("Part 2: FDA database, laryngograph reference F0 (speech resampled 20 -> 16 kHz)")
    for speaker, label in (("rl", "male"), ("sb", "female")):
        rows = {"so.f0_track": [], "Cepstrum.f0": []}
        envelope_change, ap_given_ref = [], {"D4C": [], "harmonic residual": []}
        for sig in sorted((root / speaker).glob("*.sig")):
            x = resample_poly(np.fromfile(sig, ">i2").astype(float) / 32768, 4, 5)
            snd = so.Sound(x, FS)
            segments = read_fx(sig.with_suffix(".fx"))
            track = so.f0_track(snd)
            ref, ok = reference(segments, track.t)
            rows["so.f0_track"].append((ref, track.f0[0], ok))
            ct, cf, _ = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=HOP)).f0()
            cref, cok = reference(segments, ct)
            rows["Cepstrum.f0"].append((cref, cf[0], cok))
            # Where the reference voices: how much the envelope moves when the
            # tracked F0 replaces the reference, and the noise share given the reference.
            env_ref = so.cheaptrick(snd, (track.t, ref))
            env_trk = so.cheaptrick(snd, track)
            both = (ref > 0) & (track.f0[0] > 0)
            band = (env_ref.f >= 100) & (env_ref.f <= 5000)
            diff = env_trk.db[0][band][:, both] - env_ref.db[0][band][:, both]
            envelope_change.append(np.sqrt(np.mean(diff**2, axis=0)))
            for name, measure in (("D4C", so.d4c), ("harmonic residual", so.harmonic_aperiodicity)):
                ap = measure(snd, (track.t, ref))
                shares = ap.bands(AP_BANDS, envelope=env_ref)[0][:, ref > 0]
                ap_given_ref[name].append(shares)
        print(f"  {speaker} ({label})")
        for name, r in rows.items():
            scores, by_band = f0_scores(r)
            print("    " + f"{name:12s} " + ", ".join(f"{k} {v:.1%}" for k, v in scores.items()))
            print("    " + " " * 13 + "gross >20% by reference F0: " + band_summary(by_band))
        change = np.concatenate(envelope_change)
        print(
            f"    CheapTrick with tracked instead of reference F0: RMS change 100-5000 Hz, "
            f"median {np.median(change):.2f} dB, 95th percentile {np.percentile(change, 95):.2f} dB"
        )
        for name, shares in ap_given_ref.items():
            s = np.concatenate(shares, axis=1)
            medians = 10 * np.log10(np.median(s, axis=1))
            print(
                f"    {name:17s} noise share given reference F0, median dB in "
                + ", ".join(
                    f"{lo / 1000:g}-{hi / 1000:g} kHz {m:+.1f}"
                    for lo, hi, m in zip(AP_BANDS[:-1], AP_BANDS[1:], medians, strict=True)
                )
            )


# -------------------------------------------------------------- part 3: ARCTIC
def harvest_at(t, t_h, f0_h):
    """Harvest's F0 read at times t, 0 where the nearer side is unvoiced."""
    return np.interp(t, t_h, f0_h) * (np.interp(t, t_h, f0_h > 0) > 0.5)


def part3(folder):
    import pyworld

    print()
    print("Part 3: CMU ARCTIC, agreement with WORLD's Harvest (pyworld, 71-800 Hz, 5 ms)")
    for speaker, label in (("bdl", "male"), ("slt", "female")):
        files = sorted(folder.glob(f"{speaker}_arctic_*.wav"))
        files += sorted(folder.glob(f"{speaker}_arctic_*.flac"))
        rows = {"so.f0_track": [], "Cepstrum.f0": []}
        harvest_f0 = []
        for path in files:
            snd = so.load(str(path))
            x = np.ascontiguousarray(snd.data[:, 0], dtype=float)
            f0_h, t_h = pyworld.harvest(x, int(snd.fs), frame_period=HOP * 1000)
            harvest_f0.append(f0_h[f0_h > 0])
            track = so.f0_track(snd)
            ref = harvest_at(track.t, t_h, f0_h)
            rows["so.f0_track"].append((ref, track.f0[0], np.ones(len(track.t), bool)))
            ct, cf, _ = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=HOP)).f0()
            ref = harvest_at(ct, t_h, f0_h)
            rows["Cepstrum.f0"].append((ref, cf[0], np.ones(len(ct), bool)))
        h = np.concatenate(harvest_f0)
        print(
            f"  {speaker} ({label}), {len(files)} sentences: Harvest F0 5th/50th/95th percentile "
            + "/".join(f"{v:.0f}" for v in np.percentile(h, [5, 50, 95]))
            + " Hz"
        )
        for name, r in rows.items():
            scores, by_band = f0_scores(r)
            print("    " + f"{name:12s} " + ", ".join(f"{k} {v:.1%}" for k, v in scores.items()))
            print("    " + " " * 13 + "gross >20% by Harvest F0: " + band_summary(by_band))


if __name__ == "__main__":
    args = sys.argv[1:]
    part1()
    if "--fda" in args:
        part2(Path(args[args.index("--fda") + 1]))
    if "--arctic" in args:
        part3(Path(args[args.index("--arctic") + 1]))
