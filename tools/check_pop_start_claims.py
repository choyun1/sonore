"""Check the claims of docs/design/texture/pop-start.md.

    python tools/check_pop_start_claims.py [--paper DIR] [--iter N] [--names NAME ...]

Synthesizes each recording from Gaussian noise and from ``pop_start`` (the
recording's own length, ``--iter`` iterations, seed 1, the start's seed 5)
and measures both against the recording. ``docs/textures/`` supplies our
fire, stream and applause; ``--paper DIR`` names the folder of McDermott &
Simoncelli's recordings for Fire1, "Fire - in chimney" and Fire2 (not in the
repository, licenses unknown). About 20 minutes on four cores for all six.

Synthesis and the cochlear filterbank come from sonore; every measurement
below is the checker's own NumPy/SciPy code so that it does not share
``envelope_peaks``' or ``pop_rate``'s blind spots:

- SNR: average over the paper's statistic classes of ``TextureStats.snr``.
- exponent: Hill estimator on the largest 5% of 5 ms envelope maxima of
  the sound high-passed at 300 Hz (Hilbert envelope, 500 Hz smoothing).
- gap: 99.9th percentile of those maxima over their median, in dB.
- crest: peak over RMS, in dB.
- width: median width at half maximum of envelope peaks more than 20 dB
  above the median envelope, in ms.
- pops/s: envelope peaks more than 10 dB above the median envelope, at
  least 10 ms apart, per second; also by ``pop_rate``'s crossing count.
- bg5: the envelope's 5th percentile below its median, in dB (how deep the
  background dips between pops).
- together: at the 20 biggest pops, the fraction of the 30 cochlear channels
  whose own envelope peaks (10 dB over its median) within 1 ms.
"""

import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

import sonore as so
from sonore.texture import TextureModel, TextureStats
from sonore.texture.synth import pop_rate, pop_start, synthesize

warnings.simplefilter("ignore")
HERE = Path(__file__).parent.parent


def recordings(paper):
    files = {name: HERE / "docs" / "textures" / f"{name}.flac" for name in ("fire", "stream", "applause")}
    if paper is not None:
        for name, file in (
            ("Fire1", "Fire1.wav"),
            ("chimney", "Fire - in chimney.wav"),
            ("Fire2", "Fire2.wav"),
        ):
            files[name] = Path(paper) / file
    return files


def envelope(x, fs, cutoff=300.0):
    high = sosfiltfilt(butter(4, cutoff, "highpass", fs=fs, output="sos"), x)
    return sosfiltfilt(butter(2, 500.0, "lowpass", fs=fs, output="sos"), np.abs(hilbert(high)))


def measure(sound):
    fs, x = sound.fs, sound.data[:, 0]
    env = envelope(x, fs)
    bin_length = int(round(0.005 * fs))
    maxima = env[: len(env) // bin_length * bin_length].reshape(-1, bin_length).max(axis=1)
    ordered = np.sort(maxima)[::-1]
    k = max(10, int(0.05 * len(maxima)))
    out = {
        "exponent": k / np.sum(np.log(ordered[:k] / ordered[k])),
        "gap": 20 * np.log10(np.percentile(maxima, 99.9) / np.median(maxima)),
        "crest": 20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x**2))),
        "bg5": 20 * np.log10(np.median(env) / np.percentile(env, 5)),
    }
    median = np.median(env)
    loud, _ = find_peaks(env, height=median * 10 ** (20 / 20), distance=int(0.01 * fs))
    widths = []
    for i in loud:
        half = env[i] / 2
        left, right = i, i
        while left > 0 and env[left] > half:
            left -= 1
        while right < len(env) - 1 and env[right] > half:
            right += 1
        widths.append((right - left) / fs * 1000)
    out["width"] = float(np.median(widths)) if widths else float("nan")
    peaks, _ = find_peaks(env, height=median * 10 ** (10 / 20), distance=int(0.01 * fs))
    out["pops/s"] = len(peaks) / sound.duration
    out["crossings/s"] = pop_rate(sound)
    # channel coherence at the 20 biggest pops
    model = TextureModel()
    prepared = model.prepare(sound)
    subbands = model.subbands(prepared)[:, 1:-1]
    channel_env = sosfiltfilt(
        butter(2, 500.0, "lowpass", fs=model.fs, output="sos"), np.abs(hilbert(subbands, axis=0)), axis=0
    )
    broad = envelope(prepared, model.fs)
    idx, _ = find_peaks(broad, height=np.median(broad) * 10, distance=int(0.05 * model.fs))
    top = idx[np.argsort(broad[idx])[::-1][:20]]
    window = int(0.001 * model.fs)
    medians = np.median(channel_env, axis=0)
    hits = [
        np.mean(channel_env[i - window : i + window + 1].max(axis=0) / medians >= 10 ** (10 / 20))
        for i in top
        if i > window
    ]
    out["together"] = float(np.mean(hits)) if hits else float("nan")
    return out


def one(job):
    name, file, start_kind, iterations = job
    sound = so.load(file)
    sound = sound.mono() if sound.n_channels > 1 else sound
    target = TextureStats.measure(sound)
    start = pop_start(sound, sound.duration, rng=5) if start_kind == "pops" else None
    synth, _ = synthesize(target, duration=sound.duration, rng=1, max_iter=iterations, init=start)
    snr = target.snr(TextureStats.measure(synth, window="uniform"))
    return name, start_kind, float(np.mean(list(snr.values()))), measure(synth), measure(sound)


COLUMNS = ("exponent", "gap", "crest", "width", "pops/s", "crossings/s", "bg5", "together")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=None)
    parser.add_argument("--iter", type=int, default=30)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    files = recordings(args.paper)
    names = args.names or list(files)
    jobs = [(name, files[name], kind, args.iter) for name in names for kind in ("noise", "pops")]
    with Pool(args.workers) as pool:
        results = pool.map(one, jobs)
    header = f"{'recording':10s} {'start':6s} {'SNR':>5s} " + " ".join(f"{c:>11s}" for c in COLUMNS)
    print(header)
    for name in names:
        rows = [r for r in results if r[0] == name]
        original = rows[0][4]
        print(f"{name:10s} {'(orig)':6s} {'':>5s} " + " ".join(f"{original[c]:11.2f}" for c in COLUMNS))
        for _, kind, snr, synth, _ in rows:
            print(f"{'':10s} {kind:6s} {snr:5.1f} " + " ".join(f"{synth[c]:11.2f}" for c in COLUMNS))


if __name__ == "__main__":
    main()
