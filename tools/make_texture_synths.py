"""Synthesize the texture recordings in docs/textures/ (milestone 5).

    python tools/make_texture_synths.py [NAME ...] [--iter N] [--dur S] [--backgrounds-only]

For each excerpt: measure its statistics (the paper's model, ramped
measurement window), synthesize ``--dur`` seconds from noise, and write
``docs/textures/synth/<name>.flac`` plus ``<name>.json`` (the per-iteration
SNR history). Ablations (``ABLATIONS``) impose only some statistic classes
and are written as ``<name>__<ablation>.flac``. For the recordings in
``BACKGROUNDS``, the recording with its pops limited away (``remove_pops``) is
synthesized too, 10 s long, as ``<name>__background.flac``: the gallery adds
pops back with ``so.event_train``. Synthesis takes about 2 s per
iteration for 5 s of sound (about a minute per texture at 30 iterations), so
results are cached here and the gallery build only reads them.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sonore as so
from sonore.texture import PAPER_CLASSES, TextureStats
from sonore.texture.synth import synthesize

HERE = Path(__file__).parent.parent / "docs" / "textures"
OUT = HERE / "synth"
NAMES = ["rain", "stream", "crickets", "applause", "fire", "mud", "wind_rain"]
MARGINALS = ("env_mean", "env_var", "env_skew", "env_kurt")
# Which statistics are imposed; the full model is PAPER_CLASSES.
ABLATIONS = {
    "marginals": MARGINALS,
    "marginals_modpower": MARGINALS + ("mod_power",),
}
ABLATED = ["applause", "stream", "fire"]
BACKGROUNDS = ["fire"]
BACKGROUND_SECONDS = 10.0
SEED = 1


def remove_pops(sound):
    """The sound with its pops limited away: each 5 ms bin is scaled so that its largest envelope
    value above 300 Hz (as `so.envelope_peaks` measures it) is at most twice the 80th percentile
    of those values. The gain is smoothed over 2 ms so the cuts do not click. The same function
    is on the Sound textures gallery page, where it is run and heard."""
    peaks = so.envelope_peaks(sound, cutoffs_hz=(300.0,))
    maxima = peaks.maxima[:, 0]
    threshold = 2 * np.percentile(maxima, 80)
    bin_length = int(round(peaks.bin_seconds * sound.fs))
    gain = np.ones(len(sound))
    gain[: len(maxima) * bin_length] = np.repeat(np.minimum(1.0, threshold / maxima), bin_length)
    smoothing = np.hanning(int(0.002 * sound.fs))
    gain = np.minimum(gain, np.convolve(gain, smoothing / smoothing.sum(), "same"))
    return sound * gain[:, None]


def run(name, classes, tag, iters, dur, sound=None):
    target = TextureStats.measure(sound if sound is not None else so.load(HERE / f"{name}.flac"))
    t0 = time.time()
    snd, rep = synthesize(target, duration=dur, classes=classes, rng=SEED, max_iter=iters)
    # score every synthesis on all classes, including those it didn't impose
    full = target.snr(TextureStats.measure(snd, window="uniform"))
    out = f"{name}__{tag}" if tag else name
    (snd * (0.05 / snd.rms)).save(OUT / f"{out}.flac")
    info = {
        "classes": list(classes),
        "best_iteration": rep["best_iteration"],
        "iterations": rep["iterations"],
        "average_snr_imposed": rep["average_snr"],
        "snr_all_classes": full,
        "history": rep["snr"],
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / f"{out}.json").write_text(json.dumps(info, indent=1))
    avg = np.mean(list(full.values()))
    print(
        f"{out:30s} {info['seconds']:6.0f} s  it {rep['best_iteration']:2d}  avg(all) {avg:5.1f} dB  "
        + " ".join(f"{k} {v:.0f}" for k, v in full.items()),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=NAMES)
    ap.add_argument("--iter", type=int, default=30)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--backgrounds-only", action="store_true", help="only the pop-free backgrounds")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in args.names:
        if not args.backgrounds_only:
            run(name, PAPER_CLASSES, "", args.iter, args.dur)
        if name in ABLATED and not args.no_ablations and not args.backgrounds_only:
            for tag, classes in ABLATIONS.items():
                run(name, classes, tag, args.iter, args.dur)
        if name in BACKGROUNDS:
            background = remove_pops(so.load(HERE / f"{name}.flac"))
            run(name, PAPER_CLASSES, "background", args.iter, BACKGROUND_SECONDS, sound=background)


if __name__ == "__main__":
    main()
