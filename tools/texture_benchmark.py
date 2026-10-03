"""Texture synthesis benchmark on synthetic textures (milestone 4).

    python tools/texture_benchmark.py [out_dir] [--iter N] [--dur S]

For each texture: measure the original, synthesize a new sample, report the
per-class SNR, and write original/synthesis FLACs plus one comparison figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sonore as so  # noqa: E402
from sonore.texture import TextureStats  # noqa: E402
from sonore.texture.synth import synthesize  # noqa: E402

FS = 20000


def am_noise(dur, rng=0):
    """Broadband noise, 90% sinusoidal AM at 6 Hz, the same in every band."""
    t = np.arange(int(dur * FS)) / FS
    return so.gaussian_noise(dur, FS, rng=rng) * so.Sound(1 + 0.9 * np.sin(2 * np.pi * 6 * t), FS)


def clicks(dur, rate=20, rng=0):
    """Random-polarity single-sample clicks at Poisson-like times, 20/s."""
    r = np.random.default_rng(rng)
    x = np.zeros(int(dur * FS))
    k = int(rate * dur)
    x[r.choice(len(x), k, replace=False)] = r.choice([-1, 1], k)
    return so.Sound(x, FS)


def bursts(dur, rate=3, rng=0):
    """Comodulated noise bursts: 30 ms Hann-windowed, 3/s, over a faint floor."""
    r = np.random.default_rng(rng)
    n, L = int(dur * FS), int(0.03 * FS)
    env = np.zeros(n)
    for o in r.choice(n - L, int(rate * dur), replace=False):
        env[o : o + L] += np.hanning(L)
    return so.gaussian_noise(dur, FS, rng=rng) * so.Sound(env + 0.02, FS)


TEXTURES = {"am_noise": am_noise, "clicks": clicks, "bursts": bursts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path, default=Path("texture_benchmark"))
    ap.add_argument("--iter", type=int, default=20)
    ap.add_argument("--dur", type=float, default=4.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TEXTURES), 3, figsize=(15, 3.2 * len(TEXTURES)), layout="constrained")
    for row, (name, make) in zip(axes, TEXTURES.items(), strict=True):
        orig = make(args.dur)
        target = TextureStats.measure(orig)
        snd, rep = synthesize(target, duration=args.dur, rng=1, max_iter=args.iter)
        best = rep["snr"][rep["best_iteration"] - 1]
        print(
            f"{name:9s} avg {rep['average_snr']:5.1f} dB (iteration {rep['best_iteration']})  "
            + "  ".join(f"{k} {v:.1f}" for k, v in best.items())
        )
        orig_n = so.Sound(target.model.prepare(orig), target.model.fs)
        gain = min(0.1 / orig_n.rms, 0.9 / max(orig_n.peak, snd.peak))  # same gain for both, no clipping
        for label, s in (("original", orig_n), ("synthesis", snd)):
            (s * gain).save(args.out / f"{name}_{label}.flac")
        for ax, (label, s) in zip(row[:2], (("original", orig_n), ("synthesis", snd)), strict=True):
            envelopes = so.cosine_filterbank(30, 50, 9500).analyze(s).envelopes(fs=400)
            envelopes.plot(ax, db_range=50, colorbar=False)
            ax.set_title(f"{name}: {label}")
        avg = [np.mean(list(h.values())) for h in rep["snr"]]
        for c in best:
            row[2].plot(range(1, len(avg) + 1), [h[c] for h in rep["snr"]], lw=0.8, label=c)
        row[2].plot(range(1, len(avg) + 1), avg, "k", lw=2, label="average")
        row[2].axhline(20, color="k", ls=":", lw=0.8)
        row[2].set(xlabel="Iteration", ylabel="SNR [dB]", title="Statistic SNR by iteration")
    axes[0, 2].legend(fontsize=7, ncol=2)
    fig.savefig(args.out / "benchmark.png", dpi=90)


if __name__ == "__main__":
    main()
