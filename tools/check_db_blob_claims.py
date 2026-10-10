"""Numbers for D8 of docs/design/sources/gaussian-spectrogram.md: a blob
target drawn in dB (log amplitude) instead of linear amplitude, with
ModulationSpectrum.from_blobs(..., scale="db"). Uses sonore.

  G8  for one blob (4 Hz, 1 cyc/oct), 5 seeds: the drawn envelopes' linear
      rms depth, and the share of the sound's modulation power (rate at
      least 0.5 Hz) that falls within an octave of 4 Hz and 0.5 cyc/oct of
      1 cyc/oct, with the measured peak, for the linear draw at rms depth
      0.2 and for dB draws with standard deviations of 3, 6 and 10 dB
"""

import collections
import warnings

import numpy as np

import sonore as so

fs, dur = 16000, 2.0
blob = so.ModulationBlob(4.0, 1.0)


def blob_share(sound, scale):
    ms = so.ModulationSpectrum.octave(sound, f_lo=250, f_hi=7000, scale=scale)
    power = 10 ** (ms.level / 10)
    rate, dens = ms.w_t[None, :], ms.w_f[:, None]
    moving = np.broadcast_to(np.abs(rate) >= 0.5, power.shape)
    near = moving & (rate > 0) & (np.abs(np.log2(np.abs(rate) / 4)) <= 1) & (np.abs(dens - 1) <= 0.5)
    pr, pd = ms.peak()
    return power[near].sum() / power[moving].sum(), pr, pd


def depth(env):
    x = env.data[:, 1:-1, 0]
    return np.sqrt(np.mean((x - x.mean(axis=0)) ** 2)) / x.mean()


warnings.simplefilter("ignore")
rows = []
for seed in range(5):
    lin = so.ModulationSpectrum.from_blobs(blob, dur, f_lo=250, f_hi=7000, rms_depth=0.2)
    env = lin.to_envelopes(rng=seed)
    s = env.to_sound("noise", fs=fs, rng=100 + seed)
    rows.append(("linear, rms_depth 0.2", depth(env), *blob_share(s, "linear"), *blob_share(s, "db")))
    for sd in (3, 6, 10):
        d = so.ModulationSpectrum.from_blobs(blob, dur, f_lo=250, f_hi=7000, scale="db", sd_db=sd)
        env = d.to_envelopes(rng=seed)
        s = env.to_sound("noise", fs=fs, rng=100 + seed)
        rows.append((f"dB, sd {sd} dB", depth(env), *blob_share(s, "linear"), *blob_share(s, "db")))
agg = collections.defaultdict(list)
for r in rows:
    agg[r[0]].append(r[1:])
for name, values in agg.items():
    v = np.array(values, float)
    print(
        f"G8 {name}: depth {v[:, 0].mean():.2f}; linear spectrum: share {v[:, 1].mean():.2f}, "
        f"peak {np.median(v[:, 2]):.1f} Hz {np.median(v[:, 3]):.2f} cyc/oct; dB spectrum: share "
        f"{v[:, 4].mean():.2f}, peak {np.median(v[:, 5]):.1f} Hz {np.median(v[:, 6]):.2f} cyc/oct"
    )
