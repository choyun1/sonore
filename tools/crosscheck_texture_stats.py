"""Dev-time cross-check of sonore.texture against wil-j-wil/texture_stats (MIT),
a Python port of the MATLAB toolbox. Not part of the test suite.

    git clone https://github.com/wil-j-wil/texture_stats /tmp/ts
    python tools/crosscheck_texture_stats.py /tmp/ts
"""
import sys
import numpy as np
import sonore as so
from sonore.texture import TextureStats

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/tmp/ts")
import texture_stats as ts  # noqa: E402

def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.max(np.abs(a[m] - b[m])) / np.max(np.abs(b[m]))

rng = np.random.default_rng(1)
fs = 20000
t = np.arange(5 * fs) / fs
sounds = {
    "pink": so.gaussian_noise(5, fs, tilt=-3, rng=rng),
    "AM noise": so.gaussian_noise(5, fs, rng=rng) * so.Sound(1 + 0.9 * np.sin(2 * np.pi * 7 * t), fs),
}
for name, s in sounds.items():
    p = ts.SoundTexture(s.data[:, 0].copy(), fs)
    q = TextureStats.measure(s)
    B = q.env_mean.shape[0]
    # port's C/C1 use unweighted means/stds; compare offsets by pulling the diagonals
    pc = np.array([[p.env_c[j, j + d] if j + d < B else np.nan for d in q.model.corr_offsets] for j in range(B)])
    pc1 = np.stack([np.array([[p.mod_c1[j, j + d, k] if j + d < B else np.nan for d in q.model.c1_offsets]
                              for j in range(B)]) for k in range(6)], axis=1)
    print(f"{name}: max relative difference")
    print(f"  env_mean {rel(q.env_mean, p.env_mean):.2e}  env_var {rel(q.env_var, p.env_var**2):.2e}  "
          f"skew {rel(q.env_skew, p.env_skew):.2e}  kurt {rel(q.env_kurt, p.env_kurt):.2e}")
    print(f"  mod_power {rel(q.mod_power, p.mod_power):.2e}  env_corr {rel(q.env_corr, pc):.2e}  "
          f"c1 {rel(q.c1, pc1):.2e}  c2 {rel(np.stack([q.c2.real, q.c2.imag], -1), p.mod_c2):.2e}")
