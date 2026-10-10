"""Checks behind the speed-up of so.move_sound in docs/design/spatial/moving-sound.md
("As built"): finding each direction's triangle from its nearest measured directions
picks the same triangles and weights as testing every triangle, the tabulated
windowed sinc reads as accurately as the sinc computed for every sample, and how
long a render takes with each.

    python tools/check_move_sound_speed.py PKU_IOA_DIR

PKU_IOA_DIR holds Cho's original PKU-IOA .dat files (any layout below it).
"""

import sys
import time
import warnings

import numpy as np
from scipy.special import i0

import sonore as so
from sonore.spatial import spatialization as sp

warnings.simplefilter("ignore")  # the one empty PKU-IOA file
hrirs = so.HRIRSet.from_pku_ioa(sys.argv[1])
rng = np.random.default_rng(1)


def direct_read(signal, positions):
    """The windowed sinc computed afresh for every position, as before the table."""
    half, beta, margin = sp._READ_HALF_WIDTH, sp._READ_KAISER_BETA, 2 * sp._READ_HALF_WIDTH
    padded = np.concatenate([np.zeros(margin), signal, np.zeros(margin)])
    offsets = np.arange(-half + 1, half + 1)
    out = np.zeros(len(positions))
    inside = np.flatnonzero((positions > -half) & (positions < len(signal) + half - 1))
    for start in range(0, len(inside), 32768):
        rows = inside[start : start + 32768]
        whole = np.floor(positions[rows]).astype(int)
        distance = positions[rows, None] - (whole[:, None] + offsets[None, :])
        window = i0(beta * np.sqrt(np.clip(1 - (distance / half) ** 2, 0, None)))
        taps = np.sinc(distance) * window / i0(beta)
        out[rows] = np.sum(padded[whole[:, None] + offsets[None, :] + margin] * taps, axis=1)
    return out


def search_all(self, shell, directions):
    """Barycentric weights from testing every triangle of the shell, as before."""
    _, triangles, inverse = self._shells[shell]
    triangle = self._search_all_triangles(shell, directions)
    found = np.einsum("nij,nj->ni", inverse[triangle], directions)
    return triangles[triangle], found / found.sum(axis=1, keepdims=True)


def db(error, reference):
    return 20 * np.log10(np.sqrt(np.mean(error**2) / np.mean(reference**2)))


print("Triangle search: nearest measured directions against every triangle")
directions = rng.standard_normal((20000, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
for shell, distance in enumerate(hrirs.distances):
    fast, full = hrirs._direction_weights(shell, directions), search_all(hrirs, shell, directions)
    same = np.array_equal(fast[0], full[0]) and np.array_equal(fast[1], full[1])
    print(f"  {distance:4.2f} m: {len(directions)} directions, identical: {same}")

print("Reading between samples: table against the sinc computed per sample")
fs, n = 48000, 48000
positions = np.sort(rng.uniform(2000, n - 2000, 20000))
noise = rng.standard_normal(n)
difference = sp._read_between_samples(noise, positions) - direct_read(noise, positions)
print(f"  white noise: table differs by {db(difference, noise):.0f} dB")
for frequency in (1000, 8000, 16000, 20000):
    tone = np.sin(2 * np.pi * frequency * np.arange(n) / fs)
    truth = np.sin(2 * np.pi * frequency * positions / fs)
    errors = [db(read(tone, positions) - truth, truth) for read in (direct_read, sp._read_between_samples)]
    print(
        f"  {frequency / 1000:4.0f} kHz tone: error {errors[0]:.1f} dB per sample, {errors[1]:.1f} dB table"
    )

print("Render time: 3 s of noise at 44.1 kHz")
noise = so.Sound(rng.standard_normal(3 * 44100), 44100)
paths = {
    "azimuth swing, 1 m": so.hcc_trajectory(100, 0, lambda t: 30 * np.sin(2 * np.pi * 2 * t)),
    "walk from 3 m to 0.6 m": so.hcc_trajectory(([0, 3], [300, 60]), 10, 30),
}
fast_read, fast_weights = sp._read_between_samples, so.HRIRSet._direction_weights
for name, path in paths.items():
    seconds, outputs = [], []
    for read, weights in ((direct_read, search_all), (fast_read, fast_weights)):
        sp._read_between_samples, so.HRIRSet._direction_weights = read, weights
        so.move_sound(noise, path, hrirs)  # caches
        started = time.perf_counter()
        outputs.append(so.move_sound(noise, path, hrirs).data)
        seconds.append(time.perf_counter() - started)
    print(
        f"  {name}: {seconds[0]:.2f} s before, {seconds[1]:.2f} s now, "
        f"outputs differ by {db(outputs[1] - outputs[0], outputs[0]):.0f} dB"
    )
sp._read_between_samples, so.HRIRSet._direction_weights = fast_read, fast_weights
