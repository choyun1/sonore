"""Checks behind "Onsets fitted across distances" in docs/design/spatial/moving-sound.md:
PKU-IOA's horizontal ring at 1.3 m arrives early, what that did to the Doppler shift
of a source moving through it, and what fitting the onsets across distances changes.

    python tools/check_onset_fit.py PKU_IOA_DIR

PKU_IOA_DIR holds Cho's original PKU-IOA .dat files (any layout below it).
"""

import sys
import warnings

import numpy as np
from scipy.signal import hilbert

import sonore as so

warnings.simplefilter("ignore")  # the one empty PKU-IOA file
measured = so.HRIRSet.from_pku_ioa(sys.argv[1], fit_onsets=False)
fitted = so.HRIRSet.from_pku_ioa(sys.argv[1])
radii = np.linalg.norm(measured.positions, axis=1)
_, elevation, azimuth = so.rect_to_hcc(*measured.positions.T)
horizontal = np.abs(elevation) < 1e-6
mean_onset = measured._onsets.mean(axis=1) / measured.fs  # the two ears' mean [s]


def on_shell(distance):
    return np.abs(radii - distance) < 1e-3 * distance


print("Measured mean onset minus travel time r/c, median [us]")
for distance in measured.distances:
    beyond = (mean_onset - radii / so.SPEED_OF_SOUND) * 1e6
    print(
        f"  {distance:4.2f} m: all directions {np.median(beyond[on_shell(distance)]):6.0f}, "
        f"horizontal ring {np.median(beyond[on_shell(distance) & horizontal]):6.0f}"
    )

print("Horizontal ring, 1.3 m onset minus 1.0 m onset, per azimuth [us]")
step = []
for angle in np.unique(np.round(azimuth[on_shell(1.3) & horizontal])):
    pair = [np.flatnonzero(on_shell(d) & horizontal & (np.round(azimuth) == angle))[0] for d in (1.3, 1.0)]
    step.append((mean_onset[pair[0]] - mean_onset[pair[1]]) * 1e6)
print(f"  {len(step)} azimuths: min {min(step):.0f}, median {np.median(step):.0f}, max {max(step):.0f}")
print(f"  travel time over 0.3 m: {0.3 / so.SPEED_OF_SOUND * 1e6:.0f}")

print("Shift of each IR by the fit [us], median and 5th to 95th percentile")
shift = (fitted._delay_onsets - measured._onsets)[:, 0] / measured.fs * 1e6
for distance in measured.distances:
    here, ring = shift[on_shell(distance)], shift[on_shell(distance) & horizontal]
    print(
        f"  {distance:4.2f} m: {np.median(here):6.1f} ({np.percentile(here, 5):6.1f} to "
        f"{np.percentile(here, 95):5.1f}), horizontal ring {np.median(ring):6.1f}"
    )
itd_change = np.diff(fitted._delay_onsets - measured._onsets, axis=1)
print(f"  largest change of any ITD [us]: {np.abs(itd_change).max() / measured.fs * 1e6:.2g}")

print("Measured mean onset minus the fit at 20 cm, median [us]")
_, direction = np.unique(np.round(measured.positions / radii[:, None], 6), axis=0, return_inverse=True)
direction = direction.ravel()
beyond = mean_onset - radii / so.SPEED_OF_SOUND
constant_only = np.array([np.median(beyond[direction == k]) for k in range(direction.max() + 1)])[direction]
with_fit = fitted._delay_onsets.mean(axis=1) / measured.fs - radii / so.SPEED_OF_SOUND
print(
    f"  r/c + a, a the median over distances: {np.median((beyond - constant_only)[on_shell(0.2)]) * 1e6:.0f}"
)
print(f"  r/c + a + b/r, as fitted: {np.median((beyond - with_fit)[on_shell(0.2)]) * 1e6:.0f}")

print("Doppler shift at the left ear, 2 kHz tone approaching at 15 m/s from 3 m [cents]")
fs, frequency, speed, start, end = 48000, 2000.0, 15.0, 3.0, 0.4
duration = (start - end) / speed
tone = so.Sound(np.sin(2 * np.pi * frequency * np.arange(int(duration * fs)) / fs), fs)
print(f"  travel time alone: {1200 * np.log2(1 / (1 - speed / so.SPEED_OF_SOUND)):.1f}")
checked = [2.0, 1.45, 1.15, 0.9, 0.62]
for elevation_deg, azimuth_deg in [(0, 0), (0, 40), (10, 40)]:
    toward = np.array(so.hcc_to_rect(100, elevation_deg, azimuth_deg))

    def path(t, toward=toward):
        return np.outer(start - speed * np.clip(np.atleast_1d(t), 0, duration), toward)

    for label, hrirs in (("measured onsets", measured), ("fitted onsets ", fitted)):
        left = so.move_sound(tone, path, hrirs).data[:, 0]
        instantaneous = np.diff(np.unwrap(np.angle(hilbert(left)))) * fs / 2 / np.pi
        cents = []
        for distance in checked:
            center = int(((start - distance) / speed + distance / so.SPEED_OF_SOUND) * fs)
            window = instantaneous[center - 144 : center + 144]  # 6 ms
            cents.append(1200 * np.log2(np.median(window) / frequency))
        print(
            f"  elevation {elevation_deg:2d}, azimuth {azimuth_deg:2d}, {label}: "
            + "  ".join(f"{d} m {c:5.1f}" for d, c in zip(checked, cents, strict=True))
        )
