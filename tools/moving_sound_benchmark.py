"""Timing behind docs/design/moving-sound.md, D5 (c): a truly time-varying
convolution, with a fresh interpolated HRIR at every sample, against
today's switching at 200 points per second.

    python tools/moving_sound_benchmark.py [sofa_file]

Uses sonore and a SOFA HRIR set (default: the MIT KEMAR set libmysofa
installs, 1.4 m, 44.1 kHz; needs h5py). The source swings 30 degrees to
either side at 2 Hz for 3 s at 48 kHz.
"""

import sys
import time

import numpy as np

import sonore as so

SOFA = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"
FS = 48000
DURATION = 3.0
DISTANCE_CM = 140.0


def per_sample(sound, azimuth, hrirs, chunk=4096):
    """y[n] = sum_k h_n[k] x[n - k], with h_n the HRIR interpolated at sample
    n's position. Returns the output and the seconds spent interpolating and
    filtering."""
    signal = sound.data[:, 0]
    n_taps = hrirs.at(so.hcc_to_rect(DISTANCE_CM, 0.0, 0.0), fs=FS).shape[-1]
    padded = np.concatenate([np.zeros(n_taps - 1), signal])
    out = np.zeros((len(signal), 2))
    interpolating = filtering = 0.0
    for start in range(0, len(signal), chunk):
        stop = min(len(signal), start + chunk)
        started = time.perf_counter()
        positions = np.column_stack(so.hcc_to_rect(DISTANCE_CM, 0 * azimuth[start:stop], azimuth[start:stop]))
        hrir_per_sample = hrirs.at(positions, fs=FS)  # (samples, 2, taps)
        interpolated = time.perf_counter()
        past_samples = np.lib.stride_tricks.sliding_window_view(padded[start : stop + n_taps - 1], n_taps)[
            :, ::-1
        ]
        out[start:stop] = np.einsum("sct,st->sc", hrir_per_sample, past_samples)
        interpolating += interpolated - started
        filtering += time.perf_counter() - interpolated
    return out, interpolating, filtering


if __name__ == "__main__":
    hrirs = so.HRIRSet.from_sofa(SOFA)
    sound = so.gaussian_noise(DURATION, FS, rng=0)
    t = np.arange(len(sound)) / FS
    azimuth = 30 * np.sin(2 * np.pi * 2 * t)
    hop = FS // 200
    points = np.column_stack(so.hcc_to_rect(DISTANCE_CM, 0 * azimuth[::hop], azimuth[::hop]))
    started = time.perf_counter()
    so.move_sound(sound, points, hrirs)
    print(f"today's switching, 200 points/s: {time.perf_counter() - started:.2f} s")
    started = time.perf_counter()
    _, interpolating, filtering = per_sample(sound, azimuth, hrirs)
    total = time.perf_counter() - started
    print(f"a fresh HRIR every sample: {total:.1f} s, of which")
    print(f"  interpolating the HRIRs {interpolating:.1f} s, filtering {filtering:.1f} s")
