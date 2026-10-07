"""Numerical checks for the claims in docs/design/spatial/moving-sound.md (C1-C7).

Like the other claim checkers, this is independent of sonore: only NumPy and
SciPy, with every step written out from its formula. It holds a small
prototype of the proposed renderer's propagation stage (a delay that follows
the source's distance, read with fractional-delay interpolation), of the
current `move_sound` (raised-cosine switching between fixed filters), and of
a batched frequency-domain version of that switching, so the numbers in the
document can be reproduced; it is not library code. Each line prints the
claim number and the number that supports it.

    python tools/check_moving_sound_claims.py
    python tools/check_moving_sound_claims.py --pku DIR   # also C7

C7 needs the PKU-IOA files in DIR: the original .dat files (any layout), or
the SOFA copy (dist_0.2m.sofa ... dist_1.6m.sofa, with h5py); without --pku it
is skipped. The rest runs in about ten seconds.

Coordinates are sonore's head-centered Cartesian ones, in meters: x right,
y front, z up. The source's position is a function of the time at which it
emits; the sound reaches the head center r/c later.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import i0

FS = 48000.0
C_SOUND = 343.0  # m/s, air at about 20 degrees C
HEAD_RADIUS = 0.0875  # m, for the Woodworth interaural delay


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


def db(power_ratio):
    return 10 * np.log10(power_ratio)


def error_db(estimate, reference):
    """Error energy re signal energy [dB]."""
    return db(np.sum((estimate - reference) ** 2) / np.sum(reference**2))


# ------------------------------------------------------------- trajectories
def pass_by(speed, closest=2.0, duration=3.0):
    """A straight path in front of the head, left to right, at `speed` m/s,
    `closest` m ahead at the middle of `duration`."""

    def position(t):
        t = np.asarray(t, float)
        return np.stack([speed * (t - duration / 2), np.full_like(t, closest), np.zeros_like(t)], axis=-1)

    return position


def azimuth_swing(amplitude_deg=30.0, rate=2.0, distance=1.0):
    """Cho & Kidd (2022): azimuth(t) = A sin(2 pi rate t) about straight ahead,
    at a fixed distance. Azimuth is clockwise from the front, so x = r sin(az)."""

    def azimuth(t):
        return np.radians(amplitude_deg) * np.sin(2 * np.pi * rate * np.asarray(t, float))

    def position(t):
        az = azimuth(t)
        return np.stack([distance * np.sin(az), distance * np.cos(az), np.zeros_like(az)], axis=-1)

    return position, azimuth


def emission_time(t, delay_of, iterations=60):
    """Solve t_e + delay(t_e) = t by fixed-point iteration from t_e = t - delay(t).
    Converges when the delay changes by less than a second per second
    (the source is slower than sound): the contraction factor is |d delay/dt|."""
    t_e = t - delay_of(t)
    for _ in range(iterations):
        t_e = t - delay_of(t_e)
    return t_e


# ------------------------------------------------------ fractional-delay reads
def read_linear(x, n):
    i = np.floor(n).astype(int)
    frac = n - i
    return (1 - frac) * x[i] + frac * x[i + 1]


def read_lagrange3(x, n):
    """Cubic Lagrange through the samples at i-1, i, i+1, i+2."""
    i = np.floor(n).astype(int)
    d = n - i
    weights = [
        -d * (d - 1) * (d - 2) / 6,
        (d + 1) * (d - 1) * (d - 2) / 2,
        -(d + 1) * d * (d - 2) / 2,
        (d + 1) * d * (d - 1) / 6,
    ]
    return sum(w * x[i + k] for w, k in zip(weights, (-1, 0, 1, 2), strict=True))


def read_sinc(x, n, half_width=16, beta=8.0):
    """Kaiser-windowed sinc over 2*half_width samples around each read point."""
    i = np.floor(n).astype(int)
    offsets = np.arange(-half_width + 1, half_width + 1)
    distance = n[:, None] - (i[:, None] + offsets[None, :])  # read point minus tap
    window = i0(beta * np.sqrt(np.clip(1 - (distance / half_width) ** 2, 0, None))) / i0(beta)
    return np.sum(x[i[:, None] + offsets[None, :]] * np.sinc(distance) * window, axis=1)


# ----------------------------------------------- the current move_sound, written out
def switched(signal, n_points, filters_of, n_out):
    """move_sound's scheme: N raised-cosine windows over the input that sum to
    one, each windowed piece filtered by its own fixed filter, outputs added.
    `filters_of(k)` returns the (n_channels, taps) filter for point k."""
    length = len(signal)
    knots = np.linspace(0, length - 1, n_points)
    out = None
    for k in range(n_points):
        start = int(np.floor(knots[k - 1])) if k > 0 else 0
        stop = int(np.ceil(knots[k + 1])) + 1 if k < n_points - 1 else length
        samples = np.arange(start, stop)
        offset = np.interp(samples, knots, np.arange(n_points)) - k
        window = np.where(np.abs(offset) < 1, (1 + np.cos(np.pi * offset)) / 2, 0.0)
        if k == 0:
            window[samples <= knots[0]] = 1
        if k == n_points - 1:
            window[samples >= knots[-1]] = 1
        filters = filters_of(k)
        if out is None:
            out = np.zeros((filters.shape[0], n_out))
        segment = fftconvolve((signal[start:stop] * window)[None, :], filters, axes=1)
        stop_out = min(n_out, start + segment.shape[1])
        out[:, start:stop_out] += segment[:, : stop_out - start]
    return out


def switched_batched(signal, n_points, filters, n_out):
    """The same windows and filters, but every windowed piece transformed in
    one batched FFT and multiplied by every filter's spectrum at once.
    `filters` is (N, n_channels, taps)."""
    length = len(signal)
    knots = np.linspace(0, length - 1, n_points)
    starts = np.array([int(np.floor(knots[k - 1])) if k > 0 else 0 for k in range(n_points)])
    stops = np.array(
        [int(np.ceil(knots[k + 1])) + 1 if k < n_points - 1 else length for k in range(n_points)]
    )
    width = int(np.max(stops - starts))
    taps = filters.shape[-1]
    n_fft = 1 << int(np.ceil(np.log2(width + taps - 1)))
    pieces = np.zeros((n_points, width))
    for k in range(n_points):
        samples = np.arange(starts[k], stops[k])
        offset = np.interp(samples, knots, np.arange(n_points)) - k
        window = np.where(np.abs(offset) < 1, (1 + np.cos(np.pi * offset)) / 2, 0.0)
        if k == 0:
            window[samples <= knots[0]] = 1
        if k == n_points - 1:
            window[samples >= knots[-1]] = 1
        pieces[k, : len(samples)] = signal[starts[k] : stops[k]] * window
    spectra = np.fft.rfft(pieces, n_fft)[:, None, :] * np.fft.rfft(filters, n_fft)
    blocks = np.fft.irfft(spectra, n_fft)[..., : width + taps - 1]  # (N, channels, block)
    out = np.zeros((filters.shape[1], n_out + width + taps))
    for k in range(n_points):
        out[:, starts[k] : starts[k] + blocks.shape[-1]] += blocks[k]
    return out[:, :n_out]


def delay_filter(delay_samples, gain, taps, half_width=64):
    """A fixed fractional delay as a FIR: a Kaiser-windowed sinc 2*half_width
    samples wide, centered on the delay (accurate to well below the errors
    measured here)."""
    distance = np.arange(taps) - delay_samples
    window = i0(10.0 * np.sqrt(np.clip(1 - (distance / half_width) ** 2, 0, None))) / i0(10.0)
    return gain * np.sinc(distance) * window


# ------------------------------------------------------------------- claims
def claim_doppler_sizes():
    """C1: how large Doppler shifts and the convective level change are."""
    for speed, what in [(1.4, "walking"), (5.0, "running"), (15.0, "a car in town")]:
        cents = 1200 * np.log2(C_SOUND / (C_SOUND - speed))
        report("C1", f"{what}, {speed:g} m/s straight toward the head: pitch up, cents", cents)
    for speed in (1.4, 15.0):
        mach = speed / C_SOUND
        report(
            "C1",
            f"{speed:g} m/s: level change from 1/(1 - M)^2 (convective), dB",
            -20 * np.log10((1 - mach) ** 2),
        )


def claim_swing_doppler():
    """C2: the azimuth swing has no Doppler at the head center, but each ear's
    path length changes, so the ears hear slightly different pitches."""
    position, azimuth = azimuth_swing()
    t = np.arange(0, 1.0, 1 / FS)
    radius = np.linalg.norm(position(t), axis=1)
    report("C2", "30 deg swing at 2 Hz, 1 m: range of distance to head center, m", np.ptp(radius))
    speed = np.max(np.abs(np.gradient(azimuth(t), t))) * 1.0
    report("C2", "peak speed of the source along its arc, m/s", speed)

    # Woodworth: delay to each ear re head center, a/c (theta + sin theta)/2 with sign by ear
    def ear_delay(az, ear):  # ear = +1 right, -1 left
        return -ear * HEAD_RADIUS / C_SOUND * (az + np.sin(az)) / 2

    rate_right = np.gradient(ear_delay(azimuth(t), +1), t)
    rate_left = np.gradient(ear_delay(azimuth(t), -1), t)
    interaural_cents = 1200 * np.log2((1 - rate_left) / (1 - rate_right))
    report(
        "C2",
        "peak pitch difference between the ears (Woodworth delays), cents",
        np.max(np.abs(interaural_cents)),
    )


def claim_retarded_time():
    """C3: the propagation stage is exact up to the fractional-delay read."""
    position = pass_by(15.0)
    delay_of = lambda t: np.linalg.norm(position(t), axis=-1) / C_SOUND  # noqa: E731
    t = np.arange(int(3.0 * FS)) / FS
    exact = emission_time(t, delay_of, iterations=60)
    guess = t - delay_of(t)
    for iterations in range(0, 8):
        approx = guess.copy()
        for _ in range(iterations):
            approx = t - delay_of(approx)
        if np.max(np.abs(approx - exact)) < 1e-12:
            report("C3", "pass-by at 15 m/s: fixed-point iterations to reach 1e-12 s", iterations)
            break
    rate = np.max(np.abs(np.gradient(delay_of(exact), exact)))
    report("C3", "contraction factor, max |d(r/c)/dt|", rate)

    # source signal on its own clock, long enough to read from at any t_e
    pad = 64
    n_src = np.arange(-pad, int(3.0 * FS) + pad)
    inside = slice(int(0.2 * FS), int(2.8 * FS))  # away from the ends of the source signal
    t_e = exact[inside]
    gain = 1 / np.linalg.norm(position(t_e), axis=-1)
    for f in (500.0, 2000.0, 8000.0):
        source = np.sin(2 * np.pi * f * n_src / FS)
        reference = gain * np.sin(2 * np.pi * f * t_e)
        read_at = t_e * FS + pad
        for name, read in [
            ("linear", read_linear),
            ("cubic Lagrange", read_lagrange3),
            ("32-tap windowed sinc", read_sinc),
        ]:
            report(
                "C3",
                f"{f:g} Hz tone, {name} read: error re signal, dB",
                error_db(gain * read(source, read_at), reference),
            )
    ratio = np.gradient(exact, t)  # received frequency / emitted frequency
    report("C3", "received/emitted frequency, approaching, cents", 1200 * np.log2(ratio.max()))
    report("C3", "received/emitted frequency, receding, cents", 1200 * np.log2(ratio.min()))
    started = time.perf_counter()
    for _ear in range(2):
        read_sinc(np.sin(2 * np.pi * 1000 * n_src / FS), exact * FS + pad)
    report("C3", "time for the 32-tap read of 3 s, two ears, s", time.perf_counter() - started)


def claim_switching_fails_for_distance():
    """C4: switching between fixed delays at 200 points/s cannot follow a
    changing distance: neighboring windows hold copies of the sound at
    different delays, which comb-filter."""
    position = pass_by(15.0)
    duration = 3.0
    n = int(duration * FS)
    t = np.arange(n) / FS
    delay_of = lambda tt: np.linalg.norm(position(tt), axis=-1) / C_SOUND  # noqa: E731
    exact = emission_time(t, delay_of)
    gain = 1 / np.linalg.norm(position(exact), axis=-1)
    n_points = int(200 * duration)
    knot_times = np.linspace(0, duration - 1 / FS, n_points)
    taps = 3300  # longer than the largest delay plus the kernel's half width
    filters = [delay_filter(delay_of(tk) * FS, 1 / np.linalg.norm(position(tk)), taps) for tk in knot_times]
    inside = slice(int(0.3 * FS), int(2.7 * FS))
    hop_delay_change = np.max(np.abs(np.diff([delay_of(tk) for tk in knot_times])))
    report(
        "C4", "pass-by at 15 m/s, 200 points/s: largest delay step between points, us", 1e6 * hop_delay_change
    )
    report("C4", "comb-filter first notch for that step, 1/(2 step), Hz", 1 / (2 * hop_delay_change))
    for f in (500.0, 2000.0, 8000.0):
        source = np.sin(2 * np.pi * f * t)
        out = switched(source, n_points, lambda k: filters[k][None, :], n + taps)[0, :n]
        reference = gain * np.sin(2 * np.pi * f * exact)
        report(
            "C4",
            f"{f:g} Hz tone, current switching: error re signal, dB",
            error_db(out[inside], reference[inside]),
        )


def claim_swing_switching():
    """C5: for the azimuth swing the current switching is already accurate,
    measured on Woodworth delays (no level differences, so only timing counts)."""
    _, azimuth = azimuth_swing()
    duration = 3.0
    n = int(duration * FS)
    t = np.arange(n) / FS
    offset = 0.002  # s, a common delay so every ear delay is positive

    def ear_delay(az, ear):
        return offset - ear * HEAD_RADIUS / C_SOUND * (az + np.sin(az)) / 2

    n_points = int(200 * duration)
    knot_times = np.linspace(0, duration - 1 / FS, n_points)
    taps = 512
    filters = np.array(
        [[delay_filter(ear_delay(azimuth(tk), ear) * FS, 1.0, taps) for ear in (-1, +1)] for tk in knot_times]
    )
    inside = slice(int(0.2 * FS), int(2.8 * FS))
    for f in (500.0, 4000.0):
        source = np.sin(2 * np.pi * f * t)
        out = switched(source, n_points, lambda k: filters[k], n + taps)[:, :n]
        errors = []
        for channel, ear in enumerate((-1, +1)):
            exact = emission_time(t, lambda tt, e=ear: ear_delay(azimuth(tt), e))
            reference = np.sin(2 * np.pi * f * exact)
            errors.append(error_db(out[channel, inside], reference[inside]))
        report("C5", f"30 deg swing, {f:g} Hz tone, current switching: worse ear's error, dB", max(errors))


def claim_batched():
    """C6: the batched frequency-domain switching gives the same output as the
    loop, and how much faster it is."""
    rng = np.random.default_rng(0)
    duration = 3.0
    signal = rng.standard_normal(int(duration * FS))
    n_points = int(200 * duration)
    taps = 750  # a 1024-tap PKU-IOA HRIR resampled from 65536 to 48000 Hz
    filters = rng.standard_normal((n_points, 2, taps)) * np.exp(-np.arange(taps) / 60)
    n_out = len(signal) + taps - 1
    started = time.perf_counter()
    loop = switched(signal, n_points, lambda k: filters[k], n_out)
    loop_time = time.perf_counter() - started
    started = time.perf_counter()
    batched = switched_batched(signal, n_points, filters, n_out)
    batched_time = time.perf_counter() - started
    report(
        "C6",
        "3 s, 200 points/s, 750 taps: batched vs loop, max difference re peak",
        np.max(np.abs(batched - loop)) / np.max(np.abs(loop)),
    )
    report("C6", "loop time, s", loop_time)
    report("C6", "batched time, s", batched_time)


def pku_shells(directory):
    """The PKU-IOA responses per distance [m] -> (irs (M, 2, n), fs), from the
    original .dat files (azi{A}_elev{E}_dist{D}.dat, float64, left then
    right, 65536 Hz) anywhere below `directory`, else from the SOFA copy
    (dist_*m.sofa, needs h5py). Empty files are skipped."""
    shells = {}
    pattern = re.compile(r"azi(-?\d+)_elev(-?\d+)_dist(\d+)\.dat$")
    for path in sorted(Path(directory).rglob("*.dat")):
        match = pattern.search(path.name)
        if not match or path.stat().st_size == 0:
            continue
        shells.setdefault(int(match.group(3)) / 100, []).append(np.fromfile(path, "<f8").reshape(2, -1))
    if shells:
        return {distance: (np.array(irs), 65536.0) for distance, irs in sorted(shells.items())}
    import h5py

    for path in sorted(Path(directory).rglob("dist_*m.sofa")):
        with h5py.File(path, "r") as sofa:
            fs = float(np.ravel(sofa["Data.SamplingRate"][()])[0])
            distance = float(np.median(sofa["SourcePosition"][()][:, 2]))
            shells[distance] = (sofa["Data.IR"][()], fs)
    return dict(sorted(shells.items()))


def claim_pku_shells(directory):
    """C7 (needs the PKU-IOA files): does each distance shell carry the 1/r
    level and the r/c delay, or were they removed? Mean over directions of
    the two ears' energy and of the onset (first sample within 20 dB of the
    peak, as HRIRSet does)."""
    rows = []
    for distance, (irs, fs) in pku_shells(directory).items():
        energy = np.mean(np.sum(irs**2, axis=-1))
        envelope = np.abs(irs)
        onsets = np.argmax(envelope >= envelope.max(axis=-1, keepdims=True) * 10 ** (-20 / 20), axis=-1)
        rows.append((distance, db(energy), np.mean(onsets) / fs, len(irs)))
    if not rows:
        print("C7   no PKU-IOA .dat or SOFA files found")
        return
    ref_distance, ref_level, ref_onset, _ = min(rows, key=lambda row: abs(row[0] - 1.0))
    for distance, level, onset, n_positions in rows:
        level_predicted = -20 * np.log10(distance / ref_distance)
        onset_predicted = 1e3 * (distance - ref_distance) / C_SOUND
        label = f"{distance:g} m re {ref_distance:g} m"
        report(
            "C7",
            f"{label} ({n_positions} positions): mean level, dB (1/r predicts {level_predicted:+.1f})",
            level - ref_level,
        )
        onset_ms = 1e3 * (onset - ref_onset)
        report("C7", f"{label}: mean onset, ms (r/c predicts {onset_predicted:+.2f})", onset_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pku", help="folder with the PKU-IOA .dat or SOFA files, for C7")
    args = parser.parse_args()
    claim_doppler_sizes()
    claim_swing_doppler()
    claim_retarded_time()
    claim_switching_fails_for_distance()
    claim_swing_switching()
    claim_batched()
    if args.pku:
        claim_pku_shells(args.pku)
    else:
        print("C7   skipped: pass --pku DIR with the PKU-IOA SOFA files")
