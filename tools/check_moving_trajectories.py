"""Numerical checks for the paths on the gallery's Moving talkers page: six straight
lines through the horizontal plane, and one path no source could take.

Unlike tools/check_moving_sound_claims.py, which is independent of sonore, this one
runs `so.move_sound` itself: it measures what the renderer does with these paths,
including the fast one, which is outside the speeds the renderer was designed for
(docs/design/moving-sound.md, "Out of scope"). Each line prints what was measured.

    python tools/check_moving_trajectories.py --pku DIR

DIR holds the PKU-IOA HRIRs: the original .dat files (any layout). Without --pku it
uses so.load_hrirs(distances="all"), which downloads them. It takes about two minutes.

Coordinates are sonore's head-centered Cartesian ones, in meters: x right, y front.
The paths and the buzz are the same as on the page (docs/gallery/spatial/moving.py).
"""

import argparse
import warnings

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

import sonore as so
import sonore.spatial.spatialization as spatialization

FS = 16000  # the gallery's sampling rate
SPEED = 15.0  # m/s
DURATION = 2.5  # s
BUZZ_F0 = 120.0  # Hz
F_MAX = 5000.0  # Hz, room for Doppler below 8 kHz
HARMONICS = np.arange(1, 42)  # what f_max leaves: up to 4920 Hz, fading from 4500 Hz
HOLD, MOVE = 0.10, 0.05  # s, the fast path: still, then a jump
GRAVITY = 9.81  # m/s^2
HEAD_RADIUS = 0.0875  # m, the ears on the x axis


def report(text, value, unit=""):
    print(f"{text:<78s} {value:>9.4g} {unit}")


def buzz(fs, harmonics=None):
    """The page's buzz (harmonics below F_MAX, faded at the top), or exactly ``harmonics``."""
    if harmonics is None:
        return so.harmonic_complex(DURATION, fs, BUZZ_F0, f_max=F_MAX).normalize()
    return so.harmonic_complex(DURATION, fs, BUZZ_F0, harmonics=harmonics).normalize()


emitted = (round(DURATION * FS) - 1) / FS  # the time of the buzz's last sample [s]


def straight(point, heading, at=None):
    """At ``point`` (x, y) [m] at time ``at`` (default the middle), moving along ``heading``."""
    point = np.array([*point, 0.0])
    velocity = SPEED * np.array([*heading, 0.0]) / np.hypot(*heading)
    at = emitted / 2 if at is None else at
    return lambda t: point + np.outer(np.atleast_1d(t) - at, velocity)


toward = np.array([np.sin(np.radians(30)), np.cos(np.radians(30))])  # 30 degrees right
STRAIGHT = {
    "far in front, 10 m": straight((0, 10), (1, 0)),
    "close in front, 50 cm": straight((0, 0.5), (1, 0)),
    "down the right side, 1 m": straight((1, 0), (0, -1)),
    "behind, 2 m": straight((0, -2), (-1, 0)),
    "crossing at an angle, 1 m": straight((np.sqrt(0.5), np.sqrt(0.5)), (1, -1)),
    "straight at the listener, to 40 cm": straight(tuple(0.4 * toward), tuple(-toward), at=emitted),
}


def jumping_path(seed=7):
    """Still for HOLD, then a jump of 60-150 degrees and to a new distance (40 cm to 4 m)
    in MOVE, along a quintic smoothstep in azimuth and log distance, so position, velocity
    and acceleration are all continuous."""
    rng = np.random.default_rng(seed)
    n_jumps = int(np.ceil(DURATION / (HOLD + MOVE))) + 1
    azimuths, distances = [0.0], [1.0]
    for _ in range(n_jumps):
        azimuths.append(azimuths[-1] + rng.choice([-1, 1]) * rng.uniform(60, 150))
        distances.append(np.exp(rng.uniform(np.log(0.4), np.log(4.0))))
    azimuths, distances = np.array(azimuths), np.array(distances)

    def where(t):
        t = np.atleast_1d(np.asarray(t, float))
        jump = np.clip(np.floor(t / (HOLD + MOVE)).astype(int), 0, n_jumps - 1)
        progress = np.clip((t - jump * (HOLD + MOVE) - HOLD) / MOVE, 0, 1)
        weight = progress**3 * (10 - 15 * progress + 6 * progress**2)
        distance = np.exp(np.log(distances[jump]) * (1 - weight) + np.log(distances[jump + 1]) * weight)
        return distance, azimuths[jump] * (1 - weight) + azimuths[jump + 1] * weight

    return where, so.hcc_trajectory(lambda t: 100 * where(t)[0], 0.0, lambda t: where(t)[1])


def doppler_cents(path, t):
    distance = np.linalg.norm(path(t), axis=1)
    return distance, -1200 * np.log2(1 + np.gradient(distance, t) / so.SPEED_OF_SOUND)


def pitch_error(rendered, path, f_lo, f_hi):
    """Median and 95th percentile of |measured - Doppler| [cents] over both ears, away
    from the first and last 0.1 s."""
    t = np.linspace(0, emitted, 5001)
    distance, cents = doppler_cents(path, t)
    arrival = t + distance / so.SPEED_OF_SOUND
    errors, voiced_fraction = [], []
    for channel in (rendered.left, rendered.right):
        track = so.f0_track(channel, f_lo=f_lo, f_hi=f_hi)
        f0 = track.f0[0]
        voiced = (f0 > 0) & (track.t > 0.1) & (track.t < arrival[-1] - 0.1)
        voiced_fraction.append(voiced.sum() / np.sum((track.t > 0.1) & (track.t < arrival[-1] - 0.1)))
        measured = 1200 * np.log2(f0[voiced] / BUZZ_F0)
        errors.append(np.abs(measured - np.interp(track.t[voiced], arrival, cents)))
    errors = np.concatenate(errors)
    return np.median(errors), np.percentile(errors, 95), min(voiced_fraction)


def difference_db(estimate, reference, window=None):
    """Difference energy re reference energy [dB], both ears together; with ``window``
    [samples], the worst and the median over windows where the reference is not silent."""
    n = min(len(estimate), len(reference))
    difference, reference = estimate[:n] - reference[:n], reference[:n]
    if window is None:
        return 10 * np.log10(np.sum(difference**2) / np.sum(reference**2))
    count = n // window
    error = np.sum(difference[: count * window].reshape(count, window, -1) ** 2, axis=(1, 2))
    power = np.sum(reference[: count * window].reshape(count, window, -1) ** 2, axis=(1, 2))
    sounding = power > 1e-3 * power.max()
    ratio = 10 * np.log10(error[sounding] / power[sounding] + 1e-30)
    return ratio.max(), np.median(ratio)


def main(hrirs):
    print("Straight paths at 15 m/s, a 120 Hz buzz with f_max 5 kHz (41 harmonics, to 4920 Hz)")
    source = buzz(FS)
    for name, path in STRAIGHT.items():
        t = np.linspace(0, emitted, 5001)
        distance, cents = doppler_cents(path, t)
        rendered = so.move_sound(source, path, hrirs)
        median, worst, voiced = pitch_error(rendered, path, 80, 200)
        print(f"  {name}")
        report("    distance, nearest [m]", distance.min())
        report("    distance, farthest [m]", distance.max())
        report(
            "    level change, 1/r from farthest to nearest [dB]",
            20 * np.log10(distance.max() / distance.min()),
        )
        report("    Doppler shift, highest [cents]", cents.max())
        report("    Doppler shift, lowest [cents]", cents.min())
        report("    pitch measured by f0_track vs Doppler, median |error| [cents]", median)
        report("    pitch measured by f0_track vs Doppler, 95th percentile [cents]", worst)
        report("    fraction of time windows f0_track calls voiced (worse ear)", voiced)
        # time over which the shift goes from +90% to -90% of its range: closest / speed scale
        span = cents.max() - cents.min()
        if span > 1:
            arrival = t + distance / so.SPEED_OF_SOUND
            inside = (cents < cents.max() - 0.05 * span) & (cents > cents.min() + 0.05 * span)
            report("    time the glide takes, 95% to 5% of its range, as heard [s]", np.ptp(arrival[inside]))

    print("\nThe jumping path: still 0.1 s, then a jump in 0.05 s")
    where, path = jumping_path()
    t = np.linspace(0, DURATION, 250001)
    position = path(t)
    velocity = np.gradient(position, t, axis=0)
    acceleration = np.gradient(velocity, t, axis=0)
    distance, cents = doppler_cents(path, t)
    report("  speed, highest [m/s]", np.linalg.norm(velocity, axis=1).max())
    report(
        "  speed, highest, re the speed of sound", np.linalg.norm(velocity, axis=1).max() / so.SPEED_OF_SOUND
    )
    report("  speed toward or away from the head, highest [m/s]", np.abs(np.gradient(distance, t)).max())
    report("  acceleration, highest [g]", np.linalg.norm(acceleration, axis=1).max() / GRAVITY)
    starts = np.arange(HOLD, DURATION, HOLD + MOVE)
    ends = starts + MOVE
    jump_m = np.linalg.norm(path(ends) - path(starts), axis=1)
    jump_deg = np.abs(where(ends)[1] - where(starts)[1])
    report("  longest jump, straight line [m]", jump_m.max())
    report("  largest jump in azimuth [deg]", jump_deg.max())
    report("  number of jumps", len(starts))
    angular = np.abs(np.gradient(where(t)[1], t))
    report("  fastest turn, azimuth per 5 ms [deg]", angular.max() * 5e-3)
    report("  distance, nearest [m]", distance.min())
    report("  distance, farthest [m]", distance.max())
    report("  Doppler shift, highest [cents]", cents.max())
    report("  Doppler shift, lowest [cents]", cents.min())
    highest_harmonic = BUZZ_F0 * HARMONICS[-1] * 2 ** (cents.max() / 1200)
    report("  highest harmonic as heard, at the largest upward shift [Hz]", highest_harmonic)

    # The page uses hop 0.5 ms; the default is 5 ms; 0.0625 ms stands in for the limit.
    rendered = {hop: so.move_sound(source, path, hrirs, hop=hop) for hop in (5e-3, 0.5e-3, 0.0625e-3)}
    median, worst, voiced = pitch_error(rendered[0.5e-3], path, 60, 250)
    report("  pitch measured vs Doppler (hop 0.5 ms), median |error| [cents]", median)
    report("  pitch measured vs Doppler (hop 0.5 ms), 95th percentile [cents]", worst)
    report("  fraction of time windows f0_track calls voiced (worse ear)", voiced)
    window = round(0.02 * FS)
    finest = rendered[0.0625e-3].data
    for hop in (5e-3, 0.5e-3):
        name = f"HRIR shapes switched every {1e3 * hop:g} ms vs every 0.0625 ms"
        report(f"  {name}, whole sound [dB]", difference_db(rendered[hop].data, finest))
        worst, median = difference_db(rendered[hop].data, finest, window)
        report("    worst 20 ms window [dB]", worst)
        report("    median 20 ms window [dB]", median)

    default_step = spatialization._DELAY_STEP
    spatialization._DELAY_STEP = default_step / 8
    finer = so.move_sound(source, path, hrirs, hop=0.5e-3)
    spatialization._DELAY_STEP = default_step
    report(
        "  ear delays looked up every 1 ms vs every 0.125 ms, whole sound [dB]",
        difference_db(rendered[0.5e-3].data, finer.data),
    )

    # f0_track loses the pitch during the jumps, so check the shift there on a 1 kHz tone:
    # its instantaneous frequency at each ear (from the analytic signal, smoothed below
    # 200 Hz) against the shift computed from the straight-line distance to that ear.
    tone = so.Sound(np.cos(2 * np.pi * 1000 * np.arange(round(DURATION * FS)) / FS), FS)
    tone_rendered = so.move_sound(tone, path, hrirs, hop=0.5e-3)
    smoothing = butter(4, 200, fs=FS, output="sos")
    errors = []
    for ear, ear_x in enumerate((-HEAD_RADIUS, HEAD_RADIUS)):
        to_ear = np.linalg.norm(position - [ear_x, 0, 0], axis=1)
        shift = -1200 * np.log2(1 + np.gradient(to_ear, t) / so.SPEED_OF_SOUND)
        arrival = t + to_ear / so.SPEED_OF_SOUND
        phase = np.unwrap(np.angle(hilbert(tone_rendered.data[:, ear])))
        frequency = sosfiltfilt(smoothing, np.diff(phase) * FS / (2 * np.pi))
        times = (np.arange(len(frequency)) + 0.5) / FS
        expected = np.interp(times, arrival, shift)
        jumping_now = (np.abs(expected) > 50) & (times > 0.1) & (times < arrival[-1] - 0.1)
        measured = 1200 * np.log2(np.clip(frequency[jumping_now], 1, None) / 1000)
        errors.append(np.abs(measured - expected[jumping_now]))
    errors = np.concatenate(errors)
    report(
        "  1 kHz tone in the jumps, shift at each ear vs path, median |error| [cents]",
        np.median(errors),
    )
    report("  1 kHz tone in the jumps, 95th percentile [cents]", np.percentile(errors, 95))

    print("\nAliasing, measured on the path straight at the listener, where the shift is constant")
    path = STRAIGHT["straight at the listener, to 40 cm"]
    factor = 2 ** (doppler_cents(path, np.array([0.5, 1.0]))[1][0] / 1200)
    to_nyquist = np.arange(1, int(np.ceil(FS / 2 / BUZZ_F0)))  # what harmonic_complex makes by default
    for label, harmonics in (
        ("below 5 kHz, as on the page", None),
        ("every harmonic below 8 kHz", to_nyquist),
    ):
        rendered = so.move_sound(buzz(FS, harmonics), path, hrirs)
        middle = rendered.data[round(0.5 * FS) : round(2.0 * FS), 1]  # right ear, 1.5 s
        spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle)))) ** 2
        frequencies = np.fft.rfftfreq(len(middle), 1 / FS)
        heard = BUZZ_F0 * (HARMONICS if harmonics is None else harmonics) * factor
        folded = FS - heard[heard > FS / 2]

        def energy_near(targets, frequencies=frequencies, spectrum=spectrum):
            near = np.zeros(len(frequencies), bool)
            for target in targets:
                near |= np.abs(frequencies - target) < 15
            return np.sum(spectrum[near])

        print(f"  {label}")
        report("    harmonics shifted above 8 kHz", len(folded))
        if len(folded):
            report(
                "    energy folded back below 8 kHz, re the harmonics' [dB]",
                10 * np.log10(energy_near(folded) / energy_near(heard[heard < FS / 2])),
            )
            report("    lowest folded frequency [Hz]", folded.min())

    print("\nFast sources, coming straight at the head from the front, ending 1 m away")
    for speed in (100.0, 200.0, 250.0, 300.0, 400.0):

        def fast(t, speed=speed):
            return np.outer(1 + speed * (emitted - np.atleast_1d(t)), [0.0, 1.0, 0.0])

        try:
            so.move_sound(source, fast, hrirs)
            print(f"  {speed:.0f} m/s: rendered")
        except ValueError as error:
            print(f"  {speed:.0f} m/s: refused ({error})")

    # Why 300 m/s is refused: the ear delays come from the measured HRIR onsets, which
    # change with distance a little faster than the travel time does.
    def at_300(t):
        return np.outer(1 + 300.0 * (emitted - np.atleast_1d(t)), [0.0, 1.0, 0.0])

    lookup = np.linspace(0, emitted, 2501)
    measured_points, radii = hrirs._nearest_measured(at_300(lookup))
    measured_radii = np.linalg.norm(measured_points, axis=1)
    ear_delay = hrirs._onset_times(measured_points)[:, 0] + (radii - measured_radii) / so.SPEED_OF_SOUND
    rate = np.diff(ear_delay) / np.diff(lookup)
    steepest = np.argmin(rate)
    report("  at 300 m/s, travel time changes by [s per s]", -300.0 / so.SPEED_OF_SOUND)
    report("  at 300 m/s, the left ear's delay changes by at most [s per s]", rate[steepest])
    report("    there, the distance [m]", radii[steepest])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pku", help="directory with the PKU-IOA .dat files")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # one empty .dat file in the distribution
        hrirs = so.HRIRSet.from_pku_ioa(args.pku) if args.pku else so.load_hrirs(distances="all")
    main(hrirs)
