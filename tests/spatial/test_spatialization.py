"""HRIR sets and spatialization."""

import numpy as np
import pytest
from helpers import toy_hrirs
from scipy.interpolate import CubicSpline

import sonore as so
from sonore.spatial.spatialization import _emission_times


def error_db(estimate, reference):
    return 10 * np.log10(np.sum((estimate - reference) ** 2) / np.sum(reference**2))


class TestSpatialization:
    def test_coordinate_roundtrip(self):
        h = (150.0, 20.0, 250.0)
        np.testing.assert_allclose(so.rect_to_hcc(*so.hcc_to_rect(*h)), h)
        x, y, z = so.hcc_to_rect(100, 0, 90)
        np.testing.assert_allclose([x, y, z], [1, 0, 0], atol=1e-12)  # 90 deg = right

    def test_linear_trajectory_endpoints_and_even_steps(self):
        path = so.linear_trajectory((0, 0, 0), (3, -6, 9), 4)
        np.testing.assert_allclose(path, [[0, 0, 0], [1, -2, 3], [2, -4, 6], [3, -6, 9]], atol=1e-15)

    def test_distance_gain(self):
        assert so.distance_gain_db(2.0) == pytest.approx(-6.02, abs=0.01)

    def test_interpolation_at_measured_point_is_exact(self):
        hs, _ = toy_hrirs()
        h = hs.at(hs.positions[5], fs=hs.fs)[0]
        n = hs.irs.shape[-1]
        np.testing.assert_allclose(h[:, :n], hs.irs[5], atol=1e-6)

    def test_no_interpolation_below_the_lowest_ring(self):
        # the toy set's lowest ring is at -40 degrees; the hull closes the hole under it
        # with triangles whose corners are far apart in azimuth
        hs, _ = toy_hrirs()
        for elev, azim in ((-50, 0), (-60, 90), (-90, 0)):
            with pytest.raises(ValueError, match="gap in the measured directions"):
                hs.at(np.column_stack(so.hcc_to_rect(100, elev, azim)))
        with pytest.raises(ValueError, match="gap in the measured directions"):
            so.move_sound(so.gaussian_noise(0.1, hs.fs, rng=1), so.hcc_trajectory(100, -60, 0), hs)
        # on and above the lowest ring, including its measured directions, it interpolates
        on_ring = np.column_stack(so.hcc_to_rect(100, [-40, -40, -30, 85], [0, 15, 200, 10]))
        assert np.all(np.isfinite(hs.at(on_ring)))
        n = hs.irs.shape[-1]
        np.testing.assert_allclose(hs.at(hs.positions)[..., :n], hs.irs, atol=1e-6)

    def test_moving_sound_itd_sweeps(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(2, 48000, band=(200, 3000), rng=0)
        traj = so.circular_trajectory((100, 0, 90), (100, 0, -90), 200)
        y = so.move_sound(x, traj, hs)
        c = so.interaural_cues(y, win_dur=50e-3)
        itd = c.itd[np.isfinite(c.itd)]
        assert itd[:3].mean() > 400e-6  # starts right
        assert itd[-3:].mean() < -400e-6  # ends left
        assert abs(np.median(itd[len(itd) // 2 - 2 : len(itd) // 2 + 2])) < 100e-6

    def test_move_sound_still_matches_spatialize(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(1, 48000, rng=0)
        still = so.spatialize(x, so.hcc_to_rect(100, 0, 40), hs)
        moving = so.move_sound(x, np.repeat([so.hcc_to_rect(100, 0, 40)], 50, axis=0), hs)
        n = min(len(still), len(moving))
        assert error_db(moving.data[:n], still.data[:n]) < -90
        assert np.abs(moving.data[n:]).max(initial=0) < 1e-9

    def test_trajectory_forms_agree(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(0.5, 48000, rng=1)
        duration = (len(x) - 1) / x.fs
        start, end = np.array(so.hcc_to_rect(100, 0, 20)), np.array(so.hcc_to_rect(100, 0, -20))
        from_array = so.move_sound(x, so.linear_trajectory(start, end, 2), hs)
        from_pair = so.move_sound(x, (np.array([0, duration]), np.array([start, end])), hs)
        from_function = so.move_sound(x, lambda t: start + np.outer(t / duration, end - start), hs)
        np.testing.assert_allclose(from_pair.data, from_array.data, atol=1e-12)
        np.testing.assert_allclose(from_function.data, from_array.data, atol=1e-12)

    def test_hcc_trajectory(self):
        swing = so.hcc_trajectory(100, 0, lambda t: 30 * np.sin(2 * np.pi * 2 * t))
        points = swing(np.array([0.0, 0.125, 0.375]))
        np.testing.assert_allclose(so.rect_to_hcc(*points.T)[0], 100)
        np.testing.assert_allclose(
            np.mod(so.rect_to_hcc(*points.T)[2] + 180, 360) - 180, [0, 30, -30], atol=1e-9
        )
        approach = so.hcc_trajectory(([0, 2], [300, 50]), 0, 20)
        np.testing.assert_allclose(
            np.linalg.norm(approach(np.array([0, 1, 2, 5])), axis=1), [3, 1.75, 0.5, 0.5]
        )

    def test_pass_by_matches_closed_form(self):
        """Through HRIRs that are pure impulses, a tone passing by at 15 m/s is
        the tone read at the emission time, over the distance: Doppler included."""
        toy, _ = toy_hrirs()
        fs = toy.fs
        onset = 140  # samples, as if measured 1 m away
        irs = np.zeros_like(toy.irs)
        irs[:, :, onset] = 1.0
        hs = so.HRIRSet(irs, toy.positions, fs)
        speed, closest, f = 15.0, 2.0, 1000.0
        duration = 2.0
        x = so.Sound(np.sin(2 * np.pi * f * np.arange(int(duration * fs)) / fs), fs)
        emitted = (len(x) - 1) / fs

        def path(t):
            return np.column_stack([speed * (t - emitted / 2), np.full_like(t, closest), np.zeros_like(t)])

        y = so.move_sound(x, path, hs)
        t = np.arange(len(y)) / fs
        emission = t - onset / fs
        for _ in range(60):
            emission = (
                t - onset / fs - (np.linalg.norm(path(np.clip(emission, 0, emitted)), axis=1) - 1) / 343.0
            )
        expected = np.sin(2 * np.pi * f * emission) / np.linalg.norm(
            path(np.clip(emission, 0, emitted)), axis=1
        )
        middle = slice(int(0.3 * fs), int(1.7 * fs))
        assert error_db(y.data[middle, 0], expected[middle]) < -80
        assert error_db(y.data[middle, 1], expected[middle]) < -80

    def test_fast_sources(self):
        """Through impulse HRIRs, a tone coming straight at the head at 250 m/s
        rises by 1 / (1 - 250/343) as the closed form says; receding at 400 m/s
        renders; coming toward the head at 400 m/s is refused."""
        toy, _ = toy_hrirs()
        fs = toy.fs
        irs = np.zeros_like(toy.irs)
        irs[:, :, 140] = 1.0
        hs = so.HRIRSet(irs, toy.positions, fs)
        f = 200.0
        x = so.Sound(np.sin(2 * np.pi * f * np.arange(int(0.5 * fs)) / fs), fs)
        emitted = (len(x) - 1) / fs

        def straight_at(speed):  # ends 2 m in front
            return lambda t: np.outer(2 + speed * (emitted - np.atleast_1d(t)), [0.0, 1.0, 0.0])

        y = so.move_sound(x, straight_at(250.0), hs)
        arriving = np.flatnonzero(np.abs(y.data[:, 0]) > 1e-3)
        middle = y.data[arriving[0] + 200 : arriving[-1] - 200, 0]
        crossings = np.flatnonzero(np.diff(np.signbit(middle)))
        heard = (len(crossings) - 1) / 2 / ((crossings[-1] - crossings[0]) / fs)
        assert heard == pytest.approx(f / (1 - 250.0 / 343.0), rel=1e-3)
        receding = so.move_sound(x, lambda t: np.outer(2 + 400.0 * np.atleast_1d(t), [0.0, 1.0, 0.0]), hs)
        assert receding.duration > x.duration + 400.0 * emitted / 343.0
        with pytest.raises(ValueError, match="faster than"):
            so.move_sound(x, straight_at(400.0), hs)

    def test_beyond_the_measured_distance(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(
            0.2, 48000, band=(20, 16000), rng=2
        )  # the delay read is exact below ~0.8 Nyquist
        near = so.spatialize(x, so.hcc_to_rect(200, 0, 30), hs)
        far = so.spatialize(x, so.hcc_to_rect(400, 0, 30), hs)
        assert 10 * np.log10(np.sum(far.data**2) / np.sum(near.data**2)) == pytest.approx(-6.02, abs=0.05)
        lag = np.argmax(np.correlate(far.data[:, 0], near.data[:, 0], "full")) - (len(near) - 1)
        assert lag == pytest.approx(2.0 / 343.0 * 48000, abs=1)
        moving = so.move_sound(x, np.repeat([so.hcc_to_rect(400, 0, 30)], 10, axis=0), hs)
        middle = slice(1000, len(x) - 1000)  # away from the sound's abrupt start and end
        assert error_db(moving.data[middle], far.data[middle]) < -80

    def test_room_level_stays_while_direct_falls(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(0.5, 48000, rng=3)
        tail = so.synth_ir(0.3, 48000, n_channels=2, rng=4)
        parts = {}
        for distance in (100, 200):
            point = np.repeat([so.hcc_to_rect(distance, 0, 0)], 4, axis=0)
            dry = so.move_sound(x, point, hs)
            wet = so.move_sound(x, point, hs, room=tail, drr_db=5.0)
            parts[distance] = (dry.data, wet.data[: len(dry)] - dry.data, wet.data[len(dry) :])
        direct_change = 10 * np.log10(np.sum(parts[200][0] ** 2) / np.sum(parts[100][0] ** 2))
        reverberant = {d: np.sum(p[1] ** 2) + np.sum(p[2] ** 2) for d, p in parts.items()}
        assert direct_change == pytest.approx(-6.02, abs=0.1)
        assert 10 * np.log10(reverberant[200] / reverberant[100]) == pytest.approx(0, abs=0.2)
        expected = np.sum(x.data**2) * hs._diffuse_field(48000)[0] * 10 ** (-5.0 / 10) * 2
        assert 10 * np.log10(reverberant[100] / expected) == pytest.approx(0, abs=0.5)


class TestHRIRFiles:
    def test_concat_merges_distances(self):
        hs, _ = toy_hrirs()
        far = so.HRIRSet(hs.irs * 0.5, hs.positions * 2, hs.fs)
        both = so.HRIRSet.concat([hs, far])
        assert len(both.positions) == 2 * len(hs.positions)
        # halfway between the shells, the IR is between the two
        p = hs.positions[5] * 1.5
        assert not both._single_distance
        h = both.at(p)[0]
        assert 0.5 * np.abs(hs.irs[5]).max() < np.abs(h).max() < np.abs(hs.irs[5]).max()

    def test_onsets_fitted_across_distances(self):
        # five distances whose onsets hold the travel time, but one arrives 5 samples early
        hs, _ = toy_hrirs(taps=512)
        distances = [0.5, 0.75, 1.0, 1.3, 1.6]
        early = {1.3: 5}
        sets = []
        for distance in distances:
            shift = int(round(distance / so.SPEED_OF_SOUND * hs.fs)) - early.get(distance, 0)
            sets.append(so.HRIRSet(np.roll(hs.irs, shift, axis=-1), hs.positions * distance, hs.fs))
        fitted = so.HRIRSet.concat(sets)
        measured = so.HRIRSet.concat(sets, fit_onsets=False)
        radii = np.linalg.norm(fitted.positions, axis=1)
        beyond_travel = fitted._delay_onsets.mean(axis=1) - radii / so.SPEED_OF_SOUND * hs.fs
        # in each direction, every distance, the early one included, now lines up with the travel time
        by_direction = beyond_travel.reshape(len(distances), -1)
        assert np.ptp(by_direction, axis=0).max() < 1.0
        shift = fitted._delay_onsets - measured._onsets
        np.testing.assert_allclose(shift[:, 0], shift[:, 1], atol=1e-9)  # ITDs as measured
        assert np.median(shift[np.isclose(radii, 1.3), 0]) == pytest.approx(5, abs=1)

    def test_concat_rejects_mixed_rates(self):
        hs, _ = toy_hrirs()
        other, _ = toy_hrirs(fs=44100)
        with pytest.raises(ValueError, match="sampling rate"):
            so.HRIRSet.concat([hs, other])

    def test_pku_ioa_dat_in_subfolders(self, tmp_path):
        rng = np.random.default_rng(0)
        for d in (20, 50):
            for e in (-40, 0, 40, 90):
                folder = tmp_path / f"dist{d}" / f"elev{e}"
                folder.mkdir(parents=True)
                for a in (0,) if e == 90 else range(0, 360, 90):
                    rng.standard_normal(2 * 1024).tofile(folder / f"azi{a}_elev{e}_dist{d}.dat")
        hs = so.HRIRSet.from_pku_ioa(tmp_path)
        assert hs.irs.shape == (2 * 13, 2, 1024)
        assert sorted(np.unique(np.round(np.linalg.norm(hs.positions, axis=1), 6))) == [0.2, 0.5]


def impulse_hrirs(onset=(140, 140), gain=None, fs=48000):
    toy, _ = toy_hrirs(fs=fs)
    irs = np.zeros_like(toy.irs)
    for ear in (0, 1):
        irs[:, ear, onset[ear]] = 1.0 if gain is None else gain(toy.positions, ear)
    return so.HRIRSet(irs, toy.positions, fs)


def test_trajectory_refusals():
    hs, _ = toy_hrirs()
    x = so.gaussian_noise(0.05, hs.fs, rng=0)
    p = np.array(so.hcc_to_rect(100, 0, 20))
    with pytest.raises(ValueError, match="times must increase"):
        so.move_sound(x, (np.array([0.0, 0.0]), np.array([p, p])), hs)
    with pytest.raises(ValueError, match=r"\(N,\) times and \(N, 3\) points"):
        so.move_sound(x, (np.array([0.0, 0.01, 0.02]), np.array([p, p])), hs)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        so.move_sound(x, np.ones((4, 2)), hs)


def test_duplicate_positions_are_dropped():
    hs, _ = toy_hrirs()
    doubled = so.HRIRSet(
        np.concatenate([hs.irs, hs.irs[:3]]), np.concatenate([hs.positions, hs.positions[:3]]), hs.fs
    )
    assert len(doubled.positions) == len(hs.positions)


def test_onset_is_the_first_sample_within_20_db_and_the_shapes_keep_what_comes_before():
    hs, _ = toy_hrirs()
    irs = np.zeros_like(hs.irs)
    irs[:, :, 48:51] = [0.05, 0.2, 1.0]  # 0.05 is more than 20 dB down, 0.2 is not
    ramped = so.HRIRSet(irs, hs.positions, hs.fs, fit_onsets=False)
    assert np.all(ramped._onsets == 49)
    np.testing.assert_allclose(ramped.at(hs.positions[7])[0, :, :256], irs[7], atol=1e-9)


def test_onset_fit_follows_a_plus_b_over_r():
    hs, _ = toy_hrirs(taps=512)
    sets = []
    for distance in (0.25, 0.5, 1.0, 1.6):
        # onsets that follow r/c + a + b/r, to the nearest sample
        shift = int(round((distance / so.SPEED_OF_SOUND + 20e-6 + 30e-6 / distance) * hs.fs))
        sets.append(so.HRIRSet(np.roll(hs.irs, shift, axis=-1), hs.positions * distance, hs.fs))
    fitted = so.HRIRSet.concat(sets)
    measured = so.HRIRSet.concat(sets, fit_onsets=False)
    assert np.abs(fitted._delay_onsets - measured._onsets).max() < 0.5  # the rounding, no more


def test_between_distances_the_weights_are_linear_in_distance():
    hs, _ = toy_hrirs()
    far = so.HRIRSet(hs.irs * 0.5, hs.positions * 2, hs.fs)
    both = so.HRIRSet.concat([hs, far])
    peak = np.abs(both.at(hs.positions[5] * 1.25)[0]).max()
    assert peak == pytest.approx(0.875 * np.abs(hs.irs[5]).max(), rel=1e-6)
    with pytest.raises(ValueError, match="outside the measured distances"):
        both.at(hs.positions[5] * 0.5)
    x = so.gaussian_noise(0.05, hs.fs, rng=0)
    with pytest.raises(ValueError, match="outside the measured distances"):
        so.spatialize(x, hs.positions[5] * 0.5, both)


def test_without_align_the_raw_irs_are_blended():
    hs, _ = toy_hrirs()
    raw = so.HRIRSet(hs.irs, hs.positions, hs.fs, align=False)
    point = (hs.positions[0] + hs.positions[1]) / 2
    indices, weights = raw._weights(np.atleast_2d(point))
    np.testing.assert_allclose(raw.at(point)[0], np.einsum("k,kij->ij", weights[0], hs.irs[indices[0]]))


def test_emission_times_solve_the_delay_equation():
    duration, lookup = 2.0, np.linspace(0, 2.0, 2001)
    delay = CubicSpline(lookup, 0.003 + 0.002 * np.sin(2 * np.pi * 1.3 * lookup))
    arrival = np.linspace(-0.01, 2.02, 50001)
    emission = _emission_times(arrival, delay, duration, lookup)
    inside = (emission > 0) & (emission < duration)
    assert np.abs(emission + delay(emission) - arrival)[inside].max() < 1e-12
    assert np.allclose(emission[~inside], arrival[~inside] - delay(np.clip(emission[~inside], 0, duration)))


def test_move_sound_refuses_a_source_nearer_than_its_delay_allows():
    hs, _ = (
        toy_hrirs()
    )  # onsets about 0.4 ms: a source 50 cm nearer than measured would arrive before it left
    with pytest.raises(ValueError, match="closer than sound can travel"):
        so.move_sound(
            so.gaussian_noise(0.05, hs.fs, rng=0), np.repeat([so.hcc_to_rect(50, 0, 0)], 4, axis=0), hs
        )


def test_shapes_cross_fade_with_raised_cosines():
    hs = impulse_hrirs(gain=lambda positions, ear: 2.0 + positions[:, 0] * (1 if ear else -1))
    fs = hs.fs
    x = so.Sound(np.ones(fs), fs)
    duration = (len(x) - 1) / fs
    y = so.move_sound(x, so.hcc_trajectory(100, 0, ([0, duration], [0, 80])), hs, hop=0.25)
    shape_times = np.linspace(0, duration, 5)
    gains = np.array([hs.at(so.hcc_to_rect(100, 0, 80 * t / duration))[0, 1].sum() for t in shape_times])
    delay = (140 - hs._pre) / fs + hs._pre / fs  # every onset is 140 samples
    for k, fraction in ((1, 0.25), (2, 0.5), (3, 0.75)):
        emitted = shape_times[k] + fraction * (shape_times[1] - shape_times[0])
        weight = (1 + np.cos(np.pi * fraction)) / 2
        expected = weight * gains[k] + (1 - weight) * gains[k + 1]
        assert y.data[int(round((emitted + delay) * fs)), 1] == pytest.approx(expected, rel=1e-3)


def test_the_room_is_driven_at_the_mean_of_the_ears_delays():
    hs = impulse_hrirs(onset=(100, 200))
    fs = hs.fs
    click = np.zeros(4800)
    click[0] = 1.0
    room = so.Sound(np.r_[1.0, np.zeros(99)], fs)
    point = np.repeat([so.hcc_to_rect(100, 0, 0)], 4, axis=0)
    dry = so.move_sound(so.Sound(click, fs), point, hs)
    wet = so.move_sound(so.Sound(click, fs), point, hs, room=room, drr_db=0.0)
    reverberant = wet.data[: len(dry)] - dry.data
    assert np.argmax(np.abs(reverberant[:, 0])) == pytest.approx(150 - hs._pre, abs=2)


def test_diffuse_field_is_per_ear_and_scaled_to_one_meter():
    hs = impulse_hrirs(gain=lambda positions, ear: 1.0 + 0.5 * ear)
    energy, filters = hs._diffuse_field(hs.fs)
    assert energy == pytest.approx((1 + 1.5**2) / 2)
    assert np.sum(filters[1] ** 2) / np.sum(filters[0] ** 2) == pytest.approx(1.5**2, rel=1e-3)
    near = so.HRIRSet(hs.irs * 2, hs.positions / 2, hs.fs)  # the same set at 50 cm, 1/r louder
    assert near._diffuse_field(hs.fs)[0] == pytest.approx(energy)


def test_room_must_be_one_or_two_channels_at_the_sound_rate():
    hs, _ = toy_hrirs()
    x = so.gaussian_noise(0.05, hs.fs, rng=0)
    point = np.repeat([so.hcc_to_rect(100, 0, 0)], 4, axis=0)
    with pytest.raises(ValueError, match="one or two channels"):
        so.move_sound(x, point, hs, room=so.Sound(np.ones((10, 3)), hs.fs))
    with pytest.raises(ValueError, match="one or two channels"):
        so.move_sound(x, point, hs, room=so.Sound(np.ones(10), 44100))


def test_concat_refusals():
    hs, _ = toy_hrirs()
    with pytest.raises(ValueError, match="no HRIR sets"):
        so.HRIRSet.concat([])
    short, _ = toy_hrirs(taps=128)
    with pytest.raises(ValueError, match="IR shape"):
        so.HRIRSet.concat([hs, short])


def test_sofa_cartesian_positions(tmp_path):
    h5py = pytest.importorskip("h5py")
    hs, _ = toy_hrirs()
    x, y, z = hs.positions.T
    with h5py.File(tmp_path / "c.sofa", "w") as f:
        f["Data.IR"] = hs.irs
        f["Data.SamplingRate"] = np.array([hs.fs])
        pos = f.create_dataset("SourcePosition", data=np.column_stack([y, -x, z]))  # SOFA: x front, y left
        pos.attrs["Type"] = b"cartesian"
    np.testing.assert_allclose(so.HRIRSet.from_sofa(tmp_path / "c.sofa").positions, hs.positions, atol=1e-12)


def test_at_keeps_room_for_the_latest_onset():
    hs, _ = toy_hrirs()
    n_taps = hs.irs.shape[-1]
    assert hs.at(hs.positions[0]).shape == (1, 2, n_taps + int(np.ceil(hs._delay_onsets.max())))


def test_resampling_keeps_an_hrirs_gain():
    hs = impulse_hrirs(fs=65536)
    point = hs.positions[3]
    for fs in (48000, 44100, 16000):
        np.testing.assert_allclose(hs.at(point, fs=fs)[0].sum(axis=-1), 1.0, rtol=2e-3)


@pytest.mark.parametrize("distance_cm", [90, 200])
def test_spatialize_matches_a_still_move_sound_off_the_measured_distance(distance_cm):
    hs, _ = toy_hrirs()
    x = so.gaussian_noise(0.05, hs.fs, rng=0)
    point = np.array(so.hcc_to_rect(distance_cm, 0, 30))
    still = so.spatialize(x, point, hs)
    moving = so.move_sound(x, np.repeat([point], 4, axis=0), hs)
    n = min(len(still), len(moving))
    assert error_db(still.data[:n], moving.data[:n]) < -60


def test_directions_must_surround_the_head():
    hs, _ = toy_hrirs()
    _, elev, _ = so.rect_to_hcc(*hs.positions.T)
    for keep in (np.abs(elev) < 1e-6, hs.positions[:, 1] > 0):  # one plane, the front half
        part = so.HRIRSet(hs.irs[keep], hs.positions[keep], hs.fs)
        with pytest.raises(ValueError, match="must surround the head"):
            part.at(part.positions[0])


def test_a_source_at_the_center_of_the_head_is_refused():
    hs, _ = toy_hrirs()
    x = so.gaussian_noise(0.05, hs.fs, rng=0)
    with pytest.raises(ValueError, match="center of the head"):
        hs.at([0, 0, 0])
    with pytest.raises(ValueError, match="center of the head"):
        so.spatialize(x, [0, 0, 0], hs)
    with pytest.raises(ValueError, match="center of the head"):
        so.move_sound(x, np.zeros((4, 3)), hs)


def test_sofa_delays_are_added_to_the_irs(tmp_path):
    h5py = pytest.importorskip("h5py")
    hs, _ = toy_hrirs()
    x, y, z = hs.positions.T
    delays = np.zeros((len(hs.positions), 2))
    delays[:, 1] = 3
    delays[0] = [5, 2]
    with h5py.File(tmp_path / "d.sofa", "w") as f:
        f["Data.IR"] = hs.irs
        f["Data.SamplingRate"] = np.array([hs.fs])
        f["Data.Delay"] = delays
        pos = f.create_dataset("SourcePosition", data=np.column_stack([y, -x, z]))
        pos.attrs["Type"] = b"cartesian"
    loaded = so.HRIRSet.from_sofa(tmp_path / "d.sofa")
    n_taps = hs.irs.shape[-1]
    assert loaded.irs.shape[-1] == n_taps + 5
    for position in (0, 1):
        for ear in (0, 1):
            shift = int(delays[position, ear])
            np.testing.assert_allclose(
                loaded.irs[position, ear, shift : shift + n_taps], hs.irs[position, ear], atol=1e-12
            )
            np.testing.assert_allclose(loaded.irs[position, ear, :shift], 0, atol=1e-12)
