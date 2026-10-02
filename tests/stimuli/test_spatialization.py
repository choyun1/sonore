"""HRIR sets and spatialization."""

import numpy as np
import pytest
from helpers import toy_hrirs

import sonore as so


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

    def test_moving_sound_itd_sweeps(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(2, 48000, band=(200, 3000), rng=0)
        traj = so.circular_trajectory((100, 0, 90), (100, 0, -90), 200)
        y = so.move_sound(x, traj, hs)
        c = so.interaural_cues(y, 50e-3)
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
        expected = np.sum(x.data**2) * hs._direct_energy_at_1m(48000) * 10 ** (-5.0 / 10) * 2
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
