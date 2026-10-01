"""HRIR sets and spatialization."""

import numpy as np
import pytest
from helpers import toy_hrirs

import sonore as so


class TestSpatialization:
    def test_coordinate_roundtrip(self):
        h = (150.0, 20.0, 250.0)
        np.testing.assert_allclose(so.rect_to_hcc(*so.hcc_to_rect(*h)), h)
        x, y, z = so.hcc_to_rect(100, 0, 90)
        np.testing.assert_allclose([x, y, z], [1, 0, 0], atol=1e-12)  # 90 deg = right

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

    def test_move_sound_crossfade_preserves_level(self):
        hs, _ = toy_hrirs()
        x = so.gaussian_noise(1, 48000, rng=0)
        still = so.spatialize(x, so.hcc_to_rect(100, 0, 0), hs)
        moving = so.move_sound(x, np.repeat([so.hcc_to_rect(100, 0, 0)], 50, axis=0), hs)
        np.testing.assert_allclose(moving.data[: len(still)], still.data, atol=1e-9)


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
