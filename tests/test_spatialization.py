"""HRIR sets and spatialization."""

import numpy as np
import pytest

import sonore as so


def toy_hrirs(fs=48000, taps=256):
    """Spherical-head-ish toy set: ITD from Woodworth, ILD from azimuth."""
    az = np.arange(0, 360, 10)
    el = np.arange(-40, 91, 20)
    hcc = np.array([(100, e, a) for e in el for a in az])
    pos = np.column_stack(so.hcc_to_rect(*hcc.T))
    theta = np.radians(hcc[:, 2])
    lat = np.arcsin(np.sin(theta) * np.cos(np.radians(hcc[:, 1])))
    itd = 0.0875 / 343 * (lat + np.sin(lat))
    irs = np.zeros((len(hcc), 2, taps))
    for i, d in enumerate(itd):
        base = 20
        irs[i, 0, base + int(round(max(d, 0) * fs))] = 10 ** (-np.sin(lat[i]) * 5 / 20)
        irs[i, 1, base + int(round(max(-d, 0) * fs))] = 10 ** (np.sin(lat[i]) * 5 / 20)
    return so.HRIRSet(irs, pos, fs), itd


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
