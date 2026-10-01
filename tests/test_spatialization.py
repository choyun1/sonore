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


def write_sofa(path, hrirs, dist_m):
    """A minimal SimpleFreeFieldHRIR file: spherical positions, azimuth counter-clockwise."""
    h5py = pytest.importorskip("h5py")
    _, elev, azim = so.rect_to_hcc(*hrirs.positions.T)
    with h5py.File(path, "w") as f:
        f["Data.IR"] = hrirs.irs
        f["Data.SamplingRate"] = np.array([hrirs.fs])
        pos = f.create_dataset(
            "SourcePosition", data=np.column_stack([np.mod(-azim, 360), elev, np.full(len(elev), dist_m)])
        )
        pos.attrs["Type"] = b"spherical"


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


class TestLoadHRIRs:
    @pytest.fixture
    def server(self, tmp_path, monkeypatch):
        """Serve fake PKU-IOA SOFA files from a local folder instead of the network."""
        from sonore.stimuli import hrir_data

        pytest.importorskip("h5py")
        hs, _ = toy_hrirs()
        remote = tmp_path / "remote"
        remote.mkdir()
        files = {}
        for d, (fname, _) in hrir_data.HRIR_DATABASES["pku-ioa"].files.items():
            write_sofa(remote / fname, so.HRIRSet(hs.irs, hs.positions * d / 100, hs.fs), d / 100)
            files[d] = (fname, hrir_data._sha256(remote / fname))
        db = hrir_data._Database(hrir_data.HRIR_DATABASES["pku-ioa"].base_url, files, (100,), "")
        monkeypatch.setitem(hrir_data.HRIR_DATABASES, "pku-ioa", db)
        calls = []

        def download(url, dest):
            calls.append(url)
            dest.write_bytes((remote / url.rsplit("/", 1)[-1]).read_bytes())

        monkeypatch.setattr(hrir_data, "_download", download)
        monkeypatch.setenv("SONORE_DATA_DIR", str(tmp_path / "cache"))
        return hrir_data, remote, calls, hs

    def test_default_is_one_meter_and_cached(self, server, tmp_path):
        _, _, calls, hs = server
        a = so.load_hrirs()
        b = so.load_hrirs()
        assert calls == ["https://sofacoustics.org/data/database/pku-ioa/dist_1.0m.sofa"]
        assert (tmp_path / "cache" / "pku-ioa" / "dist_1.0m.sofa").exists()
        np.testing.assert_allclose(a.positions, hs.positions, atol=1e-12)
        np.testing.assert_array_equal(a.irs, b.irs)

    def test_several_distances(self, server):
        _, _, calls, hs = server
        s = so.load_hrirs(distances=[20, 160])
        assert len(calls) == 2
        r = np.round(np.linalg.norm(s.positions, axis=1), 6)
        assert sorted(np.unique(r)) == [0.2, 1.6]
        assert len(so.load_hrirs(distances="all").positions) == 8 * len(hs.positions)

    def test_checksum_mismatch_leaves_no_file(self, server, tmp_path, monkeypatch):
        hrir_data, *_ = server
        db = hrir_data.HRIR_DATABASES["pku-ioa"]
        bad = {**db.files, 100: ("dist_1.0m.sofa", "0" * 64)}
        monkeypatch.setitem(
            hrir_data.HRIR_DATABASES, "pku-ioa", hrir_data._Database(db.base_url, bad, (100,), "")
        )
        with pytest.raises(OSError, match="checksum mismatch"):
            so.load_hrirs()
        assert list((tmp_path / "cache" / "pku-ioa").iterdir()) == []

    def test_corrupted_cache_is_fetched_again(self, server, tmp_path):
        _, _, calls, _ = server
        so.load_hrirs()
        (tmp_path / "cache" / "pku-ioa" / "dist_1.0m.sofa").write_bytes(b"junk")
        so.load_hrirs()
        assert len(calls) == 2

    def test_unknown_names_and_distances(self, server):
        with pytest.raises(ValueError, match="unknown HRIR database"):
            so.load_hrirs("cipic")
        with pytest.raises(ValueError, match="no distance"):
            so.load_hrirs(distances=[60])

    def test_pku_ioa_names_files_of_the_wrong_length(self, tmp_path):
        for a in range(0, 360, 90):
            np.zeros(2048).tofile(tmp_path / f"azi{a}_elev0_dist20.dat")
        np.zeros(1024).tofile(tmp_path / "azi0_elev10_dist20.dat")
        with pytest.raises(
            ValueError, match=r"1 of 5 .dat files(.|\n)*azi0_elev10_dist20.dat \(1024 values\)"
        ):
            so.HRIRSet.from_pku_ioa(tmp_path)
