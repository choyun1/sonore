"""Downloading and caching published HRIR sets (so.load_hrirs)."""

import numpy as np
import pytest
from helpers import toy_hrirs

import sonore as so


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

    def test_pku_ioa_skips_empty_files(self, tmp_path):
        for a in range(0, 360, 90):
            np.ones(2048).tofile(tmp_path / f"azi{a}_elev0_dist20.dat")
        (tmp_path / "azi210_elev-30_dist20.dat").touch()
        with pytest.warns(UserWarning, match="skipped 1 empty"):
            hs = so.HRIRSet.from_pku_ioa(tmp_path)
        assert len(hs.positions) == 4

    def test_clockwise_sofa_copy_is_mirrored_back(self, server, monkeypatch):
        """The PKU-IOA SOFA copy stores clockwise azimuth; the loader undoes the mirror image."""
        import dataclasses

        hrir_data, _, _, hs = server
        assert hrir_data.HRIR_DATABASES["pku-ioa"].azimuth_clockwise is False  # the fixture's fake
        db = dataclasses.replace(hrir_data.HRIR_DATABASES["pku-ioa"], azimuth_clockwise=True)
        monkeypatch.setitem(hrir_data.HRIR_DATABASES, "pku-ioa", db)
        s = so.load_hrirs()
        np.testing.assert_allclose(s.positions[:, 0], -hs.positions[:, 0], atol=1e-12)
        np.testing.assert_allclose(s.positions[:, 1:], hs.positions[:, 1:], atol=1e-12)


def test_pku_ioa_registry_corrects_the_mirrored_copy():
    from sonore.stimuli import hrir_data

    assert hrir_data.HRIR_DATABASES["pku-ioa"].azimuth_clockwise
