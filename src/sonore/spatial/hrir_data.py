"""Public HRIR databases, downloaded on demand and cached.

sonore bundles no HRIRs. :func:`load_hrirs` downloads the files a database
needs the first time they are asked for, checks each against a pinned SHA-256,
and keeps them in a cache directory: ``$SONORE_DATA_DIR`` if set, else the
platform's user cache directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from sonore.spatial.spatialization import HRIRSet

__all__ = ["HRIR_DATABASES", "data_dir", "load_hrirs"]


@dataclass(frozen=True)
class _Database:
    base_url: str
    files: dict[float, tuple[str, str | None]]  # distance [cm] -> (file name, sha256)
    default: tuple[float, ...]
    citation: str
    # The file stores azimuth clockwise in SOFA's counter-clockwise field, so
    # read as SOFA it is the left-right mirror image of the measurement.
    azimuth_clockwise: bool = False

    def read(self, path: Path, **kwargs) -> HRIRSet:
        hrir_set = HRIRSet.from_sofa(path, **kwargs)
        if self.azimuth_clockwise:
            hrir_set = HRIRSet(hrir_set.irs, hrir_set.positions * [-1.0, 1.0, 1.0], hrir_set.fs, **kwargs)
        return hrir_set


HRIR_DATABASES: dict[str, _Database] = {
    # KEMAR, 793 directions per distance, 65536 Hz. Its terms of use are not
    # published, so sonore only downloads the copy the SOFA project serves.
    # That copy keeps PKU-IOA's clockwise azimuth: compared with the original
    # .dat files, it is mirrored left to right, and a source on the right is
    # louder in the right ear only after the correction
    # (tools/check_pku_ioa_sofa.py).
    "pku-ioa": _Database(
        base_url="https://sofacoustics.org/data/database/pku-ioa/",
        files={
            20: ("dist_0.2m.sofa", "a05019362ead5ff30499a5dbe3b4a7a11af56469a13346efe7d9af4b8ea748d7"),
            30: ("dist_0.3m.sofa", "4b8aa087ec8c8b08291bbe89dc9f03ffca66ad68ffe4a800871075f0e85eec1d"),
            40: ("dist_0.4m.sofa", "4c8643895e0ef9e408eb2ae083d1e55cd60e57bd7469e3d1a6193eecbe8b4247"),
            50: ("dist_0.5m.sofa", "6bebe6c61f88fb123ab6e792a95e4faf4f36f7bb4ea4afe1ec968f43e0f553a5"),
            75: ("dist_0.75m.sofa", "1fbd2f60dc622f7a4e9a6ae986d4292e884dae305e348c057e08f8343c2b6f55"),
            100: ("dist_1.0m.sofa", "5c8948a719034153f48bd8886f3654e1d373dd9e8f9911d67070c40c4066b914"),
            130: ("dist_1.3m.sofa", "7ed5d5d48da6e61d85b08d4873459b22a2b4b2cd5715980cb3e27af6acd1677f"),
            160: ("dist_1.6m.sofa", "b07c5a1fa120dace81916f94594af050f19126447595da75987ae3c488ae591f"),
        },
        default=(100,),
        azimuth_clockwise=True,
        citation="Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related transfer "
        "functions measured with high spatial resolution using a spark gap. IEEE TASLP 17(6).",
    ),
}


def data_dir() -> Path:
    """Where downloaded data is cached: ``$SONORE_DATA_DIR``, else the user cache directory."""
    if env_dir := os.environ.get("SONORE_DATA_DIR"):
        return Path(env_dir).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "sonore" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sonore"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sonore"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, open(dest, "wb") as file:
        shutil.copyfileobj(response, file)


def _fetch(url: str, path: Path, sha256: str | None) -> Path:
    """Download ``url`` to ``path`` unless it is already there, and check its hash.

    The file is written under a temporary name and moved into place only after
    the check, so an interrupted or corrupted download never lands in the cache.
    """
    if path.exists():
        if sha256 is None or _sha256(path) == sha256:
            return path
        path.unlink()  # stale or corrupted; fetch it again
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_path)
    try:
        _download(url, tmp_path)
        digest = _sha256(tmp_path)
        if sha256 is None:
            warnings.warn(f"no pinned checksum for {path.name}; sha256 is {digest}", stacklevel=3)
        elif digest != sha256:
            raise OSError(f"checksum mismatch for {url}: expected {sha256}, got {digest}")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def load_hrirs(
    name: str = "pku-ioa",
    distances=None,
    cache_dir: str | PathLike | None = None,
    **kwargs,
) -> HRIRSet:
    """Load a public HRIR database, downloading its files on first use.

    Parameters
    ----------
    name
        Database name, a key of :data:`HRIR_DATABASES`. ``"pku-ioa"``: KEMAR at
        20, 30, 40, 50, 75, 100, 130 and 160 cm, about 13 MB per distance. Cite
        Qu et al. (2009) when you use it.
    distances
        Source distances [cm] to load. Several distances give a set that
        interpolates across distance as well as direction. ``None`` loads the
        database's default (1 m for PKU-IOA), ``"all"`` every distance.
    cache_dir
        Overrides :func:`data_dir`.
    **kwargs
        Passed to :class:`HRIRSet`, e.g. ``align``.
    """
    try:
        database = HRIR_DATABASES[name]
    except KeyError:
        raise ValueError(f"unknown HRIR database {name!r}; known: {sorted(HRIR_DATABASES)}") from None
    if distances is None:
        distances = database.default
    elif isinstance(distances, str) and distances == "all":
        distances = tuple(database.files)
    else:
        distances = tuple(
            float(distance) for distance in (distances if hasattr(distances, "__iter__") else [distances])
        )
    missing = [distance for distance in distances if distance not in database.files]
    if missing:
        raise ValueError(f"{name} has no distance {missing} cm; available: {sorted(database.files)}")
    root = Path(cache_dir).expanduser() if cache_dir is not None else data_dir()
    sets = []
    for distance in distances:
        file_name, sha256 = database.files[distance]
        path = _fetch(database.base_url + file_name, root / name / file_name, sha256)
        sets.append(database.read(path, **kwargs))
    return HRIRSet.concat(sets, **kwargs)
