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

from sonore.stimuli.spatialization import HRIRSet

__all__ = ["HRIR_DATABASES", "data_dir", "load_hrirs"]


@dataclass(frozen=True)
class _Database:
    base_url: str
    files: dict[float, tuple[str, str | None]]  # distance [cm] -> (file name, sha256)
    default: tuple[float, ...]
    citation: str


HRIR_DATABASES: dict[str, _Database] = {
    # KEMAR, 793 directions per distance, 65536 Hz. Its terms of use are not
    # published, so sonore only downloads the copy the SOFA project serves.
    "pku-ioa": _Database(
        base_url="https://sofacoustics.org/data/database/pku-ioa/",
        files={
            20: ("dist_0.2m.sofa", None),
            30: ("dist_0.3m.sofa", None),
            40: ("dist_0.4m.sofa", None),
            50: ("dist_0.5m.sofa", None),
            75: ("dist_0.75m.sofa", None),
            100: ("dist_1.0m.sofa", None),
            130: ("dist_1.3m.sofa", None),
            160: ("dist_1.6m.sofa", None),
        },
        default=(100,),
        citation="Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related transfer "
        "functions measured with high spatial resolution using a spark gap. IEEE TASLP 17(6).",
    ),
}


def data_dir() -> Path:
    """Where downloaded data is cached: ``$SONORE_DATA_DIR``, else the user cache directory."""
    if env := os.environ.get("SONORE_DATA_DIR"):
        return Path(env).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "sonore" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sonore"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sonore"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)


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
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    os.close(fd)
    tmp = Path(tmp)
    try:
        _download(url, tmp)
        digest = _sha256(tmp)
        if sha256 is None:
            warnings.warn(f"no pinned checksum for {path.name}; sha256 is {digest}", stacklevel=3)
        elif digest != sha256:
            raise OSError(f"checksum mismatch for {url}: expected {sha256}, got {digest}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
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
        db = HRIR_DATABASES[name]
    except KeyError:
        raise ValueError(f"unknown HRIR database {name!r}; known: {sorted(HRIR_DATABASES)}") from None
    if distances is None:
        distances = db.default
    elif isinstance(distances, str) and distances == "all":
        distances = tuple(db.files)
    else:
        distances = tuple(float(d) for d in (distances if hasattr(distances, "__iter__") else [distances]))
    missing = [d for d in distances if d not in db.files]
    if missing:
        raise ValueError(f"{name} has no distance {missing} cm; available: {sorted(db.files)}")
    root = Path(cache_dir).expanduser() if cache_dir is not None else data_dir()
    sets = []
    for d in distances:
        fname, sha256 = db.files[d]
        path = _fetch(db.base_url + fname, root / name / fname, sha256)
        sets.append(HRIRSet.from_sofa(path, **kwargs))
    return HRIRSet.concat(sets, **kwargs)
