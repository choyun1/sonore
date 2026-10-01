# Changelog

Notable changes to sonore. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until 1.0, a minor version (0.x.0) may change import paths or behaviour; a patch
version (0.x.y) only fixes bugs.

## [Unreleased]

### Added
- `so.load_hrirs()` downloads the PKU-IOA HRIR database on first use (1 m by
  default, any of its 8 distances on request), checks each file's SHA-256 and
  caches it in `$SONORE_DATA_DIR` or the user cache directory.
- `HRIRSet.concat` merges HRIR sets, e.g. one per distance.
- Continuous integration: tests and lint on Python 3.10 and 3.14 for every push
  and pull request, plus a check that the PyPI files build and pass their tests.
- Release workflow publishing to TestPyPI and PyPI with trusted publishing
  (see `docs/releasing.md`).

### Changed
- `HRIRSet.from_pku_ioa` also finds `.dat` files in subfolders, so the
  database's own `dist*/elev*/` layout loads directly.
- The source distribution now holds the code, tests and tools only; the docs,
  gallery and audio samples stay in the repository.

## [0.2.0] - 2026-09-29

Renamed to sonore, with the version kept in one place (`src/sonore/__init__.py`).
Not published to PyPI.

[Unreleased]: https://github.com/choyun1/sonore/commits/main
[0.2.0]: https://github.com/choyun1/sonore/commit/213e21a
