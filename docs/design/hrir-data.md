# HRIR data on demand

sonore spatializes with measured HRIRs but ships none. This note covers how
it fetches them.

## Which database

PKU-IOA (Qu et al., 2009): KEMAR, 8 distances from 20 to 160 cm, elevation
-40 to 90 degrees, 793 directions per distance, 65536 Hz, 1024 taps. It is one
of the few public sets that varies distance as well as direction, so a source
can move toward or away from the head.

Its terms of use are not published: the database readme carries a copyright
line only. So sonore never bundles or rehosts it. It downloads the copy that
the SOFA conventions project serves, one SOFA file of about 13 MB per distance,
from <https://sofacoustics.org/data/database/pku-ioa/>. Users cite Qu et al.
(2009).

## Behaviour

- `load_hrirs("pku-ioa", distances=(100,))` returns one `HRIRSet`. Distances
  are in cm, as in head-centered coordinates. The default is 1 m only (13 MB);
  ask for several distances to get distance-dependent interpolation, or
  `distances="all"` for all 8 (104 MB).
- Each file is downloaded once into a cache directory, checked against a
  pinned SHA-256, and moved into place only after the check passes. A
  half-finished or tampered download never lands in the cache.
- The cache directory is `$SONORE_DATA_DIR` if set, else the platform's user
  cache directory (`~/.cache/sonore` on Linux, `~/Library/Caches/sonore` on
  macOS, `%LOCALAPPDATA%\sonore\Cache` on Windows).
- `HRIRSet.concat` merges sets with the same sampling rate and length, so the
  per-distance files become one set.
- Downloading uses the standard library, so it adds no dependency. Reading SOFA
  needs `h5py`, already the `sofa` extra.
- `HRIRSet.from_pku_ioa` also searches subfolders, so the original `.dat`
  layout (`dist20/elev-40/azi0_elev-40_dist20.dat`, ...) loads directly.

## Verification

`tools/check_pku_ioa_sofa.py`, run where the network allows it:

1. downloads the 8 SOFA files and prints their SHA-256 for the registry;
2. given a local copy of the original `.dat` files, checks that every SOFA
   position and IR matches the `.dat` set, which catches a coordinate
   convention mismatch between the two sources.

Tests never touch the network: they write small SOFA files and serve them
through a patched download function.

## Not covered

Other databases. The registry is a dictionary, so adding one is a name, a
base URL, file names and hashes.
