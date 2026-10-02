"""Stores HTK's own MFCC output for the MFCC tests (tests/data/htk_mfcc_reference.npz).

HTK's HCopy (Young et al., The HTK Book) is the implementation that fixed
the MFCC recipe used in speech recognition, so it is the primary reference
for so.MFCC. HTK is free but needs registration, and its licence does not
allow redistribution, so this script is run by hand by someone who has
built it; the tests then compare against the stored numbers and need no HTK.

It writes the 0.8 s excerpt of the gallery sentence that the librosa fixture
uses (tests/data/librosa_mfcc_reference.npz) to a 16-bit WAV file, runs
HCopy on it with four configurations, reads HTK's parameter files, and
stores the results:

- ``standard``: MFCC_0 with HTK's usual front end: 25 ms Hamming window,
  10 ms hop, pre-emphasis 0.97, 26 channels, 12 cepstra plus c0, lifter 22,
  magnitude spectrum (HTK's default).
- ``plain``: the same with pre-emphasis 0, no lifter and the power spectrum.
- ``fbank_plain``: FBANK, the log channel outputs, with the ``plain`` settings.
- ``fbank_magnitude``: FBANK with magnitude, no pre-emphasis.

Comparing these isolates each of HTK's steps (filterbank, log, DCT,
lifter, pre-emphasis) against so.MFCC.

    # build HTK 3.4.1 (only HTKLib and HTKTools are needed), then:
    python tools/make_htk_fixtures.py /path/to/HCopy
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LIBROSA_FIXTURE = ROOT / "tests" / "data" / "librosa_mfcc_reference.npz"
OUT = ROOT / "tests" / "data" / "htk_mfcc_reference.npz"

COMMON = """SOURCEFORMAT = WAV
TARGETRATE = 100000.0
WINDOWSIZE = 250000.0
USEHAMMING = T
NUMCHANS = 26
NUMCEPS = 12
ZMEANSOURCE = F
ENORMALISE = F
SAVECOMPRESSED = F
SAVEWITHCRC = F
NATURALWRITEORDER = F
"""

CONFIGS = {
    "standard": "TARGETKIND = MFCC_0\nPREEMCOEF = 0.97\nCEPLIFTER = 22\nUSEPOWER = F\n",
    "plain": "TARGETKIND = MFCC_0\nPREEMCOEF = 0.0\nCEPLIFTER = 0\nUSEPOWER = T\n",
    "fbank_plain": "TARGETKIND = FBANK\nPREEMCOEF = 0.0\nUSEPOWER = T\n",
    "fbank_magnitude": "TARGETKIND = FBANK\nPREEMCOEF = 0.0\nUSEPOWER = F\n",
}


def read_htk(path):
    """An HTK parameter file: a 12-byte big-endian header (number of vectors,
    sample period in 100 ns units, bytes per vector, parameter kind) and
    big-endian float32 vectors. Returns (vectors, period in s, kind)."""
    data = Path(path).read_bytes()
    n_vectors, period, vector_bytes, kind = struct.unpack(">iihh", data[:12])
    values = np.frombuffer(data[12:], dtype=">f4").astype(float)
    return values.reshape(n_vectors, vector_bytes // 4), period * 1e-7, kind


def main(hcopy):
    reference = np.load(LIBROSA_FIXTURE)
    samples, fs = reference["samples_int16"], int(reference["fs"])
    arrays = {"samples_int16": samples, "fs": np.array(fs)}
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        wav = folder / "excerpt.wav"
        sf.write(wav, samples, fs, subtype="PCM_16")
        for name, settings in CONFIGS.items():
            config_text = COMMON + settings
            config, output = folder / f"{name}.conf", folder / f"{name}.htk"
            config.write_text(config_text)
            subprocess.run([hcopy, "-C", str(config), str(wav), str(output)], check=True)
            vectors, period, kind = read_htk(output)
            arrays[name] = vectors
            arrays[f"{name}_config"] = np.array(config_text)
            arrays[f"{name}_kind"] = np.array(kind)
            n_vectors, vector_length = vectors.shape
            print(f"{name}: {n_vectors} vectors of {vector_length}, period {period * 1e3:g} ms, kind {kind}")
        version = subprocess.run([hcopy, "-V"], capture_output=True, text=True)
        arrays["hcopy_version"] = np.array((version.stdout + version.stderr).strip())
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/make_htk_fixtures.py /path/to/HCopy")
    main(sys.argv[1])
