"""Stores Kaldi's MFCCs for the MFCC tests (tests/data/kaldi_mfcc_reference.npz).

Kaldi's ``compute-mfcc-feats`` is the speech-recognition standard and
follows HTK closely. HTK's own site was unreachable when this was written
(2026-10-02), so Kaldi stands in for it as the primary reference. The
numbers come from kaldi-native-fbank (k2-fsa), a C++ re-implementation of
Kaldi's feature code that installs with pip; it is not Kaldi itself.

It runs on the 0.8 s excerpt that the librosa fixture holds
(tests/data/librosa_mfcc_reference.npz), at 16-bit integer scale as Kaldi
reads WAV files, with dither off (Kaldi's default dither is random), and
stores four outputs:

- ``plain``: MFCCs with no DC removal, no pre-emphasis, a Hamming window,
  26 bins from 0 Hz, no lifter, and c0 kept (``use_energy=false``);
- ``fbank_plain``: the log mel energies of the same analysis;
- ``kaldi_window``: Kaldi's other defaults that sonore can express: the
  "povey" window, 23 bins from 20 Hz, lifter 22, still no DC removal,
  pre-emphasis or energy;
- ``defaults``: Kaldi's defaults (DC removal, pre-emphasis 0.97 inside each
  time window, c0 replaced by the raw log energy), which sonore does not
  reproduce by design; the difference is measured, not tested.

    pip install kaldi-native-fbank            # development only
    python tools/make_kaldi_fixtures.py
"""

from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBROSA_FIXTURE = ROOT / "tests" / "data" / "librosa_mfcc_reference.npz"
OUT = ROOT / "tests" / "data" / "kaldi_mfcc_reference.npz"


def options(kind, setting):
    opts = knf.MfccOptions() if kind == "mfcc" else knf.FbankOptions()
    frame = opts.frame_opts
    frame.samp_freq = 16000
    frame.dither = 0.0
    if setting in ("plain", "kaldi_window"):
        frame.remove_dc_offset = False
        frame.preemph_coeff = 0.0
        opts.use_energy = False
    if setting == "plain":
        frame.window_type = "hamming"
        opts.mel_opts.low_freq = 0.0
        opts.mel_opts.num_bins = 26
        if kind == "mfcc":
            opts.cepstral_lifter = 0.0
    return opts


def compute(kind, setting, waveform):
    opts = options(kind, setting)
    computer = knf.OnlineMfcc(opts) if kind == "mfcc" else knf.OnlineFbank(opts)
    computer.accept_waveform(16000, waveform.tolist())
    computer.input_finished()
    vectors = np.array([computer.get_frame(i) for i in range(computer.num_frames_ready)])
    return vectors, str(opts)


def main():
    reference = np.load(LIBROSA_FIXTURE)
    samples = reference["samples_int16"]
    waveform = samples.astype(float)  # Kaldi reads 16-bit WAV files at integer scale
    arrays = {"kaldi_native_fbank_version": np.array(knf.__version__)}
    for name, kind, setting in (
        ("plain", "mfcc", "plain"),
        ("fbank_plain", "fbank", "plain"),
        ("kaldi_window", "mfcc", "kaldi_window"),
        ("defaults", "mfcc", "defaults"),
    ):
        vectors, described = compute(kind, setting, waveform)
        arrays[name] = vectors
        arrays[f"{name}_options"] = np.array(described)
        print(f"{name}: {vectors.shape}")
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
