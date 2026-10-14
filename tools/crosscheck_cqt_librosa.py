"""How exactly librosa's inverse constant-Q transform gives a sound back, for
docs/design/views/chroma.md (K7). Needs librosa, which sonore does not depend on.

The test sound is 2 s of white noise band-limited to 70-5000 Hz, inside the transform's
range. Each line prints the relative error after the best single gain (librosa's icqt does
not return the input's level exactly), measured away from the first and last 0.25 s.

    python tools/crosscheck_cqt_librosa.py
"""

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 22050

rng = np.random.default_rng(0)
noise = sosfiltfilt(butter(8, [70, 5000], "band", fs=FS, output="sos"), rng.standard_normal(2 * FS))
middle = slice(FS // 4, -FS // 4)
print(f"librosa {librosa.__version__}")
for bins_per_octave, hop in [(12, 256), (36, 256), (36, 64), (48, 32)]:
    n_bins = 7 * bins_per_octave
    coefs = librosa.cqt(noise, sr=FS, fmin=55, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop)
    back = librosa.icqt(coefs, sr=FS, fmin=55, bins_per_octave=bins_per_octave, hop_length=hop, length=len(noise))
    gain = np.dot(back[middle], noise[middle]) / np.dot(back[middle], back[middle])
    error = np.linalg.norm(noise[middle] - gain * back[middle]) / np.linalg.norm(noise[middle])
    print(f"K7   {bins_per_octave:2d} bins per octave, hop {hop:3d}: relative error {error:.4f}, gain {gain:.4f}")
