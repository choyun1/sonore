"""Cross-check the MFCC recipes in docs/design/mfcc.md against reference
implementations, and compare MFCCs with sonore's CheapTrick envelope.

- librosa (Slaney mel, area-normalized triangles, power in dB with an 80 dB
  floor below the loudest cell): ``librosa.feature.mfcc`` with its defaults,
  and ``librosa.filters.mel(htk=True, norm=None)`` against HTK triangles.
- python_speech_features 0.6 (HTK mel, triangles on rounded FFT bins,
  rectangular window, pre-emphasis 0.97, lifter 22, c0 replaced by the log
  energy): ``python_speech_features.mfcc`` with its defaults.
- CheapTrick: MFCCs of a synthetic vowel taken from ``so.cheaptrick``'s
  envelope instead of its power spectrum, at several F0s (claim C5).

The formulas are the claim checker's (``tools/check_mfcc_claims.py``). The
two libraries are development-time dependencies only:

    pip install librosa python_speech_features
    python tools/crosscheck_mfcc.py

(If python_speech_features will not build, unpack its sdist and put the
folder on PYTHONPATH; the package is pure Python.)
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import python_speech_features as psf
import soundfile as sf
from scipy.fft import dct
from scipy.signal import savgol_filter

import sonore as so

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_mfcc_claims import (  # noqa: E402
    DB_PER_NEPER,
    SPEECH,
    VOWELS,
    harmonic_vowel,
    htk_mel,
    htk_mel_to_hz,
    mel_matrix,
)


def report(text, value):
    print(f"{text:<80s} {value:.4g}")


speech, fs = sf.read(SPEECH)
print(f"librosa {librosa.__version__}, python_speech_features 0.6, sonore {so.__version__}")

# ------------------------------------------------------------ librosa
n_fft, hop_len, n_mels, n_mfcc = 2048, 512, 128, 20
padded = np.pad(speech, n_fft // 2)  # center=True, pad_mode="constant"
n_windows = 1 + (len(padded) - n_fft) // hop_len
segments = padded[hop_len * np.arange(n_windows)[:, None] + np.arange(n_fft)[None, :]]
periodic_hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
power = np.abs(np.fft.rfft(segments * periodic_hann, n_fft)) ** 2
weights, _ = mel_matrix(n_mels, n_fft, fs, scale="slaney", area_normalized=True)
mel_db = 10 * np.log10(np.maximum(power @ weights.T, 1e-10))
mel_db = np.maximum(mel_db, mel_db.max() - 80)
ours = dct(mel_db, type=2, norm="ortho", axis=1)[:, :n_mfcc].T
theirs = librosa.feature.mfcc(y=speech, sr=fs)
report(
    "librosa.feature.mfcc defaults vs formula: max abs diff / max abs value",
    np.abs(ours - theirs).max() / np.abs(theirs).max(),
)
report("  share of mel cells raised by the 80 dB floor [%]", 100 * np.mean(mel_db == mel_db.min()))
unclipped = dct(10 * np.log10(np.maximum(power @ weights.T, 1e-10)), type=2, norm="ortho", axis=1)[
    :, :n_mfcc
].T
report("  floor's effect on the coefficients: max abs change", np.abs(unclipped - theirs).max())
report("  as a share of the coefficients' range", np.abs(unclipped - theirs).max() / np.ptp(theirs))
for scale, norm in (("slaney", "slaney"), ("htk", None)):
    ours_weights, _ = mel_matrix(26, 512, fs, scale=scale, area_normalized=norm == "slaney")
    theirs_weights = librosa.filters.mel(sr=fs, n_fft=512, n_mels=26, htk=scale == "htk", norm=norm)
    report(
        f"librosa.filters.mel({scale}, norm={norm}) vs formula: max abs diff / max",
        np.abs(ours_weights - theirs_weights).max() / theirs_weights.max(),
    )

# ------------------------------------------------------------ python_speech_features
win_len, hop_len, n_fft, n_mels, n_mfcc, lifter_length = 400, 160, 512, 26, 13, 22
emphasized = np.concatenate([speech[:1], speech[1:] - 0.97 * speech[:-1]])
n_windows = 1 + int(np.ceil((len(emphasized) - win_len) / hop_len))
emphasized = np.pad(emphasized, (0, (n_windows - 1) * hop_len + win_len - len(emphasized)))
segments = emphasized[hop_len * np.arange(n_windows)[:, None] + np.arange(win_len)[None, :]]
power = np.abs(np.fft.rfft(segments, n_fft)) ** 2 / n_fft
eps = np.finfo(float).eps
energy = np.where(power.sum(1) == 0, eps, power.sum(1))
edge_bins = np.floor((n_fft + 1) * htk_mel_to_hz(np.linspace(0, htk_mel(fs / 2), n_mels + 2)) / fs)
bin_index = np.arange(n_fft // 2 + 1)
rounded_weights = np.zeros((n_mels, n_fft // 2 + 1))
for band in range(n_mels):
    low, centre, high = edge_bins[band : band + 3]
    rounded_weights[band] = np.where(
        (bin_index >= low) & (bin_index < centre),
        (bin_index - low) / (centre - low),
        np.where((bin_index >= centre) & (bin_index < high), (high - bin_index) / (high - centre), 0.0),
    )


def psf_recipe(band_weights):
    band_power = power @ band_weights.T
    coefficients = dct(np.log(np.where(band_power == 0, eps, band_power)), type=2, norm="ortho", axis=1)[
        :, :n_mfcc
    ]
    coefficients *= 1 + lifter_length / 2 * np.sin(np.pi * np.arange(n_mfcc) / lifter_length)
    coefficients[:, 0] = np.log(energy)
    return coefficients


theirs = psf.mfcc(speech, fs)
ours = psf_recipe(rounded_weights)
report(
    "python_speech_features.mfcc defaults vs formula: max abs diff / max abs value",
    np.abs(ours - theirs).max() / np.abs(theirs).max(),
)
exact_weights, _ = mel_matrix(n_mels, n_fft, fs, scale="htk")
exact = psf_recipe(exact_weights)
loud = energy > 1e-6 * energy.max()
report(
    "  rounded vs exact triangles, loud time windows: rms diff / rms of c1..c12",
    np.sqrt(np.mean((exact - theirs)[loud, 1:] ** 2)) / np.sqrt(np.mean(theirs[loud, 1:] ** 2)),
)
report(
    "  max |difference| between the two filterbanks' weights", np.abs(exact_weights - rounded_weights).max()
)

# ------------------------------------------------------------ CheapTrick (C5)
n_mels, n_mfcc = 26, 13


def smoothed_shape(band_power):
    coefficients = dct(np.log(band_power), type=2, norm="ortho")[:n_mfcc]
    coefficients[0] = 0
    return coefficients


def distance_db(first, second):
    return DB_PER_NEPER * np.sqrt(np.sum((first - second) ** 2) / n_mels)


f0s = (100, 150, 200, 250, 300)
raw, smooth = {}, {}
for vowel in VOWELS:
    for f0 in f0s:
        signal = harmonic_vowel(f0, vowel, fs, duration=0.4)
        middle = len(signal) // 2
        segment = signal[middle - 200 : middle + 200] * np.hamming(400)
        raw[vowel, f0] = smoothed_shape(
            mel_matrix(n_mels, 512, fs)[0] @ (np.abs(np.fft.rfft(segment, 512)) ** 2)
        )
        envelope = so.cheaptrick(so.Sound(signal, fs), (np.array([middle / fs]), np.array([float(f0)])))
        envelope_weights, _ = mel_matrix(n_mels, envelope.n_fft, fs)
        smooth[vowel, f0] = smoothed_shape(envelope_weights @ envelope.data[0, :, 0])

for name, shapes in (("power spectrum", raw), ("CheapTrick envelope", smooth)):
    same_vowel = [
        distance_db(shapes[v, low], shapes[v, high])
        for v in VOWELS
        for low in f0s
        for high in f0s
        if low < high
    ]
    across = [distance_db(shapes["a", f0], shapes["i", f0]) for f0 in f0s]
    report(f"MFCCs from the {name}: same vowel, two F0s, largest distance [dB]", max(same_vowel))
    report("  median [dB]", np.median(same_vowel))
    report("  /a/ vs /i/ at the same F0, smallest distance [dB]", min(across))

# ------------------------------------------------------------ sonore's STFT on librosa's time grid
# The tests compare so.MFCC with tests/data/librosa_mfcc_reference.npz
# (tools/make_mfcc_fixtures.py). This section shows the comparison can be
# exact: sonore's Hann STFT has librosa's time windows (plus one before and
# two after), and the formulas on its power reproduce the stored output.
reference = np.load(Path(__file__).resolve().parents[1] / "tests" / "data" / "librosa_mfcc_reference.npz")
sentence = so.load(SPEECH)
settings = {
    "defaults": (2048, 512, 2048, 128, 20, "slaney"),
    "speech_htk": (400, 160, 512, 26, 13, "htk"),
    "speech_slaney": (400, 160, 512, 40, 13, "slaney"),
}
for name, (win_len, hop_len, n_fft, n_mels, n_mfcc, scale) in settings.items():
    frame = so.GaborFrame(win_len / fs, hop_len / fs, window="hann", n_fft=n_fft)
    sonore_power = np.abs(so.STFT(sentence, frame=frame).data[0]) ** 2
    librosa_power = np.abs(librosa.stft(speech, n_fft=n_fft, hop_length=hop_len, win_length=win_len)) ** 2
    n_windows = librosa_power.shape[1]
    sonore_power = sonore_power[:, 1 : 1 + n_windows]  # drop the time window centred before the first sample
    report(
        f"{name}: sonore STFT power vs librosa.stft, max abs diff / max",
        np.abs(sonore_power - librosa_power).max() / librosa_power.max(),
    )
    weights, _ = mel_matrix(n_mels, n_fft, fs, scale=scale, area_normalized=scale == "slaney")
    mel_power = weights @ sonore_power
    mel_db = 10 * np.log10(np.maximum(mel_power, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 80)
    coefficients = dct(mel_db, type=2, norm="ortho", axis=0)[:n_mfcc]
    stored_mel, stored_mfcc = reference[f"{name}_mel_power"], reference[f"{name}_mfcc"]
    report(
        "  mel power from sonore's STFT vs stored librosa, max abs diff / max",
        np.abs(mel_power - stored_mel).max() / stored_mel.max(),
    )
    report(
        "  MFCCs vs stored librosa, max abs diff / max abs value",
        np.abs(coefficients - stored_mfcc).max() / np.abs(stored_mfcc).max(),
    )

for width in (5, 9):
    for order in (1, 2):
        key = f"speech_htk_delta{width}" + ("_order2" if order == 2 else "")
        ours = savgol_filter(
            reference["speech_htk_mfcc"], width, polyorder=order, deriv=order, axis=-1, mode="interp"
        )
        report(
            f"deltas, width {width}, order {order}: Savitzky-Golay vs stored librosa, max abs diff",
            np.abs(ours - reference[key]).max(),
        )
half_width = 2
offsets = np.arange(-half_width, half_width + 1)
regression = np.apply_along_axis(
    lambda row: np.convolve(
        np.pad(row, half_width, mode="edge"), offsets[::-1] / np.sum(offsets**2), mode="valid"
    ),
    -1,
    reference["speech_htk_mfcc"],
)
stored = reference["speech_htk_delta5"]
report(
    "HTK regression (+-2, edges repeated) vs librosa width 5: interior max abs diff",
    np.abs(regression - stored)[:, half_width:-half_width].max(),
)
report(
    "  first and last two time windows: max abs diff",
    np.abs(regression - stored)[:, np.r_[:half_width, -half_width:0]].max(),
)
