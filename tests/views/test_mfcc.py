"""MFCCs against librosa's own output, stored by tools/make_mfcc_fixtures.py
(librosa is not needed here) together with the 0.8 s of the gallery sentence
it analyzed, and the properties the design relies on."""

from pathlib import Path

import numpy as np
import pytest

import sonore as so
from sonore.core.utils import freq_to_mel, mel_to_freq
from sonore.views.mfcc import delta_features, symmetric_hamming

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = np.load(ROOT / "tests" / "data" / "librosa_mfcc_reference.npz")
SENTENCE = so.Sound(REFERENCE["samples_int16"] / 32768.0, float(REFERENCE["fs"]))

# name: (window, hop, n_fft) in samples, n_mels, n_mfcc, mel scale, triangles
LIBROSA_SETTINGS = {
    "defaults": ((2048, 512, 2048), 128, 20, "slaney", "area"),
    "speech_htk": ((400, 160, 512), 26, 13, "htk", "height"),
    "speech_slaney": ((400, 160, 512), 40, 13, "slaney", "area"),
}


def librosa_like(name):
    """so.MFCC set up as librosa was, and the slice of time windows librosa has
    (sonore's grid has one more before the first sample and some after)."""
    (window, hop, n_fft), n_mels, n_mfcc, scale, triangles = LIBROSA_SETTINGS[name]
    frame = so.GaborFrame(window / SENTENCE.fs, hop / SENTENCE.fs, window="hann", n_fft=n_fft)
    mfcc = so.MFCC(
        so.STFT(SENTENCE, frame=frame),
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        mel_scale=scale,
        triangles=triangles,
        triangle_axis="hz",
        floor_db=-80,
    )
    n_windows = REFERENCE[f"{name}_mfcc"].shape[1]
    return mfcc, slice(1, 1 + n_windows)


@pytest.mark.parametrize("name", LIBROSA_SETTINGS)
def test_matches_librosa(name):
    mfcc, librosa_windows = librosa_like(name)
    stored_power = REFERENCE[f"{name}_mel_power"]
    stored_mfcc = REFERENCE[f"{name}_mfcc"]
    power_error = np.abs(mfcc.mel_power[0][:, librosa_windows] - stored_power).max()
    # librosa's filter weights are float32, so each is off by up to about 6e-8 of itself.
    assert power_error < 3e-7 * stored_power.max()
    mfcc_error = np.abs(mfcc.db[0][:, librosa_windows] - stored_mfcc).max()
    assert mfcc_error < 1e-8 * np.abs(stored_mfcc).max()


@pytest.mark.parametrize("width", [5, 9])
@pytest.mark.parametrize("order", [1, 2])
def test_deltas_match_librosa(width, order):
    stored_mfcc = REFERENCE["speech_htk_mfcc"]
    stored = REFERENCE[f"speech_htk_delta{width}" + ("_order2" if order == 2 else "")]
    assert np.abs(delta_features(stored_mfcc, order=order, width=width) - stored).max() < 1e-12
    # The method on sonore's own grid agrees away from the ends of librosa's.
    mfcc, librosa_windows = librosa_like("speech_htk")
    interior = slice(width, -width)
    ours = mfcc.deltas(order=order, width=width)[0] * 10 / np.log(10)
    tolerance = 1e-8 * np.abs(stored_mfcc).max()  # as for the coefficients themselves
    assert np.abs(ours[:, librosa_windows][:, interior] - stored[:, interior]).max() < tolerance


def test_first_deltas_are_the_regression_slope():
    rng = np.random.default_rng(0)
    track = rng.standard_normal(40)
    offsets = np.arange(-2, 3)
    slopes = [np.polyfit(offsets, track[i - 2 : i + 3], 1)[0] for i in range(2, 38)]
    assert np.allclose(delta_features(track)[2:38], slopes, atol=1e-13)


def test_sound_input_uses_the_speech_settings():
    from_sound = so.MFCC(SENTENCE)
    frame = so.GaborFrame(0.025, 0.010, window=symmetric_hamming, n_fft=512)
    from_stft = so.MFCC(so.STFT(SENTENCE, frame=frame))
    assert from_sound.n_fft == 512
    assert np.array_equal(from_sound.data, from_stft.data)
    assert from_sound.data.shape == (1, 13, len(from_sound.t))
    with pytest.raises(TypeError, match="win_dur"):
        so.MFCC(so.STFT(SENTENCE), win_dur=0.02)


def test_scaling_moves_only_c0():
    quiet, loud = so.MFCC(SENTENCE), so.MFCC(SENTENCE * 3.7)
    n_mels = len(quiet.cfs)
    # c0 is the band mean of ln(power) times sqrt(n_mels); power scales by 3.7 ** 2.
    assert np.allclose(loud.data[:, 0] - quiet.data[:, 0], 2 * np.log(3.7) * np.sqrt(n_mels))
    assert np.allclose(loud.data[:, 1:], quiet.data[:, 1:], atol=1e-10)


def test_area_triangles_shift_every_time_window_by_the_same_vector():
    height = so.MFCC(SENTENCE)
    area = so.MFCC(SENTENCE, triangles="area")
    shift = area.data - height.data
    assert np.abs(shift - shift[:, :, :1]).max() < 1e-10
    assert np.allclose(area.deltas(), height.deltas(), atol=1e-10)


def test_silent_channel_is_finite():
    silent_first = so.Sound(np.column_stack([np.zeros(SENTENCE.n_samples), SENTENCE.data[:, 0]]), SENTENCE.fs)
    mfcc = so.MFCC(silent_first)
    assert np.all(np.isfinite(mfcc.data))
    assert np.allclose(mfcc.data[0, 1:], 0)


def test_envelope_returns_the_band_powers_when_nothing_is_dropped():
    mfcc = so.MFCC(SENTENCE, n_mfcc=26, lifter=22)
    floored = 10 ** (mfcc.mel_db / 10)
    assert np.allclose(mfcc.envelope(mfcc.cfs), floored, rtol=1e-10)


def test_lifter_is_a_fixed_gain_per_coefficient():
    plain, liftered = so.MFCC(SENTENCE), so.MFCC(SENTENCE, lifter=22)
    gains = 1 + 11 * np.sin(np.pi * np.arange(13) / 22)
    assert np.allclose(liftered.data, plain.data * gains[:, None])


@pytest.mark.parametrize("scale", ["htk", "slaney"])
def test_mel_scale_round_trip_and_landmarks(scale):
    freqs = np.array([0.0, 200.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_freq(freq_to_mel(freqs, scale), scale), freqs)
    assert freq_to_mel(1000.0, scale) == pytest.approx(1000.0 if scale == "htk" else 15.0, rel=1e-3)


def test_height_triangles_sum_to_one_between_the_outer_centers():
    mfcc = so.MFCC(SENTENCE)
    bin_freqs = np.arange(mfcc.n_fft // 2 + 1) * mfcc.fs / mfcc.n_fft
    inside = (bin_freqs >= mfcc.cfs[0]) & (bin_freqs <= mfcc.cfs[-1])
    assert np.allclose(mfcc.weights.sum(0)[inside], 1)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_mfcc"):
        so.MFCC(SENTENCE, n_mfcc=30)
    with pytest.raises(ValueError, match="f_hi"):
        so.MFCC(SENTENCE, f_hi=9000)
    with pytest.raises(ValueError, match="width"):
        delta_features(np.zeros(10), width=4)
    with pytest.raises(TypeError):
        so.MFCC(np.zeros(100))


@pytest.mark.parametrize("kind", ["mfcc", "mel"])
def test_plot(kind):
    import matplotlib

    matplotlib.use("Agg")
    ax = so.MFCC(SENTENCE).plot(kind=kind)
    assert ax.get_xlabel() == "Time [s]"


KALDI = np.load(ROOT / "tests" / "data" / "kaldi_mfcc_reference.npz")
KALDI_SAMPLES = REFERENCE["samples_int16"].astype(float)  # Kaldi reads WAV files at integer scale


def kaldi_like(window, **settings):
    """so.MFCC on Kaldi's time windows and the matching slices of both.

    Kaldi's time windows start at multiples of the hop (160 samples); sonore's
    are centered on them. Trimming 40 samples (half the 400-sample window,
    modulo the hop) makes sonore's window 3 cover Kaldi's window 1; Kaldi's
    window 0 would need the trimmed samples, so it is left out."""
    sound = so.Sound(KALDI_SAMPLES[40:], SENTENCE.fs)
    frame = so.GaborFrame(400 / SENTENCE.fs, 160 / SENTENCE.fs, window=window, n_fft=512)
    mfcc = so.MFCC(so.STFT(sound, frame=frame), **settings)
    n_windows = len(KALDI["plain"])
    return mfcc, slice(3, n_windows + 2), slice(1, None)


def povey(n_samples):
    """Kaldi's default window: a symmetric Hann window to the power 0.85."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_samples) / (n_samples - 1))) ** 0.85


def test_matches_kaldi_plain():
    mfcc, ours, theirs = kaldi_like(symmetric_hamming)
    stored = KALDI["plain"][theirs]
    # kaldi-native-fbank computes in float32.
    assert np.abs(mfcc.data[0].T[ours] - stored).max() < 1e-6 * np.abs(stored).max()
    log_mel = np.log(mfcc.mel_power[0]).T[ours]
    stored_log_mel = KALDI["fbank_plain"][theirs]
    assert np.abs(log_mel - stored_log_mel).max() < 1e-5 * np.abs(stored_log_mel).max()


def test_matches_kaldi_window_bins_and_lifter():
    mfcc, ours, theirs = kaldi_like(povey, n_mels=23, f_lo=20, lifter=22)
    stored = KALDI["kaldi_window"][theirs]
    assert np.abs(mfcc.data[0].T[ours] - stored).max() < 1e-5 * np.abs(stored).max()
