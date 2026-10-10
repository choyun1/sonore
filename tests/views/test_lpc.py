"""so.LPC, so.levinson and so.formant_track against their formulas, Praat and known formants."""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

import sonore as so
from sonore.views.lpc import _track
from sonore.views.mfcc import symmetric_hamming

ROOT = Path(__file__).resolve().parents[2]
PRAAT = np.load(ROOT / "tests" / "data" / "praat_formants_reference.npz")
FS = 16000


def autocorrelation(segment, order):
    """r[0..order] of a windowed segment, straight from the sum."""
    return np.array([segment[: len(segment) - lag] @ segment[lag:] for lag in range(order + 1)])


def emphasized(snd):
    return so.Sound(lfilter([1, -0.97], 1, snd.data, axis=0), snd.fs)


@pytest.fixture(scope="module")
def vowel():
    """The vowel of "hod" with Peterson and Barney's male formants, at 120 Hz."""
    return so.klatt_synthesize(0.5, FS, F0=120, F1=730, F2=1090, F3=2440, F4=3500, F5=4500)


def test_levinson_solves_the_normal_equations():
    rng = np.random.default_rng(0)
    segments = rng.standard_normal((3, 400)) * symmetric_hamming(400)
    r = np.array([autocorrelation(segment, 12) for segment in segments])
    predictors, reflections, error = so.levinson(r, 12)
    for row in range(3):
        direct = solve_toeplitz(r[row, :12], -r[row, 1:])
        assert np.abs(predictors[row, 1:] - direct).max() < 1e-12
        assert error[row] == pytest.approx(r[row, 0] + predictors[row, 1:] @ r[row, 1:], rel=1e-12)
    assert np.all(predictors[:, 0] == 1)
    assert np.abs(reflections).max() < 1


def test_levinson_of_silence_is_finite():
    predictors, reflections, error = so.levinson(np.zeros(6), 5)
    assert np.array_equal(predictors, [1, 0, 0, 0, 0, 0])
    assert np.array_equal(reflections, np.zeros(5)) and error == 0


def test_sound_analysis_is_the_autocorrelation_method(vowel):
    snd = emphasized(vowel)
    lpc = so.LPC(snd)
    assert lpc.order == 18 and lpc.data.shape[:2] == (1, 19)
    samples = snd.data[:, 0]
    window = symmetric_hamming(400)
    for index in (20, 30):
        start = round(lpc.t[index] * FS) - 200
        predictors, _, error = so.levinson(autocorrelation(samples[start : start + 400] * window, 18), 18)
        # The FFT route rounds differently, and a peaky vowel's normal equations amplify that.
        assert np.abs(lpc.data[0, :, index] - predictors).max() < 1e-6 * np.abs(predictors).max()
        assert lpc.error_power[0, index] == pytest.approx(error, rel=1e-6)


def test_stft_needs_room_for_the_lags(vowel):
    with pytest.raises(ValueError, match="n_fft"):
        so.LPC(so.STFT(vowel, frame=so.GaborFrame(0.025, 0.01)), order=18)
    frame = so.GaborFrame(0.025, 0.01, window=symmetric_hamming, n_fft=512)
    assert np.allclose(so.LPC(so.STFT(vowel, frame=frame), order=18).data, so.LPC(vowel).data)


def test_every_root_is_inside_the_unit_circle():
    snd = so.Sound(PRAAT["slt_samples_int16"] / 32768.0, int(PRAAT["slt_fs"]))
    lpc = so.LPC(emphasized(snd))
    assert np.abs(lpc.reflection).max() < 1
    _, bandwidths = lpc.candidates()
    assert np.nanmin(bandwidths) > 0  # a root on or outside the circle would give a bandwidth <= 0


def test_candidates_hold_the_formants(vowel):
    lpc = so.LPC(emphasized(vowel))
    freqs, _ = lpc.candidates()
    middle = freqs[0, :, 20]
    for formant in (730, 1090, 2440):
        assert np.nanmin(np.abs(middle - formant)) < 0.03 * formant


def test_envelope_matches_the_power_spectrum_on_average(vowel):
    lpc = so.LPC(emphasized(vowel))
    power = np.abs(lpc.source.data[0, :, 20]) ** 2
    model = lpc.envelope(lpc.source.f)[0, :, 20]
    ratio = power / model
    full_mean = (ratio[0] + ratio[-1] + 2 * ratio[1:-1].sum()) / lpc.n_fft  # over all n_fft bins
    assert full_mean == pytest.approx(1, rel=1e-6)


def test_envelope_view_reads_like_any_envelope(vowel):
    view = so.LPC(emphasized(vowel)).envelope_view()
    assert isinstance(view, so.GridEnvelope)
    warped = so.warp_frequency(view, 1.2)
    assert np.all(np.isfinite(warped(np.array([0.2]), np.array([500.0, 1000.0]))))


def test_silence_gives_finite_output():
    lpc = so.LPC(so.silence(0.2, FS))
    freqs, bandwidths = lpc.candidates()
    assert np.isnan(freqs).all() and np.isnan(bandwidths).all()
    assert np.all(lpc.error_power == 0)
    track = so.formant_track(so.silence(0.2, FS))
    assert np.isnan(track.frequencies).all()


@pytest.mark.parametrize(
    "f0, ceiling, formants, higher, tolerance",
    [(120, 5000, (730, 1090, 2440), (3500, 4500), 25), (220, 5500, (850, 1220, 2810), (4100, 4900), 40)],
)
def test_formant_track_recovers_synthesized_formants(f0, ceiling, formants, higher, tolerance):
    f1, f2, f3 = formants
    snd = so.klatt_synthesize(0.5, FS, F0=f0, F1=f1, F2=f2, F3=f3, F4=higher[0], F5=higher[1])
    track = so.formant_track(snd, ceiling=ceiling)
    steady = (track.t > 0.1) & (track.t < 0.4)
    error = np.abs(track.frequencies[0][:, steady] - np.array(formants)[:, None])
    assert error.max() < tolerance


@pytest.mark.parametrize("speaker", ["bdl", "slt"])
def test_formant_track_matches_praat(speaker):
    snd = so.Sound(PRAAT[f"{speaker}_samples_int16"] / 32768.0, int(PRAAT[f"{speaker}_fs"]))
    track = so.formant_track(snd, ceiling=float(PRAAT[f"{speaker}_ceiling"]))
    times = PRAAT[f"{speaker}_times"]
    index = np.searchsorted(np.round(track.t, 9), np.round(times, 9))
    assert np.allclose(track.t[index], times)
    compared = PRAAT[f"{speaker}_voiced"] & (times > 0.03) & (times < 0.97)
    difference = np.abs(track.frequencies[0][:, index] - PRAAT[f"{speaker}_formants"])[:, compared]
    assert np.all(np.nanmedian(difference, axis=1) < 10)


def test_tracker_skips_a_spurious_resonance():
    n_windows = 9
    freqs = np.tile(np.array([[500.0], [1500.0], [2500.0], [3500.0], [np.nan]]), n_windows)
    freqs[:, 4] = [500.0, 1000.0, 1500.0, 2500.0, 3500.0]  # a narrow root between F1 and F2
    bandwidths = np.where(np.isnan(freqs), np.nan, 80.0)
    tracked, _ = _track(freqs, bandwidths, (500, 1500, 2500), 1.0, 1.0, 2.0, 5.0)
    assert np.allclose(tracked[:, 4], [500, 1500, 2500])
    assert np.allclose(tracked, np.array([[500.0], [1500.0], [2500.0]]))


def test_track_feeds_klatt_synthesize(vowel):
    track = so.formant_track(vowel)
    times, values = track.track(1)
    assert len(times) == len(values) and np.all(np.isfinite(values))
    copy = so.klatt_synthesize(0.5, FS, F0=120, F1=track.track(1), F2=track.track(2), F3=track.track(3))
    assert np.all(np.isfinite(copy.data))


def test_plots_draw():
    import matplotlib

    matplotlib.use("Agg")
    snd = so.Sound(PRAAT["bdl_samples_int16"] / 32768.0, 16000)
    so.LPC(emphasized(snd)).plot()
    so.formant_track(snd).plot(candidates=True)
