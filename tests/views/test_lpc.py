"""so.LPC, so.levinson and so.formant_track against their formulas, Praat and known formants."""

import itertools
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
    assert isinstance(view, so.SpectralEnvelope)
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


def test_reflection_is_the_last_coefficient_at_each_order():
    rng = np.random.default_rng(1)
    segment = rng.standard_normal(400) * symmetric_hamming(400)
    r = np.array([segment[: 400 - lag] @ segment[lag:] for lag in range(9)])
    _, reflections, _ = so.levinson(r, 8)
    for order in range(1, 9):
        predictors, _, _ = so.levinson(r[: order + 1], order)
        assert reflections[order - 1] == pytest.approx(predictors[-1], abs=1e-12)


def test_poles_of_an_all_pole_response_come_back_exactly():
    """The whole impulse response inside one rectangular window: its autocorrelation is
    the all-pole model's own, so the roots are the resonators' poles."""
    impulse = np.zeros(FS)
    impulse[FS // 2 - 2000] = 1.0
    response = impulse
    for freq, bandwidth in ((1000, 100), (2500, 200)):
        radius, angle = np.exp(-np.pi * bandwidth / FS), 2 * np.pi * freq / FS
        response = lfilter([1], [1, -2 * radius * np.cos(angle), radius**2], response)
    frame = so.GaborFrame(0.5, 0.25, window="boxcar", n_fft=8192)
    lpc = so.LPC(so.STFT(so.Sound(response, FS), frame=frame), order=4)
    freqs, bandwidths = lpc.candidates()
    assert np.allclose(freqs[0, :, 2], [1000, 2500], atol=1e-6)
    assert np.allclose(bandwidths[0, :, 2], [100, 200], atol=1e-6)


def test_sound_fft_leaves_room_for_the_lags(vowel):
    lpc = so.LPC(vowel, win_dur=0.031875)  # 510 samples plus 18 lags pass 512
    assert lpc.n_fft == 1024


def test_stft_fft_of_exactly_window_plus_order_is_enough(vowel):
    ok = so.GaborFrame(0.025, 0.01, window=symmetric_hamming, n_fft=418)
    assert np.allclose(so.LPC(so.STFT(vowel, frame=ok), order=18).data, so.LPC(vowel).data)
    short = so.GaborFrame(0.025, 0.01, window=symmetric_hamming, n_fft=417)
    with pytest.raises(ValueError, match="n_fft"):
        so.LPC(so.STFT(vowel, frame=short), order=18)


def test_odd_fft_length(vowel):
    odd = so.GaborFrame(0.025, 0.01, window=symmetric_hamming, n_fft=419)
    assert np.allclose(so.LPC(so.STFT(vowel, frame=odd), order=18).data, so.LPC(vowel).data)


def test_stft_default_order(vowel):
    frame = so.GaborFrame(0.025, 0.01, window=symmetric_hamming, n_fft=512)
    assert so.LPC(so.STFT(vowel, frame=frame)).order == 18


def test_tvstft_needs_room_for_its_longest_window(vowel):
    frame = so.TVGaborFrame([0.1, 0.2, 0.3], [0.02, 0.03, 0.025], window=symmetric_hamming, n_fft=480 + 17)
    with pytest.raises(ValueError, match="longest window \\(480\\)"):
        so.LPC(frame.analyze(vowel), order=18)
    roomy = so.TVGaborFrame([0.1, 0.2, 0.3], [0.02, 0.03, 0.025], window=symmetric_hamming, n_fft=480 + 18)
    assert so.LPC(roomy.analyze(vowel), order=18).data.shape == (1, 19, 3)


def test_formant_candidates_stay_inside_50_hz_of_the_band(vowel):
    t = np.arange(vowel.n_samples) / FS
    peak = np.abs(vowel.data).max()
    for hum in (40.0, 4980.0):
        amplitude = 100 if hum < 100 else 3
        snd = so.Sound(vowel.data[:, 0] + amplitude * peak * np.sin(2 * np.pi * hum * t), FS)
        track = so.formant_track(snd)
        assert np.nanmin(track.candidates) > 50 and np.nanmax(track.candidates) < 4950


def path_cost(path, freqs, widths, skipped, nominal, weights):
    nominal_weight, bandwidth_weight, skip_cost, transition_weight = weights
    total = 0.0
    for window, state in enumerate(path):
        values, bandwidths = freqs[window][state], widths[window][state]
        filled = np.isfinite(values)
        total += np.sum(nominal_weight * np.abs(np.log(values[filled] / nominal[filled])))
        total += np.sum(bandwidth_weight * bandwidths[filled] / values[filled])
        total += skip_cost * skipped[window][state]
        if window:
            before = freqs[window - 1][path[window - 1]]
            both = filled & np.isfinite(before)
            total += transition_weight * np.sum(np.abs(np.log(values[both] / before[both])))
    return total


def states(candidates, bandwidths):
    """The docstring's states, written out: increasing choices of three, or of all when fewer,
    filling F1 first; and how many candidates each skips below its highest."""
    count = min(3, len(candidates))
    values, widths, skipped = [], [], []
    for chosen in itertools.combinations(range(len(candidates)), count):
        values.append(np.r_[candidates[list(chosen)], np.full(3 - count, np.nan)])
        widths.append(np.r_[bandwidths[list(chosen)], np.full(3 - count, np.nan)])
        skipped.append((chosen[-1] + 1 - count) if count else 0)
    return np.array(values), np.array(widths), skipped


@pytest.mark.parametrize("seed", range(40))
def test_tracker_finds_the_cheapest_path(seed):
    rng = np.random.default_rng(seed)
    counts = [5, 2, 0, 4] if seed % 4 == 0 else list(rng.integers(3, 6, 4))
    n_windows = len(counts)
    freqs = np.full((5, n_windows), np.nan)
    bandwidths = np.full((5, n_windows), np.nan)
    for window, count in enumerate(counts):
        freqs[:count, window] = np.sort(rng.uniform(200, 4500, count))
        bandwidths[:count, window] = rng.uniform(30, 600, count)
    nominal = np.array([500.0, 1500.0, 2500.0])
    weights = tuple(rng.uniform(0.2, 5.0, 4))
    per_window = [states(freqs[: counts[w], w], bandwidths[: counts[w], w]) for w in range(n_windows)]
    all_freqs = [s[0] for s in per_window]
    all_widths = [s[1] for s in per_window]
    all_skipped = [s[2] for s in per_window]
    best = min(
        itertools.product(*[range(len(f)) for f in all_freqs]),
        key=lambda path: path_cost(path, all_freqs, all_widths, all_skipped, nominal, weights),
    )
    tracked, tracked_widths = _track(freqs, bandwidths, nominal, *weights)
    expected = np.array([all_freqs[w][state] for w, state in enumerate(best)]).T
    assert np.array_equal(np.isnan(tracked), np.isnan(expected))
    assert np.allclose(tracked[np.isfinite(tracked)], expected[np.isfinite(expected)])


def test_track_leaves_out_missing_windows():
    freqs = np.array([[600.0, 600.0, 600.0], [1500.0, np.nan, 1500.0], [2500.0, np.nan, 2500.0]])
    track = so.FormantTrack(
        t=np.array([0.0, 0.005, 0.01]),
        frequencies=freqs[None],
        bandwidths=np.full((1, 3, 3), 80.0),
        candidates=freqs[None],
        candidate_bandwidths=np.full((1, 3, 3), 80.0),
        ceiling=5000.0,
    )
    times, values = track.track(2)
    assert np.allclose(times, [0.0, 0.01]) and np.allclose(values, 1500)


def test_formant_candidates_are_sorted_with_nan_last(vowel):
    t = np.arange(vowel.n_samples) / FS
    snd = so.Sound(vowel.data[:, 0] + 100 * np.abs(vowel.data).max() * np.sin(2 * np.pi * 40 * t), FS)
    candidates = so.formant_track(snd).candidates[0]
    for column in candidates.T:
        finite = np.isfinite(column)
        assert np.all(np.diff(column[finite]) > 0)
        assert not np.any(finite[np.argmin(finite) :]) or finite.all()


def test_refusals(vowel):
    stft = so.STFT(vowel, frame=so.GaborFrame(0.025, 0.01, window=symmetric_hamming, n_fft=512))
    with pytest.raises(TypeError, match="win_dur and hop_dur"):
        so.LPC(stft, win_dur=0.02)
    with pytest.raises(TypeError, match="expected a Sound"):
        so.LPC(vowel.data)
    with pytest.raises(ValueError, match="order must be at least 1"):
        so.LPC(vowel, 0)
    with pytest.raises(ValueError, match="ceiling"):
        so.formant_track(vowel, ceiling=8000)
    with pytest.raises(ValueError, match=r"lags 0 to 5, not only 3"):
        so.levinson(np.array([3.0, 1.0, 0.5]), 5)
    with pytest.raises(ValueError, match="order must be 0 or more"):
        so.levinson(np.array([3.0, 1.0, 0.5]), -1)
    for n_samples in (0, 199):
        with pytest.raises(ValueError, match="shorter than half the window"):
            so.formant_track(so.Sound(np.ones((n_samples, 1)), FS))
    assert so.formant_track(so.Sound(np.ones(200), FS)).frequencies.shape[:2] == (1, 3)


def test_reprs():
    assert repr(so.LPC(so.silence(0.1, FS))) == "LPC(order 18 x 13 time windows, 1 ch, fs 16000 Hz)"
    assert repr(so.formant_track(so.silence(0.2, FS))) == (
        "FormantTrack(45 time windows, 1 ch, ceiling 5000 Hz, medians F1 none, F2 none, F3 none Hz)"
    )


def test_formant_track_times_are_rounded_to_samples(vowel):
    assert np.allclose(np.diff(so.formant_track(vowel, ceiling=5250).t), 52 / 10500)
