from pathlib import Path

import numpy as np
import pytest

import sonore as so

FS = 16000.0
SPEECH = Path(__file__).resolve().parents[2] / "docs" / "speech"
FORMANTS = [(730, 60), (1090, 100), (2440, 120), (3400, 175)]


def vowel_gain(freqs):
    z = np.exp(-2j * np.pi * np.asarray(freqs) / FS)
    h = np.ones_like(z)
    for f, bw in FORMANTS:
        r = np.exp(-np.pi * bw / FS)
        h = h / (1 - 2 * r * np.cos(2 * np.pi * f / FS) * z + r * r * z * z)
    return np.abs(h)


def vowel(f0):
    """Harmonics of a per-sample F0 contour with /a/ gains, unit RMS. The
    fundamental is about 26 dB below the harmonics near F1, as in speech."""
    phase = 2 * np.pi * np.cumsum(f0) / FS
    y = np.zeros(len(f0))
    for k in range(1, int(0.95 * FS / 2 / f0.min()) + 1):
        y += vowel_gain(k * f0) * (k * f0 < 0.95 * FS / 2) * np.cos(k * phase)
    return y / np.sqrt(np.mean(y**2))


def errors(track, f0):
    """Largest relative error [%] and number of unvoiced time windows, away from the ends."""
    truth = np.interp(track.t, np.arange(len(f0)) / FS, f0)
    inner = (track.t > 0.05) & (track.t < len(f0) / FS - 0.05)
    est = track.f0[0][inner]
    return np.max(np.abs(est[est > 0] / truth[inner][est > 0] - 1)) * 100, int(np.sum(est == 0))


N = int(0.3 * FS)
TT = np.arange(N) / FS


@pytest.mark.parametrize("f0", [80.0, 200.0, 350.0])
def test_steady_vowels_are_tracked_exactly(f0):
    err, unvoiced = errors(so.f0_track(so.Sound(vowel(np.full(N, f0)), FS)), np.full(N, f0))
    assert unvoiced == 0
    assert err < 0.01


@pytest.mark.parametrize("f0", [250.0, 300.0, 400.0])
def test_steady_high_vowels_are_not_tracked_at_a_subharmonic(f0):
    # A perfectly periodic sound has a difference-function minimum at every
    # multiple of its period; the period itself must stay among the
    # candidates, and its multiples (F0/2, F0/3, ...) must not win.
    gains = [(436, 60), (2761, 100), (3372, 120), (4100, 175)]  # female heed (Hillenbrand et al. 1995)
    phase = 2 * np.pi * f0 * TT
    y = np.zeros(N)
    for k in range(1, int(0.95 * FS / 2 / f0) + 1):
        z = np.exp(-2j * np.pi * k * f0 / FS)
        h = 1.0
        for f, bw in gains:
            r = np.exp(-np.pi * bw / FS)
            h = h / (1 - 2 * r * np.cos(2 * np.pi * f / FS) * z + r * r * z * z)
        y += np.abs(h) * np.cos(k * phase)
    err, unvoiced = errors(so.f0_track(so.Sound(y, FS)), np.full(N, f0))
    assert unvoiced == 0
    assert err < 0.01


@pytest.mark.parametrize(
    "contour",
    [100 * 2**TT, 150 * (1 + 0.06 * np.sin(2 * np.pi * 5.5 * TT))],
    ids=["glide 1 oct/s", "vibrato"],
)
def test_moving_f0_is_tracked_within_a_fifth_of_a_percent(contour):
    err, unvoiced = errors(so.f0_track(so.Sound(vowel(contour), FS)), contour)
    assert unvoiced == 0
    assert err < 0.2


def test_missing_fundamental_is_tracked():
    from scipy.signal import butter, sosfiltfilt

    contour = 150 * (1 + 0.06 * np.sin(2 * np.pi * 5.5 * TT))
    x = sosfiltfilt(butter(8, 300, "highpass", fs=FS, output="sos"), vowel(contour))
    err, unvoiced = errors(so.f0_track(so.Sound(x, FS)), contour)
    assert unvoiced == 0
    assert err < 0.2


def test_score_in_noise_follows_the_power_ratio():
    from sonore.views.f0 import _periodicity

    x = vowel(np.full(N, 120.0))
    noise = np.random.default_rng(0).standard_normal(N)
    tc = np.arange(0.05, 0.25, 0.005)
    for snr_db in (10, 0):
        s = 10 ** (snr_db / 10)
        r = _periodicity(x + noise / np.sqrt(s), tc, np.full(len(tc), 120.0), FS)
        assert abs(np.median(r) - s / (1 + s)) < 0.03


def test_noise_and_silence_are_unvoiced():
    noise = np.random.default_rng(2).standard_normal(N)
    assert not so.f0_track(so.Sound(noise, FS)).voiced.any()
    with np.errstate(all="raise"):
        trk = so.f0_track(so.Sound(np.zeros(N), FS))
    assert not trk.voiced.any()
    assert np.all(trk.score == 0)


def test_channels_are_tracked_separately():
    a, b = vowel(np.full(N, 110.0)), vowel(np.full(N, 220.0))
    trk = so.f0_track(so.Sound(np.column_stack([a, b]), FS))
    assert trk.f0.shape == (2, len(trk.t)) == trk.score.shape
    assert trk.candidates.shape == (2, len(trk.t), 8)
    inner = (trk.t > 0.05) & (trk.t < 0.25)
    np.testing.assert_allclose(trk.f0[:, inner] / [[110.0], [220.0]], 1, rtol=1e-4)


def test_time_windows_and_arguments():
    trk = so.f0_track(so.Sound(vowel(np.full(N, 120.0)), FS), hop=0.01)
    np.testing.assert_allclose(trk.t, np.arange(0, 0.3, 0.01))
    bad = ({"f_lo": 0}, {"f_lo": 300, "f_hi": 200}, {"f_hi": 9000}, {"hop": 0}, {"subharmonic_margin": -0.1})
    for kw in bad:
        with pytest.raises(ValueError):
            so.f0_track(so.Sound(np.zeros(100), FS), **kw)


def test_subharmonic_rule_can_be_turned_off():
    """Anything periodic at 220 Hz is periodic at 110 Hz too. When the 110 Hz
    candidate scores a little higher, the rule still picks 220 Hz; without
    the rule the higher score wins."""
    from sonore.views.f0 import _viterbi

    cand = np.tile([110.0, 220.0, np.nan, np.nan], (20, 1))
    score = np.tile([0.92, 0.9, np.nan, np.nan], (20, 1))
    on, _ = _viterbi(cand, score, 0.5, 2.0, 0.5, 0.05)
    off, _ = _viterbi(cand, score, 0.5, 2.0, 0.5, None)
    assert np.all(on == 220) and np.all(off == 110)


@pytest.mark.skipif(not SPEECH.exists(), reason="docs are not in the sdist")
def test_sentence_agrees_with_harvest_where_both_voice():
    snd = so.load(SPEECH / "bdl_arctic_a0131.flac")
    trk = so.f0_track(snd)
    ref = np.loadtxt(SPEECH / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    h = np.interp(trk.t, ref[:, 0], ref[:, 1])
    hv = np.interp(trk.t, ref[:, 0], (ref[:, 1] > 0).astype(float)) > 0.5
    v = trk.voiced[0]
    both = v & hv
    # 98.6% with the two-period score (99.7% with three periods), which voices
    # 83% of Harvest's time windows instead of 79%.
    assert np.mean(np.abs(trk.f0[0][both] / h[both] - 1) < 0.05) >= 0.98
    assert not np.any(v & ~hv)
    assert np.mean(v[hv]) > 0.8


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    trk = so.f0_track(so.Sound(vowel(np.full(N // 3, 120.0)), FS))
    ax = trk.plot(candidates=True)
    assert ax.get_ylabel() == "F0 [Hz]"
    assert "voiced" in repr(trk)


# ---------------------------------------------------------------- audit gaps
HEED = [(436, 60), (2761, 100), (3372, 120), (4100, 175)]  # female heed (Hillenbrand et al. 1995)


def heed(f0, snr_db, seed=0):
    """A steady female /i/ at f0 in white noise snr_db below it."""
    phase = 2 * np.pi * f0 * TT
    y = np.zeros(N)
    for k in range(1, int(0.95 * FS / 2 / f0) + 1):
        z = np.exp(-2j * np.pi * k * f0 / FS)
        h = 1.0
        for f, bw in HEED:
            r = np.exp(-np.pi * bw / FS)
            h = h / (1 - 2 * r * np.cos(2 * np.pi * f / FS) * z + r * r * z * z)
        y += np.abs(h) * np.cos(k * phase)
    noise = np.random.default_rng(seed).standard_normal(N)
    return so.Sound(y / np.std(y) + noise * 10 ** (-snr_db / 20), FS)


@pytest.mark.parametrize("f0", [250.0, 300.0, 350.0])
def test_noisy_high_vowels_are_not_tracked_at_a_third(f0):
    # In noise the candidates at F0/2 and F0/3 score about as well as F0; the
    # rule must look at every whole multiple above a candidate, not only the
    # octave (which leaves F0/3 standing), and is needed at all here.
    err, unvoiced = errors(so.f0_track(heed(f0, 20)), np.full(N, f0))
    assert unvoiced == 0
    assert err < 2
    err_off, _ = errors(so.f0_track(heed(f0, 20), subharmonic_margin=None), np.full(N, f0))
    assert err_off > 40


def test_unvoiced_cost_is_one_minus_threshold():
    # A vowel at 0 dB SNR scores about 0.5: voiced at the default threshold,
    # unvoiced at 0.95, voiced again at 0.05.
    x = so.Sound(vowel(np.full(N, 150.0)) + np.random.default_rng(1).standard_normal(N), FS)
    inner = slice(10, -10)
    assert np.mean(so.f0_track(x, threshold=0.95).voiced[0][inner]) == 0
    assert np.mean(so.f0_track(x, threshold=0.05).voiced[0][inner]) == 1


def test_chosen_score_is_the_chosen_candidates():
    trk = so.f0_track(so.Sound(vowel(150 * 2**TT), FS))
    for i in np.flatnonzero(trk.voiced[0]):
        j = np.nanargmin(np.abs(trk.candidates[0, i] - trk.f0[0, i]))
        assert trk.candidates[0, i, j] == trk.f0[0, i]
        assert trk.score[0, i] == trk.candidate_scores[0, i, j]


def test_subharmonic_rule_reaches_a_near_multiple():
    # 224 Hz is 1.8% from twice 110 Hz, within the rule's 5%: 110 is ruled out.
    from sonore.views.f0 import _viterbi

    cand = np.tile([110.0, 224.0, np.nan, np.nan], (20, 1))
    score = np.tile([0.92, 0.9, np.nan, np.nan], (20, 1))
    on, chosen_score = _viterbi(cand, score, 0.5, 2.0, 0.5, 0.05)
    assert np.all(on == 224)
    assert np.all(chosen_score == 0.9)  # the score of the candidate chosen, not of the first


@pytest.mark.parametrize(
    "f0, kwargs",
    [(60.1, {}), (490.0, {}), (35.0, {"f_lo": 30.0, "subharmonic_margin": None})],
    ids=["f_lo", "f_hi", "f_lo 30"],
)
def test_f0_near_the_ends_of_the_range(f0, kwargs):
    err, unvoiced = errors(so.f0_track(so.Sound(vowel(np.full(N, f0)), FS), **kwargs), np.full(N, f0))
    assert unvoiced == 0
    assert err < 0.2


def test_candidates_stay_near_the_range():
    # A sound below f_lo: refined candidates that leave [0.9 f_lo, 1.1 f_hi]
    # are dropped.
    trk = so.f0_track(so.harmonic_complex(0.3, FS, 45.0, harmonics=np.arange(1, 40)))
    assert np.nanmin(trk.candidates) >= 0.9 * 60
    assert np.nanmax(trk.candidates) <= 1.1 * 500


def test_raw_candidates_are_refined_by_a_parabola():
    from sonore.views.f0 import _candidates

    x = vowel(np.full(N, 123.4))  # a period of 129.66 samples
    t = np.arange(0.05, 0.25, 0.01)
    cand = _candidates(x, t, FS, 60.0, 500.0)
    assert np.nanmin(np.abs(cand / 123.4 - 1), axis=1).max() < 1e-4


def test_weak_periodicity_still_offers_its_period():
    # At -3 dB SNR the difference function's dip at the period is shallow
    # (above 0.5), yet it is offered, so the score decides.
    from sonore.views.f0 import _candidates

    x = vowel(np.full(N, 150.0)) + np.random.default_rng(3).standard_normal(N) * 10 ** (3 / 20)
    t = np.arange(0.05, 0.25, 0.005)
    cand = _candidates(x, t, FS, 60.0, 500.0)
    assert np.mean(np.any(np.abs(cand / 150 - 1) < 0.02, axis=1)) > 0.9


def test_score_is_exact_at_a_fractional_period():
    from sonore.views.f0 import _periodicity

    for f0 in (123.4, 333.3):  # flat harmonics up to 7 kHz
        x = so.harmonic_complex(0.3, FS, f0, harmonics=np.arange(1, int(7000 / f0))).data[:, 0]
        t = np.arange(0.05, 0.25, 0.01)
        assert _periodicity(x, t, np.full(len(t), f0), FS).min() > 1 - 1e-6


def test_harmonics_above_nyquist_are_left_out_of_refinement():
    from sonore.views.f0 import _refine

    # 1400 Hz: harmonics 6 (8400 Hz) lies above Nyquist and would alias.
    x = so.harmonic_complex(0.3, FS, 1400.0, harmonics=np.arange(1, 6)).data[:, 0]
    t = np.arange(0.05, 0.25, 0.01)
    refined = _refine(x, t, np.full(len(t), 1400.0), FS)
    assert np.max(np.abs(refined / 1400 - 1)) < 0.005


def test_many_time_windows():
    # More time windows than the candidate stage handles at once.
    n = int(2.2 * FS)
    trk = so.f0_track(so.Sound(vowel(np.full(n, 150.0)), FS), hop=0.001)
    assert len(trk.t) > 2048
    assert trk.voiced[0][50:-50].all()


@pytest.mark.parametrize("f0", [600.0, 800.0, 1000.0])
def test_singing_range(f0):
    # With f_hi raised, a high voice has more period multiples in the search
    # range than there are candidates; its period must still be offered.
    for sound in (so.Sound(vowel(np.full(N, f0)), FS), heed(f0, 40)):
        err, unvoiced = errors(so.f0_track(sound, f_hi=1200.0), np.full(N, f0))
        assert unvoiced == 0
        assert err < 0.2


def test_an_empty_sound_gives_an_empty_track():
    trk = so.f0_track(so.Sound(np.zeros(0), FS))
    assert trk.f0.shape == trk.score.shape == (1, 0)
    assert trk.candidates.shape == (1, 0, 8)
    assert "0 time windows" in repr(trk)


@pytest.mark.parametrize(
    "contour",
    [70 * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * TT)), 60 * 2**TT],
    ids=["vibrato 70 Hz", "glide 60-74 Hz"],
)
def test_moving_low_voices_keep_their_period(contour):
    # Below 100 Hz the power of the /a/ is in the harmonics near F1, the 10th
    # and up, which drift out of line with their copy one period later when
    # F0 moves within the score's stretch. With three periods the vibrato
    # was unvoiced or at a sixth of the period on 62% of time windows.
    err, unvoiced = errors(so.f0_track(so.Sound(vowel(contour), FS)), contour)
    assert unvoiced == 0
    assert err < 0.5


def test_low_voice_in_noise_is_refined_from_its_strong_harmonics():
    # At 60 Hz the first six harmonics hold 0.6% of the /a/'s power; refined
    # from them alone in 10 dB noise, the F0 came out a percent off and the
    # true period scored below 0.5 (81% of time windows wrong).
    from sonore.views.f0 import _refine

    x = vowel(np.full(N, 60.0)) + np.random.default_rng(4).standard_normal(N) * 10 ** (-10 / 20)
    t = np.arange(0.06, 0.24, 0.005)
    assert np.max(np.abs(_refine(x, t, np.full(len(t), 60.0), FS) / 60 - 1)) < 0.001
    err, unvoiced = errors(so.f0_track(so.Sound(x, FS)), np.full(N, 60.0))
    assert unvoiced == 0
    assert err < 1
