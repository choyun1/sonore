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
    gains = [(436, 60), (2761, 100), (3372, 120), (4100, 175)]  # women's heed (Hillenbrand et al. 1995)
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
    from sonore.analysis.f0 import _periodicity

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
    from sonore.analysis.f0 import _viterbi

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
    assert np.mean(np.abs(trk.f0[0][both] / h[both] - 1) < 0.05) >= 0.99
    assert not np.any(v & ~hv)
    assert np.mean(v[hv]) > 0.75


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    trk = so.f0_track(so.Sound(vowel(np.full(N // 3, 120.0)), FS))
    ax = trk.plot(candidates=True)
    assert ax.get_ylabel() == "F0 [Hz]"
    assert "voiced" in repr(trk)
