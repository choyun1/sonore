from pathlib import Path

import numpy as np
import pytest

import sonore as so

FS = 16000.0
FORMANTS = [(730, 60), (1090, 100), (2440, 120)]


def vowel_response(freqs):
    z = np.exp(-2j * np.pi * np.asarray(freqs) / FS)
    h = np.ones_like(z)
    for f, bw in FORMANTS:
        r = np.exp(-np.pi * bw / FS)
        h = h / (1 - 2 * r * np.cos(2 * np.pi * f / FS) * z + r * r * z * z)
    return h


def vowel(f0, dur=0.3):
    """A band-limited pulse train at f0 through three formants of /a/."""
    t = np.arange(int(dur * FS)) / FS
    harm = np.arange(1, int(0.95 * FS / 2 // f0) + 1) * f0
    h = vowel_response(harm)
    x = np.sum(np.abs(h)[:, None] * np.cos(2 * np.pi * harm[:, None] * t + np.angle(h)[:, None]), axis=0)
    return so.Sound(x / np.abs(x).max() * 0.5, FS)


@pytest.fixture(scope="module")
def noise():
    rng = np.random.default_rng(1)
    return so.Sound(rng.standard_normal((4000, 2)) * [1.0, 0.1], FS)


def test_round_trip_stft(noise):
    cep = so.Cepstrum(so.STFT(noise, win_dur=0.02))
    assert cep.data.shape == (2, cep.n_fft // 2 + 1, len(cep.t))
    assert np.isrealobj(cep.data)
    assert np.allclose(cep.to_sound().data, noise.data, atol=1e-12)


@pytest.mark.parametrize("n_fft", [512, 513])
def test_round_trip_tvstft_even_and_odd_fft(noise, n_fft):
    frame = so.TVGaborFrame.from_function(lambda t: 0.01 + 0.04 * t, t_end=0.3, n_fft=n_fft)
    cep = so.Cepstrum(frame.analyze(noise))
    assert cep.n_fft == n_fft
    assert np.allclose(cep.envelope(), np.abs(cep.source.data), rtol=1e-12)
    assert np.allclose(cep.to_sound().data, noise.data, atol=1e-12)


def test_scaling_moves_only_c0(noise):
    a = so.Cepstrum(so.STFT(noise))
    b = so.Cepstrum(so.STFT(noise * 3.7))
    assert np.allclose(b.data[:, 0] - a.data[:, 0], np.log(3.7))
    assert np.allclose(b.data[:, 1:], a.data[:, 1:], atol=1e-12)


def test_minimum_phase_fold():
    """A minimum-phase FIR alone in a time window comes back with its phase; a
    mixed-phase one keeps only its magnitude."""
    roots = np.array([0.8 * np.exp(0.4j), 0.6 * np.exp(1.7j), 0.5 * np.exp(2.6j)])
    h_min = np.real(np.poly(np.concatenate([roots, roots.conj()])))
    frame = so.GaborFrame(256 / FS, 64 / FS, window="boxcar")
    x = np.zeros(1024)
    x[512 : 512 + len(h_min)] = h_min  # starts at time window 8's phase reference
    cep = so.Cepstrum(frame.analyze(so.Sound(x, FS)))
    i = int(np.argmin(np.abs(cep.t - 512 / FS)))
    got = cep.to_stft(phase="minimum").data[0, :, i]
    want = np.fft.rfft(h_min, 256)
    assert np.allclose(got, want, atol=1e-9 * np.abs(want).max())

    mixed = np.concatenate([roots, roots.conj()])
    mixed[[0, 3]] = 1 / mixed[[0, 3]].conj()
    x[512 : 512 + len(h_min)] = np.real(np.poly(mixed))
    cep = so.Cepstrum(frame.analyze(so.Sound(x, FS)))
    got = cep.to_stft(phase="minimum").data[0, :, i]
    assert np.allclose(np.abs(got), np.abs(cep.source.data[0, :, i]), rtol=1e-9)
    assert not np.allclose(got, cep.source.data[0, :, i], atol=1e-3)


@pytest.mark.parametrize("f0", [100.0, 200.0])
def test_f0_of_a_vowel(f0):
    snd = vowel(f0)
    cep = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.01))
    t, est, peak = cep.f0()
    inside = (t > 0.03) & (t < snd.duration - 0.03)
    assert est.shape == peak.shape == (1, len(t))
    assert np.all(np.abs(est[0, inside] / f0 - 1) < 0.01)


def test_f0_needs_three_periods():
    cep = so.Cepstrum(so.STFT(vowel(150.0), win_dur=0.030))
    with pytest.raises(ValueError, match="three periods"):
        cep.f0(f_lo=75)
    cep.f0(f_lo=101)


def test_silence_is_finite_and_unvoiced():
    x = np.concatenate([np.zeros(2000), vowel(120.0).data[:, 0]])
    cep = so.Cepstrum(so.STFT(so.Sound(x, FS), win_dur=0.040))
    assert np.all(np.isfinite(cep.data))
    _, est, _ = cep.f0()
    assert est[0, 0] == 0
    assert np.all(np.isfinite(so.Cepstrum(so.STFT(so.silence(0.1, FS))).data))


def test_lifter_splits_the_cepstrum(noise):
    cep = so.Cepstrum(so.STFT(noise))
    low, high = cep.lifter(2e-3), cep.lifter(2e-3, keep="high")
    assert np.allclose(low.data + high.data, cep.data)
    assert np.all(low.data[:, cep.q >= 2e-3] == 0)
    per_window = cep.lifter(np.full(len(cep.t), 2e-3))
    assert np.array_equal(per_window.data, low.data)
    with pytest.raises(ValueError, match="one value per time window"):
        cep.lifter(np.ones(3) * 1e-3)
    with pytest.raises(ValueError, match="keep"):
        cep.lifter(1e-3, keep="middle")


def test_low_lifter_follows_the_envelope_shape():
    f0 = 100.0
    snd = vowel(f0)
    frame = so.TVGaborFrame.pitch_adaptive([0.0, snd.duration], [f0, f0], t_end=snd.duration, n_fft=4096)
    cep = so.Cepstrum(frame.analyze(snd))
    env = cep.lifter(0.5 / f0).envelope()[0, :, len(cep.t) // 2]
    harm = np.arange(1, 40) * f0
    idx = np.round(harm / FS * 4096).astype(int)
    diff = 20 * np.log10(env[idx]) - 20 * np.log10(np.abs(vowel_response(harm)))
    assert np.std(diff) < 1.0


def test_plot(noise):
    import matplotlib

    matplotlib.use("Agg")
    ax = so.Cepstrum(so.STFT(noise, win_dur=0.04)).plot()
    assert ax.get_ylabel() == "Quefrency [ms]"


def test_the_floor_is_per_channel():
    """A stretch 260 dB below the loud part lies under the floor, which is set
    by the channel's loudest time window, so it reads flat."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(4000), rng.standard_normal(4000) * 1e-13])
    cep = so.Cepstrum(so.STFT(so.Sound(x, FS), win_dur=0.02))
    quiet = np.flatnonzero(cep.t > 0.3)
    assert np.abs(cep.data[0, 1:][:, quiet]).max() < 1e-12


@pytest.mark.parametrize("floor_db", [0.0, 10.0])
def test_a_floor_at_or_above_the_maximum_is_refused(noise, floor_db):
    with pytest.raises(ValueError, match="floor_db must be negative"):
        so.Cepstrum(so.STFT(noise), floor_db=floor_db)


@pytest.mark.parametrize("n_fft", [512, 513])
def test_minimum_phase_keeps_the_envelope_as_its_magnitude(noise, n_fft):
    cep = so.Cepstrum(so.STFT(noise, frame=so.GaborFrame(0.02, 0.005, n_fft=n_fft)))
    assert np.allclose(np.abs(cep.to_stft(phase="minimum").data), cep.envelope(), rtol=1e-9)


def test_a_liftered_cepstrum_keeps_the_source_phase_on_its_own_envelope(noise):
    cep = so.Cepstrum(so.STFT(noise)).lifter(2e-3)
    coefs = cep.to_stft()
    assert np.allclose(np.abs(coefs.data), cep.envelope(), rtol=1e-12)
    assert np.allclose(np.angle(coefs.data), np.angle(cep.source.data), atol=1e-12)


@pytest.mark.parametrize(("f0", "f_lo", "f_hi"), [(400.0, 75.0, 400.0), (80.0, 80.0, 400.0)])
def test_f0_at_either_end_of_the_search_range(f0, f_lo, f_hi):
    snd = vowel(f0, dur=0.4)
    t, est, _ = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.01)).f0(f_lo=f_lo, f_hi=f_hi)
    inside = (t > 0.03) & (t < snd.duration - 0.03)
    assert np.all(np.abs(est[0, inside] / f0 - 1) < 0.002)


@pytest.mark.parametrize("f0", [130.0, 260.0])
def test_f0_between_whole_sample_periods(f0):
    """The parabola through the peak refines the period between samples
    (measured within 0.04% at 130 Hz and 0.07% at 260 Hz; the nearest whole
    sample alone is up to 0.8% off at 260 Hz)."""
    snd = vowel(f0)
    t, est, _ = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.01)).f0()
    inside = (t > 0.03) & (t < snd.duration - 0.03)
    assert np.all(np.abs(est[0, inside] / f0 - 1) < 0.002)


def test_the_voicing_threshold():
    cep = so.Cepstrum(so.STFT(vowel(150.0), win_dur=0.040, hop_dur=0.01))
    _, _, peak = cep.f0()
    window = len(cep.t) // 2
    _, at_peak, _ = cep.f0(threshold=peak[0, window])
    _, above, _ = cep.f0(threshold=np.nextafter(peak[0, window], np.inf))
    assert at_peak[0, window] > 0
    assert above[0, window] == 0


def test_refusals_and_repr(noise):
    with pytest.raises(TypeError, match="STFT or TVSTFT"):
        so.Cepstrum(noise)
    cep = so.Cepstrum(so.STFT(noise, win_dur=0.04))
    with pytest.raises(ValueError, match="phase"):
        cep.to_stft(phase="zero")
    for f_lo, f_hi in ((0.0, 400.0), (400.0, 100.0), (75.0, FS)):
        with pytest.raises(ValueError, match="f_lo < f_hi"):
            cep.f0(f_lo=f_lo, f_hi=f_hi)
    n_quefrencies = cep.n_fft // 2 + 1
    assert repr(cep).startswith(f"Cepstrum({n_quefrencies} quefrencies x {len(cep.t)} time windows, 2 ch")


SPEECH = Path(__file__).resolve().parents[2] / "docs" / "speech"


@pytest.mark.skipif(not SPEECH.exists(), reason="docs are not in the sdist")
def test_f0_agrees_with_harvest_as_the_docstring_says():
    """The docstring's numbers on the gallery sentence (measured 78.5% within
    5% of Harvest, and 95.6% where the peak exceeds 0.1)."""
    snd = so.load(SPEECH / "bdl_arctic_a0131.flac")
    reference = np.loadtxt(SPEECH / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    cep = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.005))
    t, est, peak = cep.f0(threshold=0)
    on_grid = np.searchsorted(np.round(t, 6), np.round(reference[:, 0], 6))
    assert np.allclose(t[on_grid], reference[:, 0])
    harvest, est, peak = reference[:, 1], est[0, on_grid], peak[0, on_grid]
    inside = (reference[:, 0] >= 0.02) & (reference[:, 0] <= snd.duration - 0.02)
    voiced = inside & (harvest > 0)
    agrees = np.abs(est / np.where(voiced, harvest, 1) - 1) < 0.05
    assert voiced.sum() == 432
    assert np.mean(agrees[voiced]) > 0.77
    assert np.mean(agrees[voiced & (peak > 0.1)]) > 0.94
