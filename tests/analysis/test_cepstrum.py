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
    got = cep.to_stft("minimum").data[0, :, i]
    want = np.fft.rfft(h_min, 256)
    assert np.allclose(got, want, atol=1e-9 * np.abs(want).max())

    mixed = np.concatenate([roots, roots.conj()])
    mixed[[0, 3]] = 1 / mixed[[0, 3]].conj()
    x[512 : 512 + len(h_min)] = np.real(np.poly(mixed))
    cep = so.Cepstrum(frame.analyze(so.Sound(x, FS)))
    got = cep.to_stft("minimum").data[0, :, i]
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
