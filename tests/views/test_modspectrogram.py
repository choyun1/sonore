import gc
import shutil

import numpy as np
import pytest

import sonore as so

FS = 16000.0
FE = 1000.0


def am_tone(rate=4.0, depth=0.5, dur=3.0, on=(1.0, 2.0), fc=1000.0):
    t = np.arange(int(dur * FS)) / FS
    gate = (t >= on[0]) & (t < on[1])
    return so.Sound(0.1 * (1 + depth * gate * np.sin(2 * np.pi * rate * t)) * np.sin(2 * np.pi * fc * t), FS)


@pytest.fixture(scope="module")
def tone_env():
    fb = so.ERBFilterbank(n_bands=24, f_lo=100, f_hi=7000)
    return fb.analyze(am_tone()).envelopes(fs=FE)


@pytest.fixture(scope="module")
def tone_msg(tone_env):
    return so.ModulationSpectrogram(tone_env, f_lo=2.0, f_hi=32.0)


def test_bank_ignores_the_mean():
    for bank in (so.HannModulationFilterbank(), so.HannModulationFilterbank(window=1.0, f_hi=32)):
        dc = [abs(h.sum()) for h, _ in bank.kernels(FE)]
        assert 20 * np.log10(max(dc)) < -60
        assert np.all(bank.response([0.0]) < 1e-12)


def test_bank_layout():
    cq = so.HannModulationFilterbank()
    assert cq.n_bands == 15 and cq.cfs[0] == 0.5 and np.isclose(cq.cfs[-1], 64)
    assert np.allclose(cq.durations(), 3 / cq.cfs)
    fixed = so.HannModulationFilterbank(window=0.5, f_lo=1.0, f_hi=10.0)
    assert np.allclose(fixed.cfs, [4, 6, 8, 10])  # from 2 / T, steps of 1 / T
    with pytest.raises(ValueError, match="whole number"):
        so.HannModulationFilterbank(cycles=2.5)
    with pytest.raises(ValueError, match="whole number"):
        so.HannModulationFilterbank(cycles=1)


def test_filter_matches_response():
    bank = so.HannModulationFilterbank(f_lo=2.0, f_hi=32.0)
    t = np.arange(int(8 * FE)) / FE
    for f in (3.0, 4.0, 9.0):
        y = bank.filter(np.cos(2 * np.pi * f * t), FE, analytic=True)
        assert np.allclose(np.abs(y[len(t) // 2]), bank.response([f])[0], atol=5e-3)


def test_causal_is_centred_delayed():
    bank = so.HannModulationFilterbank(f_lo=2.0, f_hi=32.0)
    x = np.random.default_rng(0).random(3000)
    c = bank.filter(x, FE, analytic=True, align="causal")
    z = bank.filter(x, FE, analytic=True, align="center")
    for k, n in enumerate(bank.lengths(FE)):
        d = (n - 1) // 2
        assert np.allclose(c[d:, k], z[: len(x) - d, k], atol=1e-12)


def test_am_tone_in_the_right_cell(tone_msg):
    msg = tone_msg
    c, b, k, f = msg.power.shape
    assert (c, b, k) == (1, 24, 9) and f == len(msg.t) == 300
    band = int(np.argmin(np.abs(msg.f - 1000)))
    rate = int(np.flatnonzero(np.isclose(msg.fm, 4.0))[0])
    assert np.unravel_index(np.argmax(msg.average()[0]), (b, k)) == (band, rate)
    steady = (msg.t > 1.4) & (msg.t < 1.6)
    depth_db = 20 * np.log10(msg.depth[0, band, rate, steady])
    assert np.allclose(depth_db, 20 * np.log10(0.5), atol=0.1)
    quiet = msg.valid[band, rate] & ((msg.t < 0.6) | (msg.t > 2.4))  # window clear of 1-2 s
    assert quiet.any() and np.all(msg.depth[0, band, rate, quiet] < 0.01)
    snap = msg.at(1.0)
    assert snap.shape == (1, 24, 9) and np.isclose(snap[0, band, rate], msg.depth[0, band, rate, 100])
    assert msg.average().shape == (1, 24, 9)


def test_valid_marks_ends_and_fast_rates(tone_env, tone_msg):
    msg = tone_msg
    n = msg.valid.shape[-1]
    half = msg.bank.lengths(FE) // 2 / FE
    for k, h in enumerate(half):
        ok = msg.valid[-1, k]  # the widest band: only the ends are marked
        assert not ok[msg.t < h - 0.01].any() and not ok[msg.t > msg.t[-1] - h + 0.01].any()
        assert ok[(msg.t > h + 0.01) & (msg.t < msg.t[-1] - h - 0.01)].all()
    assert n == len(msg.t)
    # the lowest band (143 Hz) is about 46 Hz wide: it can carry 32 Hz, not 64
    fast = so.ModulationSpectrogram(tone_env, f_lo=16.0, f_hi=64.0)
    assert fast.valid[0, list(fast.fm).index(32.0)].any()
    assert not fast.valid[0, -1].any() and fast.valid[-1, -1].any()


def test_glide_is_tracked():
    dur, r0, r1 = 3.0, 4.0, 16.0  # two octaves in 3 s
    t = np.arange(int(dur * FS)) / FS
    phase = 2 * np.pi * r0 * dur / np.log(r1 / r0) * ((r1 / r0) ** (t / dur) - 1)
    snd = so.Sound(0.1 * (1 + 0.5 * np.sin(phase)) * np.sin(2 * np.pi * 1000 * t), FS)
    env = so.ERBFilterbank(n_bands=12, f_lo=500, f_hi=2000).analyze(snd).envelopes(fs=FE)
    msg = so.ModulationSpectrogram(env, f_lo=2.0, f_hi=32.0, per_octave=4)
    band = int(np.argmin(np.abs(msg.f - 1000)))
    D = np.log(msg.depth[0, band] + 1e-12)  # (K, F)
    sel = (msg.t > 0.8) & (msg.t < 2.2)
    k = np.clip(np.argmax(D[:, sel], axis=0), 1, len(msg.fm) - 2)
    i = np.flatnonzero(sel)
    y0, y1, y2 = D[k - 1, i], D[k, i], D[k + 1, i]
    est = msg.fm[k] * 2.0 ** (0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) / 4)
    true = r0 * (r1 / r0) ** (msg.t[sel] / dur)
    assert np.max(np.abs(np.log2(est / true))) < 0.1


def test_checks():
    fb = so.ERBFilterbank(n_bands=8, f_lo=200, f_hi=4000)
    env = fb.analyze(am_tone(dur=0.5, on=(0, 0.5))).envelopes(fs=150)
    with pytest.raises(ValueError, match="too low"):
        so.ModulationSpectrogram(env)
    with pytest.raises(TypeError):
        so.ModulationSpectrogram(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="align"):
        so.ModulationSpectrogram(env, f_hi=32, align="left")


def test_stereo_and_silence():
    rng = np.random.default_rng(2)
    x = np.zeros((8000, 2))
    x[2000:6000, 0] = rng.standard_normal(4000)
    fb = so.ERBFilterbank(n_bands=8, f_lo=200, f_hi=4000)
    msg = so.ModulationSpectrogram(fb.analyze(so.Sound(x, FS)).envelopes(fs=FE), f_lo=4.0, f_hi=32.0)
    assert msg.power.shape[0] == 2
    assert np.isnan(msg.depth[1]).all()  # a silent channel has no depth
    assert np.isfinite(msg.depth[0][:, :, (msg.t > 0.2) & (msg.t < 0.3)]).all()


def test_pooled_depth(tone_msg):
    msg = tone_msg
    pooled = msg.pooled_depth()
    assert pooled.shape == (1, len(msg.fm), len(msg.t))
    assert np.isnan(pooled[0, :, 0]).all()  # every window runs off the start
    rate = int(np.flatnonzero(np.isclose(msg.fm, 4.0))[0])
    steady = (msg.t > 1.4) & (msg.t < 1.6)
    # one tone: pooling over the bands that pass it gives its depth back
    assert np.allclose(pooled[0, rate, steady], 0.5, atol=0.01)


@pytest.mark.filterwarnings("ignore:Animation was deleted")
def test_plots(tone_msg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    msg = tone_msg
    for kwargs in ({}, {"band": 1000}, {"rate": 4}):
        ax = msg.plot(**kwargs)
        assert ax.get_yscale() == "log"
        plt.close(ax.figure)
    with pytest.raises(ValueError, match="not both"):
        msg.plot(band=1000, rate=4)
    fig = msg.slices(1.5)
    assert len(fig.axes) == 4  # three cuts and the colour bar
    anim = msg.animate(fps=10)
    assert isinstance(anim, FuncAnimation)
    del anim  # unrendered: let it go while the warning filter applies
    gc.collect()
    plt.close("all")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_animation_with_audio(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snd = am_tone(dur=0.3, on=(0, 0.3))
    env = so.ERBFilterbank(n_bands=4, f_lo=500, f_hi=2000).analyze(snd).envelopes(fs=FE)
    msg = so.ModulationSpectrogram(env, f_lo=16.0, f_hi=32.0)
    path = tmp_path / "msg.mp4"
    msg.animate(path, sound=snd, fps=10, figsize=(2, 2), dpi=50)
    assert path.stat().st_size > 0
    plt.close("all")
