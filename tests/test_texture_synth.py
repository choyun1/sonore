import numpy as np
import pytest

import sonore as so
from sonore import texture_grad as tg
from sonore import texture_synth as synth
from sonore.texture import TextureModel, TextureStats
from sonore.texture_synth import ChannelObjective, impose_channel

M = TextureModel()
FS = 20000


@pytest.fixture(scope="module")
def setup():
    """Target: comodulated 6 Hz AM noise. Start: envelopes of plain noise,
    with channels 13, 14, 16, 17 already set to the target's envelopes."""
    t = np.arange(2 * FS) / FS
    tex = so.gaussian_noise(2, FS, rng=0) * so.Sound(1 + 0.9 * np.sin(2 * np.pi * 6 * t), FS)
    sb = M.subbands(M.prepare(tex))
    env_t = M.envelopes(sb)
    target = TextureStats.from_subbands(sb, M, window="uniform")
    env = M.envelopes(M.subbands(M.prepare(so.gaussian_noise(2, FS, rng=5))))
    adjusted = np.zeros(env.shape[1], bool)
    for j in (13, 14, 16, 17):
        env[:, j], adjusted[j] = env_t[:, j], True
    return target, env_t, env, adjusted


def test_objective_is_zero_at_the_target(setup):
    """With every channel at the target's own envelopes, each channel's
    objective vanishes: checks that every pairwise term (both directions)
    is compared with the right stored target."""
    target, env_t, _, _ = setup
    ctx = tg.ChannelContext.build(M, env_t.shape[0])
    for k in (0, 1, 15, 30, 31):
        obj = ChannelObjective(ctx, target, k, env_t, np.ones(env_t.shape[1], bool))
        names = [name for name, _, _ in obj.terms()]
        assert {"env_corr", "c1"} <= set(names)
        assert obj(env_t[:, k])[0] < 1e-20


def test_objective_gradient(setup):
    target, _, env, adjusted = setup
    ctx = tg.ChannelContext.build(M, env.shape[0])
    obj = ChannelObjective(ctx, target, 15, env, adjusted)
    s = env[:, 15]
    v = np.random.default_rng(0).standard_normal(s.shape) * s.std()
    eps = 1e-6
    fd = (obj(s + eps * v)[0] - obj(s - eps * v)[0]) / (2 * eps)
    assert obj(s)[1] @ v == pytest.approx(fd, rel=1e-6)


def test_impose_channel_reduces_every_class(setup):
    target, _, env, adjusted = setup
    k = 15
    s, report = impose_channel(target, env, k, adjusted, n_iter=5)
    assert s.min() >= 0
    assert report["after"] < report["before"] / 100
    for name, before in report["terms_before"].items():
        assert report["terms_after"][name] < before / 5, name
    # measured the ordinary way, channel k's statistics moved toward the target
    new = env.copy()
    new[:, k] = s
    old_st = TextureStats.from_envelopes(env, target.subband_var, M)
    new_st = TextureStats.from_envelopes(new, target.subband_var, M)
    # (the mean barely moves: see the module docstring; it is checked not to get worse)
    for c in ("env_var", "env_skew", "env_kurt", "mod_power", "c2"):
        e_old = np.sum(np.abs(old_st.get(c)[k] - target.get(c)[k]) ** 2)
        e_new = np.sum(np.abs(new_st.get(c)[k] - target.get(c)[k]) ** 2)
        assert e_new < e_old / 2, c
    mean_err = [abs(st.env_mean[k] - target.env_mean[k]) for st in (old_st, new_st)]
    assert mean_err[1] <= mean_err[0] * 1.01


def test_more_iterations_do_better(setup):
    target, _, env, adjusted = setup
    losses = [impose_channel(target, env, 15, adjusted, n_iter=n)[1]["after"] for n in (1, 5, 20)]
    assert losses[0] > losses[1] > losses[2]


def test_channel_order():
    assert synth.channel_order(np.array([0, 1, 5, 2, 0])) == [2, 3, 1, 4, 0]
    assert sorted(synth.channel_order(np.random.default_rng(0).random(32))) == list(range(32))


def test_affine_correction_sets_mean_and_variance(setup):
    target, _, env, adjusted = setup
    ctx = tg.ChannelContext.build(M, env.shape[0])
    s, _ = impose_channel(target, env, 15, adjusted, n_iter=2, ctx=ctx)
    mom = tg.env_moments(s, ctx)[0]
    assert mom[0] == pytest.approx(target.env_mean[15], rel=1e-9)
    assert mom[1] == pytest.approx(target.env_var[15], rel=1e-9)


def test_synthesis_converges_on_am_noise():
    """Full loop, kept short for the suite (1 s, 6 iterations); see
    tools/texture_benchmark.py for the synthetic-texture benchmark."""
    t = np.arange(FS) / FS
    tex = so.gaussian_noise(1, FS, rng=0) * so.Sound(1 + 0.9 * np.sin(2 * np.pi * 6 * t), FS)
    target = TextureStats.measure(tex)
    seen = []
    snd, report = synth.synthesize(
        target, duration=1, rng=1, max_iter=6, callback=lambda i, s, snr: seen.append(i)
    )
    assert seen == list(range(1, report["iterations"] + 1))
    avg = [np.mean(list(h.values())) for h in report["snr"]]
    assert avg[-1] > avg[0] + 5
    assert report["converged"] and report["average_snr"] >= 20
    assert snd.fs == M.fs and len(snd) == FS and snd.rms == pytest.approx(M.rms)
    # the result is a new sound, not the original
    assert abs(np.corrcoef(snd.data[:, 0], M.prepare(tex))[0, 1]) < 0.1


def test_synthesis_restricted_classes():
    target = TextureStats.measure(so.gaussian_noise(1, FS, tilt=-3, rng=0))
    classes = ("env_mean", "env_var", "mod_power")
    _, report = synth.synthesize(target, duration=1, rng=1, max_iter=2, classes=classes)
    assert set(report["snr"][0]) == set(classes)


@pytest.mark.parametrize("k", [0, 15, 31])
@pytest.mark.parametrize("classes", [None, ("env_var", "c1"), ("mod_power", "c2", "env_corr")])
def test_fused_objective_matches_reference(setup, k, classes):
    target, _, env, adjusted = setup
    ctx = tg.ChannelContext.build(M, env.shape[0])
    args = () if classes is None else (classes,)
    obj = ChannelObjective(ctx, target, k, env, np.ones(env.shape[1], bool), *args)
    s = env[:, k] * 1.1 + 0.01
    fast, ref = obj(s, per_term=True), obj.reference(s, per_term=True)
    assert fast[0] == pytest.approx(ref[0], rel=1e-10)
    assert np.allclose(fast[1], ref[1], rtol=1e-8, atol=1e-12 * np.abs(ref[1]).max())
    assert fast[2].keys() == ref[2].keys()
