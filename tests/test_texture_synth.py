import numpy as np
import pytest

import sonore as so
from sonore import texture_grad as tg
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
