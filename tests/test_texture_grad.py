import numpy as np
import pytest

import sonore as so
from sonore.texture import TextureModel, TextureStats, measurement_window
from sonore.texture import grad as tg

M = TextureModel()
FS = 20000


@pytest.fixture(scope="module")
def envs():
    """Compressed envelopes of 1 s of AM noise (realistic, positive, modulated)."""
    t = np.arange(FS) / FS
    snd = so.gaussian_noise(1, FS, rng=0) * so.Sound(1 + 0.8 * np.sin(2 * np.pi * 6 * t), FS)
    return M.envelopes(M.subbands(M.prepare(snd)))


def fd_check(f, s, ctx, complex_out=False, seed=0, eps=1e-6):
    """Compare vjp(g) . v with the central difference of g . f along v."""
    rng = np.random.default_rng(seed)
    val, vjp = f(s)
    v = rng.standard_normal(s.shape) * s.std()
    g = rng.standard_normal(val.shape)
    if complex_out:
        g = g + 1j * rng.standard_normal(val.shape)

    def proj(x):
        out = f(x)[0]
        return np.sum(np.real(np.conj(g) * out)) if complex_out else np.sum(g * out)

    fd = (proj(s + eps * v) - proj(s - eps * v)) / (2 * eps)
    an = vjp(g) @ v
    assert an == pytest.approx(fd, rel=1e-5, abs=1e-9 * abs(an) + 1e-12)


@pytest.mark.parametrize("window", ["uniform", "ramped"])
@pytest.mark.parametrize("ch", [5, 17, 28])
def test_gradients_match_finite_differences(envs, window, ch):
    n = envs.shape[0]
    w = None if window == "uniform" else measurement_window(n, 1)
    ctx = tg.ChannelContext.build(M, n, w)
    s = envs[:, ch]
    fd_check(lambda x: tg.env_moments(x, ctx), s, ctx)
    fd_check(lambda x: tg.mod_power(x, ctx), s, ctx)
    fd_check(lambda x: tg.env_corr(x, envs[:, [ch - 3, ch - 1, ch + 2]], ctx), s, ctx)
    fd_check(lambda x: tg.c1(x, envs[:, [ch - 2, ch - 1]], ctx), s, ctx)
    fd_check(lambda x: tg.c2(x, ctx), s, ctx, complex_out=True)


def test_values_equal_measured_stats(envs):
    """The per-channel functions compute exactly what TextureStats measures."""
    sb = M.subbands(M.prepare(so.gaussian_noise(1, FS, rng=3)))
    st = TextureStats.from_subbands(sb, M, window="ramped")
    env = M.envelopes(sb)
    ctx = tg.ChannelContext.build(M, env.shape[0], measurement_window(env.shape[0], 1))
    for ch in (0, 9, 31):
        s = env[:, ch]
        mom = tg.env_moments(s, ctx)[0]
        assert np.allclose(mom, [st.env_mean[ch], st.env_var[ch], st.env_skew[ch], st.env_kurt[ch]])
        assert np.allclose(tg.mod_power(s, ctx)[0], st.mod_power[ch])
        assert np.allclose(tg.c2(s, ctx)[0], st.c2[ch])
        ok = [i for i, d in enumerate(M.corr_offsets) if ch + d < env.shape[1]]
        others = env[:, [ch + M.corr_offsets[i] for i in ok]]
        assert np.allclose(tg.env_corr(s, others, ctx)[0], st.env_corr[ch, ok])
        ok = [i for i, d in enumerate(M.c1_offsets) if ch + d < env.shape[1]]
        others = env[:, [ch + M.c1_offsets[i] for i in ok]]
        assert np.allclose(tg.c1(s, others, ctx)[0], st.c1[ch][:, ok])
