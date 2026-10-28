import numpy as np
import pytest

import sonore as so
from sonore.texture import TextureModel, TextureStats, measurement_window

FS = 20000
M = TextureModel()


def am_noise(f_mod, dur=2.0, rng=0, comodulated=True):
    """Noise with sinusoidal AM; independent AM phases per band if not comodulated."""
    rng = np.random.default_rng(rng)
    t = np.arange(int(dur * FS)) / FS
    if comodulated:
        return so.gaussian_noise(dur, FS, rng=rng) * so.Sound(1 + 0.9 * np.sin(2 * np.pi * f_mod * t), FS)
    sb = M.subbands(so.gaussian_noise(dur, FS, rng=rng).data[:, 0])
    ph = rng.uniform(0, 2 * np.pi, sb.shape[1])
    return so.Sound((sb * (1 + 0.9 * np.sin(2 * np.pi * f_mod * t[:, None] + ph))).sum(1), FS)


def test_counts_match_paper():
    st = TextureStats.measure(so.gaussian_noise(2, FS, rng=0))
    assert st.count() == 1515
    counts = [st.count(classes=c) for c in ("env_mean", "env_corr", "mod_power", "c1", "c2")]
    assert counts == [32, 189, 640, 366, 192]


def test_filterbanks():
    assert np.allclose(M.filterbank.cfs[[1, -2]], [51.7, 8844.5], atol=0.1)
    assert np.allclose(M.oct_bank.cfs, 100 / 2.0 ** np.arange(6, -1, -1))
    b = M.mod_bank
    f = np.linspace(b.cfs[3], b.cfs[-4], 2000)
    s = (b.response(f) ** 2).sum(1)
    assert abs(s.mean() - 1) < 1e-3 and np.all(np.abs(s - 1) < 0.05)


def test_analytic_modulation_filtering_matches_real():
    x = np.random.default_rng(0).standard_normal((800, 3))
    b = M.oct_bank
    assert np.allclose(b.filter(x, 400, analytic=True).real, b.filter(x, 400))


def test_measure_applies_the_window():
    """``measure(window=...)`` reaches the statistics: the ramped window
    downweights the ends, so a sound that is loud only in its first quarter
    has a smaller envelope mean under it than under the uniform window."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(2 * FS)
    noise[FS // 2 :] *= 0.1
    sound = so.Sound(noise, FS)
    ramped = TextureStats.measure(sound)
    uniform = TextureStats.measure(sound, window="uniform")
    assert np.all(ramped.env_mean[3:29] < 0.95 * uniform.env_mean[3:29])


def test_envelope_is_the_analytic_magnitude():
    """A 60 Hz tone's subband envelope is flat: the Hilbert magnitude, not
    the rectified waveform, whose ripple at 120 Hz would survive the
    downsampling to 400 Hz."""
    t = np.arange(FS) / FS
    env = M.envelopes(M.subbands(M.prepare(so.Sound(np.cos(2 * np.pi * 60 * t), FS))))
    loudest = int(np.argmax(env.mean(0)))
    assert env[:, loudest].std() < 1e-3 * env[:, loudest].mean()


def test_c2_phase_convention():
    """C2 of an envelope with components at f and 2f is exp(i phi), phi the
    phase of the upper component relative to the lower one doubled: the
    imaginary part is positive when the upper band leads by a quarter cycle
    (the toolbox's convention)."""
    t = np.arange(2000) / M.env_fs
    f = M.oct_bank.cfs[2]  # 6.25 Hz; band 3 is centered at 12.5 Hz
    for phi in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        env = 1 + 0.3 * np.cos(2 * np.pi * f * t) + 0.3 * np.cos(2 * np.pi * 2 * f * t + phi)
        st = TextureStats.from_envelopes(env[:, None], np.ones(1), M, window="uniform")
        assert abs(st.c2[0, 2] - np.exp(1j * phi)) < 0.05, phi


def test_measurement_window():
    w = measurement_window(2000, 5)
    assert np.isclose(w.sum(), 1)
    r = 2000 // 6
    assert np.allclose(w[r:-r], w[1000]) and w[0] < 0.01 * w[1000] and np.allclose(w, w[::-1])


def test_level_invariant_and_modulation_peak():
    s = am_noise(10)
    a, b = TextureStats.measure(s), TextureStats.measure(s * 7.0)
    assert np.allclose(a.mod_power, b.mod_power) and np.allclose(a.env_mean, b.env_mean)
    peak = M.mod_bank.cfs[np.argmax(a.mod_power[5:28].mean(0))]
    assert 7 < peak < 14
    assert a.mod_power[5:28].sum(1).mean() == pytest.approx(1, abs=0.1)  # powers sum to env variance


def test_comodulation_raises_c1_and_corr():
    co, ind = TextureStats.measure(am_noise(6)), TextureStats.measure(am_noise(6, comodulated=False))
    band = 1  # c1 index 1 = octave band c1_bands[1] = 2, centered at 6.25 Hz
    assert np.nanmean(co.c1[3:28, band, 0]) > 0.9
    assert np.nanmean(co.c1[3:28, band, 0]) > np.nanmean(ind.c1[3:28, band, 0]) + 0.5
    assert np.nanmean(co.env_corr[3:28]) > np.nanmean(ind.env_corr[3:28]) + 0.3


def test_noise_is_lower_bound_of_envelope_moments():
    """Paper Fig. 2: noise envelopes have lower variance, skew and kurtosis
    than sparse or modulated textures."""
    noise = TextureStats.measure(so.gaussian_noise(2, FS, tilt=-3, rng=0))
    rng = np.random.default_rng(1)
    clicks = np.zeros(2 * FS)
    clicks[rng.choice(len(clicks), 8, replace=False)] = rng.choice(
        [-1, 1], 8
    )  # 4/s: sparse in low channels too
    sparse = TextureStats.measure(so.Sound(clicks, FS))
    ch = slice(3, 29)
    for c in ("env_var", "env_skew", "env_kurt"):
        assert np.all(sparse.get(c)[ch] > noise.get(c)[ch]), c
    assert np.all(noise.env_var[ch] < 0.05) and np.all(np.abs(noise.env_skew[ch]) < 0.6)


def test_save_load_and_replace(tmp_path):
    a = TextureStats.measure(so.gaussian_noise(2, FS, rng=0))
    a.save(tmp_path / "a.npz")
    b = TextureStats.load(tmp_path / "a.npz")
    assert b.model == a.model and b.duration == a.duration
    for c in so.texture.STAT_CLASSES:
        assert np.array_equal(a.get(c), b.get(c), equal_nan=True)
    h = a.replace(mod_power=np.zeros_like(a.mod_power))
    assert np.all(h.mod_power == 0) and h.env_mean is a.env_mean


def test_snr():
    a = TextureStats.measure(so.gaussian_noise(2, FS, tilt=-3, rng=0))
    b = TextureStats.measure(so.gaussian_noise(2, FS, tilt=-3, rng=1))
    c = TextureStats.measure(am_noise(10))
    assert all(v == float("inf") for v in a.snr(a).values())
    same, diff = a.snr(b), a.snr(c)
    assert set(same) == set(so.texture.PAPER_CLASSES)
    # another sample of the same texture matches better than a different texture
    for k in ("env_var", "env_kurt", "mod_power"):
        assert same[k] > diff[k] + 3, k
    # channels outside the 30 dB range are ignored
    quiet = a.replace(subband_var=np.where(np.arange(32) < 16, 1e-6, 1.0) * a.subband_var)
    assert quiet.channel_mask().sum() == 16


def test_a_stereo_sound_is_measured_as_its_channel_average():
    noise = so.gaussian_noise(2, FS, n_channels=2, rng=0).data
    stereo = so.Sound(noise * [1.0, 0.3], FS)
    average = so.Sound(noise @ [0.5, 0.15], FS)
    a, b = TextureStats.measure(stereo), TextureStats.measure(average)
    for name in so.texture.STAT_CLASSES:
        np.testing.assert_allclose(a.get(name), b.get(name), rtol=1e-10, atol=1e-14)


def test_another_rate_is_resampled_to_the_models():
    t = np.arange(2 * 44100) / 44100
    tone = so.Sound(np.sin(2 * np.pi * 1000 * t), 44100)
    prepared = M.prepare(tone)
    assert len(prepared) == 2 * FS
    loudest = np.argmax(TextureStats.measure(tone).subband_var)
    assert M.filterbank.cfs[loudest] == pytest.approx(1000, rel=0.15)


def test_measurement_window_ramps():
    n, seconds = 2400, 3
    w = measurement_window(n, seconds)
    ramp_len = n // (seconds + 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, ramp_len + 1) / ramp_len)
    expected = np.ones(n)
    expected[:ramp_len], expected[-ramp_len:] = ramp, ramp[::-1]
    np.testing.assert_allclose(w, expected / expected.sum(), rtol=1e-12)


def test_measure_weights_by_the_sounds_whole_seconds():
    """A 3 s original is weighted by measurement_window(n_env, 3)."""
    sound = so.gaussian_noise(3, FS, rng=2) * so.Sound(np.linspace(0.2, 1.0, 3 * FS), FS)
    env = M.envelopes(M.subbands(M.prepare(sound)))
    weights = measurement_window(len(env), 3)
    np.testing.assert_allclose(TextureStats.measure(sound).env_mean, weights @ env, rtol=1e-12)


def test_subband_variance_is_weighted_and_centered():
    """With a DC offset the lowpass channel has a nonzero mean, which the
    variance removes; the weights are the envelope window held over each
    envelope sample."""
    sound = so.gaussian_noise(2, FS, rng=3) * so.Sound(np.linspace(0.2, 1.0, 2 * FS), FS) + np.full(
        2 * FS, 0.5
    )
    x = M.prepare(sound)
    subbands = M.subbands(x)
    weights = np.repeat(measurement_window(len(x) // M.decimation, 2), M.decimation) / M.decimation
    expected = weights @ (subbands - weights @ subbands) ** 2
    np.testing.assert_allclose(TextureStats.measure(sound).subband_var, expected, rtol=1e-12)


def test_duration_is_whole_envelope_samples():
    assert TextureStats.measure(so.gaussian_noise(2.0012, FS, rng=0)).duration == pytest.approx(2.0)


class TestSnrInDetail:
    stats = TextureStats.measure(so.gaussian_noise(2, FS, tilt=-3, rng=0))

    def test_is_relative_to_the_target(self):
        other = self.stats.replace(env_mean=1.1 * self.stats.env_mean)
        assert self.stats.snr(other)["env_mean"] == pytest.approx(20)

    def test_a_pair_counts_only_if_both_channels_are_loud(self):
        quiet = np.where(np.arange(32) == 20, 1e-6, 1.0)
        target = self.stats.replace(subband_var=quiet * self.stats.subband_var)
        corr = target.env_corr.copy()
        corr[19, 0] += 0.5  # the pair (19, 20): channel 20 is quiet
        assert target.snr(target.replace(env_corr=corr))["env_corr"] == float("inf")


def test_a_sound_shorter_than_one_envelope_sample_is_refused():
    with pytest.raises(ValueError, match="shorter than one envelope sample"):
        TextureStats.measure(so.Sound(np.ones(49), FS))


def test_snr_refuses_statistics_from_another_model():
    a = TextureStats.measure(so.gaussian_noise(1, FS, rng=0))
    b = TextureStats.measure(so.gaussian_noise(1, FS, rng=0), model=TextureModel(n_bands=20))
    with pytest.raises(ValueError, match="different models"):
        a.snr(b)
