import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import sonore as so
from sonore.plotting import plot_envelopes, plot_lissajous


def test_plot_lissajous_draws_left_against_right():
    fs = 8000
    t = np.arange(fs // 10) / fs
    sound = so.Sound(np.stack([np.cos(2 * np.pi * 200 * t), np.cos(2 * np.pi * 300 * t)], axis=1), fs)
    ax = plot_lissajous(sound, duration=0.01, start=0.02)
    x, y = ax.lines[0].get_data()
    assert len(x) == 80
    np.testing.assert_array_equal(x, sound.data[160:240, 0])
    np.testing.assert_array_equal(y, sound.data[160:240, 1])


def test_plot_lissajous_needs_two_channels():
    with pytest.raises(ValueError, match="two-channel"):
        plot_lissajous(so.pure_tone(0.1, 8000, 200))


def test_plot_envelopes_shares_a_color_scale():
    bank = so.cosine_filterbank(8, 100, 3000)
    noise = so.gaussian_noise(0.2, 8000, rng=0)
    loud, quiet = bank.analyze(noise).envelopes(), bank.analyze(noise * 0.1).envelopes()
    top = float(loud.db[:, 1:-1].max())  # the edge bands are not drawn
    own = plot_envelopes(quiet, colorbar=False).collections[0].get_clim()
    shared = plot_envelopes(quiet, db_max=top, colorbar=False).collections[0].get_clim()
    assert own[1] == pytest.approx(top - 20, abs=1e-6)  # its own maximum, 20 dB lower
    assert shared == pytest.approx((top - 40, top))
