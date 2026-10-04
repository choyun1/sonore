import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import sonore as so
from sonore.plotting import plot_lissajous


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
