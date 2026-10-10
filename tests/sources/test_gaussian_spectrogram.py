"""Random spectrograms with natural correlations (McDermott et al., 2011)."""

import numpy as np
import pytest

import sonore as so
from sonore.sources.gaussian_spectrogram import _correlated_field

FS = 16000


def _cells_db(env: so.Envelopes, hop: float = 0.010) -> np.ndarray:
    """Cell levels [dB] (band, window): at a window's center only that window is nonzero."""
    n_windows = int(np.ceil(env.t[-1] / hop)) + 1
    centers = np.minimum(np.round(hop * np.arange(n_windows) * env.fs).astype(int), len(env.t) - 1)
    return 20 * np.log10(env.data[centers, 1:-1, 0].T)


def test_field_covariance_is_the_separable_exponential():
    # the recursion is linear, so its covariance follows from the unit impulse responses
    n_bands, n_windows, rho_band, rho_time = 5, 7, 0.8, 0.6

    class Unit:
        def __init__(self, index):
            self.index = index

        def standard_normal(self, shape):
            white = np.zeros(shape)
            white.flat[self.index] = 1.0
            return white

    columns = [_correlated_field(n_bands, n_windows, rho_band, rho_time, Unit(i)).ravel() for i in range(35)]
    transform = np.array(columns).T
    lags_band = np.abs(np.arange(n_bands)[:, None] - np.arange(n_bands)[None, :])
    lags_time = np.abs(np.arange(n_windows)[:, None] - np.arange(n_windows)[None, :])
    expected = np.kron(rho_band**lags_band, rho_time**lags_time)
    np.testing.assert_allclose(transform @ transform.T, expected, atol=1e-12)


def test_shape_reproducibility_and_edges():
    env = so.gaussian_spectrogram(0.3, FS, rng=4)
    assert isinstance(env, so.Envelopes)
    assert env.data.shape == (4800, 41, 1)
    assert env.filterbank.n_bands == 39
    np.testing.assert_array_equal(env.data, so.gaussian_spectrogram(0.3, FS, rng=4).data)
    assert not np.array_equal(env.data, so.gaussian_spectrogram(0.3, FS, rng=5).data)
    assert np.all(env.data[:, [0, -1]] == 0) and np.all(env.data[:, 1:-1] > 0)


def test_cell_statistics():
    cells = np.array([_cells_db(so.gaussian_spectrogram(0.4, FS, rng=seed)) for seed in range(100)])
    band_means = cells.mean(axis=(0, 2))
    cells = cells - band_means[None, :, None]
    assert cells.std() == pytest.approx(14.1, rel=0.05)
    spacing = so.cosine_filterbank(39, 20, 4000).spacing
    band_lag1 = np.corrcoef(cells[:, :-1].ravel(), cells[:, 1:].ravel())[0, 1]
    time_lag1 = np.corrcoef(cells[:, :, :-1].ravel(), cells[:, :, 1:].ravel())[0, 1]
    assert band_lag1 == pytest.approx(np.exp(-spacing / 8.78), abs=0.02)
    assert time_lag1 == pytest.approx(np.exp(-0.010 / 0.154), abs=0.02)
    # flat on average: mean band level rises with bandwidth, 10 dB per decade of width
    bank = so.cosine_filterbank(39, 20, 4000)
    widths = np.gradient(bank.band_cfs)
    slope = np.polyfit(10 * np.log10(widths), band_means, 1)[0]
    assert slope == pytest.approx(1.0, abs=0.1)


def test_sound_keeps_the_drawn_cells():
    env = so.gaussian_spectrogram(0.4, FS, rng=2)
    sound = env.to_sound(so.gaussian_noise(0.4, FS, rng=3))
    drawn = env.data[:, 1:-1, 0]
    got = np.abs(env.filterbank.analyze(sound).envelopes().data[:, 1:-1, 0])
    # compare smoothed band levels over 20 ms
    kernel = np.hanning(int(0.02 * FS))
    smooth = lambda x: 10 * np.log10(np.apply_along_axis(np.convolve, 0, x**2, kernel, "same") + 1e-30)  # noqa: E731
    assert np.corrcoef(smooth(drawn).ravel(), smooth(got).ravel())[0, 1] > 0.9


def test_sd_zero_is_flat_in_time():
    env = so.gaussian_spectrogram(0.2, FS, sd_db=0, rng=0)
    middle = env.data[200:-200, 1:-1, 0]
    np.testing.assert_allclose(middle, np.broadcast_to(middle[0], middle.shape), rtol=1e-9)


@pytest.mark.parametrize(
    "kwargs", [{"band_correlation_erb": 0}, {"time_correlation": -1}, {"sd_db": -1}, {"window": 0}]
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        so.gaussian_spectrogram(0.1, FS, **kwargs)
