import inspect
import math

import numpy as np
import pytest

import sonore as so

FS = 16000.0


def clicks(alpha, *, seconds=20.0, rate=20.0, seed=0):
    """Clicks at random times with Pareto amplitudes of exponent ``alpha``
    (at least 1), over a faint noise floor so that no bin is silent."""
    rng = np.random.default_rng(seed)
    samples = 1e-4 * rng.standard_normal(int(seconds * FS))
    times = rng.integers(0, len(samples), int(seconds * rate))
    samples[times] += rng.pareto(alpha, len(times)) + 1
    return so.Sound(samples, FS)


@pytest.mark.parametrize("alpha", [1.0, 2.5])
def test_recovers_the_exponent_of_sparse_clicks(alpha):
    peaks = so.envelope_peaks(clicks(alpha))
    # 200 values are fitted, so one standard error is about 7%
    np.testing.assert_allclose(peaks.exponent, alpha, rtol=0.2)
    assert peaks.maxima.shape == (4000, 2)


def test_gaussian_noise_has_a_light_tail():
    peaks = so.envelope_peaks(so.gaussian_noise(20.0, fs=FS, rng=1))
    assert np.all(peaks.exponent > 10)


def test_uncertainty_is_exponent_over_root_of_values_fitted():
    peaks = so.envelope_peaks(clicks(1.5), fraction=0.1)
    assert peaks.n_fitted == 400
    np.testing.assert_allclose(peaks.uncertainty, peaks.exponent / math.sqrt(400))
    assert np.all(peaks.maxima <= peaks.maxima.max(axis=0))
    assert np.sum(peaks.maxima[:, 0] > peaks.threshold[0]) <= 400


def test_matches_the_hill_formula_on_the_maxima():
    peaks = so.envelope_peaks(clicks(1.5), cutoffs_hz=(1000.0,))
    ordered = np.sort(peaks.maxima[:, 0])[::-1]
    k = peaks.n_fitted
    expected = k / np.sum(np.log(ordered[:k] / ordered[k]))
    np.testing.assert_allclose(peaks.exponent[0], expected)


def test_warns_when_too_few_values_are_fitted():
    with pytest.warns(UserWarning, match="fitted only 20 values"):
        so.envelope_peaks(clicks(1.5, seconds=2.0))


def test_refuses_to_go_back_to_a_sound_and_names_event_train():
    peaks = so.envelope_peaks(clicks(1.5))
    for method in (peaks.synthesize, peaks.to_sound):
        with pytest.raises(so.NotInvertibleError, match="event_train"):
            method()


def test_options_are_keyword_only_and_checked():
    parameters = inspect.signature(so.envelope_peaks).parameters
    options = [parameter for name, parameter in parameters.items() if name != "sound"]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in options)
    sound = clicks(1.5, seconds=2.0)
    with pytest.raises(ValueError, match="Nyquist"):
        so.envelope_peaks(sound, cutoffs_hz=(9000.0,))
    with pytest.raises(ValueError, match="fraction"):
        so.envelope_peaks(sound, fraction=1.5)
    with pytest.raises(ValueError, match="mono"):
        so.envelope_peaks(sound.to_channels(2))
    with pytest.raises(ValueError, match="too few"):
        so.envelope_peaks(so.Sound(np.ones(400), FS))


def test_plot_draws_one_survival_curve_and_one_fit_per_cutoff():
    pytest.importorskip("matplotlib")
    import matplotlib
    import matplotlib.pyplot

    matplotlib.use("Agg")
    ax = so.envelope_peaks(clicks(1.5)).plot()
    assert len(ax.get_lines()) == 4
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    matplotlib.pyplot.close(ax.figure)
