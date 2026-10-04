"""Numeric helpers: levels, the ERB and mel scales, sample counts, and the
private helpers other modules share."""

import numpy as np
import pytest

import sonore as so
from sonore.core.utils import (
    _fit_length,
    _parabola_vertex,
    erb_bandwidth,
    freq_to_mel,
    mel_to_freq,
    n_samples,
)


def test_power_to_db_is_half_amp_to_db():
    assert so.power_to_db(0.0) == -300
    assert so.power_to_db(0.0, floor_db=-120) == -120
    assert so.power_to_db(100.0) == pytest.approx(20)
    assert so.power_to_db(4.0, ref=2.0) == pytest.approx(so.amp_to_db(2.0, ref=np.sqrt(2.0)))


def test_power_to_db_puts_negative_powers_at_the_floor():
    """Never the level of the absolute value: round-off and bugs alike land
    where true zeros do."""
    np.testing.assert_array_equal(so.power_to_db([-4.0, -1e-17, 0.0]), [-300, -300, -300])
    assert so.power_to_db(-4.0, floor_db=-120) == -120


def test_db_to_power_inverts_power_to_db():
    levels = np.array([-60.0, -3.0, 0.0, 20.0])
    np.testing.assert_allclose(so.power_to_db(so.db_to_power(levels)), levels, rtol=1e-12)
    np.testing.assert_allclose(so.db_to_power(levels), so.db_to_amp(levels) ** 2, rtol=1e-12)


def test_amp_to_db_floor():
    assert so.amp_to_db(0.0) == -300
    assert so.amp_to_db(0.0, floor_db=-120) == -120
    assert so.amp_to_db(10.0) == pytest.approx(20)


def test_erb_at_1khz_matches_glasberg_and_moore():
    # Glasberg & Moore (1990): ERB = 24.7 (4.37 F + 1), F in kHz; ERB-number
    # printed as 21.4 log10(4.37 F + 1). sonore integrates 1/ERB exactly, which
    # differs from the printed rounding by about 0.3%.
    assert erb_bandwidth(1000) == pytest.approx(132.639)
    assert so.freq_to_erb(1000) == pytest.approx(21.4 * np.log10(5.37), rel=4e-3)


def test_erb_number_is_the_integral_of_one_over_erb():
    freqs = np.linspace(0, 10000, 100001)
    integrand = 1 / erb_bandwidth(freqs)
    integral = np.concatenate([[0], np.cumsum((integrand[1:] + integrand[:-1]) / 2 * np.diff(freqs))])
    np.testing.assert_allclose(so.freq_to_erb(freqs), integral, rtol=1e-3, atol=1e-9)


def test_n_samples_rounds():
    assert n_samples(0.0015, 1000) == 2  # 1.5 samples: half to even
    assert n_samples(0.0026, 1000) == 3  # floor would give 2
    assert n_samples(0.0024, 1000) == 2


@pytest.mark.parametrize("convert", [freq_to_mel, mel_to_freq])
def test_mel_scales_refuse_an_unknown_name(convert):
    with pytest.raises(ValueError, match="htk"):
        convert(1000.0, "bark")


def test_parabola_vertex_refines_peaks_and_dips():
    """Samples of y = -(x - 0.3)**2 at x = -1, 0, 1: the vertex is at 0.3."""
    samples = -((np.array([-1.0, 0.0, 1.0]) - 0.3) ** 2)
    assert _parabola_vertex(*samples) == pytest.approx(0.3)
    assert _parabola_vertex(*-samples, dip=True) == pytest.approx(0.3)


def test_parabola_vertex_is_zero_where_there_is_nothing_to_refine():
    flat = np.array([2.0, 2.0, 2.0])
    assert _parabola_vertex(*flat) == 0.0
    peak = -((np.array([-1.0, 0.0, 1.0]) - 0.3) ** 2)
    assert _parabola_vertex(*peak, dip=True) == 0.0  # a peak is no dip


def test_fit_length_cuts_and_pads_at_the_end():
    data = np.arange(1.0, 6.0)
    np.testing.assert_array_equal(_fit_length(data, 3), [1, 2, 3])
    np.testing.assert_array_equal(_fit_length(data, 7), [1, 2, 3, 4, 5, 0, 0])
    np.testing.assert_array_equal(_fit_length(data, 7, mode="edge"), [1, 2, 3, 4, 5, 5, 5])
    assert _fit_length(np.ones((5, 2)), 7).shape == (7, 2)


def test_cents_scale_counts_cents_from_the_reference():
    scale = so.cents_scale()
    np.testing.assert_allclose(
        scale.to_scale([440.0, 880.0, 220.0, 440 * 2 ** (1 / 12)]), [0, 1200, -1200, 100]
    )
    np.testing.assert_allclose(scale.from_scale([-1200.0, 0.0, 700.0]), [220.0, 440.0, 440 * 2 ** (7 / 12)])
    # 432 Hz is 31.8 cents below 440 Hz
    assert scale.to_scale(432.0) == pytest.approx(-31.77, abs=0.01)
    assert so.cents_scale(261.63).to_scale(261.63) == 0
    assert so.cents_scale().name == "cents" and so.cents_scale().unit == "cent"


def test_cents_scale_needs_a_positive_reference():
    with pytest.raises(ValueError, match="above 0 Hz"):
        so.cents_scale(0.0)
