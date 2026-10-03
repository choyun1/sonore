"""Numeric helpers: levels, the ERB scale, sample counts."""

import numpy as np
import pytest

import sonore as so
from sonore.core.utils import erb_bandwidth, n_samples


def test_power_to_db_is_half_amp_to_db():
    assert so.power_to_db(0.0) == -300
    assert so.power_to_db(0.0, floor_db=-120) == -120
    assert so.power_to_db(100.0) == pytest.approx(20)
    assert so.power_to_db(4.0, ref=2.0) == pytest.approx(so.amp_to_db(2.0, ref=np.sqrt(2.0)))


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
