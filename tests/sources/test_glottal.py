"""The Liljencrants-Fant glottal pulse and the voiced source made from it."""

import numpy as np
import pytest
from scipy.integrate import quad

import sonore as so

FS = 16000


def fant_parameters(rd):
    """Ra, Rg, Rk and the excitation time te (fraction of a period) from Rd,
    by Fant's (1995) Eqs. 2-4."""
    ra = (-1 + 4.8 * rd) / 100
    rk = (22.4 + 11.8 * rd) / 100
    rg = rk * (0.5 + 1.2 * rk) / (4 * (0.11 * rd - ra * (0.5 + 1.2 * rk)))
    return ra, rg, rk, (1 + rk) / (2 * rg)


@pytest.mark.parametrize("rd", [0.3, 0.7, 1.0, 2.7])
def test_harmonics_match_numerical_integration(rd):
    """The closed-form coefficients are the Fourier integrals of the pulse."""
    te = fant_parameters(rd)[3]  # where the slope jumps
    for k in (1, 2, 7, 30):

        def part(trig, k=k):
            def integrand(x):
                return so.lf_pulse(x, rd) * trig(2 * np.pi * k * x)

            return quad(integrand, 0, 1, points=[te], limit=400)[0]

        want = part(np.cos) - 1j * part(np.sin)
        np.testing.assert_allclose(so.lf_harmonics([k], rd)[0], want, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("rd", [0.3, 0.7, 2.7])
def test_pulse_has_zero_net_flow_and_is_minus_one_at_the_excitation(rd):
    ra, rg, rk, te = fant_parameters(rd)
    assert so.lf_pulse(te, ra=ra, rg=rg, rk=rk) == pytest.approx(-1, abs=1e-12)
    assert so.lf_pulse(te, rd) == pytest.approx(-1, abs=1e-12)
    assert so.lf_pulse(1 - 1e-12, rd, flow=True) == pytest.approx(0, abs=1e-10)
    assert so.lf_pulse(0.0, rd, flow=True) == 0


def test_flow_coefficients_are_integrated_derivative_coefficients():
    harmonic_numbers = np.arange(1, 20)
    np.testing.assert_allclose(
        so.lf_harmonics(harmonic_numbers, 1.2, flow=True),
        so.lf_harmonics(harmonic_numbers, 1.2) / (2j * np.pi * harmonic_numbers),
    )


def test_rd_gives_fant_1995_figure_5a_parameters():
    """Rd = 1 gives Ra = 0.038, Rk = 0.342 (Fant 1995, Eqs. 2-3); the pulse's
    own Rd, U0 / (0.11 Ee T0), is close to the request (Fant's Eq. 4 is an
    approximation)."""
    x = np.linspace(0, 1, 200001)
    for rd in (0.5, 1.0, 2.0):
        own_rd = so.lf_pulse(x, rd, flow=True).max() / 0.11
        assert own_rd == pytest.approx(rd, rel=0.12)
    ra, rg, rk, _ = fant_parameters(1.0)
    assert (ra, rk) == pytest.approx((0.038, 0.342))
    np.testing.assert_allclose(so.lf_harmonics([1, 5], ra=ra, rg=rg, rk=rk), so.lf_harmonics([1, 5], 1.0))


def test_parameters_are_checked():
    with pytest.raises(ValueError, match="Rd = 0.1"):
        so.lf_harmonics([1], 0.1)
    with pytest.raises(ValueError, match="all of ra, rg and rk"):
        so.lf_harmonics([1], ra=0.03)
    with pytest.raises(ValueError, match="whole numbers"):
        so.lf_harmonics([0.5])


def test_source_spectrum_is_the_coefficients():
    x = so.glottal_source(1.0, FS, 100.0, 0.7)
    harmonic_numbers = np.arange(1, 80)
    spectrum = np.fft.rfft(x.data[:, 0])[100 * harmonic_numbers]
    want = so.lf_harmonics(harmonic_numbers, 0.7)
    ratio = spectrum / want
    np.testing.assert_allclose(ratio / ratio[0], 1, atol=1e-10)


def test_constant_rd_track_is_the_fixed_source():
    f0 = (np.array([0.0, 0.5]), np.array([100.0, 160.0]))
    fixed = so.glottal_source(0.5, FS, f0, 1.3)
    tracked = so.glottal_source(0.5, FS, f0, ([0.0, 0.5], [1.3, 1.3]))
    np.testing.assert_allclose(tracked.data, fixed.data, atol=1e-10)


def test_rd_track_interpolates_between_exact_coefficients():
    """Off the table's grid, a changing Rd is within 0.1% of the exact
    coefficients at every sample."""
    rd_track = ([0.0, 0.2], [0.5, 1.7])
    tracked = so.glottal_source(0.2, FS, 100.0, rd_track, flow=True)
    t = np.arange(len(tracked)) / FS
    rd_now = np.interp(t, *rd_track)
    exact = np.zeros_like(t)
    harmonic_numbers = np.arange(1, 80)
    for index in range(len(t)):
        c = so.lf_harmonics(harmonic_numbers, rd_now[index], flow=True)
        exact[index] = np.sum(
            2 * np.abs(c) * np.cos(2 * np.pi * harmonic_numbers * 100 * t[index] + np.angle(c))
        )
    scale = np.sqrt(np.mean(exact**2))
    np.testing.assert_allclose(tracked.data[:, 0], exact / scale, atol=1e-3)


def test_source_does_not_alias():
    """A tense pulse sampled directly folds its high harmonics back; the
    source made from harmonics has none above f_max."""
    x = so.glottal_source(1.0, FS, ([0.0, 1.0], [100.0, 300.0]), 0.3).data[:, 0]
    time_windows = np.lib.stride_tricks.sliding_window_view(x, 512)[::128] * np.hanning(512)
    power = np.abs(np.fft.rfft(time_windows, axis=1)) ** 2
    freqs = np.fft.rfftfreq(512, 1 / FS)
    assert 10 * np.log10(power[:, freqs > 0.45 * FS + 400].sum() / power.sum()) < -90


def test_rd_must_be_a_number_or_a_track():
    with pytest.raises(ValueError, match="number or a"):
        so.glottal_source(0.1, FS, 100.0, "tense")


class TestAuditSitting5:
    """Gaps found by the sitting 5 audit (LF glottal source)."""

    def test_rd_outside_fants_range_is_refused(self):
        for rd in (0.29, 2.71):
            with pytest.raises(ValueError, match="outside Fant's"):
                so.lf_harmonics([1], rd)
        with pytest.raises(ValueError, match="Rd = 3 is outside"):
            so.glottal_source(0.1, FS, 100.0, ([0.0, 0.1], [0.5, 3.0]))
        so.lf_harmonics([1], 0.3), so.lf_harmonics([1], 2.7)  # the ends are in
        # a track reaching the end of the range builds its table past it
        # (0.501 + 0.002 k passes 2.7 at 2.701)
        assert so.glottal_source(0.1, FS, 100.0, ([0.0, 0.1], [0.501, 2.7])).rms == pytest.approx(1)

    def test_impossible_timings_are_refused(self):
        # te = tp (1 + rk) beyond the period
        with pytest.raises(ValueError, match="no LF pulse"):
            so.lf_pulse(0.5, ra=0.02, rg=0.6, rk=0.5)
        # rk >= 1: the open phase's sine would pass zero before te
        for rk in (1.0, 1.2):
            with pytest.raises(ValueError, match="rk < 1"):
                so.lf_pulse(0.5, ra=0.02, rg=2.0, rk=rk)
        so.lf_pulse(0.5, ra=0.02, rg=2.0, rk=0.8)

    def test_pulse_repeats_every_period(self):
        x = np.linspace(0, 1, 101, endpoint=False)
        for flow in (False, True):
            np.testing.assert_allclose(
                so.lf_pulse(x + 1, 1.0, flow=flow), so.lf_pulse(x, 1.0, flow=flow), atol=1e-12
            )
            np.testing.assert_allclose(
                so.lf_pulse(x - 2, 1.0, flow=flow), so.lf_pulse(x, 1.0, flow=flow), atol=1e-12
            )

    def test_harmonic_numbers_must_be_whole_and_positive(self):
        for bad in ([0, 1], [1.5]):
            with pytest.raises(ValueError, match="whole numbers >= 1"):
                so.lf_harmonics(bad)

    def test_laxest_pulse_dips_just_below_the_excitation(self):
        """Near Rd 2.7 the open phase dips slightly below -1 before te (0.22%
        measured); at Rd 0.7 the lowest point is the excitation itself."""
        x = np.linspace(0, 1, 200001)
        assert -1.003 < so.lf_pulse(x, 2.7).min() < -1.001
        assert so.lf_pulse(x, 0.7).min() == pytest.approx(-1, abs=1e-4)

    def test_no_harmonic_below_f_max_is_refused(self):
        with pytest.raises(ValueError, match="no harmonic"):
            so.glottal_source(0.1, FS, 9000.0)

    def test_never_voiced_contour_is_silence_as_in_harmonic_complex(self):
        never = ([0.0, 0.1], [0.0, 0.0])
        silent = so.glottal_source(0.1, FS, never)
        assert silent.rms == 0 and len(silent) == len(so.harmonic_complex(0.1, FS, never))

    def test_rd_track_needs_one_value_per_time(self):
        with pytest.raises(ValueError, match="one value per time"):
            so.glottal_source(0.1, FS, 100.0, ([0.0, 0.1], [1.0]))

    def test_ramp_and_f_max_reach_the_contour(self):
        """On a contour with a gap, the source is harmonic_complex with the
        LF amplitudes and phases and the same ramp and f_max."""
        contour = ([0.0, 0.1, 0.1001, 0.2, 0.2001, 0.3], [120.0, 120.0, 0.0, 0.0, 120.0, 120.0])
        harmonic_numbers = np.arange(1, int(6000 // 120) + 1)
        coefficients = so.lf_harmonics(harmonic_numbers, 1.0)
        for ramp in (0.005, 0.03):
            source = so.glottal_source(0.3, FS, contour, 1.0, f_max=6000, ramp=ramp)
            want = so.harmonic_complex(
                0.3,
                FS,
                contour,
                harmonic_numbers,
                2 * np.abs(coefficients),
                np.angle(coefficients),
                f_max=6000,
                ramp=ramp,
            )
            np.testing.assert_allclose(source.data, want.data, atol=1e-12)
