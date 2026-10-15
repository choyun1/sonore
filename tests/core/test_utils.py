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
        convert(1000.0, scale="bark")


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


def test_named_cents_scale_is_cents_from_a4_at_440_hz():
    from sonore.core.utils import FREQUENCY_SCALES

    named = FREQUENCY_SCALES["cents"]
    np.testing.assert_allclose(named.to_scale([440.0, 880.0, 261.6255653]), [0, 1200, -900], atol=1e-6)


def test_cents_scales_with_one_reference_are_equal():
    from sonore.core.utils import FREQUENCY_SCALES

    assert so.cents_scale() == so.cents_scale(440) == FREQUENCY_SCALES["cents"]
    assert so.cents_scale(432.0) != so.cents_scale()
    bank = so.cosine_filterbank(f_lo=110, f_hi=880, spacing=100, scale=so.cents_scale())
    assert bank == so.cosine_filterbank(f_lo=110, f_hi=880, spacing=100, scale="cents")


def test_cents_scale_needs_a_positive_reference():
    with pytest.raises(ValueError, match="above 0 Hz"):
        so.cents_scale(0.0)


def test_ratio_to_cents_and_back():
    assert so.ratio_to_cents(2.0) == pytest.approx(1200)
    assert so.ratio_to_cents(1.5) == pytest.approx(701.955, abs=1e-3)
    assert so.ratio_to_cents(440 / 432) == pytest.approx(31.767, abs=1e-3)
    ratios = np.array([0.5, 1.0, 5 / 4, 3.0])
    np.testing.assert_allclose(so.cents_to_ratio(so.ratio_to_cents(ratios)), ratios)


def test_cents_scale_uses_the_same_formula():
    freqs = np.array([110.0, 432.0, 1000.0])
    np.testing.assert_array_equal(so.cents_scale().to_scale(freqs), so.ratio_to_cents(freqs / 440.0))


def test_note_to_freq_is_equal_tempered_from_a4():
    assert so.note_to_freq("A4") == 440.0
    assert so.note_to_freq("A4", a4=432.0) == 432.0
    assert so.note_to_freq("C4") == pytest.approx(261.6256, abs=1e-4)
    assert so.note_to_freq("A3") == pytest.approx(220.0)
    assert so.note_to_freq("C#3") == so.note_to_freq("Db3") == pytest.approx(138.5913, abs=1e-4)
    assert so.note_to_freq("B♭2") == so.note_to_freq("Bb2")
    # an accidental can carry into the next octave's letter
    assert so.note_to_freq("B#3") == pytest.approx(so.note_to_freq("C4"))
    assert so.note_to_freq("Cbb5") == pytest.approx(so.note_to_freq("Bb4"))


def test_note_to_freq_reads_every_spelling_it_documents():
    assert so.note_to_freq("C♯3") == so.note_to_freq("C#3")
    assert so.note_to_freq("C##4") == pytest.approx(so.note_to_freq("D4"))
    assert so.note_to_freq(" a4 ") == 440.0  # either case, spaces around ignored
    assert so.note_to_freq("A-1") == pytest.approx(440 / 32)  # octaves may be negative


def test_note_to_freq_needs_a_positive_a4():
    with pytest.raises(ValueError, match="above 0 Hz"):
        so.note_to_freq("A4", a4=0.0)


@pytest.mark.parametrize("note", ["H4", "A", "C#x", "", "A 4", "A+4", "A4.0", "A\u0664"])
def test_note_to_freq_refuses_names_it_cannot_read(note):
    with pytest.raises(ValueError):
        so.note_to_freq(note)


# -- reading between samples ---------------------------------------------------
def _sinc_read(signal, positions, half=16, beta=8.0):
    """The 32-tap Kaiser-windowed sinc computed afresh for every position, written out
    from its formula (no table), as the reference for the tabulated reader."""
    from scipy.special import i0

    padded = np.concatenate([np.zeros(2 * half), signal, np.zeros(2 * half)])
    offsets = np.arange(-half + 1, half + 1)
    whole = np.floor(positions).astype(int)
    distance = positions[:, None] - (whole[:, None] + offsets[None, :])
    window = i0(beta * np.sqrt(np.clip(1 - (distance / half) ** 2, 0, None))) / i0(beta)
    return np.sum(padded[whole[:, None] + offsets[None, :] + 2 * half] * np.sinc(distance) * window, axis=1)


def _db(error, reference):
    return 20 * np.log10(np.sqrt(np.mean(error**2) / np.mean(reference**2)))


def test_reading_on_a_sample_gives_it_back():
    from sonore.core.utils import _read_between_samples

    signal = np.random.default_rng(0).standard_normal(400)
    np.testing.assert_allclose(_read_between_samples(signal, np.arange(400.0)), signal, rtol=0, atol=1e-12)


def test_the_table_matches_the_sinc_computed_per_sample():
    # 512 tabulated fractions, interpolated: -117 dB for white noise when measured
    from sonore.core.utils import _read_between_samples

    rng = np.random.default_rng(1)
    noise = rng.standard_normal(8000)
    positions = rng.uniform(100, 7900, 4000)
    difference = _read_between_samples(noise, positions) - _sinc_read(noise, positions)
    assert _db(difference, noise) < -110


@pytest.mark.parametrize("fraction, limit_db", [(0.04, -90), (0.33, -85), (0.42, -77)])
def test_a_tone_read_between_samples_is_accurate(fraction, limit_db):
    # measured at 48 kHz: -95 dB at 1 kHz, -91 dB at 16 kHz, -80 dB at 20 kHz
    from sonore.core.utils import _read_between_samples

    n = 20000
    positions = np.random.default_rng(2).uniform(1000, n - 1000, 5000)
    tone = np.sin(2 * np.pi * fraction * np.arange(n))
    truth = np.sin(2 * np.pi * fraction * positions)
    assert _db(_read_between_samples(tone, positions) - truth, truth) < limit_db


@pytest.mark.parametrize("stretch", [2.0, 3.0, 4.5])
def test_a_stretched_read_cuts_off_at_the_new_nyquist(stretch):
    # measured: -0.3 dB at 0.9 of the new Nyquist frequency, -30 dB at 1.1, below -85 dB from 1.3
    from sonore.core.utils import _read_between_samples

    n = 20000
    positions = np.arange(1000, n - 1000, stretch)

    def gain_db(of_new_nyquist):
        tone = np.sin(2 * np.pi * of_new_nyquist / (2 * stretch) * np.arange(n))
        read = _read_between_samples(tone, positions, stretch=stretch)
        return 20 * np.log10(np.sqrt(np.mean(read**2) / 0.5))

    assert abs(gain_db(0.5)) < 0.01 and abs(gain_db(0.8)) < 0.05
    assert -0.5 < gain_db(0.9) < -0.1
    assert -32 < gain_db(1.1) < -28
    assert gain_db(1.3) < -80 and gain_db(1.8) < -95


def test_reading_outside_the_signal_gives_zero_and_stretch_is_at_least_one():
    from sonore.core.utils import _read_between_samples

    signal = np.ones(100)
    assert np.all(_read_between_samples(signal, np.array([-40.0, -16.5, 116.5, 140.0])) == 0)
    assert (
        _read_between_samples(signal, np.array([-0.5]))[0] > 0.4
    )  # the sinc reaches in from the first sample
    with pytest.raises(ValueError, match="at least 1"):
        _read_between_samples(signal, np.array([10.0]), stretch=0.5)
