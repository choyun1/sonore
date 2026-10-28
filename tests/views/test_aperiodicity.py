"""The aperiodicities (so.harmonic_aperiodicity and so.band_aperiodicity)
against vowels whose noise share is known. What they share with the envelope
is in test_spectral_envelope.py."""

import numpy as np
import pytest

import sonore as so

FS = 16000


# ------------------------------------------------- harmonic residual
def vowel(noise_db, vibrato=0.0, duration=0.5, seed=3, *, above_3k_db=None, tremolo=0.0):
    """Harmonics of 120 Hz falling 6 dB per octave, plus noise whose power
    in every band is noise_db relative to the harmonics' power there.

    The noise is shaped like the harmonics (white noise through the same
    tilt), so its share is the same at every frequency: its density is the
    harmonics' power per hertz times 10 ** (noise_db / 10). With
    ``above_3k_db`` the noise is that much relative to the harmonics above
    3 kHz instead. ``tremolo`` changes the whole sound's amplitude by that
    fraction at 6 Hz, which leaves the share unchanged.
    """
    rng = np.random.default_rng(seed)
    sample_times = np.arange(int(duration * FS)) / FS
    f0_per_sample = 120 * (1 + vibrato * np.sin(2 * np.pi * 5.5 * sample_times))
    phase = np.cumsum(2 * np.pi * f0_per_sample / FS)
    harmonics = np.zeros_like(sample_times)
    for number in range(1, int(0.45 * FS / 120 / (1 + vibrato))):
        harmonics += np.cos(number * phase + rng.uniform(0, 2 * np.pi)) / number
    # white noise with the harmonics' 1/f amplitude tilt: power per Hz of
    # the harmonics near f is (1/k)^2 / 2 / 120 with k = f / 120
    spectrum = np.fft.rfft(rng.standard_normal(len(sample_times)))
    freqs = np.fft.rfftfreq(len(sample_times), 1 / FS)
    harmonic_density = 0.5 / 120 / np.maximum(freqs / 120, 1) ** 2
    relative_db = np.where(freqs < 3000, noise_db, noise_db if above_3k_db is None else above_3k_db)
    spectrum *= np.sqrt(harmonic_density * 10 ** (relative_db / 10) * FS / 2)
    spectrum[freqs > 0.45 * FS] = 0
    noise = np.fft.irfft(spectrum, len(sample_times))
    signal = (harmonics + noise) * (1 + tremolo * np.sin(2 * np.pi * 6 * sample_times))
    window_times = np.arange(0, duration, 0.005)
    window_f0 = 120 * (1 + vibrato * np.sin(2 * np.pi * 5.5 * window_times))
    return so.Sound(signal, FS), (window_times, window_f0)


def share_db(noise_db):
    """The noise share of the power, in dB, for noise noise_db relative to the harmonics."""
    return 10 * np.log10(10 ** (noise_db / 10) / (1 + 10 ** (noise_db / 10)))


INSIDE = slice(11, 90)  # the time windows from 0.055 to 0.445 s, clear of the ends


@pytest.mark.parametrize("vibrato", [0.0, 0.03])
@pytest.mark.parametrize("noise_db", [-20.0, -6.0])
def test_harmonic_aperiodicity_reads_a_known_noise_share(noise_db, vibrato):
    sound, track = vowel(noise_db, vibrato)
    aperiodicity = so.harmonic_aperiodicity(sound, track)
    assert aperiodicity.method == "harmonic residual"
    inside = (track[0] > 0.05) & (track[0] < 0.45)
    shares = aperiodicity.bands([200, 1000, 3000, 6000], envelope=so.pitch_adaptive_envelope(sound, track))[
        0
    ][:, inside]
    # measured within 0.36 dB in every band and case
    assert np.allclose(10 * np.log10(shares.mean(axis=1)), share_db(noise_db), atol=0.7)


def test_harmonic_aperiodicity_follows_a_tremolo():
    # the fitted harmonics change amplitude across the window; without that
    # the bands read about 1 dB too noisy (measured within 0.14 dB with it)
    sound, track = vowel(-20.0, tremolo=0.3)
    envelope = so.pitch_adaptive_envelope(sound, track)
    shares = so.harmonic_aperiodicity(sound, track).bands([200, 1000, 3000, 6000], envelope=envelope)
    assert np.allclose(10 * np.log10(shares[0][:, INSIDE].mean(axis=1)), share_db(-20.0), atol=0.7)


def test_harmonic_aperiodicity_follows_a_step_in_the_share():
    """Noise 20 dB below the harmonics under 3 kHz and 6 dB above: the step
    is as sharp as the cells (two harmonics, 240 Hz) and the dB interpolation
    between their centers make it."""
    sound, track = vowel(-20.0, above_3k_db=-6.0)
    aperiodicity = so.harmonic_aperiodicity(sound, track)
    median = np.median(aperiodicity.db[0][:, INSIDE], axis=1)

    def at(freq):
        return median[np.searchsorted(aperiodicity.f, freq)]

    # measured -20.4, -21.1, -16.7, -7.3 and -7.7 dB
    assert abs(at(1500) - share_db(-20.0)) < 1.0
    assert at(2700) < -19.5  # still the low side one cell below the step (-18.3 with cells twice as wide)
    assert at(2850) < -15.0  # partway (-12.6 when interpolated in share instead of dB)
    assert abs(at(3300) - share_db(-6.0)) < 1.0 and abs(at(4500) - share_db(-6.0)) < 1.0


def test_harmonic_aperiodicity_refuses_a_window_too_short_for_its_fit():
    sound, track = vowel(-20.0, duration=0.1)
    with pytest.raises(ValueError, match="at least 2.49"):
        so.harmonic_aperiodicity(sound, track, periods=2.25)
    assert np.all(np.isfinite(so.harmonic_aperiodicity(sound, track, periods=2.49).data))


def test_harmonic_aperiodicity_of_a_periodic_sound_is_low():
    sound, track = vowel(-300.0)
    aperiodicity = so.harmonic_aperiodicity(sound, track)
    inside = (track[0] > 0.05) & (track[0] < 0.45)
    below = aperiodicity.f < 6000
    assert aperiodicity.db[0][below][:, inside].max() < -60


def test_unvoiced_windows_are_noise():
    sound, (times, f0) = vowel(-20.0)
    f0 = f0.copy()
    f0[:20] = 0
    aperiodicity = so.harmonic_aperiodicity(sound, (times, f0))
    assert np.all(aperiodicity.data[0, :, :20] == 1)
    assert np.all(so.band_aperiodicity(sound, (times, f0)).data[0, :, :20] == 1)


@pytest.mark.parametrize("noise_db", [-20.0, -10.0])
def test_band_aperiodicity_reads_a_known_noise_share_at_3_khz(noise_db):
    # with its 8 dB correction; at 16 kHz the 3 kHz band is the only one
    sound, track = vowel(noise_db)
    aperiodicity = so.band_aperiodicity(sound, track)
    assert aperiodicity.method == "D4C"
    inside = (track[0] > 0.05) & (track[0] < 0.45)
    at_3k = np.searchsorted(aperiodicity.f, 3000)
    expected = 10 * np.log10(10 ** (noise_db / 10) / (1 + 10 ** (noise_db / 10)))
    assert abs(np.median(aperiodicity.db[0][at_3k, inside]) - expected) < 1.5
    # the paper's values, uncorrected, read 8 dB more periodic
    paper = so.band_aperiodicity(sound, track, correction_db=0.0)
    assert np.allclose(paper.db[0][at_3k, inside], aperiodicity.db[0][at_3k, inside] - 8.0)


def test_band_aperiodicity_of_a_periodic_sound_is_low():
    sound, track = vowel(-300.0)
    aperiodicity = so.band_aperiodicity(sound, track)
    inside = (track[0] > 0.05) & (track[0] < 0.45)
    assert aperiodicity.db[0][:, inside].max() < -40


def test_bands_weight_by_the_envelope():
    """The vowel's top 800 Hz hold neither harmonics nor noise, so they read as
    all noise; weighted by the envelope they count for almost nothing."""
    sound, track = vowel(-20.0)
    aperiodicity = so.harmonic_aperiodicity(sound, track)
    envelope = so.pitch_adaptive_envelope(sound, track)
    plain = aperiodicity.bands([0, 8001])[0, 0, INSIDE]
    in_band = aperiodicity.f < 8001
    assert np.allclose(plain, aperiodicity.share[0][in_band][:, INSIDE].mean(axis=0))
    weighted = aperiodicity.bands([0, 8001], envelope=envelope)[0, 0, INSIDE]
    # measured -11.7 dB plain and -19.3 dB weighted
    assert abs(10 * np.log10(np.median(weighted)) - share_db(-20.0)) < 1.0
    assert 10 * np.log10(np.median(plain)) > share_db(-20.0) + 5
