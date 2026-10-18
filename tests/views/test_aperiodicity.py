"""The aperiodicities (so.harmonic_aperiodicity and so.band_aperiodicity)
against vowels whose noise share is known. What they share with the envelope
is in test_spectral_envelope.py."""

import numpy as np
import pytest

import sonore as so

FS = 16000


# ------------------------------------------------- harmonic residual
def vowel(noise_db, vibrato=0.0, duration=0.5, seed=3):
    """Harmonics of 120 Hz falling 6 dB per octave, plus noise whose power
    in every band is noise_db relative to the harmonics' power there.

    The noise is shaped like the harmonics (white noise through the same
    tilt), so its share is the same at every frequency: its density is the
    harmonics' power per hertz times 10 ** (noise_db / 10).
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
    spectrum *= np.sqrt(harmonic_density * 10 ** (noise_db / 10) * FS / 2)
    spectrum[freqs > 0.45 * FS] = 0
    noise = np.fft.irfft(spectrum, len(sample_times))
    window_times = np.arange(0, duration, 0.005)
    window_f0 = 120 * (1 + vibrato * np.sin(2 * np.pi * 5.5 * window_times))
    return so.Sound(harmonics + noise, FS), (window_times, window_f0)


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
    expected = 10 ** (noise_db / 10) / (1 + 10 ** (noise_db / 10))
    assert np.allclose(10 * np.log10(shares.mean(axis=1)), 10 * np.log10(expected), atol=1.5)


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
