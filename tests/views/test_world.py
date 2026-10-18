"""The envelope written from the CheapTrick paper against WORLD's envelope,
stored by tools/make_world_fixtures.py (pyworld is not needed here); the
aperiodicities against vowels whose noise share is known."""

from pathlib import Path

import numpy as np
import pytest

import sonore as so

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "world_reference.npz"
FS = 16000


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE)


def case(reference, name):
    """The sound and F0 track WORLD analyzed, and its stored outputs."""
    sound = so.Sound(reference[f"{name}_sound"], FS)
    return sound, (reference[f"{name}_t"], reference[f"{name}_f0"])


def test_fft_size():
    sound = so.Sound(np.zeros(4410), 44100)
    assert so.pitch_adaptive_envelope(sound, ([0.05], [100.0])).n_fft == 2048
    assert so.band_aperiodicity(sound, ([0.05], [100.0])).n_fft == 2048
    assert so.pitch_adaptive_envelope(so.Sound(np.zeros(1600), FS), ([0.05], [100.0])).n_fft == 1024


@pytest.mark.parametrize("name", ["vowel", "sentence"])
def test_envelope_is_close_to_worlds(reference, name):
    # written from the paper, so close to WORLD's at the paper's q1, not equal;
    # the first and last time windows (zeros beyond the ends where WORLD repeats
    # the end samples) and 0 Hz (mirrored where WORLD folds about F0 / 2) differ more
    sound, track = case(reference, name)
    step = int(reference["bin_step"])
    envelope = so.pitch_adaptive_envelope(sound, track)
    assert envelope.data.shape == (1, 513, len(track[0]))
    ours = envelope.to_world()[:, ::step]
    difference = np.abs(10 * np.log10(ours / reference[f"{name}_envelope_paper_q1"]))[3:-3, 1:]
    assert np.median(difference) < 0.05
    assert np.percentile(difference, 95) < 0.5


def test_channels_are_analyzed_separately(reference):
    sound, track = case(reference, "vowel")
    stereo = so.Sound(np.column_stack([sound.data[:, 0], 0.5 * sound.data[:, 0]]), FS)
    envelope = so.pitch_adaptive_envelope(stereo, track)
    mono = so.pitch_adaptive_envelope(sound, track)
    assert np.array_equal(envelope.data[0], mono.data[0])
    assert np.allclose(envelope.data[1], 0.25 * mono.data[0], rtol=1e-6)
    aperiodicity = so.band_aperiodicity(stereo, track)
    assert np.array_equal(aperiodicity.data[0], so.band_aperiodicity(sound, track).data[0])


def test_views(reference):
    sound, track = case(reference, "sentence")
    envelope = so.pitch_adaptive_envelope(sound, track)
    aperiodicity = so.band_aperiodicity(sound, track)
    assert np.allclose(envelope.f, np.arange(513) * FS / 1024)
    assert np.allclose(envelope.t, track[0])
    # reading at window times and bin frequencies gives the stored values
    assert np.allclose(envelope(envelope.t[3:6], envelope.f[10:20]), envelope.data[:, 10:20, 3:6])
    assert np.allclose(aperiodicity.share, aperiodicity.data**2)
    # halfway between time windows, halfway in dB
    middle = envelope([(envelope.t[3] + envelope.t[4]) / 2], [envelope.f[40]])[0, 0, 0]
    assert np.isclose(10 * np.log10(middle), (envelope.db[0, 40, 3] + envelope.db[0, 40, 4]) / 2)
    bands = aperiodicity.bands([0, 4000, 8001], envelope=envelope)
    assert bands.shape == (1, 2, len(envelope.t))
    assert np.all((bands > 0) & (bands <= 1))
    assert "D4C" in repr(aperiodicity)


def test_bad_tracks_are_refused(reference):
    sound, (times, f0) = case(reference, "vowel")
    with pytest.raises(ValueError, match="values"):
        so.pitch_adaptive_envelope(sound, (times, f0[:-1]))
    with pytest.raises(TypeError):
        so.band_aperiodicity(sound, f0)
    # any sampling rate is analyzed
    assert np.all(np.isfinite(so.pitch_adaptive_envelope(so.Sound(sound.data, 16000.5), (times, f0)).data))
    with pytest.raises(ValueError, match="12 kHz"):
        so.band_aperiodicity(so.Sound(sound.data, 8000), (times, f0))
    with pytest.raises(ValueError, match="0 where unvoiced"):
        so.pitch_adaptive_envelope(sound, (times, -f0))
    with pytest.raises(ValueError, match="channels"):
        so.pitch_adaptive_envelope(sound, (times, np.vstack([f0, f0])))


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
