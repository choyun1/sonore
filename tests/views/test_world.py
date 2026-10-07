"""The WORLD ports against WORLD's own output, stored by
tools/make_world_fixtures.py (pyworld is not needed here), and the
harmonic-residual aperiodicity against vowels whose noise share is known."""

from pathlib import Path

import numpy as np
import pytest

import sonore as so
from sonore.views.world import world_fft_size, world_randn

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "world_reference.npz"
FS = 16000


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE)


def case(reference, name):
    """The sound and F0 track WORLD analyzed, and its stored outputs."""
    sound = so.Sound(reference[f"{name}_sound"], FS)
    return sound, (reference[f"{name}_t"], reference[f"{name}_f0"])


def world_randn_by_loop(n):
    """WORLD's randn written out one step at a time (matlabfunctions.cpp)."""
    x, y, z, w = 123456789, 362436069, 521288629, 88675123
    out = []
    for _ in range(n):
        total = 0
        for _ in range(12):
            shifted = (x ^ (x << 11)) & 0xFFFFFFFF
            x, y, z, w = y, z, w, (w ^ (w >> 19)) ^ (shifted ^ (shifted >> 8))
            total += w >> 4
        out.append(total / 268435456.0 - 6.0)
    return np.array(out)


def test_world_randn_matches_the_step_by_step_generator():
    expected = world_randn_by_loop(3000)
    assert np.array_equal(world_randn(3000), expected)
    # past the first block of lanes, and after the cache has grown
    long_stream = world_randn(400_000)
    assert np.array_equal(long_stream[:3000], expected)
    assert abs(long_stream.mean()) < 0.01 and abs(long_stream.std() - 1) < 0.01


def test_fft_size():
    assert world_fft_size(16000) == 1024
    assert world_fft_size(44100) == 2048


@pytest.mark.parametrize("name", ["vowel", "sentence"])
def test_cheaptrick_is_worlds(reference, name):
    sound, track = case(reference, name)
    step = int(reference["bin_step"])
    envelope = so.cheaptrick(sound, track)
    assert envelope.data.shape == (1, 513, len(track[0]))
    ours = envelope.to_world()[:, ::step]
    assert np.max(np.abs(10 * np.log10(ours / reference[f"{name}_envelope"]))) < 1e-6
    paper = so.cheaptrick(sound, track, q1=-0.09).to_world()[:, ::step]
    assert np.max(np.abs(10 * np.log10(paper / reference[f"{name}_envelope_paper_q1"]))) < 1e-6


@pytest.mark.parametrize("name", ["vowel", "sentence"])
def test_d4c_is_worlds(reference, name):
    sound, track = case(reference, name)
    step = int(reference["bin_step"])
    ours = so.d4c(sound, track).to_world()[:, ::step]
    assert np.max(np.abs(20 * np.log10(ours / reference[f"{name}_aperiodicity"]))) < 1e-8


@pytest.mark.parametrize("name", ["vowel", "sentence"])
def test_synthesis_is_worlds(reference, name):
    sound, track = case(reference, name)
    y = so.world_synthesize(track, so.cheaptrick(sound, track), so.d4c(sound, track))
    expected = reference[f"{name}_synthesized"]
    assert y.n_samples == len(expected) and y.fs == FS
    assert np.max(np.abs(y.data[:, 0] - expected)) < 1e-9 * np.max(np.abs(expected))


def test_fresh_noise_changes_only_the_noise(reference):
    sound, track = case(reference, "vowel")
    envelope, aperiodicity = so.cheaptrick(sound, track), so.d4c(sound, track)
    world = so.world_synthesize(track, envelope, aperiodicity)
    again = so.world_synthesize(track, envelope, aperiodicity)
    assert np.array_equal(world.data, again.data)
    fresh = so.world_synthesize(track, envelope, aperiodicity, rng=1)
    assert np.array_equal(fresh.data, so.world_synthesize(track, envelope, aperiodicity, rng=1).data)
    assert not np.allclose(fresh.data, world.data)
    # with no aperiodic part (WORLD keeps a noise share of 1e-6) the noise
    # hardly matters
    periodic = so.Aperiodicity(np.full_like(aperiodicity.data, 0.0), aperiodicity.t, FS, "none")
    difference = so.world_synthesize(track, envelope, periodic, rng=1) - so.world_synthesize(
        track, envelope, periodic
    )
    assert so.rms(difference.data) < 0.01 * so.rms(world.data)


def test_channels_are_analyzed_separately(reference):
    sound, track = case(reference, "vowel")
    stereo = so.Sound(np.column_stack([sound.data[:, 0], 0.5 * sound.data[:, 0]]), FS)
    envelope = so.cheaptrick(stereo, track)
    mono = so.cheaptrick(sound, track)
    assert np.array_equal(envelope.data[0], mono.data[0])
    assert np.allclose(envelope.data[1], 0.25 * mono.data[0], rtol=1e-6)
    aperiodicity = so.d4c(stereo, track)
    y = so.world_synthesize(track, envelope, aperiodicity)
    assert y.n_channels == 2
    assert np.array_equal(y.data[:, 0], so.world_synthesize(track, mono, so.d4c(sound, track)).data[:, 0])


def test_views(reference):
    sound, track = case(reference, "sentence")
    envelope = so.cheaptrick(sound, track)
    aperiodicity = so.d4c(sound, track)
    assert np.allclose(envelope.f, np.arange(513) * FS / 1024)
    assert np.allclose(envelope.t, track[0])
    # reading at window times and bin frequencies gives the stored values
    assert np.allclose(envelope(envelope.t[3:6], envelope.f[10:20]), envelope.data[:, 10:20, 3:6])
    assert np.allclose(aperiodicity.share, aperiodicity.data**2)
    # halfway between time windows, halfway in dB
    middle = envelope([(envelope.t[3] + envelope.t[4]) / 2], [envelope.f[40]])[0, 0, 0]
    assert np.isclose(10 * np.log10(middle), (envelope.db[0, 40, 3] + envelope.db[0, 40, 4]) / 2)
    bands = aperiodicity.bands([0, 4000, 8001], envelope)
    assert bands.shape == (1, 2, len(envelope.t))
    assert np.all((bands > 0) & (bands <= 1))
    assert "D4C" in repr(aperiodicity)


def test_bad_tracks_are_refused(reference):
    sound, (times, f0) = case(reference, "vowel")
    with pytest.raises(ValueError, match="values"):
        so.cheaptrick(sound, (times, f0[:-1]))
    with pytest.raises(TypeError):
        so.d4c(sound, f0)
    # an F0 track off the aperiodicity's time windows is read onto them, but
    # those time windows themselves must start at 0, as WORLD assumes
    late = (times + 0.01, f0)
    with pytest.raises(ValueError, match="evenly spaced from time 0"):
        so.world_synthesize(late, so.cheaptrick(sound, late), so.d4c(sound, late))
    with pytest.raises(ValueError, match="whole-number"):
        so.cheaptrick(so.Sound(sound.data, 16000.5), (times, f0))


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
    shares = aperiodicity.bands([200, 1000, 3000, 6000], so.cheaptrick(sound, track))[0][:, inside]
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
    assert np.all(so.d4c(sound, (times, f0)).data[0, :, :20] == 1 - 1e-12)
