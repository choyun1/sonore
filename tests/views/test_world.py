"""The synthesis port against WORLD's own output and the envelope written from
the CheapTrick paper against WORLD's envelope, both stored by
tools/make_world_fixtures.py (pyworld is not needed here); the aperiodicities
against vowels whose noise share is known."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sonore as so
from sonore.views.world import _GaussianNoise, world_randn

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


@pytest.mark.parametrize("name", ["vowel", "sentence"])
def test_synthesis_is_worlds(reference, name):
    sound, track = case(reference, name)
    y = so.source_filter_synthesize(
        track, so.pitch_adaptive_envelope(sound, track), so.band_aperiodicity(sound, track)
    )
    expected = reference[f"{name}_synthesized"]
    assert y.n_samples == len(expected) and y.fs == FS
    assert np.max(np.abs(y.data[:, 0] - expected)) < 1e-9 * np.max(np.abs(expected))


@pytest.mark.parametrize("variant", ["", "_ap_0_8", "_f0_16_3", "_f0_rising", "_hop_10_3"])
def test_synthesis_is_worlds_across_voicing_edges(reference, variant):
    # a stretch that goes from voiced to unvoiced and back, so that the
    # unvoiced pulses and noise, the voicing edges and the last time window
    # are held to WORLD as well; then with a noise share high enough at 0 Hz
    # that the periodic part is dropped, an F0 just above the lowest WORLD
    # voices (16 Hz at 16 kHz), a rising F0 (pulses in the last hop, where
    # WORLD extends the track by a step), and a hop that is not a whole
    # number of nanoseconds
    sound, (times, f0) = case(reference, "edges")
    envelope, aperiodicity = (
        so.pitch_adaptive_envelope(sound, (times, f0)),
        so.band_aperiodicity(sound, (times, f0)),
    )
    if variant == "_ap_0_8":
        aperiodicity = so.Aperiodicity(np.full_like(aperiodicity.data, 0.8), times, FS, "constant")
    if variant == "_f0_16_3":
        f0 = np.full_like(f0, 16.3)
    if variant == "_f0_rising":
        f0 = np.linspace(100.0, 200.0, len(f0))
    if variant == "_hop_10_3":
        times = np.arange(len(times)) * (10 / 3) / 1000
        envelope = so.SpectralEnvelope(envelope.data, times, FS, envelope.q1)
        aperiodicity = so.Aperiodicity(aperiodicity.data, times, FS, aperiodicity.method)
    y = so.source_filter_synthesize((times, f0), envelope, aperiodicity)
    expected = reference[f"edges_synthesized{variant}"]
    assert y.n_samples == len(expected)
    assert np.max(np.abs(y.data[:, 0] - expected)) < 1e-9 * np.max(np.abs(expected))


def test_an_f0_track_object_and_a_one_channel_envelope_function(reference):
    sound, (times, f0) = case(reference, "vowel")
    envelope, aperiodicity = (
        so.pitch_adaptive_envelope(sound, (times, f0)),
        so.band_aperiodicity(sound, (times, f0)),
    )
    expected = so.source_filter_synthesize((times, f0), envelope, aperiodicity)

    track = SimpleNamespace(t=times, f0=f0[None, :])  # anything with .t and .f0 is a track
    assert np.array_equal(so.source_filter_synthesize(track, envelope, aperiodicity).data, expected.data)
    # an envelope function giving one channel is used for every channel of
    # the aperiodicity
    stereo = so.Aperiodicity(np.repeat(aperiodicity.data, 2, axis=0), times, FS, "d4c")
    y = so.source_filter_synthesize((times, f0), lambda t, f: envelope(t, f)[:1], stereo)
    assert y.n_channels == 2
    assert np.allclose(y.data[:, 1], expected.data[:, 0], rtol=0, atol=1e-9 * np.abs(expected.data).max())


def test_a_track_off_the_grid_is_voiced_where_its_nearest_window_is(reference):
    sound, (times, f0) = case(reference, "vowel")
    envelope, aperiodicity = (
        so.pitch_adaptive_envelope(sound, (times, f0)),
        so.band_aperiodicity(sound, (times, f0)),
    )
    # a track at half the windows' spacing, offset by a quarter: every grid
    # time is a quarter of a hop from one given time and three quarters from
    # the other, so the nearest is never the one below
    half = np.arange(2 * len(times)) * 0.0025 + 0.00125 - 0.0025
    coarse = np.where(np.arange(len(half)) % 4 < 2, 150.0, 0.0)
    y = so.source_filter_synthesize((half, coarse), envelope, aperiodicity)
    nearest = coarse[np.round((times - half[0]) / 0.0025).astype(int)]
    on_grid = np.where(nearest > 0, 150.0, 0.0)
    assert np.array_equal(y.data, so.source_filter_synthesize((times, on_grid), envelope, aperiodicity).data)
    # halfway between a voiced and an unvoiced given time, the earlier one wins
    tie = so.source_filter_synthesize(([0.0, 0.01, 0.02], [150.0, 0.0, 0.0]), envelope, aperiodicity)
    on_grid = np.zeros(len(times))
    on_grid[:2] = 150.0
    assert np.array_equal(
        tie.data, so.source_filter_synthesize((times, on_grid), envelope, aperiodicity).data
    )


def test_fresh_noise_is_standard_normal():
    assert np.array_equal(_GaussianNoise(5).draw(1000), np.random.default_rng(5).standard_normal(1000))


def test_fresh_noise_changes_only_the_noise(reference):
    sound, track = case(reference, "vowel")
    envelope, aperiodicity = so.pitch_adaptive_envelope(sound, track), so.band_aperiodicity(sound, track)
    world = so.source_filter_synthesize(track, envelope, aperiodicity)
    again = so.source_filter_synthesize(track, envelope, aperiodicity)
    assert np.array_equal(world.data, again.data)
    fresh = so.source_filter_synthesize(track, envelope, aperiodicity, rng=1)
    assert np.array_equal(fresh.data, so.source_filter_synthesize(track, envelope, aperiodicity, rng=1).data)
    assert not np.allclose(fresh.data, world.data)
    # with no aperiodic part (WORLD keeps a noise share of 1e-6) the noise
    # hardly matters
    periodic = so.Aperiodicity(np.full_like(aperiodicity.data, 0.0), aperiodicity.t, FS, "none")
    difference = so.source_filter_synthesize(track, envelope, periodic, rng=1) - so.source_filter_synthesize(
        track, envelope, periodic
    )
    assert so.rms(difference.data) < 0.01 * so.rms(world.data)


def test_channels_are_analyzed_separately(reference):
    sound, track = case(reference, "vowel")
    stereo = so.Sound(np.column_stack([sound.data[:, 0], 0.5 * sound.data[:, 0]]), FS)
    envelope = so.pitch_adaptive_envelope(stereo, track)
    mono = so.pitch_adaptive_envelope(sound, track)
    assert np.array_equal(envelope.data[0], mono.data[0])
    assert np.allclose(envelope.data[1], 0.25 * mono.data[0], rtol=1e-6)
    aperiodicity = so.band_aperiodicity(stereo, track)
    y = so.source_filter_synthesize(track, envelope, aperiodicity)
    assert y.n_channels == 2
    assert np.array_equal(
        y.data[:, 0], so.source_filter_synthesize(track, mono, so.band_aperiodicity(sound, track)).data[:, 0]
    )


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
    # an F0 track off the aperiodicity's time windows is read onto them, but
    # those time windows themselves must start at 0, as WORLD assumes
    late = (times + 0.01, f0)
    with pytest.raises(ValueError, match="evenly spaced from time 0"):
        so.source_filter_synthesize(
            late, so.pitch_adaptive_envelope(sound, late), so.band_aperiodicity(sound, late)
        )
    # any sampling rate is analyzed
    assert np.all(np.isfinite(so.pitch_adaptive_envelope(so.Sound(sound.data, 16000.5), (times, f0)).data))
    with pytest.raises(ValueError, match="12 kHz"):
        so.band_aperiodicity(so.Sound(sound.data, 8000), (times, f0))
    with pytest.raises(ValueError, match="0 \\(unvoiced\\) or positive"):
        so.pitch_adaptive_envelope(sound, (times, -f0))
    with pytest.raises(ValueError, match="channels"):
        so.pitch_adaptive_envelope(sound, (times, np.vstack([f0, f0])))
    envelope, aperiodicity = (
        so.pitch_adaptive_envelope(sound, (times, f0)),
        so.band_aperiodicity(sound, (times, f0)),
    )
    one = so.Aperiodicity(aperiodicity.data[:, :, :1], times[:1], FS, "d4c")
    with pytest.raises(ValueError, match="at least two time windows"):
        so.source_filter_synthesize((times[:1], f0[:1]), lambda t, f: envelope(t, f), one)
    # a 17 Hz voice beside an unvoiced stretch on a 200 ms hop puts pulses
    # further apart than the FFT is long (WORLD itself writes past its buffer)
    slow = np.arange(12) * 0.2
    flat = so.SpectralEnvelope(np.ones((1, 513, 12)), slow, FS, -0.15)
    with pytest.raises(ValueError, match="do not fit"):
        so.source_filter_synthesize(
            (slow, np.array([0, 0, 0, 17, 17, 17, 17, 0, 0, 17, 17, 0.0])),
            flat,
            so.Aperiodicity(np.full((1, 513, 12), 0.1), slow, FS, "constant"),
        )
    with pytest.raises(TypeError, match="envelope"):
        so.source_filter_synthesize((times, f0), envelope.data, aperiodicity)
    with pytest.raises(ValueError, match="gave shape"):
        so.source_filter_synthesize((times, f0), lambda t, f: envelope(t, f)[0], aperiodicity)


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
