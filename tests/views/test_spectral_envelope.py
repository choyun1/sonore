"""The envelope written from the CheapTrick paper against WORLD's envelope,
stored by tools/make_pyworld_fixtures.py (pyworld is not needed here), and
what the envelope and the aperiodicity share: their grid, their channels and
the F0 tracks they refuse."""

from pathlib import Path

import numpy as np
import pytest

import sonore as so

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "pyworld_reference.npz"
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
    ours = np.ascontiguousarray(envelope.data[0].T)[:, ::step]
    difference = np.abs(10 * np.log10(ours / reference[f"{name}_envelope_paper_q1"]))[3:-3, 1:]
    assert np.median(difference) < 0.05
    assert np.percentile(difference, 95) < 0.5


def test_unvoiced_windows_are_close_to_worlds(reference):
    # both analyze an unvoiced time window as if F0 were 500 Hz; measured median
    # 0.006 dB and 95th percentile 1.4 dB (6.3 and 26 dB if analyzed at 71 Hz instead)
    sound, (times, _) = case(reference, "sentence")
    f0 = reference["sentence_unvoiced_f0"]
    unvoiced = f0 == 0
    assert unvoiced.sum() == 15
    envelope = so.pitch_adaptive_envelope(sound, (times, f0))
    ours = np.ascontiguousarray(envelope.data[0].T)[:, :: int(reference["bin_step"])]
    difference = np.abs(10 * np.log10(ours / reference["sentence_unvoiced_envelope_paper_q1"]))
    assert np.median(difference[unvoiced, 1:]) < 0.05
    assert np.percentile(difference[unvoiced, 1:], 95) < 2.0


def test_smoothing_is_the_moving_average_of_the_two_sided_spectrum():
    """Over +-2.7 bins of the linearly interpolated spectrum, mirrored about 0 Hz
    and Nyquist as a real signal's two-sided spectrum is, against a fine-grid sum."""
    from sonore.views.spectral_envelope import _smooth_rectangular

    values = np.random.default_rng(0).uniform(0.1, 1.0, 33)
    n_fft = 64
    two_sided = np.concatenate([values, values[-2:0:-1]])
    half_width = 2.7
    expected = []
    for center in range(len(values)):
        # the midpoints of 200000 equal steps across the width
        position = center - half_width + (np.arange(200_000) + 0.5) * (2 * half_width / 200_000)
        lower = np.floor(position).astype(int)
        fraction = position - lower
        interpolated = (1 - fraction) * two_sided[lower % n_fft] + fraction * two_sided[(lower + 1) % n_fft]
        expected.append(interpolated.mean())
    assert np.allclose(_smooth_rectangular(values, half_width), expected, rtol=1e-8)


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert envelope.plot().get_title() == "Spectral envelope (CheapTrick)"
    assert aperiodicity.plot().get_title() == "Aperiodicity (D4C)"
    plt.close("all")


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


def test_an_empty_band_does_not_make_the_envelope_nan():
    """Harmonics that stop below Nyquist leave the top of the spectrum empty. The smoothing
    clips its output at zero in case the difference of two cumulative sums comes out just
    below it there; this test guards that empty band, but does not reproduce a failure (with
    the clip removed it still passes, and no amplitude from 1e-3 to 1e6 made a NaN)."""
    fs, f0 = 16000, 200.0
    t = np.arange(round(0.4 * fs)) / fs
    harmonics = f0 * np.arange(1, int(0.45 * fs / f0) + 1)
    sound = so.Sound(np.cos(2 * np.pi * harmonics[:, None] * t).sum(0), fs)
    envelope = so.pitch_adaptive_envelope(sound, ([0.2], [f0]))
    assert np.all(np.isfinite(envelope.data)) and np.all(envelope.data > 0)


def test_method_settings_and_fft_size():
    times = np.arange(5) * 0.005
    freqs = np.arange(513) * FS / 1024
    power = np.ones((1, 513, 5))
    envelope = so.SpectralEnvelope(power, times, freqs, fs=FS, method="truth", settings={"q1": -0.15})
    assert envelope.method == "truth" and envelope.settings == {"q1": -0.15}
    assert envelope.n_fft == 1024 and envelope.fs == FS
    assert "truth" in repr(envelope)
    # a grid that is not the FFT's, or one with no sampling rate, has no FFT size
    assert so.SpectralEnvelope(power, times, freqs).n_fft is None
    assert so.SpectralEnvelope(power[:, :100], times, np.linspace(100, 5000, 100), fs=FS).n_fft is None
    bare = so.SpectralEnvelope(power, times, freqs)
    assert bare.method is None and bare.settings == {}
    with pytest.raises(ValueError, match="negative"):
        so.SpectralEnvelope(-power, times, freqs)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="SpectralEnvelope values must be finite"):
            so.SpectralEnvelope(np.where(np.arange(513)[:, None] == 7, bad, power), times, freqs)
        with pytest.raises(ValueError, match="Aperiodicity values must be finite"):
            so.Aperiodicity(np.where(np.arange(513)[:, None] == 7, bad, power), times, FS, "truth")
    # a zero reads as a very low, finite level in both views
    zeros = np.zeros((1, 513, 5))
    assert np.all(np.isfinite(so.SpectralEnvelope(zeros, times, freqs).db))
    assert np.all(np.isfinite(so.Aperiodicity(zeros, times, FS, "truth").db))
    assert np.allclose(so.Aperiodicity(0.1 * power, times, FS, "truth").db, -20.0)
    with pytest.raises(ValueError, match="shape"):
        so.SpectralEnvelope(power[:, :-1], times, freqs)


def test_estimators_label_their_views(reference):
    sound, track = case(reference, "vowel")
    cheap = so.pitch_adaptive_envelope(sound, track, q1=-0.1)
    assert cheap.method == "CheapTrick" and cheap.settings["q1"] == -0.1
    assert so.Cepstrum(so.STFT(sound)).envelope_view().method == "cepstrum"
    mfcc = so.MFCC(sound).envelope_view()
    assert mfcc.method == "MFCC" and mfcc.settings == {"n_mfcc": mfcc.settings["n_mfcc"]}
    lpc = so.LPC(sound, order=12).envelope_view()
    assert lpc.method == "LPC" and lpc.settings == {"order": 12}
    assert so.band_aperiodicity(sound, track).method == "D4C"


def test_bands_read_an_envelope_on_another_grid(reference):
    """The band shares are weighted by the envelope's power; an envelope given on a
    coarser grid is read at the aperiodicity's own times and frequencies."""
    sound, track = case(reference, "sentence")
    envelope = so.pitch_adaptive_envelope(sound, track)
    aperiodicity = so.band_aperiodicity(sound, track)
    fine = aperiodicity.bands([0, 1000, 4000, 8001], envelope=envelope)
    coarse = so.SpectralEnvelope(envelope.data[:, ::4, ::2], envelope.t[::2], envelope.f[::4])
    from_coarse = aperiodicity.bands([0, 1000, 4000, 8001], envelope=coarse)
    assert from_coarse.shape == fine.shape
    assert np.all((from_coarse > 0) & (from_coarse <= 1))
    # the same shares to within what the coarser sampling of the envelope changes
    assert np.median(np.abs(from_coarse - fine)) < 0.05
    with pytest.raises(TypeError):
        aperiodicity.bands([0, 8001], envelope=np.ones(513))
    with pytest.raises(ValueError, match="expected"):
        aperiodicity.bands([0, 8001], envelope=lambda t, f: np.ones((1, len(f) + 1, len(t))))


def test_any_grid_is_a_time_frequency_grid():
    class Flat:
        def __init__(self):
            self.t = np.arange(5) * 0.005
            self.f = np.arange(513) * FS / 1024
            self.data = np.ones((1, 513, 5))

        def __call__(self, t, f):
            return np.ones((1, len(f), len(t)))

    assert isinstance(Flat(), so.TimeFrequencyGrid)
    assert isinstance(so.SpectralEnvelope(np.ones((1, 513, 5)), Flat().t, Flat().f), so.TimeFrequencyGrid)
    assert not isinstance(np.ones(3), so.TimeFrequencyGrid)
    warped = so.warp_frequency(Flat(), 1.2)
    assert isinstance(warped, Flat) and warped.data.shape == (1, 513, 5)
