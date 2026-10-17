import numpy as np
import pytest

import sonore as so
from sonore.views.chroma import PITCH_CLASSES

FS = 22050.0


def harmonic_tone(note, *, n_harmonics=10, duration=1.0):
    harmonics = np.arange(1, n_harmonics + 1)
    return so.harmonic_complex(
        duration, FS, so.note_to_freq(note), harmonics=harmonics, amplitudes=1 / harmonics
    )


def share(view, trim=4):
    """Each pitch class's share of the power, away from the first and last time windows."""
    power = view.power[trim:-trim, :, 0].mean(axis=0)
    return power / power.sum()


def test_a_pure_tone_lands_in_its_pitch_class_in_every_octave():
    for note in ["A2", "A4", "C#5", "G7"]:
        tone = so.pure_tone(1.0, FS, so.note_to_freq(note)).ramp(0.05)
        shares = share(so.chroma(tone))
        assert PITCH_CLASSES[int(np.argmax(shares))] == note[:-1]
        assert shares.max() > 0.99  # the filters spread the tone's start and end a little


def test_a_triad_holds_most_of_its_power_in_its_three_pitch_classes():
    triad = harmonic_tone("C4") + harmonic_tone("E4") + harmonic_tone("G4")
    shares = share(so.chroma(triad))
    assert set(np.argsort(shares)[-3:]) == {0, 4, 7}  # C, E, G
    assert 0.85 < shares[[0, 4, 7]].sum() < 0.95  # the rest from overtones


def test_power_is_the_power_of_the_sound_in_the_bands():
    noise = so.gaussian_noise(2.0, FS, rng=0)
    view = so.chroma(noise, f_lo=110, f_hi=3520)
    in_bands = so.cosine_filterbank(scale=so.cents_scale(), centers=view.f_lo * 2 ** (np.arange(-1, 62) / 12))
    expected = np.mean(in_bands.analyze(noise).data[:, 1:-1, 0] ** 2, axis=0).sum()
    np.testing.assert_allclose(view.power[:, :, 0].sum(axis=1).mean(), expected, rtol=0.02)
    assert view.power.shape == (40, 12, 1)
    np.testing.assert_allclose(np.diff(view.t), 0.05, rtol=1e-3)


def test_reference_moves_the_bands():
    tone = so.pure_tone(1.0, FS, 432.0).ramp(0.05)
    assert share(so.chroma(tone, reference=432.0))[9] > 0.99  # A
    assert share(so.chroma(tone))[9] < 0.9  # 432 Hz is a third of a semitone flat of 440


@pytest.mark.parametrize("octaves", [None, "C4", 300.0, lambda freq: 1 / np.sqrt(freq)])
def test_to_sound_gives_back_the_chroma(octaves):
    triad = harmonic_tone("C4") + harmonic_tone("E4") + harmonic_tone("G4")
    view = so.chroma(triad)
    back = so.chroma(view.to_sound(FS, octaves=octaves))
    np.testing.assert_allclose(share(back), share(view), atol=0.005)
    np.testing.assert_allclose(back.power[4:-4].sum(), view.power[4:-4].sum(), rtol=0.05)


def test_to_sound_places_the_octave():
    view = so.chroma(so.pure_tone(0.5, FS, so.note_to_freq("A4")))
    low = so.chroma(view.to_sound(FS, octaves="A2"))
    high = so.chroma(view.to_sound(FS, octaves="A6"))
    assert share(low, trim=2)[9] > 0.99 and share(high, trim=2)[9] > 0.99
    for note, freq in [("A2", 110.0), ("A6", 1760.0)]:
        spectrum = so.long_term_spectrum(view.to_sound(FS, octaves=note))
        assert spectrum.f[np.argmax(spectrum.level)] == pytest.approx(freq, rel=0.05)


def test_is_a_view():
    view = so.chroma(so.pure_tone(0.5, FS, 440.0))
    with pytest.raises(so.NotInvertibleError, match="octave height"):
        view.synthesize()
    with pytest.raises(ValueError, match="octaves"):
        view.to_sound(FS, octaves=-1.0)
    with pytest.raises(ValueError, match="shorter than one time window"):
        so.chroma(so.pure_tone(0.01, FS, 440.0))


def test_view_records_its_grid():
    view = so.chroma(so.pure_tone(1.0, FS, 432.0), window=0.0501, reference=432.0, f_lo=50, f_hi=4000)
    assert view.reference == 432.0
    # both ends moved to the nearest semitone with A4 at 432 Hz
    for given, kept in [(50, view.f_lo), (4000, view.f_hi)]:
        assert kept == pytest.approx(432 * 2 ** (round(12 * np.log2(given / 432)) / 12), rel=1e-12)
    assert view.f_lo != 50 and view.f_hi != 4000
    # the window is a whole number of samples, rounded: 0.0501 s is 1104.7 samples
    assert view.window == 1105 / FS
    np.testing.assert_allclose(view.t, (np.arange(len(view.t)) + 0.5) * 1105 / FS)
    assert "time windows of" in repr(view)


@pytest.mark.parametrize("kwargs", [{"window": 0.0}, {"f_lo": 440, "f_hi": 450}])
def test_refuses_bad_options(kwargs):
    with pytest.raises(ValueError):
        so.chroma(so.pure_tone(0.5, FS, 440.0), **kwargs)


def test_windows_start_at_the_start():
    # a tone in the first time window only; a last, shorter stretch is dropped
    tone = so.pure_tone(0.05, FS, 440.0).ramp(0.005)
    sound = so.Sound(np.concatenate([tone.data, np.zeros((int(0.98 * FS), 1))]), FS)
    view = so.chroma(sound)
    assert view.power[0, 9, 0] > 10 * view.power[1:, 9, 0].max()  # the filters ring a little
    assert view.power[0, 9, 0] > 0.5  # 0.63: the whole tone (RMS 1 less its ramps) in the first window


def test_channels_are_kept_apart():
    left, right = so.pure_tone(1.0, FS, 440.0), so.pure_tone(1.0, FS, so.note_to_freq("C5"))
    view = so.chroma(so.Sound(np.hstack([left.data, right.data]), FS))
    assert view.n_channels == 2
    back = so.chroma(view.to_sound(FS))
    for sound_view in (view, back):
        shares = sound_view.power[4:-4].mean(axis=0) / sound_view.power[4:-4].mean(axis=0).sum(axis=0)
        assert shares[9, 0] > 0.99 and shares[0, 1] > 0.99


def _power_weighted_spread(sound):
    """Mean and standard deviation [octaves] of log2 frequency, weighted by power."""
    spectrum = so.long_term_spectrum(sound)
    power, octaves = 10 ** (spectrum.level / 10), np.log2(np.maximum(spectrum.f, 1.0))
    mean = np.sum(power * octaves) / power.sum()
    return mean, np.sqrt(np.sum(power * (octaves - mean) ** 2) / power.sum())


def test_shepard_bell_is_centered_and_one_octave_wide():
    view = so.chroma(so.pure_tone(1.0, 44100.0, 440.0))
    mean, spread = _power_weighted_spread(view.to_sound(44100.0))
    # amplitude bell of 1 octave: power bell of 1/sqrt(2) octave, at sqrt(55 * 7040) = 622 Hz
    assert 2**mean == pytest.approx(np.sqrt(55 * 7040), rel=0.1)
    assert spread == pytest.approx(1 / np.sqrt(2), abs=0.1)


def test_a_chosen_octave_is_narrow():
    view = so.chroma(so.pure_tone(1.0, FS, 440.0))
    spectrum = so.long_term_spectrum(view.to_sound(FS, octaves="A4"))
    power = 10 ** (spectrum.level / 10)
    near = np.abs(np.log2(np.maximum(spectrum.f, 1.0) / 440)) < 1 / 12
    assert power[near].sum() > 0.99 * power.sum()  # the next octaves are 70 dB down


def test_note_follows_the_reference():
    from sonore.views.chroma import _octave_level

    level = _octave_level("C4", 432.0, 50.0, 4000.0)
    assert level(np.array([so.note_to_freq("C4", a4=432.0)]))[0] == pytest.approx(1.0)


def _peak_frequency(sound):
    """The frequency [Hz] of the strongest FFT bin of the whole sound."""
    magnitude = np.abs(np.fft.rfft(sound.data[:, 0]))
    return np.fft.rfftfreq(len(sound.data), 1 / sound.fs)[np.argmax(magnitude)]


def test_partials_stay_within_the_measured_range():
    # bells beyond either end land on the partial at that end: A1 = 55 Hz and A8 = 7040 Hz
    view = so.chroma(so.pure_tone(1.0, 44100.0, 440.0))
    for note, freq in [("A0", 55.0), ("A9", 7040.0)]:
        assert _peak_frequency(view.to_sound(44100.0, octaves=note)) == pytest.approx(freq, rel=0.02)


def test_lowest_partial_is_kept_when_the_range_starts_on_its_pitch_class():
    # f_lo=50 moves to G1 (49.0 Hz), which rounding puts a hair below an octave of G
    view = so.chroma(so.pure_tone(1.0, FS, so.note_to_freq("G3")), f_lo=50)
    assert _peak_frequency(view.to_sound(FS, octaves="G1")) == pytest.approx(49.0, rel=0.02)


def test_to_sound_length_and_silence():
    view = so.chroma(so.pure_tone(1.0, FS, 440.0))
    assert len(view.to_sound(16000.0).data) == round(view.duration * 16000)  # 15992.7 samples
    silent = so.chroma(so.Sound(np.zeros((int(FS), 1)), FS)).to_sound(FS)
    assert np.all(silent.data == 0)


def test_level_follows_each_time_window():
    # A for the first half, C for the second: the sound made back switches in the same window
    first, second = so.pure_tone(0.5, FS, 440.0), so.pure_tone(0.5, FS, so.note_to_freq("C5"))
    view = so.chroma(so.Sound(np.concatenate([first.data, second.data]), FS))
    back = so.chroma(view.to_sound(FS))

    def switch_time(chroma_view):  # where the A row's power has fallen by half
        row = chroma_view.power[:, 9, 0]
        return np.interp(-0.5, -row / row[:4].mean(), chroma_view.t)

    # the way back smooths over a time window either side, but keeps the switch in place
    assert switch_time(back) == pytest.approx(switch_time(view), abs=0.005)


def test_plot_names_the_pitch_classes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = so.chroma(so.pure_tone(0.5, FS, 440.0)).plot()
    assert [label.get_text() for label in ax.get_yticklabels()] == list(PITCH_CLASSES)
    plt.close(ax.figure)
