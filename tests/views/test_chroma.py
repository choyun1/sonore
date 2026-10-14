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
