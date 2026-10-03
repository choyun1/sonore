"""Spectra and time-frequency power."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


def test_spectrum_db_scale():
    g = so.gaussian_noise(0.5, FS, rng=0)
    d = so.Spectrum.from_sound(2 * g).level - so.Spectrum.from_sound(g).level
    assert np.median(d[1:]) == pytest.approx(20 * np.log10(2), abs=1e-6)


def _pulses(f0, dur, fs):
    t = np.arange(int(dur * fs)) / fs
    return so.Sound(sum(np.cos(2 * np.pi * h * f0 * t) for h in range(1, int(0.45 * fs / f0))), fs)


def test_tandem_power_cancels_the_period_rate_flicker():
    """On a pulse train the averaged pair is far steadier over time windows than either window alone."""
    fs, f0 = 16000, 125.0
    snd = _pulses(f0, 0.5, fs)
    track_t = np.arange(0, 0.5, 0.005)
    track = np.full_like(track_t, f0)
    p = so.tandem_power(snd, track_t, track)
    single = so.TVGaborFrame.pitch_adaptive(track_t, track, t_end=0.5, periods=2.5, window="blackman")
    ps = np.abs(single.analyze(snd).data[0]) ** 2
    inner = (p.t > 0.1) & (p.t < 0.4)
    band = (p.f > 300) & (p.f < 4000)

    def flicker(power, keep):
        q = power[band][:, keep]
        return np.median((q.max(1) - q.min(1)) / q.mean(1))

    assert flicker(p.power[0], inner) < 0.01
    assert flicker(ps, (np.asarray(single.times) > 0.1) & (np.asarray(single.times) < 0.4)) > 0.1
    assert p.power.shape == (1, len(p.f), len(p.t))
    assert np.all(np.isfinite(p.db))


def test_tv_and_power_plots():
    import matplotlib

    matplotlib.use("Agg")
    t = np.arange(0, 0.3, 0.005)
    snd = _pulses(125.0, 0.3, 16000)
    tv = so.TVGaborFrame.pitch_adaptive(t, np.full_like(t, 125.0), t_end=0.3)
    ax = tv.analyze(snd).plot(fmax=4000)
    assert ax.get_title() == "Time-varying spectrogram" and ax.get_ylim() == (0, 4.0)
    ax = so.tandem_power(snd, t, np.full_like(t, 125.0)).plot(db_range=40, title="TANDEM")
    lo, hi = ax.collections[0].get_clim()
    assert hi - lo == 40 and ax.get_title() == "TANDEM"


def test_to_sound_noise_carrier_is_shaped_noise():
    ltass = so.long_term_spectrum(so.gaussian_noise(1.0, FS, tilt=-3, rng=2))
    noise = ltass.to_sound(2.0, FS, rng=0)
    assert np.array_equal(noise.data, so.gaussian_noise(2.0, FS, spectrum=ltass, rng=0).data)
    with pytest.raises(TypeError):
        ltass.to_sound(1.0, FS, carrier="minimum", rng=0)


def test_to_sound_with_a_sounds_phase_gives_it_back():
    sound = so.harmonic_complex(0.3, FS, 220.0, np.arange(1, 12)) + 0.1 * so.gaussian_noise(0.3, FS, rng=1)
    rebuilt = so.Spectrum.from_sound(sound).to_sound(sound.duration, FS, carrier=sound)
    assert np.max(np.abs(rebuilt.data - sound.normalize().data)) < 1e-10
    with pytest.raises(ValueError):
        so.Spectrum.from_sound(sound).to_sound(1.0, FS, carrier=sound)


def test_to_sound_minimum_phase_recovers_a_minimum_phase_response():
    # a one-pole lowpass, h[n] = 0.9**n, is minimum phase; from its magnitude
    # alone the minimum-phase route must give it back
    length = 2048
    response = 0.9 ** np.arange(length)
    freqs = np.fft.rfftfreq(length, 1 / FS)
    spectrum = so.Spectrum(freqs, 20 * np.log10(np.abs(np.fft.rfft(response))))
    rebuilt = spectrum.to_sound(length / FS, FS, carrier="minimum")
    expected = so.Sound(response, FS).normalize()
    assert np.max(np.abs(rebuilt.data - expected.data)) < 1e-9


def test_unknown_carrier_is_refused():
    spectrum = so.Spectrum(np.array([0.0, 8000.0]), np.zeros(2))
    with pytest.raises(ValueError):
        spectrum.to_sound(0.1, FS, carrier="pink")
