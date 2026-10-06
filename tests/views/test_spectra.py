"""Spectra and spectrograms that keep power: the spectrum, TANDEM power and the reassigned spectrogram."""

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
    # the -200 dB floor on the density lifts the near-zero 0 Hz bin, an offset of about 2e-10
    assert np.max(np.abs(rebuilt.data - sound.normalize().data)) < 1e-9
    with pytest.raises(ValueError, match="shorter than duration"):
        so.Spectrum.from_sound(sound).to_sound(1.0, FS, carrier=sound)
    with pytest.raises(ValueError, match="carrier fs"):
        so.Spectrum.from_sound(sound).to_sound(0.1, 22050, carrier=sound)


def test_to_sound_minimum_phase_recovers_a_minimum_phase_response():
    # a one-pole lowpass, h[n] = 0.9**n, is minimum phase; from its magnitude
    # alone the minimum-phase route must give it back
    length = 2048
    response = 0.9 ** np.arange(length)
    spectrum = so.Spectrum.from_sound(so.Sound(response, FS))
    rebuilt = spectrum.to_sound(length / FS, FS, carrier="minimum")
    expected = so.Sound(response, FS).normalize()
    assert np.max(np.abs(rebuilt.data - expected.data)) < 1e-9


def test_unknown_carrier_is_refused():
    spectrum = so.Spectrum(np.array([0.0, 8000.0]), np.zeros(2))
    with pytest.raises(ValueError):
        spectrum.to_sound(0.1, FS, carrier="pink")


def test_from_sound_and_long_term_spectrum_share_one_scale():
    """Both give a one-sided density in dB re 1 per Hz: 2 / fs for unit-variance white noise."""
    noise = so.gaussian_noise(2.0, FS, rng=3)
    expected = 10 * np.log10(2 / FS)
    single = so.Spectrum.from_sound(noise)
    averaged = so.long_term_spectrum(noise)
    inside = slice(1, -1)  # 0 Hz and Nyquist count once, not twice
    assert 10 * np.log10(np.mean(10 ** (single.level[inside] / 10))) == pytest.approx(expected, abs=0.1)
    assert 10 * np.log10(np.mean(10 ** (averaged.level[inside] / 10))) == pytest.approx(expected, abs=0.1)


def test_sound_from_a_spectrum_is_silent_above_its_top():
    """A spectrum measured at 16 kHz says nothing above 8 kHz, so a sound made from it at 44.1 kHz
    has nothing there."""
    measured = so.long_term_spectrum(so.gaussian_noise(1.0, 16000, tilt=-3, rng=4))
    for sound in (measured.to_sound(1.0, FS, rng=5), measured.to_sound(1.0, FS, carrier="minimum")):
        power = np.abs(np.fft.rfft(sound.data[:, 0])) ** 2
        freqs = np.fft.rfftfreq(len(sound), 1 / FS)
        assert power[freqs > 8001].max() < 1e-15 * power.max()
    # level_at itself still holds the end values
    assert measured.level_at(np.array([12000.0]))[0] == measured.level[-1]


def test_long_term_spectrum_pads_short_sounds():
    long, short = so.gaussian_noise(1.0, FS, rng=6), so.gaussian_noise(0.05, FS, rng=7)
    mixed = so.long_term_spectrum([long, short])
    assert np.array_equal(mixed.f, so.long_term_spectrum(long).f)
    # padding samples the short sound's own spectrum more finely and changes nothing else
    alone = so.long_term_spectrum(short, win_dur=0.05)
    np.testing.assert_allclose(so.long_term_spectrum(short).level[::2], alone.level, atol=1e-9)
    with pytest.raises(ValueError, match="at least one"):
        so.long_term_spectrum([])


# ---------------------------------------------------------------- reassignment
_RFS = 16000


_RT = np.arange(_RFS) / _RFS


_REASSIGN_FRAMES = {
    "hann": so.GaborFrame(0.032, 0.004),
    "gaussian": so.GaborFrame(0.032, 0.004, window=("gaussian", 64)),
}


def _reassign(x, frame, threshold_db):
    r = so.reassigned_spectrogram(so.Sound(x, _RFS), frame, threshold_db=threshold_db)
    return r, r.keep[0] & (r.t_hat[0] > 0.1) & (r.t_hat[0] < 0.9)


@pytest.mark.parametrize("window", _REASSIGN_FRAMES)
def test_reassignment_moves_a_tone_to_its_frequency(window):
    """0.37 bins off the grid; a sign slip in the correction would miss by about 100 Hz."""
    f_tone = 1000 + 0.37 * _RFS / 512
    r, keep = _reassign(np.cos(2 * np.pi * f_tone * _RT), _REASSIGN_FRAMES[window], -20)
    assert np.abs(r.f_hat[0][keep] - f_tone).max() < 0.2


@pytest.mark.parametrize("window", _REASSIGN_FRAMES)
def test_reassignment_moves_an_impulse_to_its_time(window):
    x = np.zeros(_RFS)
    x[8050] = 1.0
    r = so.reassigned_spectrogram(so.Sound(x, _RFS), _REASSIGN_FRAMES[window], threshold_db=-40)
    assert np.abs(r.t_hat[0][r.keep[0]] - 8050 / _RFS).max() < 1e-9


@pytest.mark.parametrize("window", _REASSIGN_FRAMES)
def test_reassignment_puts_a_chirp_on_its_line(window):
    r, keep = _reassign(np.cos(2 * np.pi * (500 * _RT + 1500 * _RT**2)), _REASSIGN_FRAMES[window], -20)
    assert np.abs(r.f_hat[0][keep] - (500 + 3000 * r.t_hat[0][keep])).max() < 0.2


def test_reassignment_binning_and_limits():
    s = so.correlated_noise(0.2, _RFS, corr=0, rng=0)
    r = so.reassigned_spectrogram(s, so.GaborFrame(0.016, 0.002), threshold_db=-200)
    assert r.t_hat.shape == r.power.shape and r.power.shape[0] == 2
    g = r.binned(np.linspace(-1, 2, 4), np.linspace(-1e5, 1e5, 3))
    # nearly all kept power lands in these wide cells; a few near-silent cells are thrown far outside
    np.testing.assert_allclose(g.power.sum(axis=(1, 2)), r.power.sum(axis=(1, 2), where=r.keep), rtol=1e-8)
    assert g.power.shape == (2, 2, 3)
    with pytest.raises(ValueError, match="hann"):
        so.reassigned_spectrogram(s, so.GaborFrame(0.016, window="hamming"))


def test_spectra_average_the_channels():
    left, right = so.gaussian_noise(0.5, FS, rng=8), so.gaussian_noise(0.5, FS, rng=9)
    stereo = so.Sound.from_channels(left, right)
    mono = so.Sound((left.data + right.data) / 2, FS)
    np.testing.assert_array_equal(so.Spectrum.from_sound(stereo).level, so.Spectrum.from_sound(mono).level)
    np.testing.assert_array_equal(so.long_term_spectrum(stereo).level, so.long_term_spectrum(mono).level)


def test_level_at_interpolates_linearly_and_relative_puts_the_peak_at_zero():
    spectrum = so.Spectrum(np.array([0.0, 1000.0, 2000.0]), np.array([-10.0, -30.0, -20.0]))
    np.testing.assert_allclose(spectrum.level_at(np.array([250.0, 1500.0])), [-15.0, -25.0])
    np.testing.assert_allclose(spectrum.relative().level, [0.0, -20.0, -10.0])


def test_smooth_is_a_power_average_over_the_fraction_of_an_octave():
    noise = so.Spectrum.from_sound(so.gaussian_noise(0.5, FS, rng=10))
    smoothed = noise.smooth(1 / 3)
    power = 10 ** (noise.level / 10)
    for index in (100, 2000, 9000):
        f = noise.f[index]
        band = (noise.f >= f / 2 ** (1 / 6)) & (noise.f <= f * 2 ** (1 / 6))
        assert smoothed.level[index] == pytest.approx(10 * np.log10(power[band].mean()), abs=1e-9)


def test_long_term_spectrum_weights_sounds_by_duration_and_resamples():
    loud, quiet = so.gaussian_noise(1.0, FS, rng=11) * 10, so.gaussian_noise(3.0, FS, rng=12)
    mixed = so.long_term_spectrum([loud, quiet])
    loud_power, quiet_power = (10 ** (so.long_term_spectrum(x).level / 10) for x in (loud, quiet))
    expected = (loud_power + 3 * quiet_power) / 4
    np.testing.assert_allclose(mixed.level, 10 * np.log10(expected), atol=1e-9)
    other_rate = so.gaussian_noise(1.0, 22050, rng=13)
    np.testing.assert_allclose(
        so.long_term_spectrum([loud, other_rate]).level,
        so.long_term_spectrum([loud, other_rate.resample(FS)]).level,
        atol=1e-9,
    )


def test_tandem_power_db_and_schedule():
    fs, f0 = 16000, 125.0
    snd = _pulses(f0, 0.5, fs)
    track_t = np.arange(0, 0.5, 0.005)
    power = so.tandem_power(snd, track_t, np.full_like(track_t, f0))
    np.testing.assert_allclose(power.db, np.maximum(10 * np.log10(power.power), -200.0))
    # the windows run past the end of the sound, so its last samples are analysed
    assert power.t[-1] >= snd.duration


def test_reassignment_threshold_is_per_channel_and_drops_silence():
    tone = np.cos(2 * np.pi * 1000 * _RT)
    quiet_right = np.column_stack([tone, 1e-3 * tone])
    quiet_right[: _RFS // 4] = 0.0  # a silent start, where the reassigned positions are 0/0
    r = so.reassigned_spectrogram(so.Sound(quiet_right, _RFS), _REASSIGN_FRAMES["hann"], threshold_db=-20)
    assert r.keep[1].any(), "the right channel, 60 dB down, keeps its own strongest cells"
    assert not np.isfinite(r.t_hat).all(), "the silent start gives positions that are not finite"
    assert np.isfinite(r.t_hat[r.keep]).all() and np.isfinite(r.f_hat[r.keep]).all()


def test_binned_centres_follow_the_edges():
    r = so.reassigned_spectrogram(so.gaussian_noise(0.2, _RFS, rng=0), so.GaborFrame(0.016, 0.002))
    grid = r.binned(np.array([0.0, 0.1, 0.2]), np.array([0.0, 2000.0, 4000.0, 8000.0]))
    np.testing.assert_allclose(grid.t, [0.05, 0.15])
    np.testing.assert_allclose(grid.f, [1000.0, 3000.0, 6000.0])
    assert grid.power.shape == (1, 3, 2)
