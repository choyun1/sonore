"""Spectra, STFTs, masks and modulation spectra."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestRepresentations:
    def test_spectrum_db_scale(self):
        g = so.gaussian_noise(0.5, FS, rng=0)
        d = so.Spectrum.from_sound(2 * g).level - so.Spectrum.from_sound(g).level
        assert np.median(d[1:]) == pytest.approx(20 * np.log10(2), abs=1e-6)

    @pytest.mark.parametrize("win", [20e-3, 20.1e-3])
    def test_stft_perfect_reconstruction(self, win):
        g = so.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(so.STFT(g, win).to_sound().data, g.data, atol=1e-10)

    def test_stft_stereo(self):
        s = so.correlated_noise(0.2, FS, corr=0, rng=0)
        S = so.STFT(s)
        assert S.data.shape[0] == 2
        np.testing.assert_allclose(S.to_sound().data, s.data, atol=1e-10)

    def test_stft_time_axis(self):
        S = so.STFT(so.gaussian_noise(1, FS, rng=0), 20e-3, 10e-3)
        assert np.diff(S.t) == pytest.approx(10e-3)

    def test_ibm_recovers_target_when_well_separated(self):
        target = so.pure_tone(1, FS, 500)
        masker = so.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
        mixture = target + masker
        mask = so.ideal_binary_mask(so.STFT(target), so.STFT(masker))
        out = (so.STFT(mixture) * mask).to_sound()
        snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
        assert snr > 20

    def test_griffin_lim_converges(self):
        x = so.harmonic_complex(0.5, 16000, 200, np.arange(1, 10))
        S = so.STFT(x, 32e-3)
        y = S.griffin_lim(n_iter=50, rng=0)
        err = np.linalg.norm(so.STFT(y, 32e-3).magnitude - S.magnitude) / np.linalg.norm(S.magnitude)
        assert err < 0.1

    def test_modulation_spectrum_peak(self):
        x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 8, depth=1)
        ms = so.ModulationSpectrum(so.STFT(x, 20e-3))
        row = ms.level[0]
        pos = ms.w_t > 2
        assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)


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


def test_tv_and_power_plots():
    import matplotlib

    matplotlib.use("Agg")
    t = np.arange(0, 0.3, 0.005)
    snd = _pulses(125.0, 0.3, _RFS)
    tv = so.TVGaborFrame.pitch_adaptive(t, np.full_like(t, 125.0), t_end=0.3)
    ax = tv.analyze(snd).plot(fmax=4000)
    assert ax.get_title() == "Time-varying spectrogram" and ax.get_ylim() == (0, 4.0)
    ax = so.tandem_power(snd, t, np.full_like(t, 125.0)).plot(db_range=40, title="TANDEM")
    lo, hi = ax.collections[0].get_clim()
    assert hi - lo == 40 and ax.get_title() == "TANDEM"
