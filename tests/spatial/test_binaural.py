"""Binaural manipulation and cue analysis."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestBinaural:
    def test_sign_convention_written_by_hand(self):
        """Positive ITD and ILD point right (module docstring): the right ear leads and is louder.
        Checked on the channels directly, so a sign flipped in both apply_itd_ild and
        interaural_cues cannot pass the round trip below unnoticed."""
        click = so.Sound(np.r_[1.0, np.zeros(99)], FS)
        b = so.apply_itd_ild(click, itd=10 / FS, ild=6)
        left, right = b.data[:, 0], b.data[:, 1]
        assert np.argmax(right) == 0 and np.argmax(left) == 10
        assert 20 * np.log10(right.max() / left.max()) == pytest.approx(6, abs=1e-12)

    def test_interaural_cues_sign_on_a_hand_made_sound(self):
        """The right channel is the left one 8 samples earlier and twice as large: ITD +8/FS, ILD +6.02 dB."""
        g = so.gaussian_noise(1, FS, band=(100, 8000), rng=0).data[:, 0]
        left, right = g[:-8], 2 * g[8:]
        c = so.interaural_cues(so.Sound(np.column_stack([left, right]), FS), win_dur=50e-3)
        assert np.nanmedian(c.itd) == pytest.approx(8 / FS, abs=1e-6)
        # each window sees slightly different noise samples in the two ears, hence not exact
        assert np.nanmedian(c.ild) == pytest.approx(20 * np.log10(2), abs=0.01)

    @pytest.mark.parametrize("itd", [300e-6, -500e-6, 123.4e-6])
    def test_itd_ild_roundtrip(self, itd):
        g = so.gaussian_noise(1, FS, band=(100, 8000), rng=0)
        b = so.apply_itd_ild(g, itd=itd, ild=6)
        c = so.interaural_cues(b, win_dur=50e-3)
        assert np.nanmedian(c.itd) == pytest.approx(itd, abs=3e-6)
        assert np.nanmedian(c.ild) == pytest.approx(6, abs=0.05)
        # fractional lags: the sampled CCF peak sits slightly below its true height
        assert np.nanmedian(c.iac) > (0.995 if abs(itd * FS - round(itd * FS)) < 1e-6 else 0.99)

    def test_simple_bir_is_level_independent_of_itd(self):
        a, b = so.simple_bir(FS, itd=100e-6), so.simple_bir(FS, itd=500e-6)
        assert np.sum(a.data**2) == pytest.approx(np.sum(b.data**2), rel=2e-2)

    def test_per_band_cues(self):
        b = so.apply_itd_ild(so.gaussian_noise(0.5, FS, rng=0), itd=200e-6)
        c = so.interaural_cues(b, win_dur=20e-3, filterbank=so.cosine_filterbank(8, 200, 1500))
        assert c.itd.shape[1] == 10
        assert np.nanmedian(c.itd[:, 2:-2]) == pytest.approx(200e-6, abs=10e-6)

    def test_oscor_and_phasewarp_correlation_oscillates(self):
        for make, fn in [(so.oscor, np.sin), (so.phasewarp, np.cos)]:
            s = make(2, FS, 2, rng=0)
            L, R = s.data.T
            w = int(0.02 * FS)
            centers = np.arange(w, len(s) - w, w)
            r = [np.corrcoef(L[c - w // 2 : c + w // 2], R[c - w // 2 : c + w // 2])[0, 1] for c in centers]
            expected = fn(2 * np.pi * 2 * centers / FS)
            assert np.corrcoef(r, expected)[0, 1] > 0.95


def test_zero_lag_correlation_tracks_oscor():
    s = so.oscor(2, FS, 2, rng=0)
    c = so.interaural_cues(s, win_dur=10e-3)
    ok = np.isfinite(c.corr0)
    r = np.corrcoef(c.corr0[ok], np.sin(2 * np.pi * 2 * c.t[ok]))[0, 1]
    assert r > 0.95


def reference_cues(sound, win_dur, hop_dur, max_itd):
    """interaural_cues written out sample by sample from its docstring: Hann
    windows, the normalized cross-correlation divided by the window's own
    autocorrelation, its peak within +-max_itd refined by a parabola."""
    fs = sound.fs
    n_win, hop = round(win_dur * fs), round(hop_dur * fs)
    max_lag = int(np.ceil(max_itd * fs))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_win) / n_win)
    window_acf = np.array([np.sum(window[k:] * window[: n_win - k]) for k in range(n_win)])
    lags = np.arange(-max_lag, max_lag + 1)
    itds, ilds, iacs, corr0s = [], [], [], []
    for start in range(0, len(sound) - n_win + 1, hop):
        left = sound.data[start : start + n_win, 0] * window
        right = sound.data[start : start + n_win, 1] * window
        norm = np.sqrt(np.sum(left**2) * np.sum(right**2))
        ccf = []
        for lag in lags:  # sum over n of left[n + lag] right[n]
            if lag >= 0:
                ccf.append(np.sum(left[lag:] * right[: n_win - lag]))
            else:
                ccf.append(np.sum(left[: n_win + lag] * right[-lag:]))
        ccf = np.array(ccf) / norm / (window_acf[np.abs(lags)] / window_acf[0])
        k = int(np.argmax(ccf))
        peak, delta = ccf[k], 0.0
        if 0 < k < len(lags) - 1:
            before, after = ccf[k - 1], ccf[k + 1]
            delta = 0.5 * (before - after) / (before - 2 * peak + after)
            peak = peak - 0.25 * (before - after) * delta
        itds.append((lags[k] + delta) / fs)
        iacs.append(min(peak, 1.0))
        ilds.append(10 * np.log10(np.sum(right**2) / np.sum(left**2)))
        corr0s.append(ccf[max_lag])
    return np.array(itds), np.array(ilds), np.array(iacs), np.array(corr0s)


class TestInterauralCuesInDetail:
    def test_matches_a_sample_by_sample_reference(self):
        # short windows, so the lag range is a large part of each window
        noise = so.gaussian_noise(0.05, 8000, n_channels=2, rng=1).data
        mixed = so.apply_itd_ild(so.Sound(noise[:, 0], 8000), itd=2.3 / 8000, ild=3) + 0.3 * so.Sound(
            np.r_[noise, np.zeros((3, 2))], 8000
        )
        c = so.interaural_cues(mixed, win_dur=4e-3, hop_dur=1e-3, max_itd=1.5e-3, silence_db=-200)
        itd, ild, iac, corr0 = reference_cues(mixed, 4e-3, 1e-3, 1.5e-3)
        np.testing.assert_allclose(c.itd, itd, atol=1e-12)
        np.testing.assert_allclose(c.ild, ild, atol=1e-9)
        np.testing.assert_allclose(c.iac, iac, atol=1e-12)
        np.testing.assert_allclose(c.corr0, corr0, atol=1e-12)

    def test_times_are_window_centers_on_the_hop(self):
        c = so.interaural_cues(so.gaussian_noise(0.2, FS, n_channels=2, rng=0), win_dur=20e-3, hop_dur=5e-3)
        n_win, hop = round(20e-3 * FS), round(5e-3 * FS)
        np.testing.assert_allclose(c.t, (np.arange(len(c.t)) * hop + n_win / 2) / FS)
        assert len(c.t) == (round(0.2 * FS) - n_win) // hop + 1

    def test_a_large_itd_keeps_its_coherence(self):
        """Dividing by the window's autocorrelation undoes the taper at long lags."""
        noise = so.gaussian_noise(1, FS, band=(100, 8000), rng=0)
        c = so.interaural_cues(so.apply_itd_ild(noise, itd=900e-6), win_dur=20e-3)
        assert np.nanmedian(c.iac) > 0.99
        assert np.nanmax(c.iac) <= 1.0

    def test_an_itd_beyond_the_range_stops_at_its_edge(self):
        # low-frequency noise, so the correlation still rises toward the edge
        noise = so.gaussian_noise(1, FS, band=(50, 150), rng=0)
        c = so.interaural_cues(so.apply_itd_ild(noise, itd=1.5e-3), win_dur=50e-3, max_itd=1e-3)
        np.testing.assert_array_equal(c.itd[np.isfinite(c.itd)], np.ceil(1e-3 * FS) / FS)

    def test_silence_threshold(self):
        noise = so.gaussian_noise(0.6, FS, n_channels=2, rng=0).data
        gains = np.r_[
            np.ones(round(0.2 * FS)),
            np.full(round(0.2 * FS), 10 ** (-30 / 20)),
            np.full(round(0.2 * FS), 10 ** (-50 / 20)),
        ]
        c = so.interaural_cues(so.Sound(noise * gains[:, None], FS), win_dur=20e-3, hop_dur=20e-3)
        assert np.all(np.isfinite(c.itd[2:8])) and np.all(np.isfinite(c.itd[12:18]))
        assert np.all(np.isnan(c.itd[22:28]))

    def test_each_band_has_its_own_silence_threshold(self):
        t = np.arange(round(0.3 * FS)) / FS
        low_and_quiet_high = np.sin(2 * np.pi * 300 * t) + 10 ** (-60 / 20) * np.sin(2 * np.pi * 3000 * t)
        fb = so.cosine_filterbank(4, 200, 5000)
        c = so.interaural_cues(so.Sound(low_and_quiet_high, FS).to_stereo(), filterbank=fb)
        band = np.argmin(np.abs(fb.cfs - 3000))
        assert np.all(np.isfinite(c.ild[:, band]))
        np.testing.assert_array_equal(c.cfs, fb.cfs)

    def test_plot(self):
        c = so.interaural_cues(so.apply_itd_ild(so.gaussian_noise(0.3, FS, rng=0), itd=200e-6))
        assert c.plot() is not None


class TestSimpleBirAcrossFrequency:
    @pytest.mark.parametrize("itd", [0.5 / FS, 100e-6, -37.3e-6])
    def test_delay_and_level_are_flat_below_0_4_fs(self, itd):
        bir = so.simple_bir(FS, itd=itd, ild=6).data
        spectrum = np.fft.rfft(bir, 8192, axis=0)
        f = np.fft.rfftfreq(8192, 1 / FS)
        band = (f > 0) & (f < 0.4 * FS)
        ratio = spectrum[band, 1] / spectrum[band, 0]  # right over left
        np.testing.assert_allclose(20 * np.log10(np.abs(ratio)), 6, atol=0.005)
        # the right ear leads by itd: its phase is ahead by 2 pi f itd
        np.testing.assert_allclose(np.unwrap(np.angle(ratio)), 2 * np.pi * f[band] * itd, atol=1e-3)


class TestOscorAndPhasewarp:
    @pytest.mark.parametrize("make", [so.oscor, so.phasewarp])
    def test_rms_is_one(self, make):
        assert make(1, FS, 2, rng=0).rms == pytest.approx(1)

    def test_phasewarp_moves_the_right_ear_up_by_f_mod(self):
        s = so.phasewarp(2, FS, 200, rng=0, band=(500, 1000)).data
        f = np.fft.rfftfreq(len(s), 1 / FS)
        power = np.abs(np.fft.rfft(s, axis=0)) ** 2
        right_above = power[(f > 1050) & (f < 1150), 1].sum()
        right_below = power[(f > 550) & (f < 650), 1].sum()
        assert right_above > 100 * power[(f > 1050) & (f < 1150), 0].sum()
        assert right_below < 0.01 * power[(f > 550) & (f < 650), 0].sum()


class TestInterauralCuesRefusals:
    def test_a_sound_shorter_than_one_window(self):
        with pytest.raises(ValueError, match="shorter than one window"):
            so.interaural_cues(so.gaussian_noise(0.01, FS, n_channels=2, rng=0), win_dur=20e-3)

    def test_a_lag_range_as_long_as_the_window(self):
        with pytest.raises(ValueError, match="max_itd"):
            so.interaural_cues(so.gaussian_noise(0.1, FS, n_channels=2, rng=0), win_dur=1e-3, max_itd=1e-3)

    def test_one_silent_ear_has_no_itd(self):
        noise = so.gaussian_noise(1, FS, rng=0).data[:, 0]
        right = np.r_[noise[: len(noise) // 2], np.zeros(len(noise) - len(noise) // 2)]
        c = so.interaural_cues(so.Sound(np.column_stack([noise, right]), FS))
        late = c.t > 0.55
        assert np.all(np.isnan(c.itd[late])) and np.all(np.isnan(c.iac[late]))
        assert np.all(c.ild[late] == -np.inf)
        assert np.all(np.isfinite(c.itd[c.t < 0.45]))
