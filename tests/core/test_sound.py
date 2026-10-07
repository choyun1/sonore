"""The Sound container."""

import numpy as np
import pytest
from helpers import FS, peak_freq

import sonore as so


class TestSound:
    def test_numpy_scalar_on_left(self):
        x = so.pure_tone(0.1, FS, 440)
        assert isinstance(np.float64(2) * x, so.Sound)
        assert isinstance(np.int64(2) * x, so.Sound)
        np.testing.assert_allclose((np.float64(2) * x).data, 2 * x.data)

    def test_reflected_division_and_subtraction(self):
        x = so.Sound(np.array([1.0, 2.0, 4.0]), FS)
        np.testing.assert_allclose((2 / x).data[:, 0], [2, 1, 0.5])
        np.testing.assert_allclose((x - x).data, 0)
        y = so.Sound(np.ones(3), FS)
        np.testing.assert_allclose((y - x).data[:, 0], [0, -1, -3])

    def test_adding_number_is_an_error_but_sum_works(self):
        x = so.pure_tone(0.1, FS, 440)
        with pytest.raises(TypeError, match=r"6\*dB"):
            x + 6
        np.testing.assert_allclose(sum([x, x]).data, 2 * x.data)

    def test_gain_db(self):
        x = so.pure_tone(0.1, FS, 440)
        assert so.amp_to_db(x.gain_db(6).rms / x.rms) == pytest.approx(6)

    def test_immutable(self):
        x = so.pure_tone(0.1, FS, 440)
        before = x.data.copy()
        x.ramp(0.01)
        np.testing.assert_array_equal(x.data, before)
        with pytest.raises(ValueError):
            x.data[0] = 1

    def test_channels_first_is_rejected(self):
        with pytest.raises(ValueError, match="channels-first"):
            so.Sound(np.zeros((2, 1000)), FS)

    def test_mono_broadcasts_against_stereo(self):
        env = so.Sound(np.linspace(0, 1, 100), FS)
        s = so.correlated_noise(100 / FS, FS, corr=0, rng=0)
        assert (env * s).n_channels == 2

    def test_mismatched_fs_is_an_error(self):
        with pytest.raises(ValueError, match="sampling rates"):
            so.silence(0.1, 44100) + so.silence(0.1, 48000)

    def test_time_slicing(self):
        x = so.pure_tone(1.0, FS, 440)
        assert len(x[0.25:0.5]) == FS // 4
        with pytest.raises(TypeError):
            x[0]

    @pytest.mark.parametrize("d", [10 / FS, 10.37 / FS])
    def test_fractional_delay(self, d):
        x = so.gaussian_noise(0.1, FS, band=(100, 15000), rng=1)
        y = x.delay(d)
        # compare in the frequency domain: phase ramp of exactly d
        n = len(y)
        X, Y = np.fft.rfft(x.data[:, 0], n), np.fft.rfft(y.data[:, 0], n)
        f = np.fft.rfftfreq(n, 1 / FS)
        band = (f > 200) & (f < 10000)
        phase = np.unwrap(np.angle(Y[band] / X[band]))
        slope = np.polyfit(f[band], phase, 1)[0]
        assert -slope / (2 * np.pi) == pytest.approx(d, rel=1e-3)

    def test_resample(self):
        x = so.pure_tone(0.5, 48000, 1000).resample(44100)
        assert x.fs == 44100 and len(x) == 22050
        assert peak_freq(x) == pytest.approx(1000, abs=2)


class TestHandWrittenValues:
    """Small cases whose expected values are written out by hand (audit sitting 2)."""

    RATE = 1000  # one sample per millisecond

    def sound(self, samples):
        return so.Sound(np.asarray(samples, dtype=float), self.RATE)

    def test_data_is_copied(self):
        samples = np.ones(3)
        snd = self.sound(samples)
        samples[0] = 5.0
        assert snd.data[0, 0] == 1.0

    def test_peak_counts_negative_samples(self):
        assert self.sound([0.5, -2.0]).peak == 2.0

    def test_mono_is_the_average_of_channels(self):
        snd = so.Sound(np.array([[1.0, 3.0], [2.0, 6.0]]), self.RATE)
        np.testing.assert_array_equal(snd.mono().data[:, 0], [2.0, 4.0])

    def test_from_channels_pads_shorter_channels_at_the_end(self):
        snd = so.Sound.from_channels([1.0, 2.0, 3.0], [4.0], fs=self.RATE)
        np.testing.assert_array_equal(snd.data, [[1, 4], [2, 0], [3, 0]])

    def test_zero_mean_removes_each_channels_own_mean(self):
        snd = so.Sound(np.array([[1.0, 10.0], [3.0, 30.0]]), self.RATE)
        np.testing.assert_array_equal(snd.zero_mean().data, [[-1, -10], [1, 10]])

    def test_normalize_to_a_peak(self):
        np.testing.assert_array_equal(self.sound([0.5, -2.0]).normalize(peak=1).data[:, 0], [0.25, -1.0])

    def test_normalize_refuses_silence(self):
        for kwargs in ({}, {"peak": 1}):
            with pytest.raises(ValueError, match="cannot normalize silence"):
                self.sound([0.0, 0.0]).normalize(**kwargs)

    def test_ramp_shapes(self):
        """Four ramp samples sample the curve at their centers, 1/8, 3/8, 5/8 and 7/8."""
        snd = self.sound(np.ones(10))
        centers = np.array([1, 3, 5, 7]) / 8
        cosine = (1 - np.cos(np.pi * centers)) / 2  # 0.038, 0.309, 0.691, 0.962
        np.testing.assert_allclose(snd.ramp(0.004).data[:, 0], [*cosine, 1, 1, *cosine[::-1]], rtol=1e-12)
        np.testing.assert_allclose(
            snd.ramp(0.004, shape="linear").data[:, 0], [*centers, 1, 1, *centers[::-1]], rtol=1e-12
        )

    def test_pad_adds_silence_before_and_after(self):
        np.testing.assert_array_equal(self.sound([1.0, 1.0]).pad(0.001, 0.003).data[:, 0], [0, 1, 1, 0, 0, 0])

    def test_pad_to_center_puts_the_odd_sample_at_the_end(self):
        np.testing.assert_array_equal(self.sound([1.0]).pad_to(4, "center").data[:, 0], [0, 1, 0, 0])

    def test_time_slices_round_to_the_nearest_sample(self):
        snd = self.sound(np.arange(10.0))
        assert snd[0.0026:].data[0, 0] == 3.0  # 2.6 samples rounds to 3
        assert snd[0.0024:].data[0, 0] == 2.0

    def test_time_slices_behave_like_python_slicing(self):
        snd = self.sound(np.arange(10.0))
        np.testing.assert_array_equal(snd[-0.003:].data[:, 0], [7, 8, 9])
        np.testing.assert_array_equal(snd[0.008:0.5].data[:, 0], [8, 9])

    def test_fractional_delay_length(self):
        """A delay of 2.5 samples makes the sound ceil(2.5) = 3 samples longer."""
        assert len(self.sound(np.ones(10)).delay(0.0025)) == 13

    def test_fractional_delay_above_half_a_sample(self):
        """A smooth pulse delayed by 10.7 samples matches the pulse written 10.7 samples later."""
        n = np.arange(64)
        pulse = self.sound(np.exp(-0.5 * ((n - 20) / 3) ** 2))
        delayed = pulse.delay(10.7 / self.RATE).data[:64, 0]
        np.testing.assert_allclose(delayed, np.exp(-0.5 * ((n - 30.7) / 3) ** 2), atol=1e-6)

    def test_envelope_is_aligned_with_the_sound(self):
        """The Hilbert envelope of (1 + 0.5 cos) times a carrier is 1 + 0.5 cos, sample for sample."""
        rate, t = 8000, np.arange(8000) / 8000
        modulation = 1 + 0.5 * np.cos(2 * np.pi * 50 * t)
        snd = so.Sound(modulation * np.cos(2 * np.pi * 1000 * t), rate)
        middle = slice(2000, 6000)
        np.testing.assert_allclose(snd.envelope().data[middle, 0], modulation[middle], atol=1e-3)


class TestAuditFixes:
    """Behavior Cho chose in audit sitting 2 (2026-10-03)."""

    def test_fs_is_read_only(self):
        snd = so.silence(0.01, FS)
        with pytest.raises(AttributeError):
            snd.fs = 8000

    def test_right_of_a_mono_sound(self):
        with pytest.raises(ValueError, match="mono sound has no right channel"):
            _ = so.silence(0.01, FS).right

    def test_number_minus_sound(self):
        snd = so.Sound(np.array([1.0, 2.0]), FS)
        with pytest.raises(TypeError, match="subtracting a Sound from a bare number"):
            2 - snd
        np.testing.assert_array_equal((0 - snd).data[:, 0], [-1, -2])
        np.testing.assert_array_equal((2 * np.ones(2) - snd).data[:, 0], [1, 0])

    def test_wrong_length_array_says_lengths_differ(self):
        snd = so.Sound(np.ones(4), FS)
        with pytest.raises(ValueError, match=r"lengths differ \(4 vs 3 samples\)"):
            snd * np.ones(3)
        with pytest.raises(ValueError, match="match_lengths"):
            snd + so.Sound(np.ones(3), FS)
        np.testing.assert_array_equal((snd * np.array([2.0])).data[:, 0], [2, 2, 2, 2])

    def test_notebook_player_plays_the_true_level(self):
        pytest.importorskip("IPython")
        quiet = so.Sound(0.1 * np.sin(np.arange(800)), 8000)
        assert "<audio" in quiet._repr_html_()
        loud = 30 * quiet  # peak 3
        assert "<audio" not in loud._repr_html_() and "normalize(peak=0.9)" in loud._repr_html_()
