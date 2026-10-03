"""List-level processing and filtering."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestProcessing:
    def test_bandpass_filters_each_channel_independently(self):
        n = so.correlated_noise(1, FS, corr=0, rng=0)
        y = so.bandpass(n, 500, 1000)
        assert abs(np.corrcoef(y.data.T)[0, 1]) < 0.1

    def test_pad_uses_zeros(self):
        a, b = so.match_lengths([so.Sound(np.ones(3), FS), so.Sound(np.ones(5), FS)], align="center")
        np.testing.assert_array_equal(a.data[:, 0], [0, 1, 1, 1, 0])

    def test_concat_upmixes_mono(self):
        s = so.concat([so.correlated_noise(0.1, FS, rng=0), so.silence(0.1, FS)])
        assert s.n_channels == 2

    def test_mix_different_lengths(self):
        m = so.mix([so.silence(0.1, FS), so.pure_tone(0.2, FS, 100)])
        assert m.duration == pytest.approx(0.2)

    @pytest.mark.parametrize(
        "align, expected", [("start", [1, 1, 0, 0]), ("center", [0, 1, 1, 0]), ("end", [0, 0, 1, 1])]
    )
    def test_mix_aligns_the_shorter_sound(self, align, expected):
        short, long = so.Sound(np.ones(2), FS), so.Sound(np.zeros(4), FS)
        np.testing.assert_array_equal(so.mix([short, long], align=align).data[:, 0], expected)

    def test_truncate_cuts_to_the_shortest(self):
        a, b = so.match_lengths([so.Sound(np.arange(5.0), FS), so.Sound(np.arange(3.0), FS)], mode="truncate")
        np.testing.assert_array_equal(a.data[:, 0], [0, 1, 2])
        assert len(b) == 3

    def test_relative_db_against_the_reference(self):
        x = so.Sound(np.ones(4), FS)
        assert so.relative_db([x, 2 * x, x / 10], ref=0) == pytest.approx(
            [0, 20 * np.log10(2), -20], abs=1e-12
        )
        assert so.relative_db([x, 2 * x], ref=1)[0] == pytest.approx(-20 * np.log10(2), abs=1e-12)

    def test_match_fs_resamples_to_the_lowest_or_highest_rate(self):
        a, b = so.silence(0.1, 16000), so.silence(0.1, 32000)
        assert [s.fs for s in so.match_fs([a, b])] == [16000, 16000]
        assert [s.fs for s in so.match_fs([a, b], mode="up")] == [32000, 32000]
        assert [s.fs for s in so.match_fs([a, b], fs=8000)] == [8000, 8000]

    def test_butter_filter_gain_at_the_cutoff(self):
        """A Butterworth filter is 3 dB down at its cutoff; run forward and backward, 6 dB."""
        n = 2**15
        imp = so.Sound(np.r_[np.zeros(n // 2), 1.0, np.zeros(n // 2 - 1)], FS)
        f = np.fft.rfftfreq(n, 1 / FS)
        at_cutoff = np.argmin(np.abs(f - 1000))
        for zero_phase, expected_db in ((False, -10 * np.log10(2)), (True, -20 * np.log10(2))):
            h = np.abs(np.fft.rfft(so.butter_filter(imp, 1000, "lowpass", zero_phase=zero_phase).data[:, 0]))
            assert 20 * np.log10(h[at_cutoff]) == pytest.approx(expected_db, abs=0.05)

    @pytest.mark.parametrize(
        "align, expected", [("start", [0, 1, 2]), ("center", [1, 2, 3]), ("end", [2, 3, 4])]
    )
    def test_truncate_keeps_start_middle_or_end(self, align, expected):
        # 2 extra samples: center cuts one from each end; 3 extra: the odd one goes at the end
        a, _ = so.match_lengths(
            [so.Sound(np.arange(5.0), FS), so.Sound(np.zeros(3), FS)], mode="truncate", align=align
        )
        np.testing.assert_array_equal(a.data[:, 0], expected)
        b, _ = so.match_lengths(
            [so.Sound(np.arange(6.0), FS), so.Sound(np.zeros(3), FS)], mode="truncate", align="center"
        )
        np.testing.assert_array_equal(b.data[:, 0], [1, 2, 3])

    def test_list_normalize_scales_each_sound_on_its_own(self):
        quiet, loud = so.Sound(0.1 * np.ones(4), FS), so.Sound(3 * np.ones(8), FS)
        assert [s.rms for s in so.normalize([quiet, loud], rms=0.5)] == pytest.approx([0.5, 0.5])
        assert [s.peak for s in so.normalize([quiet, -loud], peak=0.9)] == pytest.approx([0.9, 0.9])

    @pytest.mark.parametrize(
        "call, message",
        [
            (lambda x: so.match_lengths([x, x], align="middle"), "align must be"),
            (lambda x: so.match_lengths([x, x], mode="crop"), "mode must be 'pad' or 'truncate'"),
            (lambda x: so.match_fs([x, x], mode="lowest"), "mode must be 'down' or 'up'"),
            (lambda x: so.mix([x, so.Sound(np.ones(4), 2 * FS)]), "sampling rates differ"),
            (lambda x: so.concat([x, so.Sound(np.ones(4), 2 * FS)]), "sampling rates differ"),
            (lambda x: so.relative_db([0 * x, x]), "silent"),
            (lambda x: so.mix([]), "at least one sound"),
            (lambda x: so.concat([]), "at least one sound"),
            (lambda x: so.match_lengths([]), "at least one sound"),
            (lambda x: so.match_fs([]), "at least one sound"),
        ],
    )
    def test_bad_lists_say_what_is_wrong(self, call, message):
        with pytest.raises(ValueError, match=message):
            call(so.Sound(np.ones(4), FS))

    @staticmethod
    def gain_db(sound, freq):
        h = np.abs(np.fft.rfft(sound.data[:, 0]))
        return 20 * np.log10(h[np.argmin(np.abs(np.fft.rfftfreq(len(sound), 1 / sound.fs) - freq))])

    def impulse(self, n=2**15):
        return so.Sound(np.r_[np.zeros(n // 2), 1.0, np.zeros(n // 2 - 1)], FS)

    def test_butter_filter_order_and_type(self):
        """Digital Butterworth lowpass, order n:
        |H|^2 = 1 / (1 + (tan(pi f/fs) / tan(pi fc/fs))^(2n)).
        """
        imp = self.impulse()
        for order in (2, 4):
            low = so.butter_filter(imp, 500, "lowpass", order=order, zero_phase=False)
            ratio = np.tan(np.pi * 2000 / FS) / np.tan(np.pi * 500 / FS)
            assert self.gain_db(low, 2000) == pytest.approx(
                -10 * np.log10(1 + ratio ** (2 * order)), abs=0.05
            )
        high = so.butter_filter(imp, 4000, "highpass", zero_phase=False)
        assert self.gain_db(high, 4000) == pytest.approx(-3.01, abs=0.05)
        assert self.gain_db(high, 500) < -60
        stop = so.butter_filter(imp, (900, 1100), "bandstop", zero_phase=False)
        assert self.gain_db(stop, 1000) < -60
        assert self.gain_db(stop, 100) == pytest.approx(0, abs=0.01)

    def test_bandpass_is_butter_filter(self):
        noise = so.gaussian_noise(0.1, FS, rng=0)
        for order, zero_phase in ((2, False), (6, True)):
            np.testing.assert_array_equal(
                so.bandpass(noise, 500, 1000, order, zero_phase).data,
                so.butter_filter(noise, (500, 1000), "bandpass", order, zero_phase).data,
            )
        assert self.gain_db(so.bandpass(self.impulse(), 500, 1000, zero_phase=False), 1000) == pytest.approx(
            -3.01, abs=0.05
        )

    def test_amplitude_modulate_multiplies_by_the_envelope(self):
        ones = so.Sound(np.ones(FS), FS)
        t = np.arange(FS) / FS
        y = so.amplitude_modulate(ones, f_mod=4, depth=0.5, phase=1.0)
        np.testing.assert_allclose(y.data[:, 0], 1 + 0.5 * np.sin(2 * np.pi * 4 * t + 1.0), atol=1e-12)


class TestResonator:
    """Klatt's (1980) resonator and antiresonator."""

    def test_unit_gain_at_0_hz_peak_at_f_and_width_bw(self):
        fs = 16000
        imp = so.Sound(np.r_[1.0, np.zeros(2**17 - 1)], fs)
        h = np.abs(np.fft.rfft(so.resonator(imp, 500, 60).data[:, 0]))
        f = np.fft.rfftfreq(2**17, 1 / fs)
        assert h[0] == pytest.approx(1.0, abs=1e-9)
        assert f[np.argmax(h)] == pytest.approx(500, abs=1)
        band = f[h >= h.max() / np.sqrt(2)]
        assert band.max() - band.min() == pytest.approx(60, abs=1)

    def test_antiresonator_inverts_resonator(self):
        x = so.gaussian_noise(0.2, FS, rng=0)
        f = ([0, 0.2], [300, 2000])
        y = so.antiresonator(so.resonator(x, f, 80), f, 80)
        np.testing.assert_allclose(y.data, x.data, atol=1e-10)

    def test_constant_track_equals_number(self):
        x = so.gaussian_noise(0.1, FS, rng=0)
        a = so.resonator(x, 700, 100)
        b = so.resonator(x, ([0, 1], [700, 700]), ([0.05], [100]))
        np.testing.assert_allclose(b.data, a.data, atol=1e-10)

    def test_moving_formant_has_no_clicks(self):
        # F1 300 -> 700 Hz over 40 ms on a low-pass voiced source: nothing
        # reaches 5 kHz, as there would if the filter restarted at each step.
        fs = 16000
        k = np.arange(1, 20)
        x = so.harmonic_complex(0.06, fs, 120, harmonics=k, amplitudes=1 / k**2)
        y = so.resonator(x, ([0, 0.04], [300, 700]), 60).data[:, 0] * np.hanning(len(x))
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / fs)
        assert 10 * np.log10(s[f > 5000].sum() / s.sum()) < -80

    def test_each_channel_filtered(self):
        x = so.correlated_noise(0.1, FS, corr=0, rng=0)
        y = so.resonator(x, ([0, 0.1], [500, 900]), 100)
        z = so.resonator(so.Sound(x.data[:, 1], FS), ([0, 0.1], [500, 900]), 100)
        np.testing.assert_allclose(y.data[:, 1], z.data[:, 0])

    def test_f_0_is_a_low_pass(self):
        imp = so.Sound(np.r_[1.0, np.zeros(2**14 - 1)], FS)
        h = np.abs(np.fft.rfft(so.resonator(imp, 0, 200).data[:, 0]))
        assert h[0] == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.diff(h) <= 1e-12)

    @pytest.mark.parametrize(
        "track, message",
        [
            (([0, 0.1, 0.05], [500, 600, 700]), "times must increase"),
            (([0, 0.1, 0.1], [500, 600, 700]), "times must increase"),
            (([0, 0.1], [500, 600, 700]), "one value per time"),
            (([], []), "one value per time"),
            ("500", "number or a"),
        ],
    )
    def test_rejects_bad_tracks(self, track, message):
        with pytest.raises((ValueError, TypeError), match=message):
            so.resonator(so.silence(0.2, FS), track, 100)

    @pytest.mark.parametrize("f, bw", [(-1, 50), (FS / 2, 50), (500, 0)])
    def test_rejects_bad_values(self, f, bw):
        with pytest.raises(ValueError):
            so.resonator(so.silence(0.01, FS), f, bw)
