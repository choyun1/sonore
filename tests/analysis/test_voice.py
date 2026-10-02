"""Pitch and formant changes on any F0 contour and any envelope
(sonore.analysis.voice), and the synthesizers taking any envelope."""

import numpy as np
import pytest

import sonore as so

FS = 16000
HOP = 0.005


def bump_power(freqs, centre, width=150.0):
    """A smooth one-formant envelope: a Gaussian bump in dB on a floor."""
    return 10 ** ((-40 + 40 * np.exp(-0.5 * ((freqs - centre) / width) ** 2)) / 10)


def bump_envelope(centre=1000.0, n_windows=20):
    freqs = np.linspace(0, FS / 2, 513)
    times = np.arange(n_windows) * HOP
    power = np.repeat(bump_power(freqs, centre)[:, None], n_windows, axis=1)
    return so.GridEnvelope(power, times, freqs)


def peak(view, window=0):
    return view.f[np.argmax(view.data[0, :, window])]


@pytest.fixture(scope="module")
def vowel():
    """A steady 150 Hz vowel with one formant at 1 kHz, its F0 track, CheapTrick
    envelope and D4C aperiodicity."""
    duration = 0.4
    t = np.arange(int(duration * FS)) / FS
    harmonics = 150.0 * np.arange(1, 50)
    amplitudes = np.sqrt(bump_power(harmonics, 1000.0, width=200.0))
    sound = so.Sound(
        np.sum(amplitudes[:, None] * np.cos(2 * np.pi * np.outer(harmonics, t)), axis=0) * 0.05, FS
    )
    times = np.arange(int(duration / HOP)) * HOP
    track = (times, np.full(len(times), 150.0))
    return sound, track, so.cheaptrick(sound, track), so.d4c(sound, track)


class TestScaleF0:
    def test_pair(self):
        times = np.arange(5) * HOP
        values = np.array([0.0, 100.0, 120.0, 0.0, 150.0])
        new_times, new_values = so.scale_f0((times, values), 1.5)
        assert new_times is times
        np.testing.assert_array_equal(new_values, values * 1.5)

    def test_range_spreads_around_the_median(self):
        values = np.array([0.0, 100.0, 200.0, 400.0])
        _, flat = so.scale_f0((np.arange(4) * HOP, values), 1.0, range=0.0)
        np.testing.assert_allclose(flat, [0.0, 200.0, 200.0, 200.0])
        _, wide = so.scale_f0((np.arange(4) * HOP, values), 2.0, range=2.0)
        np.testing.assert_allclose(wide, [0.0, 400.0 * 0.25, 400.0, 400.0 * 4])

    def test_f0track_stays_an_f0track(self, vowel):
        sound = vowel[0]
        track = so.f0_track(sound)
        scaled = so.scale_f0(track, 2.0)
        assert isinstance(scaled, so.F0Track)
        np.testing.assert_array_equal(scaled.f0, track.f0 * 2.0)
        np.testing.assert_array_equal(scaled.candidates, track.candidates * 2.0)

    def test_identity_is_the_contour_itself(self):
        contour = (np.arange(3) * HOP, np.array([0.0, 100.0, 110.0]))
        assert so.scale_f0(contour, 1.0) is contour

    @pytest.mark.parametrize("kwargs", [{"ratio": 0}, {"ratio": 1.2, "range": -1}])
    def test_rejects(self, kwargs):
        with pytest.raises(ValueError):
            so.scale_f0((np.arange(2) * HOP, np.ones(2)), **kwargs)


class TestWarpFrequency:
    def test_moves_the_peak_by_the_ratio(self):
        envelope = bump_envelope(1000.0)
        for ratio in (0.8, 1.25):
            warped = so.warp_frequency(envelope, ratio)
            assert isinstance(warped, so.GridEnvelope)
            assert abs(peak(warped) - 1000.0 * ratio) <= 8.0  # one bin is 7.8 Hz

    def test_reads_the_view_at_f_over_ratio(self):
        envelope = bump_envelope()
        warped = so.warp_frequency(envelope, 1.2)
        np.testing.assert_allclose(warped.data, envelope(envelope.t, envelope.f / 1.2))

    def test_ratio_one_is_the_view_itself(self):
        envelope = bump_envelope()
        assert so.warp_frequency(envelope, 1) is envelope

    def test_a_frequency_map_matches_the_ratio(self):
        envelope = bump_envelope()
        by_map = so.warp_frequency(envelope, lambda f: f / 1.2)
        np.testing.assert_allclose(by_map.data, so.warp_frequency(envelope, 1.2).data)

    def test_a_ratio_contour(self):
        envelope = bump_envelope(n_windows=11)
        warped = so.warp_frequency(envelope, ((0.0, 0.05), (1.0, 1.5)))
        assert abs(peak(warped, 0) - 1000.0) <= 8.0
        assert abs(peak(warped, 10) - 1500.0) <= 8.0

    def test_holds_the_top_when_lowering(self):
        envelope = bump_envelope()
        warped = so.warp_frequency(envelope, 0.8)
        top = envelope.f > 0.8 * FS / 2
        np.testing.assert_allclose(warped.data[0, top, 0], envelope.data[0, -1, 0])

    def test_a_function_stays_a_function(self):
        def envelope(t, f):
            return bump_power(np.asarray(f), 1000.0)

        warped = so.warp_frequency(envelope, 1.2)
        np.testing.assert_allclose(warped(None, np.array([1200.0])), bump_power(np.array([1000.0]), 1000.0))
        with pytest.raises(TypeError):
            so.warp_frequency(envelope, ((0.0, 1.0), (1.0, 1.2)))

    def test_keeps_the_type_of_world_views(self, vowel):
        _, _, envelope, aperiodicity = vowel
        assert isinstance(so.warp_frequency(envelope, 1.2), so.SpectralEnvelope)
        warped = so.warp_frequency(aperiodicity, 1.2)
        assert isinstance(warped, so.Aperiodicity)
        assert warped.data.max() <= 1.0


class TestEnvelopeViews:
    def test_cepstrum(self, vowel):
        cepstrum = so.Cepstrum(so.STFT(vowel[0], win_dur=0.040, hop_dur=HOP)).lifter(0.5 / 150)
        view = cepstrum.envelope_view()
        np.testing.assert_allclose(view.data, cepstrum.envelope() ** 2)
        np.testing.assert_array_equal(view.t, cepstrum.t)

    def test_mfcc(self, vowel):
        mfcc = so.MFCC(vowel[0])
        view = mfcc.envelope_view()
        np.testing.assert_allclose(view.data, mfcc.envelope(view.f))
        assert view.data.shape == (1, mfcc.n_fft // 2 + 1, len(mfcc.t))


class TestSynthesizers:
    def test_identity_settings_are_world_sample_for_sample(self, vowel):
        _, track, envelope, aperiodicity = vowel
        plain = so.world_synthesize(track, envelope, aperiodicity)
        unchanged = so.world_synthesize(
            so.scale_f0(track, 1.0), so.warp_frequency(envelope, 1.0), aperiodicity
        )
        np.testing.assert_array_equal(unchanged.data, plain.data)

    def test_any_envelope(self, vowel):
        _, track, envelope, aperiodicity = vowel
        as_grid = so.GridEnvelope(envelope.data, envelope.t, envelope.f)
        plain = so.world_synthesize(track, envelope, aperiodicity)
        np.testing.assert_allclose(
            so.world_synthesize(track, as_grid, aperiodicity).data, plain.data, atol=1e-12
        )

    def test_a_contour_off_the_grid(self, vowel):
        _, (times, values), envelope, aperiodicity = vowel
        shifted = (times - 0.015, values)  # Cepstrum.f0's time windows start before 0
        out = so.world_synthesize(shifted, envelope, aperiodicity)
        assert len(out) == len(so.world_synthesize((times, values), envelope, aperiodicity))

    def test_pitch_and_formants_move_by_their_ratios(self, vowel):
        _, track, envelope, aperiodicity = vowel
        out = so.world_synthesize(so.scale_f0(track, 1.5), so.warp_frequency(envelope, 1.2), aperiodicity)
        tracked = so.f0_track(out)
        inner = (tracked.t > 0.1) & (tracked.t < 0.3)
        assert np.median(tracked.f0[0][inner]) == pytest.approx(225.0, rel=0.01)
        # the formants: the warp that best explains the output's envelope as the
        # envelope of the same pitch change without the warp, read at f / ratio
        higher = so.scale_f0(track, 1.5)
        kept = so.world_synthesize(higher, envelope, aperiodicity)
        kept_db = so.cheaptrick(kept, higher).db[0][:, 20:60].mean(axis=1)
        moved_db = so.cheaptrick(out, higher).db[0][:, 20:60].mean(axis=1)
        inside = (envelope.f > 100) & (envelope.f < 5000)

        def misfit(ratio):
            difference = moved_db[inside] - np.interp(envelope.f[inside] / ratio, envelope.f, kept_db)
            return np.std(difference)

        ratios = np.arange(1.0, 1.4, 0.005)
        assert ratios[np.argmin([misfit(ratio) for ratio in ratios])] == pytest.approx(1.2, abs=0.01)

    def test_harmonic_complex_reads_an_envelope(self):
        envelope = bump_envelope(1000.0, n_windows=60)
        out = so.harmonic_complex(0.25, FS, 200.0, amplitudes=envelope)
        spectrum = np.abs(np.fft.rfft(out.data[:, 0]))
        freqs = np.fft.rfftfreq(len(out), 1 / FS)
        levels = [spectrum[np.argmin(np.abs(freqs - k * 200.0))] for k in (2, 5, 10)]
        expected = np.sqrt(bump_power(np.array([400.0, 1000.0, 2000.0]), 1000.0))
        np.testing.assert_allclose(np.array(levels) / levels[1], expected / expected[1], rtol=1e-3)
