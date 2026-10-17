"""Spectrotemporal ripples."""

import numpy as np
import pytest
from helpers import FAST, FAST_HI, FS

import sonore as so


class TestRipples:
    @pytest.mark.parametrize("carrier", ["tones", "harmonic", "noise", "low-noise"])
    @pytest.mark.parametrize("rate, density", [(4, 1), (-8, 2), (16, 0)])
    def test_modulation_spectrum_peak(self, carrier, rate, density):
        # f0 = 40 Hz: harmonics are dense enough near 250 Hz for 2 cyc/oct
        s = so.ripple_sound(so.Ripple(rate, density), 1.0, FAST, f_hi=FAST_HI, carrier=carrier, f0=40, rng=0)
        ms = so.ModulationSpectrum.octave(s, f_lo=250, f_hi=FAST_HI)
        got_rate, got_density = ms.peak()
        bin_width = ms.w_f[1] - ms.w_f[0]
        assert got_rate == pytest.approx(abs(rate) if density == 0 else rate, abs=1.0)
        # within one bin: a density halfway between two bins can land on either
        assert got_density == pytest.approx(density, abs=bin_width)

    def test_depth_is_exact_for_tone_carrier(self):
        # with density 0 the pattern is a pure AM: output / unmodulated = 1 + m sin(...)
        m, rate = 0.7, 5.0
        mod = so.ripple_sound(so.Ripple(rate, 0, depth=m), 0.5, FS, rng=1)
        flat = so.ripple_sound(so.Ripple(rate, 0, depth=0), 0.5, FS, rng=1)
        ratio = mod.data[:, 0] / flat.data[:, 0]
        ok = np.abs(flat.data[:, 0]) > 0.05 * flat.peak
        expected = 1 + m * np.sin(2 * np.pi * rate * mod.t)
        k = np.median(ratio[ok] / expected[ok])  # normalization constant
        np.testing.assert_allclose(ratio[ok] / k, expected[ok], rtol=1e-6)

    def test_direction_convention(self):
        # positive rate & density: the envelope maximum moves down in frequency
        r = so.Ripple(2, 1, depth=1)
        x = np.linspace(0, 1, 2001)
        peak_x = [x[np.argmax(r.envelope(np.array([t]), x)[:, 0])] for t in (0.0, 0.1)]
        assert peak_x[1] < peak_x[0]
        assert r.direction == "downward" and so.Ripple(-2, 1).direction == "upward"

    def test_carriers_share_long_term_spectrum(self):
        levels = []
        for carrier in ("tones", "harmonic", "noise", "low-noise"):
            s = so.ripple_sound(so.Ripple(4, 1, depth=0), 1.0, FAST, f_hi=FAST_HI, carrier=carrier, rng=0)
            spec = so.long_term_spectrum(s, win_dur=0.512).smooth(1)
            f = np.array([500, 1000, 2000, 4000])
            levels.append(spec.level_at(f) - spec.level_at(np.array([1000]))[0])
        spread = np.ptp(np.array(levels), axis=0)
        assert np.all(spread < 3)  # all within 3 dB of each other, octave by octave

    def test_sums_and_validation(self):
        both = so.Ripple(4, 1, depth=0.5) + so.Ripple(-8, 2, depth=0.4)
        assert isinstance(both, so.RippleSum) and len(both.components) == 2
        assert sum([so.Ripple(4, 1, depth=0.3), so.Ripple(8, 2, depth=0.3)]).components[1].rate == 8
        with pytest.raises(ValueError, match="negative"):
            so.Ripple(4, 1, depth=0.6) + so.Ripple(8, 1, depth=0.6)
        with pytest.raises(ValueError, match="linear-scale and dB"):
            so.Ripple(4, 1) + so.Ripple(8, 1, depth=20, scale="db")
        with pytest.raises(ValueError):
            so.Ripple(4, 1, depth=1.5)
        s = so.ripple_sound(both, 1.0, FS, rng=0)
        assert s.rms == pytest.approx(1)

    def test_db_scale_depth(self):
        r = so.Ripple(0, 1, depth=30, scale="db")
        env = r.envelope(np.array([0.0]), np.linspace(0, 1, 1001))
        assert 20 * np.log10(env.max() / env.min()) == pytest.approx(30, abs=0.01)

    def test_resolution_warnings(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            so.ripple_sound(so.Ripple(4, 12), 0.2, FS, tones_per_octave=20, rng=0)
        with pytest.warns(UserWarning, match="lowest harmonics"):
            so.ripple_sound(so.Ripple(4, 2), 0.2, FS, carrier="harmonic", f0=200, rng=0)

    def test_callable_pattern(self):
        pattern = lambda t, x: 1 + 0.9 * np.sin(2 * np.pi * (6 * t + 1.0 * x))  # noqa: E731
        s = so.ripple_sound(pattern, 1.0, FAST, f_hi=FAST_HI, rng=0)
        rate, density = so.ModulationSpectrum.octave(s, f_lo=250, f_hi=FAST_HI).peak()
        assert rate == pytest.approx(6, abs=1) and density == pytest.approx(1, abs=0.15)

    def test_own_pattern_class(self):
        # any object with envelope(t, x) is a pattern, without inheriting from sonore's classes
        ripple = so.Ripple(4, 1)

        class OwnPattern:
            def envelope(self, t, x):
                return ripple.envelope(t, x)

        own = so.ripple_sound(OwnPattern(), 0.5, FAST, f_hi=FAST_HI, rng=0)
        reference = so.ripple_sound(ripple, 0.5, FAST, f_hi=FAST_HI, rng=0)
        np.testing.assert_array_equal(own.data, reference.data)

    def test_sound_carrier(self):
        speechlike = so.harmonic_complex(1.0, FAST, 120, harmonics=np.arange(1, 50), phases="random", rng=0)
        s = so.ripple_sound(so.Ripple(-4, 1), 1.0, FAST, f_hi=FAST_HI, carrier=speechlike, rng=0)
        rate, density = so.ModulationSpectrum.octave(s, f_lo=250, f_hi=FAST_HI).peak()
        assert rate == pytest.approx(-4, abs=1) and density == pytest.approx(1, abs=0.15)


class TestDynamicRipple:
    def test_trajectories_stay_in_range_and_vary_slowly(self):
        d = so.DynamicRipple(seed=0)
        t = np.arange(0, 10, 1e-3)
        rate, density = d.trajectories(t)
        assert rate.min() >= -350 and rate.max() <= 350 and np.ptp(rate) > 400
        assert density.min() >= 0 and density.max() <= 4 and np.ptp(density) > 2
        # slow: little power above the specified rate of change
        R = np.abs(np.fft.rfft(rate - rate.mean())) ** 2
        f = np.fft.rfftfreq(len(rate), 1e-3)
        assert R[f > 4 * d.rate_change].sum() < 0.01 * R.sum()

    def test_reproducible_and_fs_independent(self):
        d = so.DynamicRipple(seed=7)
        a = so.ripple_sound(d, 0.5, FS, rng=1)
        b = so.ripple_sound(d, 0.5, FS, rng=1)
        np.testing.assert_array_equal(a.data, b.data)
        r44, _ = d.trajectories(np.arange(0, 0.5, 1 / 44100))
        r48, _ = d.trajectories(np.arange(0, 0.5, 1 / 48000))
        assert np.interp(0.25, np.arange(0, 0.5, 1 / 48000), r48) == pytest.approx(
            np.interp(0.25, np.arange(0, 0.5, 1 / 44100), r44), abs=0.5
        )
        assert so.DynamicRipple().seed is not None  # a random seed is drawn and kept

    def test_envelope_depth(self):
        d = so.DynamicRipple(depth=45, seed=0)
        env = d.envelope(np.arange(0, 0.2, 1e-3), np.linspace(0, 5, 200))
        assert 20 * np.log10(env.max() / env.min()) == pytest.approx(45, abs=0.5)


class TestRippleGaps:
    def test_phase_and_scale_check(self):
        crest = so.Ripple(0, 0, depth=0.5, phase=np.pi / 2).envelope(np.array([0.0]), np.array([0.0]))
        assert crest[0, 0] == pytest.approx(1.5)
        with pytest.raises(ValueError, match="scale"):
            so.Ripple(4, 1, scale="log")

    def test_negative_patterns_are_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            so.ripple_sound(lambda t, x: np.sin(2 * np.pi * 4 * t) + 0 * x, 0.5, FS, rng=0)

    def test_every_component_and_dynamic_density_is_checked(self):
        with pytest.warns(UserWarning, match="Nyquist"):
            so.ripple_sound(so.Ripple(4, 1, depth=0.4) + so.Ripple(8, 12, depth=0.4), 0.1, FS, rng=0)
        with pytest.warns(UserWarning, match="Nyquist"):
            so.ripple_sound(so.DynamicRipple(density_range=(0, 12), seed=0), 0.1, FS, rng=0)

    def test_refusals(self):
        with pytest.raises(ValueError, match="Nyquist"):
            so.ripple_sound(so.Ripple(4, 1), 0.1, 16000, f_hi=8000)
        with pytest.raises(ValueError, match="no harmonics"):
            so.ripple_sound(so.Ripple(4, 1), 0.1, FS, carrier="harmonic", f0=9000)
        with pytest.raises(ValueError, match="same fs"):
            so.ripple_sound(so.Ripple(4, 1), 0.1, FS, carrier=so.gaussian_noise(0.05, FS, rng=0))
        with pytest.raises(ValueError, match="carrier must be"):
            so.ripple_sound(so.Ripple(4, 1), 0.1, FS, carrier="pink")

    @pytest.mark.parametrize("carrier", ["noise", "low-noise"])
    def test_channel_carriers_have_rms_1(self, carrier):
        sound = so.ripple_sound(so.Ripple(4, 1), 0.2, FAST, f_hi=FAST_HI, carrier=carrier, rng=0)
        assert sound.rms == pytest.approx(1)

    def test_carriers_span_f_lo_to_f_hi(self):
        def spectrum(sound):
            power = np.abs(np.fft.rfft(sound.data[:, 0])) ** 2
            return np.fft.rfftfreq(len(sound), 1 / sound.fs), power / power.sum()

        flat = so.Ripple(0, 0, depth=0)
        # tones from f_lo up to and including f_hi (250 Hz to 8 kHz is 100 steps of 1/20 octave)
        freqs, power = spectrum(so.ripple_sound(flat, 1.0, FS, rng=0))
        assert power[np.abs(freqs - 8000) < 2].sum() > 0.2 / 101
        # harmonics of 40 Hz from the first at or above 250 Hz (280 Hz), none below
        freqs, power = spectrum(
            so.ripple_sound(flat, 1.0, FAST, f_hi=FAST_HI, carrier="harmonic", f0=40, rng=0)
        )
        assert power[freqs < 260].sum() < 1e-6

    def test_noise_bands_add_up_flat(self):
        # an impulse through the noise carrier's bands: squared responses sum to a flat spectrum
        from sonore.sources.ripples import _flat_noise_bands

        bank = so.cosine_filterbank(f_lo=250, f_hi=6000, spacing=1 / 24, scale="octave")
        impulse = so.Sound(np.r_[1.0, np.zeros(FAST - 1)], FAST)
        total = _flat_noise_bands(impulse, bank).data[:, 1:-1, 0].sum(axis=1)
        level = 20 * np.log10(np.abs(np.fft.rfft(total)))
        freqs = np.fft.rfftfreq(FAST, 1 / FAST)
        inside = (freqs > 300) & (freqs < 5000)
        trend = np.polyval(np.polyfit(np.log2(freqs[inside]), level[inside], 3), np.log2(freqs[inside]))
        assert np.ptp(level[inside] - trend) < 0.2  # 0.03 dB; unsquared responses give 3 dB

    def test_plot_resolves_the_fastest_ripple(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = so.Ripple(100, 1).plot(duration=1.0)  # 8 points per cycle: 800 time points
        assert ax.collections[0].get_array().size == 200 * 800
        plt.close(ax.figure)


class TestDynamicRippleGaps:
    def test_fixed_rate_and_density_make_a_moving_ripple(self):
        # equal ends pin rate(t) and density(t); the phase integral starts one step in
        t, octaves = np.arange(0, 0.2, 1 / 8000), np.linspace(0, 5, 41)
        pinned = so.DynamicRipple(rate_range=(10, 10), density_range=(2, 2), seed=0)
        ripple = so.Ripple(10, 2, depth=45, scale="db", phase=2 * np.pi * 10 / 8000)
        np.testing.assert_allclose(pinned.envelope(t, octaves), ripple.envelope(t, octaves), rtol=1e-9)

    def test_rate_and_density_change_at_their_own_speeds(self):
        t = np.arange(0, 20, 1e-3)
        freqs = np.fft.rfftfreq(len(t), 1e-3)
        rate, density = so.DynamicRipple(seed=0, rate_change=3, density_change=6).trajectories(t)
        rate_power = np.abs(np.fft.rfft(rate - rate.mean())) ** 2
        density_power = np.abs(np.fft.rfft(density - density.mean())) ** 2
        assert rate_power[freqs > 6].sum() < 0.015 * rate_power.sum()  # 0.2-0.5%
        assert density_power[freqs > 3].sum() > 0.2 * density_power.sum()  # 42-51%

    def test_a_new_seed_each_time(self):
        assert so.DynamicRipple().seed != so.DynamicRipple().seed

    def test_a_shorter_pattern_is_the_start_of_a_longer_one(self):
        pattern = so.DynamicRipple(seed=3)
        short_t, long_t = np.arange(0, 1, 1e-3), np.arange(0, 4, 1e-3)
        for short, long in zip(pattern.trajectories(short_t), pattern.trajectories(long_t), strict=True):
            # the filters settle within the extra second at each end: 0.06 Hz at most, at the very end
            np.testing.assert_allclose(short[:500], long[:500], atol=1e-2 * np.ptp(long))
            np.testing.assert_allclose(short, long[:1000], atol=1e-3 * 700)

    def test_trajectories_are_uniform_across_draws(self):
        # each trajectory is Gaussian noise of known spread mapped through its distribution function
        rates = np.concatenate(
            [so.DynamicRipple(seed=seed).trajectories(np.arange(0, 1, 1e-3))[0] for seed in range(200)]
        )
        counts, _ = np.histogram(rates, bins=7, range=(-350, 350))
        assert counts.min() > 0.85 * counts.max()  # 0.86; standardizing each draw gave 0.72
