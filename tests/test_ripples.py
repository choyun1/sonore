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
            spec = so.long_term_spectrum(s, nperseg=8192).smooth(1)
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

    def test_sound_carrier(self):
        speechlike = so.harmonic_complex(1.0, FAST, 120, np.arange(1, 50), phases="random", rng=0)
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
