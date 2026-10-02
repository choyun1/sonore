"""Synthetic room impulse responses."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestReverb:
    def test_rt60_and_drr(self):
        ir = so.synth_ir(0.5, FS, drr_db=5, rng=0)
        direct = ir.data[0, 0] ** 2
        tail = np.sum(ir.data[1:, 0] ** 2)
        assert 10 * np.log10(direct / tail) == pytest.approx(5, abs=0.5)
        # Schroeder backward integration on the broadband tail
        e = np.cumsum(ir.data[::-1, 0] ** 2)[::-1]
        edc = 10 * np.log10(e / e[0])
        t = np.arange(len(e)) / FS
        sel = (edc < -5) & (edc > -25)
        slope = np.polyfit(t[sel], edc[sel], 1)[0]
        assert -60 / slope == pytest.approx(0.5, rel=0.25)

    def test_no_wraparound_at_the_end(self):
        # regression: circular filtering wrapped onset energy to the IR's last samples
        ir = so.synth_ir(1.0, FS, drr_db=0, rng=0).data[:, 0]
        onset = np.mean(ir[1 : int(0.05 * FS)] ** 2)
        end = np.mean(ir[-int(0.05 * FS) :] ** 2)
        assert 10 * np.log10(end / onset) < -50

    def test_binaural_tails_are_decorrelated(self):
        ir = so.synth_ir(0.5, FS, n_channels=2, rng=0)
        assert abs(np.corrcoef(ir.data.T)[0, 1]) < 0.1


# --- Traer & McDermott (2016) variants -------------------------------------
# 32 kHz: the profiles (especially "inverted", which uses the max and min over
# all bands) depend on synth_ir's bands reaching 15 kHz
IR_FS = 32000
IR_FB = so.ERBFilterbank(32, 20, 0.95 * IR_FS / 2)  # synth_ir's own bands at this rate


def measured_profile(rt60_profile, seeds=3):
    """Median over noise seeds of measured/requested RT60, per band."""
    requested = so.band_rt60s(1.0, IR_FB.cfs, rt60_profile)[1:-1]
    ratios = []
    for seed in range(seeds):
        ir = so.synth_ir(1.0, IR_FS, rt60_profile=rt60_profile, rng=seed)
        _, rt = so.measure_rt60(ir, n_bands=32, f_lo=20, f_hi=IR_FB.f_hi)
        ratios.append(rt / requested)
    return requested, np.nanmedian(np.array(ratios), axis=0)


def test_ecological_profile_is_reproduced():
    _, ratio = measured_profile("ecological")
    assert np.nanmedian(ratio) == pytest.approx(1, abs=0.05)
    assert np.all((ratio > 0.85) & (ratio < 1.15))


def test_inverted_profile_decays_slowest_at_the_extremes():
    requested, ratio = measured_profile("inverted")
    measured = requested * ratio
    mid = (IR_FB.cfs[1:-1] > 300) & (IR_FB.cfs[1:-1] < 2000)
    assert np.nanmedian(measured[~mid]) > 2 * np.nanmedian(measured[mid])


def test_exaggerated_and_reduced_profiles():
    peak = {p: np.ptp(so.band_rt60s(1.0, IR_FB.cfs, p)) for p in ("reduced", "ecological", "exaggerated")}
    assert peak["reduced"] < peak["ecological"] < peak["exaggerated"]


def test_exaggerated_and_reduced_are_twice_and_half_as_reverberant_rooms():
    """The docstring's definitions: a room twice (half) as reverberant, scaled to this length."""
    cfs = IR_FB.cfs
    eco = {f: so.band_rt60s(f, cfs) for f in (0.5, 1.0, 2.0)}
    np.testing.assert_allclose(so.band_rt60s(1.0, cfs, "exaggerated"), eco[2.0] / 2, rtol=1e-12)
    np.testing.assert_allclose(so.band_rt60s(1.0, cfs, "reduced"), eco[0.5] * 2, rtol=1e-12)


@pytest.mark.parametrize("shape", ["time_reversed", "linear_matched_start", "linear_matched_end"])
def test_atypical_decay_shapes(shape):
    ir = so.synth_ir(1.0, IR_FS, decay_shape=shape, rng=0)
    env = ir.envelope().lowpass(30).db[:, 0]
    early, late = env[len(env) // 10], env[9 * len(env) // 10]
    if shape == "time_reversed":
        assert late > early + 30  # builds up instead of decaying
    else:
        assert early > late + 15  # decays...
        # ...but linearly in amplitude: the dB curve is concave, unlike an exponential
        mid = env[len(env) // 2]
        assert mid > (early + late) / 2 + 3


def test_linear_matched_start_is_short():
    # same starting level and energy as the exponential -> zero at RT60/(2 ln 10)
    exp_ir = so.synth_ir(1.0, IR_FS, rng=0)
    lin_ir = so.synth_ir(1.0, IR_FS, decay_shape="linear_matched_start", rng=0)
    assert lin_ir.duration == pytest.approx(exp_ir.duration / 60 * 60 / (2 * np.log(10)), rel=0.02)


def test_unknown_options_raise():
    with pytest.raises(ValueError, match="decay_shape"):
        so.synth_ir(1.0, IR_FS, decay_shape="parabolic")
    with pytest.raises(ValueError, match="rt60_profile"):
        so.synth_ir(1.0, IR_FS, rt60_profile="flat")
