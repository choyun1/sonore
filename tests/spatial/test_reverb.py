"""Synthetic room impulse responses."""

import numpy as np
import pytest
from helpers import FS

import sonore as so
from sonore.spatial.reverb import _decay_envelopes, _model


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
IR_FB = so.cosine_filterbank(32, 20, IR_FS / 2)  # synth_ir's own bands at this rate


def measured_profile(rt60_profile, seeds=3):
    """Median over noise seeds of measured/requested RT60, per band."""
    requested = so.band_rt60s(1.0, IR_FB.cfs, profile=rt60_profile)[1:-1]
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
    peak = {
        p: np.ptp(so.band_rt60s(1.0, IR_FB.cfs, profile=p)) for p in ("reduced", "ecological", "exaggerated")
    }
    assert peak["reduced"] < peak["ecological"] < peak["exaggerated"]


def test_exaggerated_and_reduced_are_twice_and_half_as_reverberant_rooms():
    """The docstring's definitions: a room twice (half) as reverberant, scaled to this length."""
    cfs = IR_FB.cfs
    eco = {f: so.band_rt60s(f, cfs) for f in (0.5, 1.0, 2.0)}
    np.testing.assert_allclose(so.band_rt60s(1.0, cfs, profile="exaggerated"), eco[2.0] / 2, rtol=1e-12)
    np.testing.assert_allclose(so.band_rt60s(1.0, cfs, profile="reduced"), eco[0.5] * 2, rtol=1e-12)


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


# --- A drawn RT60 curve -------------------------------------------------------
CURVE_FS = 16000
CURVE_CFS = so.cosine_filterbank(32, 20, CURVE_FS / 2).cfs  # synth_ir's own bands at this rate


@pytest.mark.parametrize("rt60", [0.3, 1.0, 3.0])
def test_a_real_room_as_a_curve_gives_the_same_ir(rt60):
    curve = (CURVE_CFS, so.band_rt60s(rt60, CURVE_CFS))
    np.testing.assert_allclose(
        so.synth_ir(curve, CURVE_FS, drr_db=5, rng=7).data,
        so.synth_ir(rt60, CURVE_FS, drr_db=5, rng=7).data,
        rtol=0,
        atol=1e-13,  # rounding in the log-log reading; the IR peaks near 15
    )


def test_a_flat_curve_measures_flat():
    ir = so.synth_ir(((100, 4000), (1.0, 1.0)), CURVE_FS, rng=0)
    _, measured = so.measure_rt60(ir)
    assert np.nanmedian(measured) == pytest.approx(1.0, rel=0.1)


def test_a_boomy_curve_measures_boomy():
    ir = so.synth_ir(((63, 7000), (3.0, 0.3)), CURVE_FS, rng=0)
    cfs, measured = so.measure_rt60(ir)
    low = measured[np.argmin(np.abs(cfs - 150))]
    high = measured[np.argmin(np.abs(cfs - 5000))]
    assert low > 3 * high


def test_a_curve_is_read_in_log_frequency_and_log_rt60():
    # halfway between 100 Hz and 400 Hz in log frequency is 200 Hz, and
    # halfway between 0.5 s and 2 s in log RT60 is 1 s; beyond the ends it holds
    rt60s = so.band_rt60s(((100, 400), (0.5, 2.0)), np.array([50.0, 200.0, 1000.0]))
    np.testing.assert_allclose(rt60s, [0.5, 1.0, 2.0], rtol=1e-12)


@pytest.mark.parametrize(
    "curve",
    [
        ((100, 100, 4000), (1.0, 1.0, 1.0)),  # frequencies not strictly increasing
        ((4000, 100), (1.0, 1.0)),  # decreasing
        ((0, 4000), (1.0, 1.0)),  # zero frequency
        ((100, 4000), (1.0, 0.0)),  # zero RT60
        ((100, 4000), (1.0, np.inf)),  # infinite RT60
        ((100, 4000), (1.0, 1.0, 1.0)),  # mismatched lengths
        ((100, np.nan), (1.0, 1.0)),  # NaN frequency
        ((100, np.inf), (1.0, 1.0)),  # infinite frequency
        ((), ()),  # empty
        (np.ones((2, 2)), np.ones((2, 2))),  # not 1-D
    ],
)
def test_bad_curves_raise(curve):
    with pytest.raises(ValueError, match="RT60 curve"):
        so.synth_ir(curve, CURVE_FS)


def test_a_curve_refuses_other_profiles():
    with pytest.raises(ValueError, match="already a profile"):
        so.synth_ir(((100, 4000), (1.0, 1.0)), CURVE_FS, rt60_profile="inverted")


@pytest.mark.parametrize("rt60", [True, [(100, 4000), (1.0, 1.0)], "1.0"])
def test_rt60_must_be_a_number_or_a_curve(rt60):
    with pytest.raises(TypeError, match="rt60"):
        so.synth_ir(rt60, CURVE_FS)


class TestSynthIrInDetail:
    @pytest.mark.parametrize("shape", ["linear_matched_start", "linear_matched_end"])
    def test_linear_decays_keep_each_bands_energy(self, shape):
        t = np.arange(0, 3, 1e-4)
        taus, onset_db = np.array([0.3, 0.8, 1.2]), np.array([-3.0, 0.0, 2.0])
        exponential = _decay_envelopes(t, onset_db, taus, "exponential")
        linear = _decay_envelopes(t, onset_db, taus, shape)
        np.testing.assert_allclose(np.sum(linear**2, 0), np.sum(exponential**2, 0), rtol=2e-3)
        if shape == "linear_matched_start":
            np.testing.assert_allclose(linear[0], exponential[0])
        else:
            np.testing.assert_array_equal(np.argmax(linear == 0, axis=0), np.round(taus / 1e-4).astype(int))

    def test_the_onset_fits_share_one_slope(self):
        """Why synth_ir's onset levels do not depend on the room's RT60."""
        fit_drr, _, _ = _model()
        assert np.ptp(fit_drr[:, 0]) == 0

    def test_ecological_onsets_follow_the_fit(self):
        fs = 16000
        fb = so.cosine_filterbank(32, 20, fs / 2)
        natural = so.synth_ir(1.0, fs, rng=3)
        constant = so.synth_ir(1.0, fs, rng=3, drr_profile="constant")
        n = round(0.02 * fs)
        onset = lambda ir: 10 * np.log10(np.sum(fb.analyze(ir).data[:n, :, 0] ** 2, 0))  # noqa: E731
        measured = onset(natural) - onset(constant)
        fit_drr, _, fit_freqs = _model()
        wanted = np.interp(fb.cfs, fit_freqs, fit_drr[:, 1])
        measured, wanted = measured - np.median(measured), wanted - np.median(wanted)
        np.testing.assert_allclose(measured[3:], wanted[3:], atol=0.5)

    def test_a_tail_has_rms_one(self):
        assert so.synth_ir(0.5, FS, rng=0).rms == pytest.approx(1)

    def test_the_direct_impulse_is_in_every_channel(self):
        ir = so.synth_ir(0.5, FS, drr_db=20, n_channels=2, rng=0)
        np.testing.assert_allclose(ir.data[0], 1, atol=0.05)


def test_measure_rt60_returns_its_bandpass_centers():
    ir = so.synth_ir(0.5, 16000, rng=0)
    cfs, _ = so.measure_rt60(ir, n_bands=12, f_lo=100, f_hi=6000)
    np.testing.assert_array_equal(cfs, so.cosine_filterbank(12, 100, 6000).cfs[1:-1])


@pytest.mark.parametrize("rt60", [0.0, -1.0, np.nan, np.inf])
def test_rt60_must_be_positive_and_finite(rt60):
    with pytest.raises(ValueError, match="positive number of seconds"):
        so.synth_ir(rt60, CURVE_FS)


@pytest.mark.parametrize("decay_db", [0.0, -10.0])
def test_decay_db_must_be_positive(decay_db):
    with pytest.raises(ValueError, match="decay_db"):
        so.synth_ir(1.0, CURVE_FS, decay_db=decay_db)
