"""Modulation spectra."""

import re
import warnings

import numpy as np
import pytest
from helpers import FAST, FAST_HI, FS

import sonore as so


def test_modulation_spectrum_peak():
    x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 8, depth=1)
    ms = so.ModulationSpectrum(so.STFT(x, win_dur=20e-3))
    row = ms.level[0]
    pos = ms.w_t > 2
    assert ms.w_t[pos][np.argmax(row[pos])] == pytest.approx(8, abs=1)


# -- going back to a sound -------------------------------------------------
def _speech_like(rng=0):
    """Noise with a slow (4 Hz) and a fast (16 Hz) amplitude modulation, 2 s at FAST."""
    noise = so.gaussian_noise(2, FAST, rng=rng)
    return so.amplitude_modulate(so.amplitude_modulate(noise, 4, depth=0.9), 16, depth=0.5)


def _octave_spectrum(x, scale="linear"):
    bank = so.cosine_filterbank(f_lo=250, f_hi=FAST_HI, spacing=1 / 6, scale="octave")
    envelopes = bank.analyze(x).envelopes(fs=1000)
    return envelopes, envelopes.modulation_spectrum(scale=scale)


def _share_above(spectrum, rate_hz):
    """Share of the untapered modulation power faster than rate_hz."""
    rate = np.abs(np.fft.fftfreq(spectrum._magnitude.shape[1], 1 / spectrum._analysis.fs))[None, :]
    power = spectrum._magnitude**2
    return power[np.broadcast_to(rate > rate_hz, power.shape)].sum() / power.sum()


@pytest.mark.parametrize("scale", ["linear", "db"])
def test_own_sound_as_carrier_rebuilds_its_envelopes(scale):
    x = _speech_like()
    envelopes, spectrum = _octave_spectrum(x, scale)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to clip
        rebuilt = spectrum.to_envelopes(carrier=x)
    inner = slice(1, -1)
    np.testing.assert_allclose(rebuilt.data[:, inner], envelopes.data[:, inner], rtol=0, atol=1e-9)
    assert np.all(rebuilt.data[:, [0, -1]] == 0)  # the spectrum dropped the edge bands


def test_own_sound_as_carrier_gives_the_sound_back_inside_the_bands():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    y = spectrum.to_sound(carrier=x)
    bank = spectrum._analysis.filterbank
    envelopes = bank.analyze(x).envelopes(fs=1000)
    # rebuilt envelopes carry no padding
    unpadded = so.TemporalEnvelope(envelopes.data, 1000, bank).without_edges()
    expected = (unpadded * bank.analyze(x).tfs()).to_sound()
    np.testing.assert_allclose(y.data, expected.normalize().data, atol=1e-9)


def test_random_phase_keeps_the_magnitudes():
    _, spectrum = _octave_spectrum(_speech_like())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # random phase clips some cells
        rebuilt = spectrum.to_envelopes(rng=1)
    unclipped = rebuilt.data[:, 1:-1, 0].T
    assert (
        np.corrcoef(np.abs(np.fft.fft2(unclipped - unclipped.mean())).ravel(), spectrum._magnitude.ravel())[
            0, 1
        ]
        > 0.9
    )


def test_random_phase_keeps_each_bands_long_term_level():
    # Noise with its low bands 10 dB up: the zero-rate column's phase says which bands are loud,
    # so a random draw that keeps it keeps every band's mean envelope, up to what clipping adds
    # (under 1 dB here; scrambling that column too misses by 4-6 dB).
    x = so.amplitude_modulate(so.gaussian_noise(2, FAST, rng=0), 4, depth=0.3)
    x = x + so.bandpass(so.gaussian_noise(2, FAST, rng=1), 250, 1000) * 3
    envelopes, spectrum = _octave_spectrum(x)
    inner = slice(1, -1)
    own_levels = envelopes.data[:, inner].mean(axis=(0, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # noise envelopes clip a few cells
        for seed in range(3):
            levels = spectrum.to_envelopes(rng=seed).data[:, inner].mean(axis=(0, 2))
            assert np.max(np.abs(20 * np.log10(levels / own_levels))) < 1


def test_clipping_is_reported():
    _, spectrum = _octave_spectrum(_speech_like())
    with pytest.warns(UserWarning, match="were clipped"):
        spectrum.to_envelopes(rng=1)


def test_with_gain_removes_fast_modulations_on_tones():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    slow = spectrum.with_gain(lambda rate, density: np.abs(rate) <= 8)
    assert _share_above(slow, 8) == 0
    assert np.all(slow.level <= spectrum.level + 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = slow.to_sound(carrier="tone", fs=FAST, rng=2)
    _, again = _octave_spectrum(y)
    assert _share_above(again, 12) < 0.25 * _share_above(spectrum, 12)


def test_with_gain_is_symmetric():
    _, spectrum = _octave_spectrum(_speech_like())
    downward_only = spectrum.with_gain(lambda rate, density: (rate * density > 0) | (density == 0))
    magnitude = downward_only._magnitude
    mirrored = np.roll(magnitude[::-1, ::-1], 1, axis=(0, 1))
    np.testing.assert_allclose(magnitude, mirrored, rtol=1e-12, atol=1e-12)


def test_to_sound_needs_envelopes_and_a_rate():
    stft_spectrum = so.ModulationSpectrum(so.STFT(_speech_like(), win_dur=20e-3))
    with pytest.raises(TypeError, match="made from band envelopes"):
        stft_spectrum.to_sound(carrier="tone", fs=FAST)
    _, spectrum = _octave_spectrum(_speech_like())
    with pytest.raises(TypeError, match="fs is needed"):
        spectrum.to_sound(carrier="tone")
    with pytest.raises(ValueError, match="carrier must be"):
        spectrum.to_sound(carrier="pink", fs=FAST)
    with pytest.raises(ValueError, match="carrier must be 'tone' or a Sound, not 'noise'"):
        spectrum.to_sound(carrier="noise", fs=FAST)  # bands of noise would add modulation; pass a noise
    with pytest.raises(TypeError, match="carrier"):
        spectrum.to_sound(fs=FAST)  # the carrier is always named
    with pytest.raises(ValueError, match="differs"):
        spectrum.to_sound(carrier=_speech_like(), fs=FS)
    with pytest.raises(ValueError, match="shorter"):
        spectrum.to_sound(carrier=so.gaussian_noise(1, FAST, rng=0))


@pytest.mark.parametrize(
    "kwargs", [dict(carrier="tone", fs=FAST), dict(carrier=so.gaussian_noise(2, FAST, rng=9))]
)
def test_random_carriers_are_seeded(kwargs):
    _, spectrum = _octave_spectrum(_speech_like())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = spectrum.to_sound(rng=3, **kwargs)
        b = spectrum.to_sound(rng=3, **kwargs)
        c = spectrum.to_sound(rng=4, **kwargs)
    assert a.duration == pytest.approx(2) and a.rms == pytest.approx(1)
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_tone_carrier_is_the_envelopes_on_tones():
    # to_sound is a recipe: the drawn envelopes, put on tones by TemporalEnvelope.to_sound (re-filtered).
    _, spectrum = _octave_spectrum(_speech_like())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = spectrum.to_sound(carrier="tone", fs=FAST, rng=3)
        expected = spectrum.to_envelopes(rng=3).to_sound(carrier="tone", fs=FAST)
    np.testing.assert_array_equal(y.data, expected.normalize().data)


# -- targets drawn in code -------------------------------------------------
def _blob_target(rms_depth=0.2):
    return so.ModulationSpectrum.from_blobs(
        so.ModulationBlob(4, 0.5), 2, f_lo=250, f_hi=FAST_HI, bands_per_octave=6, rms_depth=rms_depth
    )


def test_blob_power_peaks_at_its_center_on_its_side_only():
    blob = so.ModulationBlob(-8, 1.0, level=-6)
    rate = np.array([[-8.0, 8.0, -16.0]])
    density = np.array([[1.0]])
    power = blob.power(rate, density)
    assert power[0, 0] == pytest.approx(10 ** (-0.6))
    assert power[0, 1] == 0
    assert power[0, 2] == pytest.approx(10 ** (-0.6) * np.exp(-1 / (2 * 0.5**2)))
    with pytest.raises(ValueError, match="nonzero rate"):
        so.ModulationBlob(0, 1)


def test_blob_target_has_the_asked_depth_and_its_drawn_magnitudes():
    target = _blob_target()
    envelopes = target.to_envelopes(rng=0).data[:, 1:-1, 0]
    assert envelopes.std() / envelopes.mean() == pytest.approx(0.2)
    deviation = envelopes.T - envelopes.mean()
    np.testing.assert_allclose(
        np.abs(np.fft.fft2(deviation)), target._magnitude, atol=1e-9 * target._magnitude.max()
    )


def test_blob_target_refuses_to_clip_and_names_the_depth_that_fits():
    with pytest.raises(ValueError, match="at most") as refusal:
        _blob_target(rms_depth=0.6).to_envelopes(rng=0)
    fits = float(str(refusal.value).split("at most ")[1].split(" ")[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _blob_target(rms_depth=0.99 * fits).to_envelopes(rng=0)


def test_blob_target_is_found_again_in_the_sound():
    target = _blob_target()
    y = target.to_sound(carrier="tone", fs=FAST, rng=0)
    _, measured = _octave_spectrum(y)
    drawn = target._magnitude > 0.1 * target._magnitude.max()
    correlation = np.corrcoef(np.log(measured._magnitude[drawn]), np.log(target._magnitude[drawn]))[0, 1]
    assert correlation > 0.9


def _db_blob_target(sd_db=10.0):
    return so.ModulationSpectrum.from_blobs(
        so.ModulationBlob(4, 0.5), 2, f_lo=250, f_hi=FAST_HI, bands_per_octave=6, scale="db", sd_db=sd_db
    )


def test_db_blob_target_has_the_asked_spread_and_goes_deeper_than_linear():
    target = _db_blob_target()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing is clipped
        envelopes = target.to_envelopes(rng=0).data[:, 1:-1, 0]
    levels = 20 * np.log10(envelopes)
    assert levels.std() == pytest.approx(10.0)
    np.testing.assert_allclose(
        np.abs(np.fft.fft2(levels.T - levels.mean())), target._magnitude, atol=1e-9 * target._magnitude.max()
    )
    # deeper than any linear draw of one blob can reach (about 0.28)
    assert envelopes.std() / envelopes.mean() > 0.5


def test_db_blob_target_is_found_again_in_the_sound():
    target = _db_blob_target(sd_db=6.0)
    y = target.to_sound(carrier="tone", fs=FAST, rng=0)
    _, measured = _octave_spectrum(y, scale="db")
    drawn = target._magnitude > 0.1 * target._magnitude.max()
    correlation = np.corrcoef(np.log(measured._magnitude[drawn]), np.log(target._magnitude[drawn]))[0, 1]
    assert correlation > 0.9


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"scale": "db", "rms_depth": 0.2}, TypeError),
        ({"sd_db": 6}, TypeError),
        ({"scale": "db", "sd_db": 0}, ValueError),
        ({"scale": "log"}, ValueError),
    ],
)
def test_blob_target_rejects_mixed_depths(kwargs, error):
    with pytest.raises(error):
        so.ModulationSpectrum.from_blobs(so.ModulationBlob(4, 0.5), 1, **kwargs)


def test_iterations_bring_the_sound_closer_to_an_edit():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    slow = spectrum.with_gain(lambda rate, density: np.abs(rate) <= 8)
    shares = []
    for iterations in (0, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = slow.to_sound(carrier=x, iterations=iterations)
        shares.append(_share_above(_octave_spectrum(y)[1], 12))
    assert shares[1] < 0.5 * shares[0]


# -- one step of the search, and drawn timing on a chosen carrier ------------
def _carriers():
    return [
        dict(carrier="tone", fs=FAST, rng=7),
        dict(carrier=so.gaussian_noise(2, FAST, rng=8), rng=7),
        dict(carrier=_speech_like(5)),
    ]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("which", [0, 1, 2])
def test_to_sound_iterations_are_calls_of_iterate(n, which):
    _, spectrum = _octave_spectrum(_speech_like())
    slow = spectrum.with_gain(lambda rate, density: np.abs(rate) <= 8)
    kwargs = _carriers()[which]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = slow.to_sound(iterations=n, **kwargs)
        start = slow.to_sound(**kwargs)
    for _ in range(n):
        start = slow.iterate(start)
    # Equal to rounding only: to_sound's own search starts from the unnormalized sound.
    np.testing.assert_allclose(y.data, start.normalize().data, rtol=0, atol=1e-12)


def test_iterate_is_the_search_step_written_out():
    _, spectrum = _octave_spectrum(_speech_like())
    slow = spectrum.with_gain(lambda rate, density: np.abs(rate) <= 8)
    x = _speech_like(5)
    subbands = slow.filterbank.analyze(x)
    envelopes = slow.to_envelopes(phase=slow.modulation_phase(subbands), quiet=True)
    expected = (envelopes * subbands.tfs()).to_sound()
    np.testing.assert_array_equal(slow.iterate(x).data, expected.data[: x.n_samples])
    with pytest.raises(ValueError, match="shorter"):
        slow.iterate(so.gaussian_noise(1, FAST, rng=0))


@pytest.mark.parametrize("scale", ["linear", "db"])
def test_phase_of_a_sound_gives_its_envelopes_back(scale):
    x = _speech_like()
    _, spectrum = _octave_spectrum(x, scale)
    phase = spectrum.modulation_phase(x)
    assert phase.shape == spectrum._magnitude.shape
    np.testing.assert_array_equal(phase, spectrum.modulation_phase(spectrum.filterbank.analyze(x)))
    np.testing.assert_allclose(
        spectrum.to_envelopes(phase=phase).data, spectrum.to_envelopes(carrier=x).data, rtol=0, atol=1e-12
    )


def test_phase_is_given_alone_and_in_shape():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    phase = spectrum.modulation_phase(x)
    with pytest.raises(TypeError, match="phase"):
        spectrum.to_envelopes(phase=phase, rng=0)
    with pytest.raises(TypeError, match="phase"):
        spectrum.to_envelopes(phase=phase, carrier=x)
    named = re.escape(f"(3, 4); the stored magnitudes have shape {phase.shape}")
    with pytest.raises(ValueError, match=named):
        spectrum.to_envelopes(phase=np.zeros((3, 4)))


def test_hooks_need_a_spectrum_made_from_envelopes():
    x = _speech_like()
    from_array = so.ModulationSpectrum.from_array(np.ones((8, 64)), 0.01, 0.1, "cyc/oct")
    with pytest.raises(TypeError, match="made from band envelopes"):
        _ = from_array.filterbank
    with pytest.raises(TypeError, match="made from band envelopes"):
        from_array.modulation_phase(x)
    with pytest.raises(TypeError, match="made from band envelopes"):
        from_array.to_envelopes(phase=np.zeros((8, 64)))
    with pytest.raises(TypeError, match="made from band envelopes"):
        from_array.iterate(x)


def test_quiet_clips_a_deep_blob_draw_without_a_word():
    target = _blob_target(rms_depth=0.5)
    with pytest.raises(ValueError, match="rms_depth"):
        target.to_envelopes(rng=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        envelopes = target.to_envelopes(rng=0, quiet=True)
        target.to_envelopes(phase=np.angle(np.fft.fft2(np.ones(target._magnitude.shape))), quiet=True)
    assert np.all(envelopes.data >= 0)


def test_sound_carrier_with_rng_draws_the_timing():
    target = _blob_target()
    x = so.harmonic_complex(2, FAST, 150, harmonics=range(1, 30))
    y = target.to_sound(carrier=x, rng=3)
    expected = (target.to_envelopes(rng=3) * target.filterbank.analyze(x).tfs()).to_sound()
    np.testing.assert_array_equal(y.data, so.Sound(expected.data[: x.n_samples], FAST).normalize().data)
    np.testing.assert_array_equal(y.data, target.to_sound(carrier=x, rng=3).data)
    assert not np.array_equal(y.data, target.to_sound(carrier=x, rng=4).data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        searched = target.to_sound(carrier=x, rng=3, iterations=2)
    start = target.to_sound(carrier=x, rng=3)
    for _ in range(2):
        start = target.iterate(start)
    # A harmonic complex leaves some bands nearly empty, whose fine structure is less exact.
    np.testing.assert_allclose(searched.data, start.normalize().data, rtol=0, atol=1e-9)


# -- the circular banks ------------------------------------------------------
def _circular_banks():
    # fs 32: the top constant-Q band (16 Hz) sits on Nyquist, and so does the octave bank's
    return [
        so.ConstantQModulationFilterbank(n_bands=4, f_lo=2.0, f_hi=16.0, Q=2.0),
        so.OctaveModulationFilterbank(n_bands=4, f_hi=16.0),
    ]


@pytest.mark.parametrize("n", [256, 255])
@pytest.mark.parametrize("bank", _circular_banks(), ids=["constant-Q", "octave"])
def test_analytic_output_is_the_real_output_plus_its_quadrature(bank, n):
    x = np.random.default_rng(0).standard_normal(n)
    real = bank.filter(x, 32.0)
    analytic = bank.filter(x, 32.0, analytic=True)
    np.testing.assert_allclose(analytic.real, real, rtol=0, atol=1e-12)  # Nyquist included


@pytest.mark.parametrize("bank", _circular_banks(), ids=["constant-Q", "octave"])
def test_analytic_output_turns_with_positive_frequency(bank):
    # cos(2 pi f t) at a band's center comes out as exp(+i 2 pi f t), scaled by the band's gain
    fs, n = 128.0, 512
    t = np.arange(n) / fs
    for k, cf in enumerate(bank.cfs):
        out = bank.filter(np.cos(2 * np.pi * cf * t), fs, analytic=True)[:, k]
        gain = bank.response([cf])[0, k]
        np.testing.assert_allclose(out, gain * np.exp(2j * np.pi * cf * t), rtol=0, atol=1e-9)


@pytest.mark.parametrize("bank", _circular_banks(), ids=["constant-Q", "octave"])
def test_filter_along_another_axis(bank):
    x = np.random.default_rng(1).standard_normal((3, 200, 2))
    along_time = bank.filter(x, 64.0, axis=1)
    assert along_time.shape == (3, 200, 2, bank.n_bands)
    np.testing.assert_array_equal(along_time, np.moveaxis(bank.filter(np.moveaxis(x, 1, 0), 64.0), 0, 1))


def test_constant_q_filters_are_half_cycle_cosines():
    bank = so.ConstantQModulationFilterbank(n_bands=6, f_lo=2.0, f_hi=64.0, Q=2.0)
    cf = bank.cfs[2]
    # the support is cf +/- cf/Q, so a quarter of it from the center reads cos(pi/4)
    edge_to_center = bank.response([cf, cf * (1 + 1 / (2 * bank.Q)), cf * (1 + 1 / bank.Q)])[:, 2]
    np.testing.assert_allclose(edge_to_center / edge_to_center[0], [1, np.cos(np.pi / 4), 0], atol=1e-12)


# -- the Hann bank's kernels and checks --------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(f_lo=0.0),
        dict(f_lo=10.0, f_hi=5.0),
        dict(window=-1.0),
        dict(window=0.1, f_hi=10.0),  # a 0.1 s window holds 2 cycles only from 20 Hz
    ],
)
def test_hann_bank_refuses_impossible_layouts(kwargs):
    with pytest.raises(ValueError):
        so.HannModulationFilterbank(**kwargs)


def test_hann_kernels_are_centered_and_symmetric():
    bank = so.HannModulationFilterbank(f_lo=2.0, f_hi=32.0)
    for kernel, window in bank.kernels(1000.0):
        np.testing.assert_allclose(window, window[::-1], rtol=0, atol=1e-15)  # sampled on half samples
        assert window.sum() == pytest.approx(1)
        np.testing.assert_allclose(
            kernel[::-1], np.conj(kernel), rtol=0, atol=1e-15
        )  # zero phase at its middle


def test_hann_windows_have_at_least_three_samples():
    bank = so.HannModulationFilterbank(f_lo=1000.0, f_hi=1000.0, window=0.002)
    assert list(bank.lengths(1000.0)) == [3]


def test_hann_filter_refuses_an_unknown_alignment():
    with pytest.raises(ValueError, match="align"):
        so.HannModulationFilterbank().filter(np.zeros(4000), 1000.0, align="left")


def test_hann_transform_is_continuous_where_its_formula_divides_by_zero():
    from sonore.views.modulation import _hann_ft

    assert _hann_ft(np.array([1.0, -1.0])) == pytest.approx([0.5, 0.5])
    assert _hann_ft(np.array([1 - 1e-4, 1 + 1e-4])) == pytest.approx([0.5, 0.5], abs=1e-3)


def test_hann_analytic_output_turns_with_positive_frequency():
    # the kernel is conjugated: a cosine at a band's rate comes out turning forward, at its amplitude
    bank, fs = so.HannModulationFilterbank(f_lo=4.0, f_hi=16.0), 1000.0
    t = np.arange(4000) / fs
    for k, rate in enumerate(bank.cfs):
        out = bank.filter(np.cos(2 * np.pi * rate * t), fs, analytic=True)[1000:3000, k]
        np.testing.assert_allclose(np.abs(out), 1, atol=0.01)
        step = np.angle(out[1:] / out[:-1])  # forward, by one sample's worth of the rate
        np.testing.assert_allclose(step, 2 * np.pi * rate / fs, atol=1e-3)


# -- blobs and the spectrum's own checks -------------------------------------
@pytest.mark.parametrize("widths", [dict(rate_width=0), dict(density_width=-0.1)])
def test_blob_widths_must_be_positive(widths):
    with pytest.raises(ValueError, match="positive"):
        so.ModulationBlob(4, 0.5, **widths)


def test_blob_density_width_is_a_standard_deviation():
    blob = so.ModulationBlob(4, 1.0, density_width=0.5)
    ratio = blob.power(np.array([[4.0]]), np.array([[1.5]])) / blob.power(
        np.array([[4.0]]), np.array([[1.0]])
    )
    assert ratio[0, 0] == pytest.approx(np.exp(-0.5))


@pytest.mark.parametrize(
    "blobs, kwargs, match",
    [
        (so.ModulationBlob(4, 0.5), dict(rms_depth=0), "positive"),
        ([], {}, "at least one"),
        (so.ModulationBlob(4, 40.0, density_width=0.01), {}, "outside"),
    ],
)
def test_blob_target_checks(blobs, kwargs, match):
    with pytest.raises(ValueError, match=match):
        so.ModulationSpectrum.from_blobs(blobs, 1, **kwargs)


def test_stft_spectrum_reads_the_asked_channel():
    left = so.amplitude_modulate(so.gaussian_noise(2, FAST, rng=0), 4, depth=1)
    right = so.amplitude_modulate(so.gaussian_noise(2, FAST, rng=1), 16, depth=1)
    stereo = so.Sound(np.column_stack([left.data[:, 0], right.data[:, 0]]), FAST)
    stft = so.STFT(stereo, win_dur=20e-3)
    assert so.ModulationSpectrum(stft, channel=1).peak()[0] == pytest.approx(16, abs=1)
    np.testing.assert_array_equal(
        so.ModulationSpectrum(stft, channel=1).level,
        so.ModulationSpectrum(so.STFT(right, win_dur=20e-3)).level,
    )


def test_a_modulation_of_fractional_cycles_peaks_in_place():
    # 7.25 Hz over 2 s is 14.5 cycles: without the time taper, the jump between the ends leaks
    x = so.amplitude_modulate(so.gaussian_noise(2, FS, rng=0), 7.25, depth=1)
    rate, _ = so.ModulationSpectrum(so.STFT(x, win_dur=20e-3)).peak()
    assert rate == pytest.approx(7.25, abs=0.3)


def test_spectrum_keeps_non_negative_densities_and_a_floored_level():
    from sonore.frames.gabor import _FLOOR_DB

    ramp = np.linspace(0, 1, 64)[None, :] * np.ones((8, 1))
    spectrum = so.ModulationSpectrum.from_array(ramp, 0.01, 0.1, "cyc/oct")
    assert np.all(spectrum.w_f >= 0) and spectrum.level.shape == (len(spectrum.w_f), 64)
    silent = so.ModulationSpectrum.from_array(np.zeros((8, 64)), 0.01, 0.1, "cyc/oct")
    assert np.all(silent.level == _FLOOR_DB)


def test_octave_spectrum_stops_at_nyquist():
    x = so.gaussian_noise(1, 8000, rng=0)
    spectrum = so.ModulationSpectrum.octave(x)  # f_hi 8000 is above this sound's 4000 Hz Nyquist
    assert spectrum.filterbank.cfs.max() <= 4000


def test_peak_skips_the_mean_only_when_asked():
    # a curved envelope leaves a large zero-rate, zero-density cell after the taper
    curved = (np.linspace(-1, 1, 200) ** 2)[None, :] * np.ones((8, 1))
    spectrum = so.ModulationSpectrum.from_array(
        curved + 0.01 * np.random.default_rng(0).random((8, 200)), 0.01, 0.1, "cyc/oct"
    )
    assert spectrum.peak(exclude_dc=False) == (0.0, 0.0)
    assert spectrum.peak() != (0.0, 0.0)


def test_with_gain_checks_and_shows_its_gain():
    from sonore.frames.gabor import _FLOOR_DB

    _, spectrum = _octave_spectrum(_speech_like())
    with pytest.raises(ValueError, match="non-negative"):
        spectrum.with_gain(-1)
    np.testing.assert_allclose(
        spectrum.with_gain(0.5).level, np.maximum(spectrum.level - 20 * np.log10(2), _FLOOR_DB)
    )
    assert np.all(spectrum.with_gain(0).level == _FLOOR_DB)


def test_carrier_and_rng_are_not_combined_in_to_envelopes():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    with pytest.raises(TypeError, match="rng"):
        spectrum.to_envelopes(carrier=x, rng=0)


def test_modulation_phase_reads_every_channel_on_the_analyzed_grid():
    x = _speech_like()
    _, spectrum = _octave_spectrum(x)
    stereo = so.Sound(np.column_stack([x.data[:, 0], _speech_like(3).data[:, 0]]), FAST)
    subbands = spectrum.filterbank.analyze(stereo)
    both = spectrum.modulation_phase(subbands)
    first = spectrum.modulation_phase(spectrum.filterbank.analyze(x))
    assert not np.allclose(both, first)  # the second channel counts
    with pytest.raises(ValueError, match="analyzed grid"):
        spectrum.modulation_phase(spectrum.filterbank.analyze(so.gaussian_noise(1.5, FAST, rng=0)))
