"""Time-frequency masks: views that apply to any frame's coefficients."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


def test_ibm_recovers_target_when_well_separated():
    target = so.pure_tone(1, FS, 500)
    masker = so.gaussian_noise(1, FS, band=(3000, 6000), rng=0)
    mixture = target + masker
    mask = so.ideal_binary_mask(so.STFT(target), so.STFT(masker))
    out = (so.STFT(mixture) * mask).to_sound()
    snr = 10 * np.log10(np.sum(target.data**2) / np.sum((out - target).data ** 2))
    assert snr > 20


def test_ideal_ratio_mask_known_cases():
    """(|T|^2 / (|T|^2 + |M|^2))**beta: equal powers give 0.5**beta, a silent masker 1, silence 0."""
    noise = so.STFT(so.gaussian_noise(0.2, FS, rng=0))
    silent = so.STFT(so.silence(0.2, FS))
    for beta in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(so.ideal_ratio_mask(noise, noise, beta=beta).values, 0.5**beta, rtol=1e-12)
    np.testing.assert_allclose(so.ideal_ratio_mask(noise, silent).values, 1.0, rtol=1e-12)
    np.testing.assert_array_equal(so.ideal_ratio_mask(silent, noise).values, 0.0)
    np.testing.assert_array_equal(so.ideal_ratio_mask(silent, silent).values, 0.0)


def test_ibm_needs_the_target_to_exceed_the_criterion():
    """Cells where the target equals the masker plus lc_db are not kept: the target must exceed it."""
    noise = so.STFT(so.gaussian_noise(0.2, FS, rng=0))
    assert not so.ideal_binary_mask(noise, noise, lc_db=0).values.any()
    assert so.ideal_binary_mask(noise, noise, lc_db=-1).values.all()


def test_mask_multiplies_from_either_side():
    target = so.STFT(so.pure_tone(0.2, FS, 500))
    masker = so.STFT(so.gaussian_noise(0.2, FS, rng=0))
    mask = so.ideal_binary_mask(target, masker)
    np.testing.assert_array_equal((mask * masker).data, (masker * mask).data)


def test_a_mask_is_a_view_that_refuses_to_be_a_sound():
    target = so.STFT(so.pure_tone(0.2, FS, 500))
    mask = so.ideal_binary_mask(target, so.STFT(so.gaussian_noise(0.2, FS, rng=0)))
    assert isinstance(mask, so.View)
    with pytest.raises(so.NotInvertibleError, match=r"\(stft \* mask\).to_sound\(\)"):
        mask.to_sound()
    with pytest.raises(so.NotInvertibleError):
        mask.synthesize()


def test_masks_work_on_subbands_through_their_envelopes():
    """On subbands the mask follows each band's Hilbert envelope: a tone and
    a noise far apart in frequency separate, and the mask does not flicker
    at the tone's frequency."""
    fs = 16000
    tone = so.pure_tone(0.5, fs, 500)
    noise = so.gaussian_noise(0.5, fs, band=(3000, 6000), rng=1)
    bank = so.cosine_filterbank(24, 80, 7000)
    target, masker = bank.analyze(tone), bank.analyze(noise)
    mask = so.ideal_binary_mask(target, masker)
    assert mask.values.shape == target._full.shape
    band = int(np.argmin(np.abs(bank.cfs - 500)))
    middle = slice(target.pad + fs // 10, target.pad + 4 * fs // 10)
    assert mask.values[middle, band, 0].all()  # no flicker: kept at every sample
    out = (bank.analyze(tone + noise) * mask).to_sound()
    snr = 10 * np.log10(np.sum(tone.data**2) / np.sum((out - tone).data ** 2))
    assert snr > 15
    soft = so.ideal_ratio_mask(target, masker)
    assert np.all((soft.values >= 0) & (soft.values <= 1))
    np.testing.assert_array_equal((mask * masker)._full, (masker * mask)._full)


def test_masks_work_on_time_varying_stfts():
    frame = so.TVGaborFrame.pitch_adaptive([0.0, 0.2], [120.0, 200.0], t_end=0.2)
    target = frame.analyze(so.pure_tone(0.2, FS, 500))
    masker = frame.analyze(so.gaussian_noise(0.2, FS, band=(3000, 6000), rng=2))
    mask = so.ideal_binary_mask(target, masker)
    assert (target * mask).data.shape == target.data.shape


def test_masks_refuse_other_grids_and_kinds():
    a = so.STFT(so.gaussian_noise(0.2, FS, rng=0))
    b = so.STFT(so.gaussian_noise(0.3, FS, rng=0))
    mask = so.ideal_binary_mask(a, a)
    with pytest.raises(ValueError, match="different grid"):
        b * mask
    with pytest.raises(ValueError, match="same frame"):
        so.ideal_binary_mask(a, b)
    bands = so.cosine_filterbank().analyze(so.gaussian_noise(0.2, FS, rng=0))
    with pytest.raises(TypeError, match="same kind"):
        so.ideal_binary_mask(a, bands)
    with pytest.raises(ValueError, match="different grid"):
        mask.apply(bands)


def test_masks_plot_on_either_grid():
    import matplotlib

    matplotlib.use("Agg")
    stft = so.STFT(so.gaussian_noise(0.1, FS, rng=0))
    so.ideal_ratio_mask(stft, stft).plot()
    bands = so.cosine_filterbank(8).analyze(so.gaussian_noise(0.1, 16000, rng=0))
    ax = so.ideal_ratio_mask(bands, bands).plot()
    assert ax.get_yscale() == "log"


def test_masks_refuse_coefficients_from_another_rate_or_frame():
    """A same-shaped grid is not enough: a mask made at 16 kHz with 20 ms
    windows does not apply to half as long a sound at 32 kHz with 10 ms ones,
    whose STFT has the same shape, nor across filterbanks."""
    low = so.STFT(so.gaussian_noise(0.2, 16000, rng=0), win_dur=0.020, hop_dur=0.005)
    high = so.STFT(so.gaussian_noise(0.1, 32000, rng=0), win_dur=0.010, hop_dur=0.0025)
    assert low.data.shape == high.data.shape
    mask = so.ideal_binary_mask(low, low)
    with pytest.raises(ValueError, match="different grid"):
        high * mask
    with pytest.raises(ValueError, match="same frame"):
        so.ideal_binary_mask(low, high)
    # the same rate and shape, another window
    hamming = so.GaborFrame(0.020, 0.005, window="hamming")
    other_window = so.STFT(so.gaussian_noise(0.2, 16000, rng=0), frame=hamming)
    assert other_window.data.shape == low.data.shape
    with pytest.raises(ValueError, match="different grid"):
        other_window * mask
    # an equal frame built anew is the same grid
    again = so.STFT(so.gaussian_noise(0.2, 16000, rng=1), win_dur=0.020, hop_dur=0.005)
    assert (again * mask).data.shape == again.data.shape
    bank = so.cosine_filterbank(8, 80, 7000)
    bands = bank.analyze(so.gaussian_noise(0.2, 16000, rng=0))
    mask = so.ideal_binary_mask(bands, bands)
    # another filterbank with the same padding and shape
    other_bank = so.cosine_filterbank(8, 80, 6000).analyze(so.gaussian_noise(0.2, 16000, rng=0))
    assert other_bank._full.shape == bands._full.shape
    with pytest.raises(ValueError, match="different grid"):
        other_bank * mask
    # the same filterbank at 32 kHz, on a sound whose padded length is the same
    faster = bank.analyze(so.gaussian_noise(1572 / 32000, 32000, rng=0))
    assert faster._full.shape == bands._full.shape
    with pytest.raises(ValueError, match="different grid"):
        faster * mask


def test_mask_values_are_checked_and_read_only():
    stft = so.STFT(so.gaussian_noise(0.1, FS, rng=0))
    with pytest.raises(ValueError, match="shape"):
        so.Mask(np.ones(3), stft)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            so.Mask(np.full(stft.data.shape, bad), stft)
    mask = so.Mask(np.ones(stft.data.shape), stft)
    with pytest.raises(ValueError, match="read-only"):
        mask.values[0, 0, 0] = 0
    assert repr(mask) == "Mask(STFT, mean gain 1)"


def test_gains_above_1_are_allowed():
    """Gains are usually between 0 and 1, but not held there: 2 doubles."""
    stft = so.STFT(so.gaussian_noise(0.1, FS, rng=0))
    doubled = stft * so.Mask(np.full(stft.data.shape, 2.0), stft)
    np.testing.assert_allclose(doubled.data, 2 * stft.data)


def test_beta_must_be_positive():
    noise = so.STFT(so.gaussian_noise(0.1, FS, rng=0))
    for beta in (0.0, -1.0):
        with pytest.raises(ValueError, match="beta must be positive"):
            so.ideal_ratio_mask(noise, noise, beta=beta)


def test_a_mask_separates_a_tone_on_a_time_varying_stft():
    """A tone and an equally loud band of noise far above it: the target
    comes back at 35.8 dB SNR with the mask (measured), 0 dB without."""
    frame = so.TVGaborFrame.pitch_adaptive([0.0, 0.2], [120.0, 200.0], t_end=0.2)
    tone = so.pure_tone(0.2, FS, 500)
    noise = so.gaussian_noise(0.2, FS, band=(3000, 6000), rng=2)
    noise = noise * (tone.rms / noise.rms)
    mask = so.ideal_binary_mask(frame.analyze(tone), frame.analyze(noise))
    out = (frame.analyze(tone + noise) * mask).to_sound()
    snr = 10 * np.log10(np.sum(tone.data**2) / np.sum((out - tone).data ** 2))
    assert snr > 30


def test_the_subband_ratio_mask_follows_the_envelope_not_the_waveform():
    """In the band of a 500 Hz tone, with quieter noise as the masker, the
    ratio mask stays near 1 at every sample (measured minimum 0.9997); from
    the waveform squared it would drop to 0 twice a cycle."""
    fs = 16000
    bank = so.cosine_filterbank(24, 80, 7000)
    tone = bank.analyze(so.pure_tone(0.5, fs, 500))
    noise = so.gaussian_noise(0.5, fs, rng=1)
    noise = bank.analyze(noise * (0.1 * so.pure_tone(0.5, fs, 500).rms / noise.rms))
    band = int(np.argmin(np.abs(bank.cfs - 500)))
    middle = slice(tone.pad + fs // 10, tone.pad + 4 * fs // 10)
    assert so.ideal_ratio_mask(tone, noise).values[middle, band, 0].min() > 0.9
