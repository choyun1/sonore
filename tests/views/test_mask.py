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
        np.testing.assert_allclose(so.ideal_ratio_mask(noise, noise, beta).values, 0.5**beta, rtol=1e-12)
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
