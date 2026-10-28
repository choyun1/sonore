"""Decimated filterbanks: each band at its own rate (docs/design/frames/decimated-filterbanks.md).

The dense-matrix oracle in test_frames.py checks the frame contract (bounds,
least squares, adjoint, energy) on small grids; this file checks exactness
and tightness on realistic banks, that a decimated band is the undecimated
band sampled, and the extremes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import sonore as so

FS = 16000


def _noise(seconds=0.5, n_channels=1, seed=0, fs=FS):
    rng = np.random.default_rng(seed)
    return so.Sound(rng.standard_normal((int(seconds * fs), n_channels)), fs)


def _error(y, x):
    return np.abs(y.data - x.data).max() / np.abs(x.data).max()


COSINE_BANKS = {
    "erb": so.cosine_filterbank(30, 50, 7600),
    "octave": so.cosine_filterbank(f_lo=125, f_hi=7600, spacing=1 / 6, scale="octave"),
    "semitones": so.cosine_filterbank(f_lo=55, f_hi=7040, scale="cents", spacing=100),
    "mel": so.cosine_filterbank(20, 50, 7600, scale="mel"),
    "linear": so.cosine_filterbank(16, 50, 7600, scale="linear"),
    "random centers": so.cosine_filterbank(
        centers=np.sort(np.random.default_rng(1).uniform(50, 7600, 24)), scale="erb"
    ),
    "wide cosines": so.cosine_filterbank(20, 50, 7600, width=2),
}


@pytest.mark.parametrize("pad", ["auto", 0])
@pytest.mark.parametrize("name", COSINE_BANKS)
def test_cosine_banks_stay_exact_and_tight(name, pad):
    bank = COSINE_BANKS[name]
    frame = so.DecimatedFilterbank(bank)
    x = _noise(n_channels=2)
    coefs = frame.analyze(x, pad=pad)
    assert _error(coefs.to_sound(), x) < 1e-13
    assert frame.is_tight(x.n_samples, FS, pad=pad)
    gain = bank.frame_bounds(x.n_samples, FS, pad=pad)[0]
    assert frame.frame_bounds(x.n_samples, FS, pad=pad) == (gain, gain)
    np.testing.assert_allclose(frame.energy(coefs), gain * np.sum(x.data**2, axis=0), rtol=1e-12)


@pytest.mark.parametrize("name", COSINE_BANKS)
def test_the_cut_leaves_cosine_banks_alone(name):
    bank = COSINE_BANKS[name]
    n = 8000
    assert np.array_equal(so.DecimatedFilterbank(bank).frame_power(n, FS), bank.frame_power(n, FS))


@pytest.mark.parametrize(
    "bank",
    [so.gammatone_filterbank(30, 70, 7000), so.morlet_filterbank(20, 70, 7000, cycles=6)],
    ids=["gammatone", "morlet"],
)
def test_filters_without_a_support_are_cut_and_stay_exact(bank):
    frame = so.DecimatedFilterbank(bank)
    x = _noise()
    coefs = frame.analyze(x, pad=0)
    assert _error(coefs.to_sound(), x) < 1e-13
    assert not frame.is_tight(x.n_samples, FS, pad=0)
    # The cut filters differ from the bank's by less than the floor.
    n = x.n_samples
    cut, full = frame.frame_power(n, FS), bank.frame_power(n, FS)
    assert np.max(np.abs(cut - full) / full) < 1e-4
    # A higher floor cuts more and keeps fewer numbers, and stays exact.
    coarse = so.DecimatedFilterbank(bank, floor_db=-40).analyze(x, pad=0)
    assert sum(map(len, coarse)) < sum(map(len, coefs))
    assert _error(coarse.to_sound(), x) < 1e-13


def test_far_fewer_numbers_than_the_undecimated_bank():
    bank = COSINE_BANKS["semitones"]
    coefs = so.DecimatedFilterbank(bank).analyze(_noise(1.0), pad=0)
    real_numbers = 2 * sum(map(len, coefs))
    assert real_numbers < 2.2 * coefs.n_padded < bank.n_filters * coefs.n_padded
    assert "real numbers per sample" in repr(coefs)


def test_a_decimated_band_is_the_undecimated_band_sampled():
    """Coefficient m of a band with M coefficients is the band's positive-frequency
    half at time m N / M, times sqrt(N / M): exactly its samples when the hop is whole,
    and its band-limited interpolation between samples otherwise."""
    n = 2**5 * 3**3 * 5**2  # a length many fast FFT lengths divide
    x = _noise(n / FS)
    bank = so.cosine_filterbank(12, 100, 6000)
    frame = so.DecimatedFilterbank(bank)
    coefs = frame.analyze(x, pad=0)
    spectrum = np.fft.fft(x.data[:, 0])
    layout = frame._layout(n, FS)
    for k in range(1, len(coefs) - 1):
        band, n_coefs = layout.bands[k], len(coefs[k])
        times = np.arange(n_coefs) * n / n_coefs  # in samples
        kernel = np.exp(2j * np.pi * np.outer(times, band.bins) / n)
        half_band = kernel @ (spectrum[band.bins] * band.response) / n
        np.testing.assert_allclose(coefs[k][:, 0] * np.sqrt(n_coefs / n), half_band, atol=1e-12)
        np.testing.assert_allclose(coefs.times[k], times / FS)
    # Where the hop is whole, twice the real part is the real subband of the undecimated bank.
    subbands = bank.analyze(x, pad=0)
    whole = [k for k in range(1, len(coefs) - 1) if n % len(coefs[k]) == 0]
    assert whole
    for k in whole:
        hop = n // len(coefs[k])
        expected = subbands.data[::hop, k, 0]
        np.testing.assert_allclose(
            2 * np.real(coefs[k][:, 0]) * np.sqrt(len(coefs[k]) / n), expected, atol=1e-12
        )


def test_band_levels_follow_the_envelope_of_a_tone():
    fs, f0 = 16000, 1000.0
    tone = so.pure_tone(0.5, fs, f0).normalize(peak=1.0)
    bank = so.cosine_filterbank(f_lo=110, f_hi=7040, scale="cents", spacing=100)
    coefs = so.DecimatedFilterbank(bank).analyze(tone, pad=0)
    k = int(np.argmin(np.abs(bank.cfs - f0)))
    level = np.median(coefs.db[k][:, 0])
    expected = 20 * np.log10(bank.response(np.array([f0]))[0, k])  # unit amplitude through the filter
    assert level == pytest.approx(expected, abs=0.01)


def test_gains_per_band_and_masked_bands_go_back():
    bank = so.cosine_filterbank(10, 100, 6000)
    frame = so.DecimatedFilterbank(bank)
    x = _noise()
    coefs = frame.analyze(x)
    np.testing.assert_allclose((0.5 * coefs).to_sound().data, 0.5 * x.data, atol=1e-13)
    gains = np.zeros(len(coefs))
    gains[4] = 1.0
    one_band = (coefs * gains).to_sound()
    np.testing.assert_allclose(one_band.data, (bank.analyze(x) * gains).to_sound().data, atol=1e-12)
    with pytest.raises(ValueError, match="one gain per band"):
        coefs * np.ones(3)


def test_extremes():
    # One bandpass filter between its edges.
    x = _noise(0.1)
    one = so.DecimatedFilterbank(so.cosine_filterbank(1, 300, 3000))
    assert _error(one.analyze(x).to_sound(), x) < 1e-13
    # A sound shorter than the lowest band's hop: that band has one or two coefficients.
    short = _noise(0.01)
    semitones = so.DecimatedFilterbank(COSINE_BANKS["semitones"])
    coefs = semitones.analyze(short, pad=0)
    assert min(map(len, coefs[1:-1])) <= 2
    assert _error(coefs.to_sound(), short) < 1e-13
    # Bands so narrow on so short a grid that some fall between bins and have no support.
    tiny = _noise(32 / FS)
    coefs = semitones.analyze(tiny, pad=0)
    assert any(len(band) == 0 for band in coefs)
    assert _error(coefs.to_sound(), tiny) < 1e-13


def test_a_bank_with_a_gap_analyzes_but_refuses_to_synthesize():
    frame = so.DecimatedFilterbank(so.cosine_filterbank(8, 300, 3000, edges=False))
    coefs = frame.analyze(_noise(0.1), pad=0)
    with pytest.raises(ValueError, match="not a frame"):
        coefs.to_sound()


def test_checks_and_plot():
    with pytest.raises(TypeError, match="needs a Filterbank"):
        so.DecimatedFilterbank(so.GaborFrame(0.01, 0.0025))
    with pytest.raises(ValueError, match="floor_db"):
        so.DecimatedFilterbank(so.cosine_filterbank(), floor_db=10)
    frame = so.DecimatedFilterbank(COSINE_BANKS["semitones"])
    coefs = frame.analyze(_noise(0.2))
    with pytest.raises(ValueError, match="expected shape"):
        coefs._with_bands([band[:-1] for band in coefs])
    assert not coefs[3].flags.writeable
    ax = coefs.plot()
    assert ax.get_yscale() == "log" and ax.get_xlim() == (0.0, 0.2)
    plt.close("all")
