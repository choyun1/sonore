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
import scipy.fft as sp_fft

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
    assert "real numbers per padded sample" in repr(coefs)


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


def _noise_samples(n, seed=0):
    return so.Sound(np.random.default_rng(seed).standard_normal(n), FS)


def _band_at(subband, times, frequencies):
    """A real band's value at fractional sample ``times``, interpolated with the DFT bins
    taken at ``frequencies`` (in bins)."""
    n = len(subband)
    return np.exp(2j * np.pi * np.outer(times, frequencies) / n) @ np.fft.fft(subband) / n


@pytest.mark.parametrize(
    "bank",
    [so.gammatone_filterbank(20, 70, 7000), so.morlet_filterbank(20, 70, 7000, cycles=6)],
    ids=["gammatone", "morlet"],
)
@pytest.mark.parametrize("floor_db", [-40, -100])
def test_the_cut_keeps_exactly_the_bins_above_the_floor(bank, floor_db):
    """Every bandpass filter is cut, the Morlet's too, though its tails underflow to
    zero far from its center: what is left is the bank's response wherever it is
    within ``floor_db`` of the peak."""
    n = FS
    layout = so.DecimatedFilterbank(bank, floor_db=floor_db)._layout(n, FS)
    half = bank.rfft_response(n, FS)
    for k in range(1, bank.n_filters - 1):
        band, column = layout.bands[k], half[:, k]
        kept = np.flatnonzero(np.abs(column) >= np.abs(column).max() * 10 ** (floor_db / 20))
        assert band.bins[0] == kept[0] and band.bins[-1] == kept[-1]
        shared = (kept == 0) | (2 * kept == n)  # 0 Hz and Nyquist, split between the two halves
        expected = np.where(shared, column[kept] / np.sqrt(2), column[kept])
        np.testing.assert_array_equal(band.response[np.isin(band.bins, kept)], expected)
        assert np.all(band.response[~np.isin(band.bins, kept)] == 0)


def test_a_cosine_filters_tail_is_never_cut():
    """A cosine filter is exactly zero outside its band, so it is not cut, even where a
    grid bin falls so close to its edge that the response is far below the floor."""
    bank = so.cosine_filterbank(centers=[999.999999, 2000, 3000], scale="linear")
    frame = so.DecimatedFilterbank(bank)
    # 1 Hz bins: the bin at 1000 Hz is a millionth of a hertz inside the 2 kHz filter, the only bandpass
    n = FS
    tail = bank.rfft_response(n, FS)[1000, 1]
    assert 0 < abs(tail) < 1e-5
    band = frame._layout(n, FS).bands[1]
    assert band.response[band.bins == 1000] == tail
    assert frame.is_tight(n, FS, pad=0)


def test_each_band_keeps_the_smallest_fast_length_that_holds_its_support():
    bank = so.cosine_filterbank(12, 100, 6000)
    n = 8000
    layout = so.DecimatedFilterbank(bank)._layout(n, FS)
    half = bank.rfft_response(n, FS)
    for k, band in enumerate(layout.bands):
        nonzero = np.flatnonzero(half[:, k])
        if k == 0:
            support = 2 * nonzero.max() + 1
        elif k == len(layout.bands) - 1:
            support = n - 2 * nonzero.min() + 1
        else:
            support = nonzero.max() - nonzero.min() + 1
        assert len(band.bins) == support
        assert band.n_coefs == sp_fft.next_fast_len(support)


def test_the_edge_bands_coefficients():
    """The lowpass's coefficients are its band sampled, real for a real sound. The
    highpass's are its band interpolated with frequencies from 0 to the sampling rate
    rather than signed ones: the band shifted down by Nyquist, times a phase, so they
    are its samples only at whole-sample times. Both levels are the magnitude of the
    coefficient, not a Hilbert envelope."""
    n = 3**4 * 5**2 * 7  # odd, so no Nyquist bin
    x = _noise_samples(n)
    bank = so.cosine_filterbank(12, 100, 6000)
    coefs = so.DecimatedFilterbank(bank).analyze(x, pad=0)
    subbands = bank.analyze(x, pad=0)
    for k, frequencies in ((0, np.fft.fftfreq(n) * n), (len(coefs) - 1, np.arange(n))):
        n_coefs = len(coefs[k])
        values = coefs[k][:, 0] * np.sqrt(n_coefs / n)
        expected = _band_at(subbands.data[:, k, 0], np.arange(n_coefs) * n / n_coefs, frequencies)
        np.testing.assert_allclose(values, expected, atol=1e-10)
        np.testing.assert_allclose(10 ** (coefs.db[k][:, 0] / 20), np.abs(values), atol=1e-10)
    np.testing.assert_allclose(coefs[0].imag, 0, atol=1e-12)


def test_times_start_before_the_sound_by_the_padding():
    frame = so.DecimatedFilterbank(so.cosine_filterbank(10, 100, 6000))
    coefs = frame.analyze(_noise_samples(4000), pad=0.01)
    assert coefs.pad == 160 and coefs.n_samples == 4000
    for times, values in zip(coefs.times, coefs, strict=True):
        assert times[0] == pytest.approx(-0.01)
        np.testing.assert_allclose(np.diff(times), coefs.n_padded / len(values) / FS)


def test_refusals_and_the_count_in_the_repr():
    with pytest.raises(ValueError, match="floor_db"):
        so.DecimatedFilterbank(so.cosine_filterbank(), floor_db=-np.inf)
    frame = so.DecimatedFilterbank(so.cosine_filterbank(10, 100, 6000))
    coefs = frame.analyze(_noise_samples(4000), pad=0)
    with pytest.raises(ValueError, match="same number of channels"):
        coefs._with_bands([np.ones((len(band), 1 + (k == 3))) for k, band in enumerate(coefs)])
    per_sample = 2 * sum(map(len, coefs)) / coefs.n_padded
    assert f"{per_sample:.2f} real numbers per padded sample" in repr(coefs)
