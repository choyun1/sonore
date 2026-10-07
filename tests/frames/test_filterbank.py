"""Cosine, gammatone and Morlet filterbanks, and subbands."""

from dataclasses import dataclass

import numpy as np
import pytest
from helpers import FS
from scipy.signal import hilbert

import sonore as so
from sonore.core.utils import FREQUENCY_SCALES, FrequencyScale
from sonore.frames.filterbank import Cosine, FilterType


class TestFilterbank:
    def test_power_complementary(self):
        fb = so.cosine_filterbank(30, 50, 8000)
        H = fb.response(np.linspace(0, 22050, 5000))
        np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)

    def test_perfect_reconstruction(self):
        g = so.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(so.cosine_filterbank().analyze(g).to_sound().data, g.data, atol=1e-10)

    def test_vocoder_runs_and_keeps_level(self):
        x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 20))
        v = so.channel_vocode(x, 8, rng=0)
        assert len(v) == len(x) and v.rms == pytest.approx(x.rms)


def test_octave_filterbank_reconstructs():
    fb = so.cosine_filterbank(f_lo=125, f_hi=6000, spacing=1 / 12, scale="octave")
    H = fb.response(np.linspace(0, FS / 2, 5000))
    np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)
    g = so.gaussian_noise(0.5, 16000, rng=0)
    np.testing.assert_allclose(fb.analyze(g).to_sound().data, g.data, atol=1e-10)


def test_subbands_copy_and_leave_the_callers_array_writeable():
    bank = so.cosine_filterbank(8, 100, 4000)
    subbands = bank.analyze(so.gaussian_noise(0.1, FS, rng=0))
    full = np.array(subbands._full)
    copied = so.Subbands(full, subbands.fs, bank, subbands.pad)
    assert full.flags.writeable and not copied._full.flags.writeable
    full[:] = 0  # the caller changing its array leaves the Subbands as built
    np.testing.assert_array_equal(copied._full, subbands._full)


def test_subband_plot_labels_and_scale():
    import matplotlib

    matplotlib.use("Agg")
    sb = so.cosine_filterbank(6, 100, 6000).analyze(so.exponential_chirp(0.2, FS, 100, 6000))
    axes = sb.plot()
    assert len(axes) == 8
    labels = [ax.get_ylabel() for ax in axes]
    assert labels[0] == "> 6000" and labels[-1] == "< 100"
    assert len({ax.get_ylim() for ax in axes}) == 1  # shared amplitude scale
    with pytest.raises(ValueError, match="need 8 axes"):
        sb.plot(axes[:6])
    chosen = sb.plot(bands=[1, 3, 5])
    assert [ax.get_ylabel() for ax in chosen] == [f"{sb.cfs[i]:.0f}" for i in (5, 3, 1)]


# ------------------------------------------- gammatone and Morlet (frames step 2)
# Checks the gammatone formulas, the edge filters' coverage, the Nyquist rule
# and the gammatone phase options (derivations in docs/design/frames/frames.md, step 2).


def _gammatone_formula(f, fc):
    """Closed-form FT of t^3 exp(-2 pi b t) cos(2 pi fc t), t >= 0, unit gain at fc."""
    b = 1.019 * 24.7 * (4.37e-3 * fc + 1)
    k = 6 / (2 * np.pi) ** 4 / 2

    def H(f):
        return k * ((b + 1j * (f - fc)) ** -4 + (b + 1j * (f + fc)) ** -4)

    return H(f) / abs(H(fc))


class TestGammatone:
    fb = so.gammatone_filterbank(2, 100, 4000, edges=False)

    def test_response_is_the_closed_form(self):
        f = np.linspace(0, 8000, 801)
        H = self.fb.response(f)
        for k, fc in enumerate(self.fb.cfs):
            np.testing.assert_allclose(H[:, k], _gammatone_formula(f, fc), rtol=1e-12, atol=1e-15)
        zero = so.gammatone_filterbank(2, 100, 4000, edges=False, phase="zero")
        assert np.array_equal(zero.response(f), np.abs(H))

    def test_group_delay_and_envelope_peak(self):
        """Group delay 4/(2 pi b) at cf, envelope peak 3/(2 pi b), within 1%."""
        cfs, df = self.fb.cfs, 1e-3
        b = 1.019 * 24.7 * (4.37e-3 * cfs + 1)
        H = self.fb.response(np.concatenate([cfs - df, cfs + df]))
        gd = -np.angle(H[[2, 3], [0, 1]] / H[[0, 1], [0, 1]]) / (2 * np.pi * 2 * df)
        np.testing.assert_allclose(gd, 4 / (2 * np.pi * b), rtol=1e-2)
        fs, n = 256000.0, 1 << 17
        h = np.fft.irfft(self.fb.rfft_response(n, fs), n=n, axis=0)
        peak = np.argmax(np.abs(hilbert(h, axis=0)), axis=0) / fs
        np.testing.assert_allclose(peak, 3 / (2 * np.pi * b), rtol=1e-2)

    @pytest.mark.parametrize("phase", ["causal", "zero"])
    @pytest.mark.parametrize("n", [4000, 4001])
    def test_exact_up_to_nyquist(self, phase, n):
        """Bands up to 7.5 kHz at 16 kHz, where Im H(fs/2) is large: exact on even and odd lengths."""
        fb = so.gammatone_filterbank(30, 50, 7500, phase=phase)
        x = so.Sound(np.random.default_rng(0).standard_normal((n, 2)), 16000)
        for pad in ("auto", 0):
            np.testing.assert_allclose(fb.analyze(x, pad=pad).to_sound().data, x.data, rtol=0, atol=1e-12)

    def test_bank_attributes(self):
        """f_lo and f_hi are the outer knots; the bandpass centers lie strictly
        inside, equally spaced on the ERB scale, and the edge filters' corners
        are the outer knots."""
        fb = so.gammatone_filterbank(40, 50, 7000)
        assert fb.n_filters == 42 and fb.unit == "ERB" and not fb.is_tight(4000, 16000)
        knots = np.linspace(so.freq_to_erb(50), so.freq_to_erb(7000), 42)
        np.testing.assert_allclose(fb.band_cfs, so.erb_to_freq(knots[1:-1]), rtol=1e-14)
        assert fb.spacing == pytest.approx(knots[1] - knots[0], rel=1e-14)
        np.testing.assert_allclose(fb.cfs[[0, -1]], [50, 7000], rtol=1e-12)
        assert so.gammatone_filterbank(40, 50, 7000, edges=False).n_filters == 40
        with pytest.raises(ValueError, match="phase"):
            so.gammatone_filterbank(phase="minimum")
        with pytest.raises(ValueError, match="n_bands"):
            so.gammatone_filterbank(0)


class TestMorlet:
    def test_response(self):
        fb = so.morlet_filterbank(3, 125, 2000, cycles=5, edges=False)  # centers 250, 500, 1000 Hz
        f = np.linspace(0, 4000, 4001)
        H = fb.response(f)
        assert H.dtype == float and np.all(H[0] == 0)  # DC correction
        assert np.allclose(H[[250, 500, 1000], [0, 1, 2]], 1.0)
        assert np.all(H >= 0) and np.array_equal(np.argmax(H, axis=0), [250, 500, 1000])
        assert fb.unit == "oct" and fb.spacing == pytest.approx(1.0)
        np.testing.assert_allclose(fb.band_cfs, [250, 500, 1000], rtol=1e-12)

    def test_bare_bank_is_not_a_frame(self):
        """Without edges A = 0 (DC); analysis works, synthesis refuses."""
        fb = so.morlet_filterbank(28, 50, 7000, edges=False)
        assert fb.frame_bounds(1000, 16000)[0] == 0
        sb = fb.analyze(so.gaussian_noise(0.05, 16000, rng=0))
        assert sb.envelopes().without_edges() is not None
        with pytest.raises(ValueError, match="not a frame"):
            sb.to_sound()

    def test_exact_with_edges(self):
        x = so.gaussian_noise(0.25, 16000, n_channels=2, rng=1)
        fb = so.morlet_filterbank(28, 50, 7000)
        np.testing.assert_allclose(fb.analyze(x).to_sound().data, x.data, rtol=0, atol=1e-12)


def _edge_banks(width):
    """The two banks of the edge-coverage table in docs/design/frames/frames.md, step 2
    (gammatone 1/ERB, Morlet 4/oct, 50-7000 Hz)."""
    e = so.freq_to_erb(50.0)
    n = len(np.arange(e, so.freq_to_erb(7000.0), 1.0))
    return (
        so.gammatone_filterbank(
            n, float(so.erb_to_freq(e - 1)), float(so.erb_to_freq(e + n)), edge_width=width
        ),
        so.morlet_filterbank(29, 50 * 2**-0.25, 50 * 2**7.25, edge_width=width),
    )


@pytest.mark.parametrize(
    ("width", "ratios", "ring_ms"), [(1, (0.53, 0.58), (107, 234)), (2, (0.18, 0.30), (None, 149))]
)
def test_edge_filters_match_design_table(width, ratios, ring_ms):
    """A/B and the edge filters' ringing at fs = N = 16000, as in that table (within 2%).
    At width 2 the causal gammatone bank itself rings longest (73 ms)."""
    for fb, ratio, ring in zip(_edge_banks(width), ratios, ring_ms, strict=True):
        lo, hi = fb.frame_bounds(16000, 16000, pad=0)
        assert lo / hi == pytest.approx(ratio, rel=2e-2)
        if ring is not None:
            assert 1e3 * fb.ringing(16000) / 16000 == pytest.approx(ring, rel=2e-2)


def test_edge_filters_are_flat_outside_and_zero_inside():
    fb = so.gammatone_filterbank(20, 200, 4000)
    f = np.array([0.0, fb.cfs[0], fb.band_cfs[0], fb.band_cfs[-1], fb.cfs[-1], 20000.0])
    H = fb.response(f)
    g = np.sqrt(fb.s_floor)
    assert np.allclose(H[:, 0], [g, g, 0, 0, 0, 0]) and np.allclose(H[:, -1], [0, 0, 0, 0, g, g])
    assert so.gammatone_filterbank(20, 200, 4000, edge_width=2).cfs[0] < fb.cfs[0]


def test_new_banks_feed_the_cochleagram_tools():
    x = so.gaussian_noise(0.25, 16000, rng=2)
    for fb in (so.gammatone_filterbank(24, 80, 6000), so.morlet_filterbank(20, 80, 6000)):
        env = fb.analyze(x).envelopes()
        assert env.data.shape == (len(x), fb.n_filters, 1)
        env.modulation_spectrum()  # needs spacing and unit
    bare = so.gammatone_filterbank(24, 80, 6000, edges=False).analyze(x).envelopes()
    assert bare.without_edges() is bare  # no edge bands to drop


# ------------------------------------------- tightness is measured (D13)
# docs/design/frames/filterbanks.md, D13: a bank must never be reported tight
# when it is not, because the tight path then loses the ripple in s silently
# (C9). Each case carries the answer the mathematics gives, independently of
# the code: cosines one gap wide are tight on any increasing centers (C1, C2),
# wider ones on equal spacing exactly when twice the width is whole (C5), and no
# other filter type is tight.


@dataclass(frozen=True)
class _UniformFormulaAnyway(Cosine):
    """The equal-spacing cosine formula applied to centers that are not
    equally spaced: the mistake the gap formula exists to avoid."""

    def responses(self, bank, freqs, edges=None):
        scale_pos = bank.scale.to_scale(freqs)[:, None]
        knots = bank._knots
        distance = (scale_pos - knots[None, :]) / (knots[1] - knots[0])
        transfer = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
        transfer[:, 0] = np.where(scale_pos[:, 0] <= knots[0], 1.0, transfer[:, 0])
        transfer[:, -1] = np.where(scale_pos[:, 0] >= knots[-1], 1.0, transfer[:, -1])
        return transfer


def _random_centers(scale, seed, f_lo=60.0, f_hi=7000.0, n=20):
    to_scale, from_scale = FREQUENCY_SCALES[scale].to_scale, FREQUENCY_SCALES[scale].from_scale
    knots = np.sort(np.random.default_rng(seed).uniform(to_scale(f_lo), to_scale(f_hi), n))
    return from_scale(knots)


def _nudged():
    bank = so.cosine_filterbank(20, 60, 7000)
    knots = np.array(bank.knots)
    knots[7] += 0.2 * (knots[1] - knots[0])
    return so.Filterbank(bank.scale, knots, _UniformFormulaAnyway())


TIGHTNESS_CASES = {
    **{
        f"cosine {scale}": (lambda scale=scale: so.cosine_filterbank(24, 60, 7000, scale=scale), True)
        for scale in FREQUENCY_SCALES
    },
    **{
        f"cosine {scale} random centers": (
            lambda scale=scale: so.cosine_filterbank(centers=_random_centers(scale, 1), scale=scale),
            True,
        )
        for scale in FREQUENCY_SCALES
    },
    **{
        f"cosine width {width}": (lambda width=width: so.cosine_filterbank(24, 60, 7000, width=width), tight)
        for width, tight in [(0.75, False), (1.25, False), (1.5, True), (1.75, False), (2, True), (2.5, True)]
    },
    "cosine f_hi at Nyquist": (lambda: so.cosine_filterbank(24, 60, 4000), True),
    "cosine f_hi beyond Nyquist": (lambda: so.cosine_filterbank(24, 60, 30000), True),
    "cosine without edges": (lambda: so.cosine_filterbank(24, 60, 7000, edges=False), False),
    "cosine nudged, uniform formula": (_nudged, False),
    "gammatone": (lambda: so.gammatone_filterbank(24, 60, 7000), False),
    "gammatone zero phase": (lambda: so.gammatone_filterbank(24, 60, 7000, phase="zero"), False),
    "gammatone random centers": (lambda: so.gammatone_filterbank(centers=_random_centers("erb", 2)), False),
    "morlet": (lambda: so.morlet_filterbank(20, 60, 7000), False),
}
GRIDS = [(8000, 1001), (16000, 1600), (44100, 4411), (96000, 9600)]


@pytest.mark.parametrize("pad", ["auto", 0])
@pytest.mark.parametrize(("fs", "n"), GRIDS)
@pytest.mark.parametrize("case", TIGHTNESS_CASES)
def test_tightness_is_reported_only_when_true(case, fs, n, pad):
    make, tight = TIGHTNESS_CASES[case]
    bank = make()
    n_grid = n + 2 * bank._pad_samples(pad, fs, n)
    power_sum = np.sum(np.abs(bank.rfft_response(n_grid, fs)) ** 2, axis=1)  # s, computed here
    measured_tight = np.ptp(power_sum) <= 1e-12 * power_sum.max()
    # the grid sees the whole of s unless the bank lies above Nyquist
    if fs / 2 > bank.f_hi * 1.05 or tight:
        assert measured_tight == tight
    assert bank.is_tight(n, fs, pad=pad) == measured_tight
    lo, hi = bank.frame_bounds(n, fs, pad=pad)
    assert (lo == hi) == measured_tight
    x = so.Sound(np.random.default_rng(3).standard_normal((n, 2)), fs)
    coefs = bank.analyze(x, pad=pad)
    if lo <= 1e-12 * hi:
        with pytest.raises(ValueError, match="not a frame"):
            coefs.to_sound()
    else:
        np.testing.assert_allclose(coefs.to_sound().data, x.data, rtol=0, atol=1e-11)


def test_wide_cosines_are_tight_at_their_width():
    """s = width when twice the width is whole (C5); the bounds report it."""
    for width in (1.5, 2.0, 2.5, 3.0):
        bank = so.cosine_filterbank(24, 60, 7000, width=width)
        lo, hi = bank.frame_bounds(4000, 16000, pad=0)
        assert lo == hi and hi == pytest.approx(width, rel=1e-12)
        assert np.ptp(bank.frame_power(4000, 16000)) < 1e-12 * width


def test_the_tight_path_divides_by_a_constant_other_than_one():
    bank = so.cosine_filterbank(16, 100, 6000, width=2.0)
    x = so.gaussian_noise(0.2, 16000, rng=4)
    masked = bank.analyze(x, pad=0)
    masked = so.Subbands(
        masked._full * np.linspace(0, 1, masked._full.shape[1])[None, :, None], 16000, bank, 0
    )
    general = _GeneralPath(bank.scale, bank.knots, bank.filter_type, edges=bank.edges)
    np.testing.assert_allclose(bank.synthesize(masked).data, general.synthesize(masked).data, atol=1e-12)


@dataclass(frozen=True)
class _GeneralPath(so.Filterbank):
    def _tight_gain(self, n, fs):
        return None


# ---------------------------------------------------------- the one bank
def test_knots_are_stored_on_the_scale():
    """C4: centers live on the scale, so building from n_bands is exactly
    the linspace the formulas use, and cfs are reported in Hz."""
    bank = so.cosine_filterbank(30, 50, 8000)
    expected = np.linspace(so.freq_to_erb(50.0), so.freq_to_erb(8000.0), 32)
    assert np.array_equal(bank._knots, expected)
    assert bank.spacing == expected[1] - expected[0] and bank.n_bands == 30 and bank.n_filters == 32
    np.testing.assert_allclose([bank.f_lo, bank.f_hi], [50, 8000], rtol=1e-12)


@pytest.mark.parametrize(
    ("scale", "unit"),
    [("erb", "ERB"), ("octave", "oct"), ("cents", "cent"), ("mel", "mel"), ("linear", "Hz")],
)
def test_scales_space_the_centers(scale, unit):
    bank = so.cosine_filterbank(10, 100, 6000, scale=scale)
    assert bank.unit == unit and bank.scale is FREQUENCY_SCALES[scale]
    np.testing.assert_allclose(np.diff(bank.scale.to_scale(bank.band_cfs)), bank.spacing, rtol=1e-9)


def test_spacing_chooses_the_number_of_bands():
    bank = so.cosine_filterbank(f_lo=125, f_hi=8000, spacing=1 / 12, scale="octave")
    assert bank.n_bands == 71 and bank.spacing == pytest.approx(1 / 12)


def test_explicit_centers_follow_the_gaps():
    centers = [100, 250, 300, 900, 4000]
    bank = so.cosine_filterbank(centers=centers)
    assert bank.spacing is None and bank.n_bands == 3
    np.testing.assert_allclose(bank.cfs, centers, rtol=1e-12)
    H = bank.response(np.array(centers, float))
    np.testing.assert_allclose(
        H, np.eye(5), atol=1e-12
    )  # each filter peaks at its center, zero at the others


def test_factory_arguments_are_checked():
    with pytest.raises(ValueError, match="scale"):
        so.cosine_filterbank(scale="bark")
    with pytest.raises(ValueError, match="not both"):
        so.cosine_filterbank(10, spacing=0.5)
    with pytest.raises(ValueError, match="not both"):
        so.cosine_filterbank(10, centers=[100, 200, 300])
    with pytest.raises(ValueError, match="f_lo < f_hi"):
        so.cosine_filterbank(10, 4000, 100)
    with pytest.raises(ValueError, match="increasing"):
        so.cosine_filterbank(centers=[100, 300, 200])
    with pytest.raises(ValueError, match="equally spaced"):
        so.cosine_filterbank(centers=[100, 250, 300, 900], width=2).response(np.array([500.0]))
    with pytest.raises(ValueError, match="width"):
        so.cosine_filterbank(width=0)


def test_ripples_need_the_octave_scale():
    with pytest.raises(TypeError, match="octave scale"):
        so.Ripple(4, 1).render(so.cosine_filterbank(10, 250, 4000), 0.1, 1000)


# Extremes and edges (Cho, 2026-10-03): the same guarantees at the limits of
# every argument and of the grid. Each case says what the mathematics gives on
# an ordinary grid (16 kHz, 1600 samples): "tight", "frame" (exact through the
# canonical dual) or "not a frame" (synthesis refuses).
EXTREME_CASES = {
    "cosine, 1 band": (lambda: so.cosine_filterbank(1, 100, 4000), "tight"),
    "cosine, 400 bands": (lambda: so.cosine_filterbank(400, 50, 7000), "tight"),
    "cosine, bands narrower than a bin": (lambda: so.cosine_filterbank(20, 1000, 1010), "tight"),
    "cosine, f_lo 1 Hz": (lambda: so.cosine_filterbank(20, 1, 7000), "tight"),
    "cosine linear, f_lo 0 Hz": (lambda: so.cosine_filterbank(20, 0, 7000, scale="linear"), "tight"),
    "cosine mel, f_lo 0 Hz": (lambda: so.cosine_filterbank(20, 0, 7000, scale="mel"), "tight"),
    "cosine ERB, f_lo 0 Hz": (lambda: so.cosine_filterbank(20, 0, 7000), "tight"),
    "cosine, f_hi 10 times Nyquist": (lambda: so.cosine_filterbank(20, 50, 80000), "tight"),
    "cosine, every filter above Nyquist": (lambda: so.cosine_filterbank(20, 9000, 12000), "tight"),
    "cosine, every filter below the first bin": (
        lambda: so.cosine_filterbank(5, 0.01, 0.5, scale="linear"),
        "tight",
    ),
    "cosine, bins exactly on the knots": (lambda: so.cosine_filterbank(8, 100, 400, scale="linear"), "tight"),
    "cosine, centers 1e-9 Hz apart": (
        lambda: so.cosine_filterbank(centers=[100, 1000, 1000 + 1e-9, 1000 + 2e-9, 5000]),
        "tight",
    ),
    "cosine, one huge gap": (lambda: so.cosine_filterbank(centers=[20, 21, 7900, 7901]), "tight"),
    "cosine, width 10": (lambda: so.cosine_filterbank(40, 50, 7000, width=10), "tight"),
    "cosine, width 10 on 3 bands": (lambda: so.cosine_filterbank(3, 50, 7000, width=10), "tight"),
    "cosine, width 1 + 1e-9": (lambda: so.cosine_filterbank(24, 60, 7000, width=1 + 1e-9), "frame"),
    "cosine, width 1.4999999": (lambda: so.cosine_filterbank(24, 60, 7000, width=1.4999999), "frame"),
    "cosine, width 0.5 (no overlap)": (lambda: so.cosine_filterbank(20, 50, 7000, width=0.5), "not a frame"),
    "cosine, width 0.01": (lambda: so.cosine_filterbank(20, 50, 7000, width=0.01), "not a frame"),
    "gammatone, 1 band": (lambda: so.gammatone_filterbank(1, 100, 4000), "frame"),
    "gammatone, order 1": (lambda: so.gammatone_filterbank(20, 50, 7000, order=1), "frame"),
    "gammatone, order 20": (lambda: so.gammatone_filterbank(20, 50, 7000, order=20), "frame"),
    "gammatone, 100 times too narrow": (
        lambda: so.gammatone_filterbank(20, 50, 7000, bandwidth_factor=0.01),
        "not a frame",
    ),
    "gammatone, 50 times too wide": (
        lambda: so.gammatone_filterbank(20, 50, 7000, bandwidth_factor=50),
        "frame",
    ),
    "gammatone, every filter above Nyquist": (lambda: so.gammatone_filterbank(20, 9000, 12000), "frame"),
    "morlet, half a cycle": (lambda: so.morlet_filterbank(20, 50, 7000, cycles=0.5), "frame"),
    "morlet, 100 cycles": (lambda: so.morlet_filterbank(20, 50, 7000, cycles=100), "not a frame"),
    "morlet, edge width 0.01": (lambda: so.morlet_filterbank(20, 50, 7000, edge_width=0.01), "frame"),
}
EXTREME_GRIDS = [(16000, 1), (16000, 2), (16000, 3), (16000, 64), (1000, 500), (384000, 3840), (16000, 1600)]


@pytest.mark.parametrize("case", EXTREME_CASES)
def test_extremes_on_an_ordinary_grid(case):
    make, expected = EXTREME_CASES[case]
    bank = make()
    lo, hi = bank.frame_bounds(1600, 16000, pad=0)
    outcome = "tight" if bank.is_tight(1600, 16000, pad=0) else "frame" if lo > 1e-12 * hi else "not a frame"
    assert outcome == expected


@pytest.mark.parametrize(("fs", "n"), EXTREME_GRIDS)
@pytest.mark.parametrize("case", EXTREME_CASES)
def test_extremes_never_lose_part_of_the_sound(case, fs, n):
    """On every grid, down to a single sample: tight only when s is constant
    there, and synthesis either exact or refused."""
    make, expected = EXTREME_CASES[case]
    bank = make()
    power_sum = np.sum(np.abs(bank.rfft_response(n, fs)) ** 2, axis=1)  # s, computed here
    measured_tight = power_sum.max() > 0 and np.ptp(power_sum) <= 1e-12 * power_sum.max()
    assert bank.is_tight(n, fs, pad=0) == measured_tight
    if expected == "tight":
        assert measured_tight  # a tight bank is tight on every grid
    x = so.Sound(np.random.default_rng(5).standard_normal((n, 1)), fs)
    coefs = bank.analyze(x, pad=0)
    lo, hi = bank.frame_bounds(n, fs, pad=0)
    if not lo > 1e-12 * hi:
        with pytest.raises(ValueError, match="not a frame"):
            coefs.to_sound()
    else:
        np.testing.assert_allclose(coefs.to_sound().data, x.data, rtol=0, atol=1e-11)


def test_octave_scale_has_no_zero():
    with pytest.raises(ValueError, match="not on the octave scale"):
        so.cosine_filterbank(20, 0, 7000, scale="octave")
    with pytest.raises(ValueError, match="octave scale"):
        so.morlet_filterbank(centers=[0, 100, 200])


# Audit sitting 8 (Cho, 2026-10-03): tests for the deliberate breaks that no
# test noticed, and for the decisions taken there.
INDEPENDENT_SCALES = {
    "erb": lambda f: 9.265 * np.log(1 + f / (24.7 * 9.265)),  # exact integral of 1 / ERB(f)
    "octave": lambda f: np.log(f) / np.log(2),
    "mel": lambda f: 2595 * np.log10(1 + f / 700),
    "linear": lambda f: f,
}


@pytest.mark.parametrize("scale", INDEPENDENT_SCALES)
def test_centers_are_equally_spaced_on_the_published_scale(scale):
    """The scale's own conversion can't check itself: equal steps on the
    formulas written out here (Glasberg & Moore 1990, HTK mel)."""
    bank = so.cosine_filterbank(10, 100, 6000, scale=scale)
    position = INDEPENDENT_SCALES[scale](bank.band_cfs)
    expected = np.linspace(INDEPENDENT_SCALES[scale](100.0), INDEPENDENT_SCALES[scale](6000.0), 12)[1:-1]
    np.testing.assert_allclose(position, expected, rtol=1e-9)


def test_scale_names_ignore_case():
    assert so.cosine_filterbank(scale="ERB").scale is FREQUENCY_SCALES["erb"]
    assert so.gammatone_filterbank(scale="Octave").scale is FREQUENCY_SCALES["octave"]


def test_a_cents_bank_is_the_octave_bank_in_other_units():
    """Spacing 100 cents is one band per equal-tempered semitone, the bank
    built on octaves with spacing 1/12."""
    cents = so.cosine_filterbank(f_lo=110, f_hi=1760, spacing=100, scale="cents")
    octaves = so.cosine_filterbank(f_lo=110, f_hi=1760, spacing=1 / 12, scale="octave")
    np.testing.assert_allclose(cents.cfs, octaves.cfs, rtol=1e-12)
    np.testing.assert_allclose(cents.band_cfs[1:] / cents.band_cfs[:-1], 2 ** (1 / 12), rtol=1e-12)
    assert cents.unit == "cent"


def test_a_scale_must_invert_itself():
    with pytest.raises(ValueError, match="invert each other"):
        FrequencyScale("erb", "ERB", so.freq_to_erb, so.mel_to_freq)
    with pytest.raises(ValueError, match="does not give f back"):
        FrequencyScale("log", "nepers", np.log, np.exp2)
    # a scale undefined at some test frequencies is checked where it is defined
    FrequencyScale("above 500 Hz", "oct", lambda f: np.log2(f - 500), lambda x: np.exp2(x) + 500)


def test_default_edge_centers_stay_at_or_above_zero():
    """Wide default edges would put the lowpass's nominal center below 0 Hz."""
    assert so.gammatone_filterbank(4, 20, 4000, edge_width=3).cfs[0] == 0.0
    assert so.gammatone_filterbank(4, 20, 4000).cfs[0] == pytest.approx(20)


@pytest.mark.parametrize(
    "bank",
    [so.gammatone_filterbank(10, 100, 6000), so.morlet_filterbank(8, 100, 6000)],
    ids=["gammatone", "morlet"],
)
def test_s_floor_is_the_minimum_between_the_outer_centers(bank):
    knots = bank._knots[1:-1]
    dense = bank.scale.from_scale(np.linspace(knots[0], knots[-1], 200_001))
    minimum = np.min(np.sum(np.abs(bank.filter_type.band_response(bank, dense)) ** 2, axis=1))
    assert minimum <= bank.s_floor <= minimum * (1 + 2e-3)


def test_wide_cosine_edges_hold_s_flat_far_outside_the_bank():
    """Width 2: s stays 2 from 0 Hz up to f_lo and from f_hi far beyond,
    where only the edge filters reach."""
    bank = so.cosine_filterbank(10, 500, 6000, width=2)
    for freqs in (np.linspace(0, 500, 101), np.linspace(6000, 40000, 101)):
        np.testing.assert_allclose(np.sum(bank.response(freqs) ** 2, axis=1), 2, rtol=1e-12)


def test_spacing_needs_exactly_even_knots():
    knots = np.linspace(2, 30, 12)
    knots[5] *= 1 + 1e-12
    assert so.Filterbank("erb", knots, Cosine()).spacing is None
    assert so.Filterbank("erb", np.linspace(2, 30, 12), Cosine()).spacing == pytest.approx(28 / 11)


def test_ringing_counts_up_to_the_last_sample_above_the_level():
    bank, fs = so.cosine_filterbank(10, 100, 6000), 16000.0
    ringing = bank.ringing(fs)
    n_grid = 1 << 16  # 4 s at 16 kHz
    impulse = np.abs(np.fft.irfft(bank.rfft_response(n_grid, fs), n=n_grid, axis=0)[: n_grid // 2])
    level = impulse.max(axis=0) * 10 ** (-60 / 20)
    assert np.any(impulse[ringing - 1] > level)
    assert np.all(impulse[ringing:] <= level)


def test_numeric_padding_is_rounded_to_samples():
    bank = so.cosine_filterbank(8)
    assert bank.analyze(so.gaussian_noise(0.1, 1000, rng=0), pad=0.0106).pad == 11


def test_peak_delay_is_refined_between_samples():
    """Against the same measurement on a grid 16 times finer: well within
    one sample (15.6 us at 64 kHz)."""
    bank = so.gammatone_filterbank(4, 100, 4000)
    fs, n_grid = 1024000.0, 1 << 21
    envelope = np.abs(hilbert(np.fft.irfft(bank.rfft_response(n_grid, fs), n=n_grid, axis=0), axis=0))
    peaks = np.argmax(envelope, axis=0)
    fine = []
    for band, peak in enumerate(peaks):
        before, at, after = envelope[[peak - 1, peak, peak + 1], band]
        fine.append((peak + 0.5 * (before - after) / (before - 2 * at + after)) / fs)
    np.testing.assert_allclose(bank.envelope_peak_delay[1:-1], fine[1:-1], atol=0.5e-6)


def test_factory_defaults():
    assert so.cosine_filterbank().n_bands == 30 and so.cosine_filterbank().scale.name == "erb"
    assert so.gammatone_filterbank().n_bands == 30 and so.gammatone_filterbank().scale.name == "erb"
    assert so.morlet_filterbank().n_bands == 30 and so.morlet_filterbank().scale.name == "octave"


def test_subbands_refuse_padding_of_half_the_data_or_more():
    bank = so.cosine_filterbank(4)
    with pytest.raises(ValueError, match="padding"):
        so.Subbands(np.zeros((10, bank.n_filters, 1)), 16000, bank, pad=5)
    so.Subbands(np.zeros((11, bank.n_filters, 1)), 16000, bank, pad=5)


def test_is_tight_measures_on_the_padded_grid():
    """A one-sample signal has one rfft bin, where any s is constant; padded,
    the grid is long enough to show a gammatone bank's ripple."""
    bank = so.gammatone_filterbank(10, 100, 6000)
    assert bank.is_tight(1, 16000, pad=0)
    assert not bank.is_tight(1, 16000)


def test_subbands_take_a_number_or_one_gain_per_band():
    bank = so.cosine_filterbank(6, 100, 6000)
    noise = so.gaussian_noise(0.2, 16000, rng=1)
    bands = bank.analyze(noise)
    np.testing.assert_allclose((bands * 0.5).to_sound().data, 0.5 * noise.data, atol=1e-12)
    np.testing.assert_allclose((2 * bands).data, 2 * bands.data)
    # one band at a time, through the bank's synthesis, the parts add up to the sound
    parts = [(bands * np.eye(len(bands))[band]).to_sound().data for band in range(len(bands))]
    np.testing.assert_allclose(np.sum(parts, axis=0), noise.data, atol=1e-12)
    assert np.std(parts[3]) < 0.5 * np.std(noise.data)
    with pytest.raises(ValueError, match="one gain per band"):
        bands * np.ones(len(bands) + 1)
    with pytest.raises(TypeError):
        bands * "loud"


def test_arguments_of_banks_and_filter_types_are_checked():
    with pytest.raises(ValueError, match="positive"):
        so.gammatone_filterbank(order=0)
    with pytest.raises(ValueError, match="phase"):
        so.gammatone_filterbank(phase="minimum")
    with pytest.raises(ValueError, match="positive"):
        so.morlet_filterbank(cycles=0)
    with pytest.raises(ValueError, match="spacing must be positive"):
        so.cosine_filterbank(spacing=0)
    with pytest.raises(ValueError, match="three knots"):
        so.Filterbank("erb", [1.0, 2.0], Cosine())
    with pytest.raises(NotImplementedError):
        FilterType().band_response(so.cosine_filterbank(4), np.array([100.0]))


def test_cosine_band_response_is_the_bandpass_part():
    bank = so.cosine_filterbank(6, 100, 6000)
    freqs = np.linspace(0, 8000, 801)
    np.testing.assert_array_equal(Cosine().band_response(bank, freqs), bank.response(freqs)[:, 1:-1])


def test_subbands_repr():
    bands = so.cosine_filterbank(6, 100, 6000).analyze(so.gaussian_noise(0.5, 16000, rng=0))
    assert repr(bands) == "Subbands(8 bands, 0.500 s, 16000 Hz, 1 ch)"
