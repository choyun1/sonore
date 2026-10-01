"""Cosine, gammatone and Morlet filterbanks, and subbands."""

import numpy as np
import pytest
from helpers import FS
from scipy.signal import hilbert

import sonore as so


class TestFilterbank:
    def test_power_complementary(self):
        fb = so.ERBFilterbank(30, 50, 8000)
        H = fb.response(np.linspace(0, 22050, 5000))
        np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)

    def test_perfect_reconstruction(self):
        g = so.gaussian_noise(0.5, FS, rng=0)
        np.testing.assert_allclose(so.subbands(g).synthesize().data, g.data, atol=1e-10)

    def test_vocoder_runs_and_keeps_level(self):
        x = so.harmonic_complex(0.5, FS, 150, np.arange(1, 20))
        v = so.noise_vocode(x, 8, rng=0)
        assert len(v) == len(x) and v.rms == pytest.approx(x.rms)


def test_octave_filterbank_reconstructs():
    fb = so.OctaveFilterbank.per_octave(12, 125, 6000)
    H = fb.response(np.linspace(0, FS / 2, 5000))
    np.testing.assert_allclose((H**2).sum(axis=1), 1, atol=1e-12)
    g = so.gaussian_noise(0.5, 16000, rng=0)
    np.testing.assert_allclose(fb.analyze(g).synthesize().data, g.data, atol=1e-10)


def test_subband_plot_labels_and_scale():
    import matplotlib

    matplotlib.use("Agg")
    sb = so.subbands(so.exponential_chirp(0.2, FS, 100, 6000), n_bands=6, f_lo=100, f_hi=6000)
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
# and the gammatone phase options (derivations in docs/design/frames.md, step 2).


def _gammatone_formula(f, fc):
    """Closed-form FT of t^3 exp(-2 pi b t) cos(2 pi fc t), t >= 0, unit gain at fc."""
    b = 1.019 * 24.7 * (4.37e-3 * fc + 1)
    k = 6 / (2 * np.pi) ** 4 / 2

    def H(f):
        return k * ((b + 1j * (f - fc)) ** -4 + (b + 1j * (f + fc)) ** -4)

    return H(f) / abs(H(fc))


class TestGammatone:
    fb = so.GammatoneFilterbank(2, 100, 4000, edges=False)

    def test_response_is_the_closed_form(self):
        f = np.linspace(0, 8000, 801)
        H = self.fb.response(f)
        for k, fc in enumerate(self.fb.cfs):
            np.testing.assert_allclose(H[:, k], _gammatone_formula(f, fc), rtol=1e-12, atol=1e-15)
        zero = so.GammatoneFilterbank(2, 100, 4000, edges=False, phase="zero")
        assert np.array_equal(zero.response(f), np.abs(H))

    def test_group_delay_and_envelope_peak(self):
        """Group delay 4/(2 pi b) at cf, envelope peak 3/(2 pi b), within 1%."""
        b, cfs, df = self.fb.b, self.fb.cfs, 1e-3
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
        fb = so.GammatoneFilterbank(30, 50, 7500, phase=phase)
        x = so.Sound(np.random.default_rng(0).standard_normal((n, 2)), 16000)
        for pad in ("auto", 0):
            np.testing.assert_allclose(fb.analyze(x, pad=pad).synthesize().data, x.data, rtol=0, atol=1e-12)

    def test_bank_attributes(self):
        fb = so.GammatoneFilterbank(40, 50, 7000)
        assert fb.n_filters == 42 and fb.unit == "ERB" and not fb.tight
        knots = np.linspace(so.freq_to_erb(50), so.freq_to_erb(7000), 40)
        assert np.allclose(fb.cfs[1:-1], so.erb_to_freq(knots))
        assert np.allclose(np.diff(so.freq_to_erb(fb.cfs[1:-1])), fb.spacing)
        assert fb.cfs[0] < 50 and fb.cfs[-1] > 7000
        assert so.GammatoneFilterbank(40, 50, 7000, edges=False).n_filters == 40
        with pytest.raises(ValueError, match="phase"):
            so.GammatoneFilterbank(phase="minimum")
        with pytest.raises(ValueError, match="n_bands"):
            so.GammatoneFilterbank(1)


class TestMorlet:
    def test_response(self):
        fb = so.MorletFilterbank(3, 250, 1000, cycles=5, edges=False)
        f = np.linspace(0, 4000, 4001)
        H = fb.response(f)
        assert H.dtype == float and np.all(H[0] == 0)  # DC correction
        assert np.allclose(H[[250, 500, 1000], [0, 1, 2]], 1.0)
        assert np.all(H >= 0) and np.array_equal(np.argmax(H, axis=0), [250, 500, 1000])
        assert fb.unit == "oct" and fb.spacing == pytest.approx(1.0)

    def test_bare_bank_is_not_a_frame(self):
        """Without edges A = 0 (DC); analysis works, synthesis refuses."""
        fb = so.MorletFilterbank(28, 50, 7000, edges=False)
        assert fb.frame_bounds(1000, 16000)[0] == 0
        sb = fb.analyze(so.gaussian_noise(0.05, 16000, rng=0))
        assert sb.envelopes().without_edges() is not None
        with pytest.raises(ValueError, match="not a frame"):
            sb.synthesize()

    def test_exact_with_edges(self):
        x = so.gaussian_noise(0.25, 16000, n_channels=2, rng=1)
        fb = so.MorletFilterbank(28, 50, 7000)
        np.testing.assert_allclose(fb.analyze(x).synthesize().data, x.data, rtol=0, atol=1e-12)


def _edge_banks(width):
    """The two banks of the edge-coverage table in docs/design/frames.md, step 2
    (gammatone 1/ERB, Morlet 4/oct, 50-7000 Hz)."""
    e = so.freq_to_erb(50.0)
    n = len(np.arange(e, so.freq_to_erb(7000.0), 1.0))
    return (
        so.GammatoneFilterbank(n, 50, float(so.erb_to_freq(e + n - 1)), edge_width=width),
        so.MorletFilterbank(29, 50, 50 * 2**7.0, edge_width=width),
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
    fb = so.GammatoneFilterbank(20, 200, 4000)
    f = np.array([0.0, fb.cfs[0], fb.band_cfs[0], fb.band_cfs[-1], fb.cfs[-1], 20000.0])
    H = fb.response(f)
    g = np.sqrt(fb.s_floor)
    assert np.allclose(H[:, 0], [g, g, 0, 0, 0, 0]) and np.allclose(H[:, -1], [0, 0, 0, 0, g, g])
    assert so.GammatoneFilterbank(20, 200, 4000, edge_width=2).cfs[0] < fb.cfs[0]


def test_new_banks_feed_the_cochleagram_tools():
    x = so.gaussian_noise(0.25, 16000, rng=2)
    for fb in (so.GammatoneFilterbank(24, 80, 6000), so.MorletFilterbank(20, 80, 6000)):
        env = fb.analyze(x).envelopes()
        assert env.data.shape == (len(x), fb.n_filters, 1)
        env.modulation_spectrum()  # needs spacing and unit
    bare = so.GammatoneFilterbank(24, 80, 6000, edges=False).analyze(x).envelopes()
    assert bare.without_edges() is bare  # no edge bands to drop
