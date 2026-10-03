"""Wrap-around (circularity) regression tests.

FFT-based operations are circular: energy near one end of a signal can wrap
around to the other. sonore pads by default so this doesn't happen, and keeps
circular behaviour available (``pad=0``) for periodic signals and texture
synthesis. Each probe puts an event near the start of 1 s of silence and
measures what appears in the last 5% of the output. With linear edge handling
there is nothing there; with wrap-around there is. Thresholds sit about 10 dB
above the values measured when the padding was introduced.
"""

import numpy as np
import pytest
from helpers import FAST, FAST_HI, FS

import sonore as so

N = FS
ERB = so.cosine_filterbank(30, 50, 8000)
OCTAVE = so.cosine_filterbank(f_lo=125, f_hi=8000, spacing=1 / 12, scale="octave")


def click(at=0.02):
    x = np.zeros(N)
    x[int(at * FS)] = 1.0
    return so.Sound(x, FS)


def burst(at=0.02):
    x = np.zeros(N)
    i, n = int(at * FS), int(0.05 * FS)
    x[i : i + n] = np.random.default_rng(0).standard_normal(n)
    return so.Sound(x, FS)


def far_end_db(y):
    """Largest |y| in the last 5% of the output, in dB re the largest |y| overall."""
    y = np.abs(np.asarray(y, float)).reshape(len(y), -1)
    tail = y[-max(1, len(y) // 20) :]
    return 20 * np.log10(max(tail.max(), 1e-300) / y.max())


@pytest.mark.parametrize(
    "name, op, limit",
    [
        ("ERB analyze", lambda: ERB.analyze(click()).data, -110),
        ("octave analyze", lambda: OCTAVE.analyze(click()).data, -110),
        ("fractional delay", lambda: click().delay(10.3 / FS).data, -90),
        ("fractional ITD", lambda: so.apply_itd_ild(click(), itd=123.4e-6).data, -90),
        ("Sound.envelope", lambda: burst().envelope().data, -70),
        ("band envelopes", lambda: ERB.analyze(burst()).envelopes().data[:, 1:-1], -76),
        ("noise vocoder", lambda: so.noise_vocode(burst(), 8, rng=0).data, -75),
    ],
)
def test_no_wraparound_by_default(name, op, limit):
    assert far_end_db(op()) < limit, name


def test_circular_mode_is_still_circular():
    # pad=0 is the intentional exception (texture synthesis): wrap-around remains
    assert far_end_db(ERB.analyze(click(), pad=0).data) > -90


def test_circular_mode_is_exact_for_periodic_signals():
    # A periodic signal analysed circularly == the middle period of the same
    # signal tiled and analysed with padding.
    x = so.gaussian_noise(0.5, FS, rng=0)  # FFT-generated noise is exactly periodic
    circular = ERB.analyze(x, pad=0).data
    tiled = so.Sound(np.tile(x.data, (9, 1)), FS)
    middle = ERB.analyze(tiled).data[4 * len(x) : 5 * len(x)]
    # equal to within 60 dB of each band's level (the residual comes from the
    # tiled signal's own abrupt outer edges, 2 s away, not the circular analysis)
    band_rms = np.sqrt(np.mean(circular**2, axis=0, keepdims=True))
    assert np.all(np.abs(circular - middle) < 1e-3 * band_rms)


def test_padding_is_hidden():
    s = burst()
    sb = ERB.analyze(s)
    assert sb.pad > 0 and ERB.analyze(s, pad=0).pad == 0
    assert sb.data.shape[0] == len(s) == sb.n_samples
    assert sb.envelopes().data.shape[0] == len(s)
    assert len(sb.to_sound()) == len(s)
    np.testing.assert_allclose(sb.to_sound().data, s.data, atol=1e-10)


def test_unpadded_envelopes_combine_with_padded_bands():
    # a rendered pattern (no padding) times the bands of a padded analysis
    fb = so.cosine_filterbank(f_lo=250, f_hi=4000, spacing=1 / 6, scale="octave")
    carrier = burst()
    env = so.Ripple(4, 1, depth=0).render(fb, carrier.duration, FS)
    out = (env * fb.analyze(carrier).tfs()).sum()
    assert len(out) == len(carrier)


def test_modulation_spectrum_with_fractional_cycles():
    # 8.5 cycles of 8 Hz AM: without a taper, leakage from the periodic
    # extension let the static spectral tilt win and misplaced the peak
    x = so.amplitude_modulate(so.gaussian_noise(1.0625, FAST, rng=1), 8, depth=1)
    rate, density = so.ModulationSpectrum.octave(x, f_hi=FAST_HI).peak()
    assert rate == pytest.approx(8, abs=1) and density == 0


def test_auto_padding_covers_the_ringing_at_a_fast_fft_length():
    from sonore.core.fft import _is_fast

    s = burst()
    for fb in (ERB, so.gammatone_filterbank(20, 80.0, 6000.0)):
        sb = fb.analyze(s)
        n = sb._full.shape[0]
        assert sb.pad >= fb.ringing(FS) and _is_fast(n)
        assert n - len(s) - 2 * fb.ringing(FS) < 0.05 * len(s)
