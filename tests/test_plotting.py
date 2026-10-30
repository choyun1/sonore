import gc

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import sonore as so
from sonore.plotting import plot_envelopes, plot_lissajous


def test_plot_lissajous_draws_left_against_right():
    fs = 8000
    t = np.arange(fs // 10) / fs
    sound = so.Sound(np.stack([np.cos(2 * np.pi * 200 * t), np.cos(2 * np.pi * 300 * t)], axis=1), fs)
    ax = plot_lissajous(sound, duration=0.01, start=0.02)
    x, y = ax.lines[0].get_data()
    assert len(x) == 80
    np.testing.assert_array_equal(x, sound.data[160:240, 0])
    np.testing.assert_array_equal(y, sound.data[160:240, 1])


def test_plot_lissajous_needs_two_channels():
    with pytest.raises(ValueError, match="two-channel"):
        plot_lissajous(so.pure_tone(0.1, 8000, 200))


def test_plot_envelopes_shares_a_color_scale():
    bank = so.cosine_filterbank(8, 100, 3000)
    noise = so.gaussian_noise(0.2, 8000, rng=0)
    loud, quiet = bank.analyze(noise).envelopes(), bank.analyze(noise * 0.1).envelopes()
    top = float(loud.db[:, 1:-1].max())  # the edge bands are not drawn
    own = plot_envelopes(quiet, colorbar=False).collections[0].get_clim()
    shared = plot_envelopes(quiet, db_max=top, colorbar=False).collections[0].get_clim()
    assert own[1] == pytest.approx(top - 20, abs=1e-6)  # its own maximum, 20 dB lower
    assert shared == pytest.approx((top - 40, top))


# What each plot draws, read back from the figure.

FS = 16000


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _image(ax, index=0):
    """The values and cell edges of the ``index``-th image (QuadMesh) on ``ax``."""
    mesh = ax.collections[index]
    return np.ma.filled(mesh.get_array(), np.nan), mesh.get_coordinates()


@pytest.fixture(scope="module")
def stereo():
    """Noise in the left ear, a 1 kHz tone in the right: the channels differ everywhere."""
    noise = so.gaussian_noise(0.5, FS, rng=0).data[:, 0]
    tone = so.pure_tone(0.5, FS, 1000).data[:, 0]
    return so.Sound(np.stack([noise, tone], axis=1), FS)


def test_plot_stft_draws_the_channel_asked_in_hz_without_the_edge_windows(stereo):
    stft = so.STFT(stereo, win_dur=0.02)
    values, coords = _image(so.plotting.plot_stft(stft, channel=1, colorbar=False))
    inner = so.plotting._interior(stft)
    np.testing.assert_array_equal(values, stft.db[1][:, inner])
    assert values.shape[1] < len(stft.t)  # the zero-padded windows at either end are left out
    assert coords[..., 1].max() == pytest.approx(FS / 2, rel=0.01)  # up to 8 kHz: drawn in Hz
    untrimmed, _ = _image(so.plotting.plot_stft(stft, colorbar=False, trim_edges=False))
    assert untrimmed.shape[1] == len(stft.t)


def test_plot_spectrum_relative_or_absolute(stereo):
    spectrum = so.long_term_spectrum(stereo.mono())
    absolute = so.plotting.plot_spectrum(spectrum, relative=False).lines[0].get_ydata()
    np.testing.assert_array_equal(absolute, spectrum.level[1:])
    relative = so.plotting.plot_spectrum(spectrum).lines[0].get_ydata()
    assert relative.max() == pytest.approx(0.0)


def test_plot_mask_draws_the_channel_asked(stereo):
    stft = so.STFT(stereo, win_dur=0.02)
    mask = so.ideal_binary_mask(stft, stft * 0.5)
    values, _ = _image(so.plotting.plot_mask(mask, channel=1))
    np.testing.assert_array_equal(values, mask.values[1])


@pytest.fixture(scope="module")
def modulation():
    t = np.arange(FS) / FS
    noise = so.gaussian_noise(1.0, FS, rng=0) * so.Sound(1 + 0.9 * np.sin(2 * np.pi * 8 * t), FS)
    envelopes = so.cosine_filterbank(8, 200, 4000).analyze(noise).envelopes()
    return so.ModulationSpectrogram(envelopes, f_lo=2.0, f_hi=32.0)


def test_plot_modulation_spectrogram_is_depth_in_db_with_invalid_cells_gray(modulation):
    pooled = so.plotting.plot_modulation_spectrogram(modulation, colorbar=False)
    values, _ = _image(pooled)
    with np.errstate(divide="ignore"):
        expected = 20 * np.log10(modulation.pooled_depth()[0])
    np.testing.assert_allclose(values, np.where(np.isfinite(expected), expected, np.nan), equal_nan=True)
    assert pooled.collections[0].get_clim() == pytest.approx((np.nanmax(values) - 30, np.nanmax(values)))
    rate = so.plotting.plot_modulation_spectrogram(modulation, rate=8.0, colorbar=False)
    rate_index = int(np.argmin(np.abs(np.log(modulation.fm / 8.0))))
    values, _ = _image(rate)
    assert np.array_equal(np.isnan(values), ~modulation.valid[:, rate_index] | ~np.isfinite(values))
    np.testing.assert_allclose(
        values[modulation.valid[:, rate_index]],
        20 * np.log10(modulation.depth[0, :, rate_index][modulation.valid[:, rate_index]]),
    )
    band = so.plotting.plot_modulation_spectrogram(modulation, band=1000.0, colorbar=False)
    band_index = int(np.argmin(np.abs(modulation.f - 1000.0)))
    assert f"{modulation.f[band_index]:.0f} Hz" in band.get_title()
    values, _ = _image(band)
    valid = modulation.valid[band_index]
    np.testing.assert_allclose(values[valid], 20 * np.log10(modulation.depth[0, band_index][valid]))


def test_modulation_slices_put_the_cursor_at_the_nearest_window(modulation):
    fig = so.plotting.plot_modulation_slices(modulation, 0.5)
    nearest = modulation.t[np.argmin(np.abs(modulation.t - 0.5))]
    assert fig.axes[0].lines[0].get_xdata()[0] == pytest.approx(nearest)
    assert f"{nearest:.2f} s" in fig.axes[2].get_title()


@pytest.mark.filterwarnings("ignore:Animation was deleted")
def test_the_animation_shows_the_window_at_each_frame_time(modulation):
    anim = so.plotting.animate_modulation_spectrogram(modulation, fps=10)
    db = so.plotting._depth_db(modulation.depth[0], modulation.valid)
    frame = 7  # 0.7 s
    anim._func(frame)  # matplotlib's update function for that frame
    shown = np.ma.filled(anim._fig.axes[0].collections[0].get_array(), np.nan).ravel()
    np.testing.assert_array_equal(shown, db[:, :, round(0.7 / modulation.hop)].ravel())
    del anim  # unrendered: let it go while the warning filter applies
    gc.collect()


def test_plot_envelopes_peak_alignment_moves_each_band_earlier_by_its_delay():
    bank = so.gammatone_filterbank(8, 200, 4000)
    env = bank.analyze(so.gaussian_noise(0.2, FS, rng=0)).envelopes()
    _, coords = _image(so.plotting.plot_envelopes(env, align="peak", colorbar=False))
    delays = bank.envelope_peak_delay[1:-1]
    unaligned = so.plotting.plot_envelopes(env, colorbar=False)
    _, plain = _image(unaligned)
    shift = coords[:-1, :, 0].mean(axis=1) - plain[:-1, :, 0].mean(axis=1)
    np.testing.assert_allclose(shift, -delays, atol=1e-3)


def test_plot_decimated_subbands_cells(stereo):
    frame = so.DecimatedFilterbank(so.cosine_filterbank(10, 200, 4000))
    coefs = frame.analyze(stereo.mono())
    assert coefs.n_padded > coefs.n_samples
    ax = so.plotting.plot_decimated_subbands(coefs, colorbar=False)
    log_cfs = np.log(np.maximum(coefs.cfs, 1.0))
    for row, k in zip(ax.collections, range(1, len(coefs) - 1), strict=True):
        coords = np.asarray(row.get_coordinates())
        widths = np.diff(coords[0, :, 0])
        np.testing.assert_allclose(widths, coefs.n_padded / len(coefs.times[k]) / coefs.fs)
        bottom, top = np.log(coords[0, 0, 1]), np.log(coords[1, 0, 1])
        assert bottom == pytest.approx((log_cfs[k - 1] + log_cfs[k]) / 2)
        assert top == pytest.approx((log_cfs[k] + log_cfs[k + 1]) / 2)


def test_plot_cepstrum_shows_positive_values_in_the_quefrency_range(stereo):
    cep = so.Cepstrum(so.STFT(stereo.mono(), win_dur=0.04))
    values, coords = _image(so.plotting.plot_cepstrum(cep, q_range=(2e-3, 10e-3)))
    in_range = (cep.q >= 2e-3) & (cep.q <= 10e-3)
    np.testing.assert_array_equal(values, np.maximum(cep.data[0][in_range], 0))
    assert values.min() == 0


def test_plot_mfcc_leaves_out_c0(stereo):
    mfcc = so.MFCC(stereo.mono())
    values, _ = _image(so.plotting.plot_mfcc(mfcc))
    np.testing.assert_array_equal(values, mfcc.data[0][1:])


def test_plot_f0_track_breaks_where_unvoiced():
    voice = so.harmonic_complex(0.3, FS, 150, harmonics=range(1, 11))
    sound = so.Sound(np.concatenate([voice.data[:, 0], np.zeros(FS // 5)]), FS)
    track = so.f0_track(sound)
    drawn = so.plotting.plot_f0_track(track).lines[0].get_ydata()
    assert np.array_equal(np.isnan(drawn), ~track.voiced[0])
    assert np.isnan(drawn).any() and not np.isnan(drawn).all()


def test_plot_lpc_is_the_envelope_in_power_db(stereo):
    lpc = so.LPC(stereo.mono(), 12)
    values, _ = _image(so.plotting.plot_lpc(lpc, candidates=False, colorbar=False))
    freqs = np.arange(lpc.n_fft // 2 + 1) * lpc.fs / lpc.n_fft
    np.testing.assert_allclose(values, 10 * np.log10(lpc.envelope(freqs)[0]))


def test_plot_formant_track_breaks_where_not_voiced():
    sound = so.gaussian_noise(0.3, FS, rng=0)
    track = so.formant_track(sound)
    voiced = np.arange(len(track.t)) % 2 == 0
    for line in so.plotting.plot_formant_track(track, voiced=voiced).lines:
        assert np.isnan(line.get_ydata()[~voiced]).all()


def test_plot_chroma_draws_the_channel_asked(stereo):
    view = so.chroma(stereo)
    values, _ = _image(so.plotting.plot_chroma(view, channel=1))
    with np.errstate(divide="ignore"):
        np.testing.assert_allclose(values, so.power_to_db(view.power[:, :, 1].T))


def test_plot_ripple_pattern_is_db_re_the_mean():
    from sonore.sources.ripples import _evaluate

    pattern = so.Ripple(4, 1, depth=0.5)
    values, _ = _image(so.plotting.plot_ripple_pattern(pattern, n_t=50, n_x=20, colorbar=False))
    t, octaves = np.linspace(0, 1, 50), np.linspace(0, np.log2(8000 / 250), 20)
    env = _evaluate(pattern, t, octaves)
    np.testing.assert_allclose(values, 20 * np.log10(env / np.mean(env)))


def test_plot_interaural_cues_draws_itd_and_ild_as_measured():
    noise = so.gaussian_noise(0.3, FS, rng=0).data[:, 0]
    sound = so.apply_itd_ild(so.Sound(noise, FS), itd=300e-6, ild=6.0)
    cues = so.interaural_cues(sound)
    ax = so.plotting.plot_interaural_cues(cues, show_iac=False)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 1e6 * cues.itd)
    ild_ax = ax.figure.axes[-1]
    np.testing.assert_allclose(ild_ax.lines[0].get_ydata(), cues.ild)
    per_band = so.interaural_cues(sound, filterbank=so.cosine_filterbank(6, 300, 3000))
    values, _ = _image(so.plotting.plot_interaural_cues(per_band))
    np.testing.assert_allclose(values, 1e6 * per_band.itd.T)


@pytest.mark.parametrize("silent_part", ["all", "second half"])
def test_plot_interaural_cues_with_one_ear_silent(silent_part):
    noise = so.gaussian_noise(0.3, FS, rng=0).data[:, 0]
    right = (
        np.zeros_like(noise)
        if silent_part == "all"
        else np.where(np.arange(len(noise)) < len(noise) // 2, noise, 0)
    )
    cues = so.interaural_cues(so.Sound(np.stack([noise, right], axis=1), FS))
    assert not np.isfinite(cues.ild).all()
    ax = so.plotting.plot_interaural_cues(cues, show_iac=False)
    for axes in ax.figure.axes:
        assert np.isfinite(axes.get_ylim()).all()
    finite = cues.ild[np.isfinite(cues.ild)]
    if finite.size:
        assert ax.figure.axes[-1].get_ylim()[1] == pytest.approx(1.1 * np.abs(finite).max())


def test_banks_without_edge_filters_draw_and_label_every_band():
    bank = so.gammatone_filterbank(16, 200, 4000, edges=False)
    sb = bank.analyze(so.gaussian_noise(0.2, FS, rng=0))
    values, _ = _image(so.plotting.plot_envelopes(sb.envelopes(), colorbar=False))
    assert values.shape[0] == 16
    decimated = so.DecimatedFilterbank(so.cosine_filterbank(10, 200, 4000, edges=False))
    ax = so.plotting.plot_decimated_subbands(decimated.analyze(sb.sum()), colorbar=False)
    assert len(ax.collections) == 10
    labels = [ax.get_ylabel() for ax in so.plotting.plot_subbands(sb)]
    assert not any(label.startswith(("<", ">")) for label in labels)
    with_edges = so.plotting.plot_subbands(so.gammatone_filterbank(16, 200, 4000).analyze(sb.sum()))
    assert with_edges[0].get_ylabel().startswith(">") and with_edges[-1].get_ylabel().startswith("<")


def test_per_band_cues_label_every_band_by_its_center():
    sound = so.apply_itd_ild(so.gaussian_noise(0.3, FS, rng=0), itd=300e-6, ild=6.0)
    cues = so.interaural_cues(sound, filterbank=so.cosine_filterbank(6, 300, 3000))
    ax = so.plotting.plot_interaural_cues(cues)
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == [f"{cf:.0f}" for cf in cues.cfs]


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ((20, 20000), [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]),
        ((55, 420), [100, 200, 300]),
        ((800, 1800), [800, 1000, 1500]),
    ],
)
def test_log_frequency_axis_labels_plain_numbers(limits, expected):
    import matplotlib.pyplot as plt

    from sonore.plotting import log_frequency_axis

    fig, ax = plt.subplots()
    log_frequency_axis(ax)
    ax.set_ylim(*limits)
    fig.canvas.draw()
    ticks = [t for t in ax.get_yticks() if limits[0] <= t <= limits[1]]
    labels = [
        label.get_text()
        for label in ax.get_yticklabels()
        if limits[0] <= label.get_position()[1] <= limits[1]
    ]
    np.testing.assert_allclose(ticks, expected)
    assert labels == [f"{t:g}" for t in expected]
    assert ax.get_yscale() == "log"
    assert all(label.get_text() == "" for label in ax.yaxis.get_ticklabels(minor=True))
    plt.close(fig)


def test_plot_stft_draws_in_khz_from_10_khz_up():
    fs = 44100
    stft = so.STFT(so.pure_tone(0.2, fs, 1000), win_dur=0.02)
    ax = so.plotting.plot_stft(stft, colorbar=False)
    assert ax.get_ylabel() == "Frequency [kHz]"
    assert _image(ax)[1][..., 1].max() == pytest.approx(fs / 2 / 1000, rel=0.01)
    ax = so.plotting.plot_stft(stft, colorbar=False, fmax=8000)
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_ylim() == (0, 8000)


def test_plot_formant_track_follows_the_spectrogram_unit():
    import matplotlib.pyplot as plt

    snd = so.Sound(np.random.default_rng(0).standard_normal(FS // 2), FS)
    track = so.formant_track(snd)
    ax = so.plotting.plot_stft(so.STFT(snd, win_dur=0.02), colorbar=False)
    so.plotting.plot_formant_track(track, ax)
    assert ax.get_ylabel() == "Frequency [Hz]"
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), track.frequencies[0, 0])
    plt.close("all")


def test_log_frequency_axis_thins_ticks_on_a_small_panel():
    import matplotlib.pyplot as plt

    from sonore.plotting import log_frequency_axis

    fig, ax = plt.subplots(figsize=(3, 2))
    log_frequency_axis(ax, "x")
    ax.set_xlim(20, 20000)
    fig.canvas.draw()
    assert [t for t in ax.get_xticks() if 20 <= t <= 20000] == [100, 1000, 10000]
    plt.close(fig)
