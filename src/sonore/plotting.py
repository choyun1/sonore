"""Matplotlib plotting. Every function takes an optional ``ax`` and returns it.

Nothing here touches global matplotlib settings; style your notebook however
you like (e.g. ``plt.style.use(...)``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_waveform",
    "plot_spectrum",
    "plot_stft",
    "plot_mask",
    "plot_modulation_spectrum",
    "plot_subbands",
    "plot_envelope",
    "plot_envelopes",
    "plot_tf_db",
    "plot_cepstrum",
    "plot_interaural_cues",
    "plot_ripple_pattern",
    "overview",
]


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def plot_waveform(sound, ax=None, labels=None, **kwargs):
    ax = _ax(ax)
    kwargs.setdefault("alpha", 0.6)
    kwargs.setdefault("lw", 0.8)
    lines = ax.plot(sound.t, sound.data, **kwargs)
    if sound.n_channels == 2 and labels is None:
        labels = ["L", "R"]
    if labels:
        for line, lab in zip(lines, labels, strict=False):
            line.set_label(lab)
        ax.legend(loc="upper right")
    ax.axhline(0, color="k", alpha=0.25, lw=0.8)
    ax.set(title="Waveform", xlabel="Time [s]", ylabel="Amplitude", xlim=(0, sound.duration))
    ax.grid(ls=":")
    return ax


def plot_spectrum(spectrum, ax=None, fscale="log", relative=True, **kwargs):
    ax = _ax(ax)
    s = spectrum.relative() if relative else spectrum
    f, lev = (s.f[1:], s.level[1:]) if fscale == "log" else (s.f, s.level)
    ax.plot(f, lev, **kwargs)
    ax.set_xscale(fscale)
    ax.set(
        title="Magnitude spectrum",
        xlabel="Frequency [Hz]",
        ylabel="Level re max [dB]" if relative else "Level [dB]",
        xlim=(f[0], f[-1]),
    )
    ax.grid(ls=":", which="both")
    return ax


def _tf_image(ax, values, t, f, cmap, vmin, vmax, colorbar, label):
    im = ax.pcolormesh(t, f / 1000, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    ax.set(xlabel="Time [s]", ylabel="Frequency [kHz]")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=label)
    return im


def _interior(stft):
    """Frames whose window lies entirely inside the signal (edge frames are
    zero-padded, which looks like a click)."""
    start = np.round(stft.t * stft.fs).astype(int) - stft.sft.m_num_mid
    ok = np.flatnonzero((start >= 0) & (start + stft.sft.m_num <= stft.n_samples))
    return slice(ok[0], ok[-1] + 1) if len(ok) else slice(None)


def plot_stft(
    stft, ax=None, channel=0, db_range=80.0, cmap="magma", colorbar=True, fmax=None, trim_edges=True
):
    ax = _ax(ax)
    keep = _interior(stft) if trim_edges else slice(None)
    d = stft.db[channel][:, keep]
    vmax = d.max()
    _tf_image(ax, d, stft.t[keep], stft.f, cmap, vmax - db_range, vmax, colorbar, "dB")
    ax.set_title("Spectrogram")
    ax.set_xlim(0, stft.n_samples / stft.fs)
    if fmax:
        ax.set_ylim(0, fmax / 1000)
    return ax


def plot_mask(mask, ax=None, channel=0, cmap="Greys_r", colorbar=False):
    ax = _ax(ax)
    v = mask.values[channel] if mask.values.ndim == 3 else mask.values
    _tf_image(ax, v, mask.t, mask.f, cmap, 0, 1, colorbar, "")
    ax.set_title("Mask")
    return ax


def plot_modulation_spectrum(
    ms, ax=None, db_range=60.0, cmap="magma", colorbar=True, wt_max=None, wf_max=None
):
    ax = _ax(ax)
    vmax = ms.level.max()
    im = ax.pcolormesh(
        ms.w_t, ms.w_f, ms.level, cmap=cmap, vmin=vmax - db_range, vmax=vmax, shading="auto", rasterized=True
    )
    ax.set(
        title="Modulation spectrum",
        xlabel="Temporal modulation [Hz]",
        ylabel=f"Spectral modulation [{getattr(ms, 'spectral_unit', 'cyc/kHz')}]",
    )
    if wt_max:
        wt_max = min(wt_max, np.abs(ms.w_t).max())
        ax.set_xlim(-wt_max, wt_max)
    if wf_max:
        ax.set_ylim(0, min(wf_max, ms.w_f.max()))
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="dB")
    return ax


def _band_labels(cfs, edges=True):
    """CF labels; the first and last filters are the lowpass/highpass edges."""
    labels = [f"{c:.0f}" for c in cfs]
    if edges:
        labels[0] = f"< {cfs[0]:.0f}"
        labels[-1] = f"> {cfs[-1]:.0f}"
    return labels


def plot_subbands(sb, axes=None, channel=0, sharey=True, color=None, bands=None):
    """One trace per band, lowest at the bottom. With ``sharey=True`` (default)
    all traces share one amplitude scale, so relative band levels are visible.
    The bottom and top traces are the lowpass and highpass edge filters
    (labelled ``< f_lo`` and ``> f_hi``).

    ``bands`` selects which bands to show (indices), e.g.
    ``range(1, len(sb) - 1, 5)`` for every fifth band of a fine filterbank.
    For an image of all bands, plot the envelopes: ``sb.envelopes().plot()``.
    """
    import matplotlib.pyplot as plt

    n = sb.data.shape[0]
    idx = np.arange(sb.data.shape[1]) if bands is None else np.asarray(list(bands))
    B = len(idx)
    t = np.arange(n) / sb.fs
    data = sb.data[:, idx, channel]
    labels = [_band_labels(sb.cfs)[i] for i in idx]
    if axes is None:
        _, axes = plt.subplots(B, 1, figsize=(8, 0.5 * B + 0.6), sharex=True)
    axes = list(axes)
    if len(axes) != B:
        raise ValueError(f"need {B} axes (one per band shown), got {len(axes)}")
    lim = 1.05 * np.max(np.abs(data)) or 1.0
    for i, ax in enumerate(axes[::-1]):
        ax.plot(t, data[:, i], lw=0.6, color=color)
        ax.set_yticks([])
        ax.set_xlim(0, n / sb.fs)
        if sharey:
            ax.set_ylim(-lim, lim)
        ax.set_ylabel(labels[i], rotation=0, ha="right", va="center", fontsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[0].set_title("Subbands [Hz]")
    axes[-1].set_xlabel("Time [s]")
    return axes


def plot_envelope(env, ax=None, db=False, **kwargs):
    """A single :class:`~sonore.analysis.envelopes.Envelope` over time."""
    ax = _ax(ax)
    ax.plot(env.t, env.db if db else env.data, **kwargs)
    ax.set(
        title="Envelope",
        xlabel="Time [s]",
        ylabel="Envelope [dB]" if db else "Envelope",
        xlim=(0, env.duration),
    )
    ax.grid(ls=":")
    return ax


def plot_envelopes(
    env,
    ax=None,
    channel=0,
    db_range=40.0,
    cmap="magma",
    colorbar=True,
    edges=False,
    align=None,
    fscale="log",
    fmax=None,
):
    """Envelopes (a cochleagram) as an image: time x band, in dB re the maximum.
    Edge bands are hidden unless ``edges=True``.

    ``align="peak"`` draws each band earlier by its filter's
    ``envelope_peak_delay`` (a causal gammatone bank's latency), so a click
    is a vertical line; only the drawing moves, not the data.
    ``fscale="linear"`` draws frequency linearly in kHz, matching
    :func:`plot_stft`, with an optional ``fmax`` [Hz].
    """
    ax = _ax(ax)
    sel = slice(None) if edges else slice(1, -1)
    db = env.db[:, sel, channel]
    vmax = db.max()
    cfs = env.cfs[sel] / (1000 if fscale == "linear" else 1)
    if fscale not in ("log", "linear"):
        raise ValueError("fscale must be 'log' or 'linear'")
    if align is None:
        t, f = env.t, cfs
    elif align == "peak":
        delay = getattr(env.filterbank, "envelope_peak_delay", None)
        if delay is None:
            raise TypeError(f"{type(env.filterbank).__name__} has no envelope_peak_delay to align by")
        t = env.t[None, :] - np.asarray(delay)[sel][:, None]  # one time axis per band
        f = np.broadcast_to(cfs[:, None], t.shape)
    else:
        raise ValueError("align must be None or 'peak'")
    im = ax.pcolormesh(
        t, f, db.T, cmap=cmap, vmin=vmax - db_range, vmax=vmax, shading="auto", rasterized=True
    )
    if fscale == "log":
        ax.set_yscale("log")
        ax.set(xlabel="Time [s]", ylabel="Frequency [Hz]", title="Envelopes (cochleagram)")
    else:
        ax.set(xlabel="Time [s]", ylabel="Frequency [kHz]", title="Envelopes (cochleagram)")
        if fmax:
            ax.set_ylim(0, fmax / 1000)
    if align == "peak":
        ax.set_xlim(env.t[0], env.t[-1])
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="dB")
    return ax


def plot_tf_db(db, t, f, ax=None, db_range=60.0, cmap="magma", colorbar=True, fmax=None, title=None):
    """A time-frequency image from levels ``db`` (shape ``(n_freqs, n_frames)``)
    at frame times ``t`` [s] and frequencies ``f`` [Hz], which need not be
    uniform: each cell extends halfway to its neighbors. Frequency is linear
    in kHz, as in :func:`plot_stft`."""
    ax = _ax(ax)
    vmax = np.max(db)
    _tf_image(ax, db, np.asarray(t), np.asarray(f), cmap, vmax - db_range, vmax, colorbar, "dB")
    if fmax:
        ax.set_ylim(0, fmax / 1000)
    if title:
        ax.set_title(title)
    return ax


def plot_cepstrum(cep, ax=None, channel=0, q_range=(1e-3, 15e-3), cmap="magma", colorbar=True):
    """A cepstrum as an image: time [s] across, quefrency [ms] up, from
    ``q_range`` [s]. The default, 1 to 15 ms, leaves out the low quefrencies
    (the spectral envelope, which would set the color scale) and covers the
    periods of voices from about 67 Hz up. Negative values are drawn as 0."""
    ax = _ax(ax)
    q = cep.q
    sel = (q >= q_range[0]) & (q <= q_range[1])
    values = np.maximum(cep.data[channel][sel], 0)
    vmax = values.max() or 1.0
    qm = q[sel] * 1e3
    im = ax.pcolormesh(cep.t, qm, values, cmap=cmap, vmin=0, vmax=vmax, shading="auto", rasterized=True)
    ax.set(title="Cepstrum", xlabel="Time [s]", ylabel="Quefrency [ms]")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="Cepstrum")
    return ax


def plot_interaural_cues(cues, ax=None, show_iac=True):
    """Broadband cues: ITD and ILD on twin axes (plus IAC underneath).
    Per-band cues: ITD as an image."""
    import matplotlib.pyplot as plt

    if cues.itd.ndim == 2:
        ax = _ax(ax)
        lim = np.nanmax(np.abs(cues.itd)) * 1e6 or 1
        im = ax.pcolormesh(
            cues.t,
            np.arange(cues.itd.shape[1]),
            1e6 * cues.itd.T,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            shading="auto",
        )
        B, step = len(cues.cfs), max(1, len(cues.cfs) // 8)
        ax.set_yticks(np.arange(B)[::step], _band_labels(cues.cfs)[::step])
        ax.set(xlabel="Time [s]", ylabel="CF [Hz]", title="ITD per band (+ = right leads)")
        ax.figure.colorbar(im, ax=ax, label="ITD [µs]")
        return ax

    if ax is None:
        nrows = 2 if show_iac else 1
        fig, axs = plt.subplots(nrows, 1, figsize=(8, 3 * nrows), sharex=True, squeeze=False)
        ax, iac_ax = axs[0, 0], (axs[1, 0] if show_iac else None)
    else:
        iac_ax = None
    ax.plot(cues.t, 1e6 * cues.itd, color="m", alpha=0.7, label="ITD")
    ax.axhline(0, color="k", alpha=0.25, lw=0.8)
    ax.set(title="Interaural cues (+ = right)", ylabel="ITD [µs]")
    ax.grid(ls=":")
    tw = ax.twinx()
    tw.plot(cues.t, cues.ild, color="g", alpha=0.7, label="ILD")
    tw.set_ylabel("ILD [dB]")
    lim = np.nanmax(np.abs(cues.ild)) * 1.1 or 1
    tw.set_ylim(-lim, lim)
    lim = np.nanmax(np.abs(cues.itd)) * 1.1e6 or 1
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left")
    tw.legend(loc="upper right")
    if iac_ax is not None:
        iac_ax.plot(cues.t, cues.iac, color="k", alpha=0.7, label="coherence (peak)")
        iac_ax.plot(cues.t, cues.corr0, color="tab:orange", alpha=0.7, label="correlation (lag 0)")
        iac_ax.axhline(0, color="k", alpha=0.25, lw=0.8)
        iac_ax.set(ylabel="Interaural corr.", xlabel="Time [s]", ylim=(-1.05, 1.05))
        iac_ax.legend(loc="lower right", fontsize=8)
        iac_ax.grid(ls=":")
    else:
        ax.set_xlabel("Time [s]")
    return ax


def plot_ripple_pattern(
    pattern, duration=1.0, f_lo=250.0, f_hi=8000.0, ax=None, cmap="RdBu_r", colorbar=True, n_t=None, n_x=200
):
    """Envelope of a ripple pattern (dB re its mean level) over time and
    log-frequency, before any sound is made."""
    from sonore.stimuli.ripples import _evaluate, _max_rate

    ax = _ax(ax)
    if n_t is None:  # resolve the fastest modulation with ~8 points per cycle
        n_t = int(max(400, np.ceil(duration * 8 * (_max_rate(pattern) or 0))))
    t = np.linspace(0, duration, n_t)
    x = np.linspace(0, np.log2(f_hi / f_lo), n_x)
    env = _evaluate(pattern, t, x)
    db = 20 * np.log10(np.maximum(env, 1e-6) / np.mean(env))
    lim = np.max(np.abs(db)) or 1.0
    im = ax.pcolormesh(t, f_lo * 2**x, db, cmap=cmap, vmin=-lim, vmax=lim, shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set(xlabel="Time [s]", ylabel="Frequency [Hz]", title="Ripple pattern")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="Envelope [dB]")
    return ax


def overview(sound, win_dur=20e-3, figsize=(12, 8), fmax=None):
    """Waveform, spectrum, spectrogram and modulation spectrum in one figure
    (replaces the old ``display_STFT``). Returns the Figure."""
    import matplotlib.pyplot as plt

    from sonore.analysis.representations import STFT, ModulationSpectrum, long_term_spectrum

    S = STFT(sound.mono(), win_dur)
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    plot_waveform(sound, axes[0, 0])
    plot_spectrum(long_term_spectrum(sound), axes[0, 1])
    plot_stft(S, axes[1, 0], fmax=fmax)
    plot_modulation_spectrum(ModulationSpectrum(S), axes[1, 1], wt_max=50, wf_max=10)
    return fig
