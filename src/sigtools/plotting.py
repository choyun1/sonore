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
    "plot_interaural_cues",
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
        title="Modulation spectrum", xlabel="Temporal modulation [Hz]", ylabel="Spectral modulation [cyc/kHz]"
    )
    if wt_max:
        wt_max = min(wt_max, np.abs(ms.w_t).max())
        ax.set_xlim(-wt_max, wt_max)
    if wf_max:
        ax.set_ylim(0, min(wf_max, ms.w_f.max()))
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="dB")
    return ax


def plot_subbands(sb, axes=None, channel=0, kind="waveform", cmap="magma"):
    """``kind="waveform"``: stacked traces (one axis per band, low at bottom).
    ``kind="image"``: bands as an image on a single axis."""
    import matplotlib.pyplot as plt

    n, B, _ = sb.data.shape
    t = np.arange(n) / sb.fs
    if kind == "image":
        ax = _ax(axes)
        im = ax.pcolormesh(
            t, np.arange(B), sb.data[:, :, channel].T, cmap=cmap, shading="auto", rasterized=True
        )
        ax.set_yticks(np.arange(B)[:: max(1, B // 8)], [f"{c:.0f}" for c in sb.cfs[:: max(1, B // 8)]])
        ax.set(xlabel="Time [s]", ylabel="CF [Hz]", title="Subbands")
        ax.figure.colorbar(im, ax=ax)
        return ax
    if axes is None:
        _, axes = plt.subplots(B, 1, figsize=(8, 0.45 * B), sharex=True)
    for i, ax in enumerate(axes[::-1]):
        ax.plot(t, sb.data[:, i, channel], lw=0.6)
        ax.set_yticks([])
        ax.set_ylabel(f"{sb.cfs[i]:.0f}", rotation=0, ha="right", va="center", fontsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_title("Subbands (CF in Hz)")
    axes[-1].set_xlabel("Time [s]")
    return axes


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
        B = len(cues.cfs)
        ax.set_yticks(np.arange(B)[:: max(1, B // 8)], [f"{c:.0f}" for c in cues.cfs[:: max(1, B // 8)]])
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


def overview(sound, win_dur=20e-3, figsize=(12, 8), fmax=None):
    """Waveform, spectrum, spectrogram and modulation spectrum in one figure
    (replaces the old ``display_STFT``). Returns the Figure."""
    import matplotlib.pyplot as plt

    from sigtools.representations import STFT, ModulationSpectrum, long_term_spectrum

    S = STFT(sound.mono(), win_dur)
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    plot_waveform(sound, axes[0, 0])
    plot_spectrum(long_term_spectrum(sound), axes[0, 1])
    plot_stft(S, axes[1, 0], fmax=fmax)
    plot_modulation_spectrum(ModulationSpectrum(S), axes[1, 1], wt_max=50, wf_max=10)
    return fig
