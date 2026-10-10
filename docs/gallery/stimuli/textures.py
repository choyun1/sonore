"""Sound textures: recordings and their syntheses from statistics.

This script is the gallery page https://choyun1.github.io/sonore/gallery/textures.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/stimuli/textures.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Sound textures
#
# Rain, a creek, crickets, applause, fire: sounds made of many similar events, whose character
# lies in their statistics rather than in any one waveform. After McDermott & Simoncelli (2011),
# sonore summarizes a texture by time-averaged statistics of a model of the cochlea, and
# synthesizes a new sample by imposing those statistics on noise.
#
# - [The model and its statistics](#h-the-model-and-its-statistics): cochlear bands, their
#   envelopes, and what is measured from them.
# - [Originals and syntheses](#h-originals-and-syntheses): each recording followed by its
#   synthesis, which shares no waveform with it, only statistics.
# - [What the statistics do](#h-what-the-statistics-do): the same textures synthesized from only
#   some of the statistics.

# %% [markdown]
# ## Code the examples share
#
# Every example below is the code shown with it, run after this cell: loading a recording and
# its synthesis, the level the gallery plays sounds at, and the plots.

# %% [setup]
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


def fetch(path):
    """A file from the sonore repository, by its path there: the local copy when this runs from
    the repository root, otherwise downloaded from GitHub to the same relative path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/choyun1/sonore/main/" + path, path)
    return path


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


# %%
import json

from sonore.texture import TextureStats

FS = 44100


def original(name):
    """A recording; docs/textures/SOURCES.md says where each comes from and which excerpt."""
    return so.load(fetch(f"docs/textures/{name}.flac"))


def synthesized(name, imposed=""):
    """A synthesis precomputed by tools/make_texture_synths.py (below)."""
    suffix = f"__{imposed}" if imposed else ""
    return so.load(fetch(f"docs/textures/synth/{name}{suffix}.flac")).resample(FS)


def report(name):
    """How closely the synthesis matches: the statistic SNR averaged over classes."""
    info = json.loads(open(fetch(f"docs/textures/synth/{name}.json")).read())
    snr = np.mean(list(info["snr_all_classes"].values()))
    return f"Average statistic SNR {snr:.0f} dB after {info['best_iteration']} iterations."


def texture_fig(snd, original, synthetic=False):
    """General-purpose plots (spectrogram, long-term spectrum) plus the texture
    model's own statistics drawn with plain matplotlib: modulation power by
    band and rate (the model's modulation spectrum), envelope sparsity, and
    band-averaged modulation power. Dashed black: the original."""
    fig = plt.figure(figsize=(10, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax_s = fig.add_subplot(gs[0, :2])
    ax_l, ax_m, ax_v, ax_p = (fig.add_subplot(gs[r, c]) for r, c in ((0, 2), (1, 0), (1, 1), (1, 2)))
    so.STFT(snd, 20e-3).plot(ax_s, fmax=10000, db_range=70, colorbar=False)
    ax_s.set_title("Spectrogram")
    so.long_term_spectrum(snd).plot(ax_l, lw=1.2, label="this sound")
    if synthetic:
        so.long_term_spectrum(original).plot(ax_l, color="k", ls="--", lw=1, label="original")
        ax_l.legend(fontsize=7)
    ax_l.set(xlim=(20, 10000), ylim=(-70, 3), title="Long-term spectrum")
    target = TextureStats.measure(original)
    this = TextureStats.measure(snd, window="uniform" if synthetic else "ramped")  # syntheses are circular
    cfs = target.model.filterbank.cfs[1:-1]
    ok = target.channel_mask()
    # The model's modulation spectrum: power at each modulation rate, in each cochlear band.
    mcf = target.model.mod_bank.cfs
    lev = 10 * np.log10(np.maximum(this.mod_power[1:-1], 1e-6))
    ax_m.pcolormesh(mcf, cfs, lev, cmap="magma", vmin=lev.max() - 25, vmax=lev.max(), shading="nearest")
    ax_m.set(
        xscale="log",
        yscale="log",
        xlabel="Modulation rate [Hz]",
        ylabel="Band center [Hz]",
        title="Modulation spectrum by band",
    )
    ax_v.semilogx(cfs, this.env_var[1:-1], lw=1.2)
    ax_p.loglog(target.model.mod_bank.cfs, this.mod_power[ok].mean(0), lw=1.2)
    if synthetic:
        ax_v.semilogx(cfs, target.env_var[1:-1], "k--", lw=1)
        ax_p.loglog(target.model.mod_bank.cfs, target.mod_power[ok].mean(0), "k--", lw=1)
    ax_v.set(xlabel="Band center [Hz]", ylabel="var / mean²", title="Envelope sparsity by band")
    ax_p.set(
        xlabel="Modulation rate [Hz]", ylabel="Power / variance", title="Modulation power (band average)"
    )
    for ax in (ax_v, ax_p):
        ax.grid(ls=":", which="both", lw=0.5)
    return fig, [ax_s]  # the playhead follows the spectrogram

# %% [markdown]
# ## The model and its statistics
#
# The sound is split into 30 cochlear bands (cosine filters on the ERB scale, 20 Hz to 10 kHz,
# plus two edge filters). Each band's envelope is compressed and downsampled to 400 Hz,
#
# $$s_k(t) = \bigl|\,x_k(t) + i\,\mathcal{H}\{x_k\}(t)\,\bigr|^{0.3},$$
#
# with $\mathcal{H}$ the Hilbert transform, and each envelope is in turn split into 20 modulation
# bands $b_{k,n}(t)$, 0.5 to 200 Hz, constant-Q. All averages $\langle\cdot\rangle$ are over time,
# weighted by a window that tapers the recording's ends. The statistics are those of the paper:
#
# - **Envelope marginals**: the mean $\mu_k = \langle s_k \rangle$, the sparsity
#   $\sigma_k^2 / \mu_k^2$, skewness and kurtosis of each band's envelope.
# - **Envelope correlations** $C_{jk}$ between bands $j$ and $k$, which make onsets coincide across
#   frequency.
# - **Modulation power** $\langle b_{k,n}^2 \rangle / \sigma_k^2$: the rhythm of each band.
# - **Modulation correlations** $C_1$, between bands in the same octave modulation band, and $C_2$,
#   between octave modulation bands within a band, which shape individual events.
#
# That makes 1515 numbers per texture. Synthesis starts from Gaussian noise and adjusts each
# band's subband signal by conjugate gradient until its statistics match, band by band, over
# several passes. The cell below is the call `tools/make_texture_synths.py` makes. It takes about
# 2 s per iteration for 5 s of sound, so the gallery's syntheses are computed once and stored.

# %%
from sonore.texture import PAPER_CLASSES  # noqa: E402
from sonore.texture.synth import synthesize  # noqa: E402


def synthesize_like(name, classes=PAPER_CLASSES, seconds=5.0, iterations=30):
    """What tools/make_texture_synths.py does for each recording (not run here)."""
    target = TextureStats.measure(original(name))
    new, rep = synthesize(target, duration=seconds, classes=classes, rng=1, max_iter=iterations)
    return new * (0.05 / new.rms)


# %% [markdown]
# ## Originals and syntheses
#
# The plots: spectrogram, long-term spectrum, and three of the model's own statistics:
# modulation power at each rate in each band (the model's modulation spectrum), envelope sparsity
# $\sigma_k^2/\mu_k^2$, and modulation power averaged over bands. For syntheses, the original's is
# overlaid in dashed black. The model's modulation spectrum is band by band, over rate only; the
# two-dimensional one of [Spectrotemporal ripples](ripples.html#h-how-a-ripple-is-made), with
# density across bands as well, is measured for the crickets, applause and rain on [Modulation
# spectrogram](modspectrogram.html#d-x1).

# %% [about]
# Steady rain (nick121087, Freesound, CC0).

# %% [demo t00a] Rain, original
sound = finish(original("rain"))
fig, playhead = texture_fig(sound, original("rain"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("rain") }}

# %% [demo t00b] Rain, synthesized
sound = finish(synthesized("rain"))
fig, playhead = texture_fig(sound, original("rain"), synthetic=True)

# %% [about]
# Water burbling over rocks in a small creek (cognito perceptu, Freesound, CC0).

# %% [demo t01a] Stream, original
sound = finish(original("stream"))
fig, playhead = texture_fig(sound, original("stream"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("stream") }}

# %% [demo t01b] Stream, synthesized
sound = finish(synthesized("stream"))
fig, playhead = texture_fig(sound, original("stream"), synthetic=True)

# %% [about]
# Crickets around midnight (sengjinn, Freesound, CC0).

# %% [demo t02a] Crickets, original
sound = finish(original("crickets"))
fig, playhead = texture_fig(sound, original("crickets"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("crickets") }}

# %% [demo t02b] Crickets, synthesized
sound = finish(synthesized("crickets"))
fig, playhead = texture_fig(sound, original("crickets"), synthetic=True)

# %% [about]
# About thirty people clapping (Breviceps, Freesound, CC0). Only 3.5 s of original.

# %% [demo t03a] Applause, original
sound = finish(original("applause"))
fig, playhead = texture_fig(sound, original("applause"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("applause") }}

# %% [demo t03b] Applause, synthesized
sound = finish(synthesized("applause"))
fig, playhead = texture_fig(sound, original("applause"), synthetic=True)

# %% [about]
# A small backyard fire: sparse pops over a low rumble (Sauron974, Freesound, CC0).

# %% [demo t04a] Fire, original
sound = finish(original("fire"))
fig, playhead = texture_fig(sound, original("fire"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("fire") }}

# %% [demo t04b] Fire, synthesized
sound = finish(synthesized("fire"))
fig, playhead = texture_fig(sound, original("fire"), synthetic=True)

# %% [about]
# Gas venting through a Yellowstone mud pot (NPS / Jennifer Jerrett, public domain).

# %% [demo t05a] Bubbling mud, original
sound = finish(original("mud"))
fig, playhead = texture_fig(sound, original("mud"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("mud") }}

# %% [demo t05b] Bubbling mud, synthesized
sound = finish(synthesized("mud"))
fig, playhead = texture_fig(sound, original("mud"), synthetic=True)

# %% [about]
# A winter storm, wind with heavy rain (sonicwars, Freesound, CC0).

# %% [demo t06a] Wind and rain, original
sound = finish(original("wind_rain"))
fig, playhead = texture_fig(sound, original("wind_rain"))

# %% [about]
# Gaussian noise adjusted until its statistics match the original's. {{ report("wind_rain") }}

# %% [demo t06b] Wind and rain, synthesized
sound = finish(synthesized("wind_rain"))
fig, playhead = texture_fig(sound, original("wind_rain"), synthetic=True)

# %% [markdown]
# ## What the statistics do
#
# The same textures synthesized with only some of the statistics imposed, by passing fewer
# classes to `synthesize_like`:
#
# - `("env_mean", "env_var", "env_skew", "env_kurt")`, the envelope marginals only;
# - the marginals plus `"mod_power"`, but no correlations across bands or modulation bands.
#
# Compare with the full syntheses above: the marginals alone give the right sparsity but not the
# right rhythm or the coordination across bands. [Hearing a modulation
# spectrum](modtargets.html#d-mt15) makes the converse point with the fire: each band's
# modulation spectrum kept, with random phases, and the crackles are gone.

# %% [about]
# Only the envelope marginals imposed (mean, variance, skew, kurtosis of each band's envelope).

# %% [demo u00] Applause, marginals only
sound = finish(synthesized("applause", "marginals"))
fig, playhead = texture_fig(sound, original("applause"), synthetic=True)

# %% [about]
# Envelope marginals plus modulation power; no correlations across bands or modulation bands.

# %% [demo u01] Applause, marginals and modulation power
sound = finish(synthesized("applause", "marginals_modpower"))
fig, playhead = texture_fig(sound, original("applause"), synthetic=True)

# %% [about]
# Only the envelope marginals imposed (mean, variance, skew, kurtosis of each band's envelope).

# %% [demo u10] Stream, marginals only
sound = finish(synthesized("stream", "marginals"))
fig, playhead = texture_fig(sound, original("stream"), synthetic=True)

# %% [about]
# Envelope marginals plus modulation power; no correlations across bands or modulation bands.

# %% [demo u11] Stream, marginals and modulation power
sound = finish(synthesized("stream", "marginals_modpower"))
fig, playhead = texture_fig(sound, original("stream"), synthetic=True)

# %% [about]
# Only the envelope marginals imposed (mean, variance, skew, kurtosis of each band's envelope).

# %% [demo u20] Fire, marginals only
sound = finish(synthesized("fire", "marginals"))
fig, playhead = texture_fig(sound, original("fire"), synthetic=True)

# %% [about]
# Envelope marginals plus modulation power; no correlations across bands or modulation bands.

# %% [demo u21] Fire, marginals and modulation power
sound = finish(synthesized("fire", "marginals_modpower"))
fig, playhead = texture_fig(sound, original("fire"), synthetic=True)

# %% [markdown]
# ## References
#
# - McDermott & Simoncelli (2011). Sound texture perception via statistics of the auditory
#   periphery. *Neuron* 71(5), 926–940.
#   [doi:10.1016/j.neuron.2011.06.032](https://doi.org/10.1016/j.neuron.2011.06.032).
#   [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py)
#   [`filterbank.Cosine`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L139)
#   [`modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py)
