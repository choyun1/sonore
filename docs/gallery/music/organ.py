"""The pipe organ: stops as additive synthesis. A template; the pipe sounds are still to come.

This script is the gallery page https://choyun1.github.io/sonore/gallery/organ.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/music/organ.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # The pipe organ
#
# Each stop of a pipe organ is a rank of pipes, one for every key, and drawing stops adds
# ranks. A registration is therefore a kind of additive synthesis: ranks at the octave and at
# other harmonics of the written note add partials to one tone, and change its timbre rather
# than adding notes. This page is a template. Its sounds wait for measured spectra of real stops,
# so for now it lays out what each section will show, and prints the arithmetic of footages and
# tuning, which needs no pipe sounds.
#
# - [Footage is a harmonic number](#h-footage-is-a-harmonic-number): what 8′, 4′ and 2⅔′ mean.
# - [One stop](#h-one-stop): principal, flute, stopped flute and reed on the same note.
# - [Building a registration](#h-building-a-registration): stops added one by one, and gap
#   registrations.
# - [Mutations are tuned pure](#h-mutations-are-tuned-pure): where the organ meets the
#   [temperament](temperament.html) page.
# - [Celeste and tremulant](#h-celeste-and-tremulant): beats on purpose.
# - [Attack](#h-attack): how a pipe starts to speak.

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
C4 = so.note_to_freq("C4")

# %% [markdown]
# ## Footage is a harmonic number
#
# A stop's footage names its pitch by the length of its lowest open pipe. The 8′ stop sounds the
# written note, 4′ an octave higher and 16′ an octave lower, and a rank at $8/n$ feet sounds
# harmonic $n$ of the 8′ stop: 2⅔′ the twelfth (harmonic 3), 1⅗′ the tierce (harmonic 5). The
# lengths are nominal: an ideal open pipe of 8 feet sounds about 70 Hz, while low C, which the
# name stands for, is 65.4 Hz. The cell prints the footages and the harmonic each reinforces.

# %%
for footage, name in [
    (16, "16′"),
    (8, "8′"),
    (4, "4′"),
    (8 / 3, "2⅔′"),
    (2, "2′"),
    (8 / 5, "1⅗′"),
    (4 / 3, "1⅓′"),
    (1, "1′"),
]:
    harmonic = 8 / footage
    print(f"{name:>4s}: harmonic {harmonic:g} of the 8′ stop, {harmonic * C4:7.1f} Hz on middle C")

# %% [markdown]
# ## One stop
#
# To come: a principal 8′, a flute 8′, a stopped flute 8′ and a reed 8′ on the same note, each
# with the spectrum of a measured stop. The principal will follow the harmonic levels Harrison &
# Thompson-Allen (1998) measured on the Great Diapason of the Newberry organ at Yale; the other
# families still need sources or recordings.

# %% [markdown]
# ## Building a registration
#
# To come: an 8′ principal, then 4′, 2⅔′, 2′ and 1⅗′ added one at a time, with the pitch track
# beside the spectrum, so that the spectrum fills in while the pitch stays on the same note. Then
# two gap registrations, 8′ + 2′ and 8′ + 1⅓′, where, in the experience of organists, the separate
# ranks can stand out as pitches of their own; the page will play them and say so, rather than
# claim a perceptual result.

# %% [markdown]
# ## Mutations are tuned pure
#
# Mutation stops, and the fifths and thirds of mixtures, are tuned pure to the harmonic they
# reinforce, not to the tempered notes of the keyboard. Tuned in equal temperament instead, as
# on a tonewheel organ, a mutation would beat with the 8′ stop's own harmonic. The cell prints
# how far, and how fast, on middle C; the sounds will play a pure and a tempered tierce.

# %%
for name, harmonic, semitones in [
    ("twelfth 2⅔′", 3, 19),
    ("tierce 1⅗′", 5, 28),
    ("larigot 1⅓′", 6, 31),
]:
    tempered = C4 * 2 ** (semitones / 12)
    pure = harmonic * C4
    print(
        f"{name:12s}: tempered {so.ratio_to_cents(tempered / pure):+6.2f} cents from harmonic {harmonic}, "
        f"beating at {abs(tempered - pure):5.2f} Hz"
    )

# %% [markdown]
# ## Celeste and tremulant
#
# To come: a celeste, a second rank tuned a few cents sharp of the first, so that the two beat
# slowly (on middle C, 3 cents sharp beats at about {{ f"{C4 * (so.cents_to_ratio(3) - 1):.2f}" }} Hz
# and 10 cents at {{ f"{C4 * (so.cents_to_ratio(10) - 1):.2f}" }} Hz; the beats of two close
# tones are on the [Synthetic sounds](classic.html#h-beats-and-roughness) page), and a tremulant,
# which shakes the wind and with it the level and pitch of every pipe.

# %% [markdown]
# ## Attack
#
# To come: how a flue pipe starts to speak, with a brief noisy transient (the chiff) and its
# harmonics building up at different rates. For instrument identification the build-up of the
# partials seems to matter more than the transient itself (Siedenburg, 2019; see the
# [Timbre](timbre.html) page), so the synthesis will model both.

# %% [markdown]
# ## References
#
# - Harrison & Thompson-Allen (1998). Steady-state spectra of diapason class stops of the
#   Newberry Memorial organ, Yale University. *J. Acoust. Soc. Am.* 103(1), 626–629.
#   [doi:10.1121/1.421134](https://doi.org/10.1121/1.421134).
# - Siedenburg (2019). Specifying the perceptual relevance of onset transients for musical
#   instrument identification. *J. Acoust. Soc. Am.* 145(2), 1078–1087.
#   [doi:10.1121/1.5091778](https://doi.org/10.1121/1.5091778).
