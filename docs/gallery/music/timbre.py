"""Timbre: what tells two sounds apart at the same pitch and loudness. A placeholder for now.

This script is the gallery page https://choyun1.github.io/sonore/gallery/timbre.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/music/timbre.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Timbre
#
# Timbre is usually defined by what it is not: whatever still tells two sounds apart when they
# have the same pitch, the same loudness and the same duration. A clarinet and a violin playing
# the same note differ in timbre, and so do two vowels sung on the same note. This page is a
# placeholder; the examples are still to be written.
#
# - [Vowels are timbres](#h-vowels-are-timbres): the link to speech.
# - [To come](#h-to-come): timbre spaces, and what the onset of a note carries.

# %% [markdown]
# ## Vowels are timbres
#
# Two vowels said on the same pitch and at the same level differ in their spectral envelopes:
# the formants, the resonances of the vocal tract, sit at different frequencies. In that sense
# different vowels are different timbres of one voice, and much of what the speech pages show
# applies here. [Formant synthesis](formants.html) builds vowels from a source and a few
# resonances and changes one formant at a time, and [Cepstral analysis](cepstrum.html) separates
# a voice's pitch from its spectral envelope, the part that carries the vowel.

# %% [markdown]
# ## To come
#
# - **Timbre spaces.** Listeners rate how different pairs of sounds are, and multidimensional
#   scaling places the sounds in a space whose axes can then be named. For 18 synthesized
#   instrument-like tones, McAdams et al. (1995) found three shared dimensions, which correlated
#   with the logarithm of the attack time, the spectral centroid (brightness) and the spectral
#   flux (how much the spectrum changes over the tone). The page will play synthetic tones that
#   move along one of those dimensions at a time.
# - **Onsets.** Much of what identifies an instrument is in the first moments of a note. In
#   Siedenburg (2019), listeners named ten instruments from 64 ms excerpts 77% of the time when
#   the excerpt came from the onset, 71% when the onset's fast transient was removed, and 52%
#   when it came from the middle of the tone (chance was 10%): the onset matters mostly through
#   the way the partials build up, and the transient itself less. The page will play the same
#   tone with its partials starting together, building up at different rates, and with a burst
#   added.

# %% [markdown]
# ## References
#
# - McAdams, Winsberg, Donnadieu, De Soete & Krimphoff (1995). Perceptual scaling of synthesized
#   musical timbres: common dimensions, specificities, and latent subject classes. *Psychol.
#   Res.* 58, 177–192. [doi:10.1007/BF00419633](https://doi.org/10.1007/BF00419633).
# - Siedenburg (2019). Specifying the perceptual relevance of onset transients for musical
#   instrument identification. *J. Acoust. Soc. Am.* 145(2), 1078–1087.
#   [doi:10.1121/1.5091778](https://doi.org/10.1121/1.5091778).
