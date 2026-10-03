# Specifying a modulation spectrum

Cho asked on 2026-10-03 how one could specify a desired modulation
spectrum: by drawing it, or in other ways. This note answers that, measures
what such a specification actually pins down, and proposes what sonore
could offer. It adds no library code; the decisions at the end come first.

Status: accepted 2026-10-03. Cho accepted D1–D9 as recommended. D2–D6
are implemented (`ModulationSpectrum.to_envelopes`, `with_gain` and
`to_sound` in `src/sonore/views/modulation.py`, tested in
`tests/views/test_modulation.py`), and so are D1's blobs and D6's
`rms_depth` (`ModulationBlob`, `ModulationSpectrum.from_blobs`), and D7
(`to_sound(iterations=...)`, after C7). D9 follows. The claims are checked by
`tools/check_modulation_targets_claims.py` (C1–C7), which uses only NumPy,
SciPy and soundfile, writes every filter and transform out from its
formula, and shares no code with sonore. It runs in about 45 s. The numbers
below come from NumPy 2.4.6 and SciPy 1.17.1, with the seeds in the script.

## How the claims are verified

As in `modulation-spectrogram.md`, each claim is tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by the checker.
- **[source]**: a published result (see References, which says how far each
  source was checked).

Anything else is marked as an estimate or as inferred.

## Which modulation spectrum

sonore already computes three things that go by the name, and a target has
to say which one it is.

1. **The 2-D modulation power spectrum** (`ModulationSpectrum`): the squared
   magnitude of the 2-D Fourier transform of a whole envelope array,
   temporal rate (Hz, signed) against spectral modulation (cycles per octave
   or per kHz). One number per cell for the whole sound. This is the plane
   of ripple stimuli and of the speech "modulation transfer function"
   experiments.
2. **Per-band modulation power** (`TextureStats.mod_power`): for every
   audio band, the power of its envelope in each modulation band. Acoustic
   frequency against rate.
3. **The modulation spectrogram** (`ModulationSpectrogram`): item 2 for
   every time window. Its time average is item 2 with slightly different
   filters (C7 of `modulation-spectrogram.md`).

These are different targets [proof]. Item 1 is a transform over the whole
frequency axis, so it does not say *where* in frequency a modulation sits:
a 4 Hz modulation in the low bands and the same one in the high bands give
the same cell. Item 2 treats each band alone, so it has no spectral
modulation and no direction: an upward and a downward sweep with the same
band envelopes up to a delay give the same item 2. Item 3 adds when, and
still has neither phase nor direction. A drawing on the rate × density
plane is item 1; a drawing on the frequency × rate plane is item 2.

## What a modulation spectrum leaves free

A power spectrum keeps magnitudes and drops phase [proof]. For item 1, the
phase of the 2-D transform holds the timing of every event and how the
bands line up across frequency. Any phase gives the same power spectrum,
and the inverse transform of each is an envelope array with exactly that
spectrum, as long as it stays non-negative. Beyond phase, a target leaves
out three more things, which a route back to sound has to supply or store:

- **the overall level**: the 2-D transform is taken after removing one
  number, the mean of the whole envelope array (in sonore as in the
  checker); the long-term spectrum's shape stays in the zero-rate column;
- **the fine structure** under the envelopes, i.e. the carrier;
- **the analysis it was measured with**: the filterbank, the envelope
  extraction, compression, and linear or dB envelopes. C10 of
  `modulation-spectrogram.md` measured that 0.3 compression multiplies depth
  by 0.45, and C3 below shows the carrier and the filterbank interact.

`docs/design/philosophy.md` ("Views may discard information") already
says a modulation spectrum is a view whose possible carriers are too many
for a canonical `to_sound`. The claims below put numbers on how much is
free.

## Claims

**C1. Two envelope arrays with the same modulation spectrum can be
unrelated, and only one of them looks like speech.** [proof, check] The
gallery sentence (`bdl_arctic_a0131`) through 61 half-cosine bands, 12 per
octave from 200 to 6400 Hz, with Hilbert envelopes at 400 Hz. Its "twin"
keeps the magnitude of the envelope array's 2-D transform and takes the
phase of white noise's.

- The two modulation power spectra agree to 4e-16 of the peak.
- The two envelope arrays correlate −0.08 over all band × time cells.
- 33% of the twin's cells are negative, which no envelope can be. Clipping
  them at zero changes the spectrum by a median of 3.1 dB where it is within
  30 dB of its peak, and puts 8.5% of the power where the original had less
  than −30 dB.
- The pauses go. The speech's pooled envelope (summed over bands) is more
  than 30 dB below its peak 9.6% of the time; the clipped twin's never is.
  Its kurtosis over time drops from 5.8 to 3.1, the value of a Gaussian.
- Randomizing phase in dB instead (the log envelope, as Elliott &
  Theunissen (2009) define the spectrum) always gives a valid envelope, but it keeps the dB
  spectrum, not the linear one: the linear spectrum moves by a median of
  7.3 dB.

So a modulation spectrum fixes a sound's *texture of change*, not the
change itself. Speech's syllables, pauses and onsets are in the phase.

**C2. Non-negativity limits how deep a random-phase pattern can be.**
[proof, check] A pattern `1 + depth · P / max |P|` is non-negative only for
`depth ≤ 1`. A random-phase field is close to Gaussian, whose peak is
several standard deviations, so its rms depth at `depth = 1` is small. For
a drawn blob at 4 Hz and 0.5 cycle/octave (downward sweeps only, one octave
wide in rate, 0.25 cycle/octave in density), over 4 s and 5 octaves: rms
depth 0.28 (median of 20 seeds, lowest 0.20), against 0.71 for one
full-depth sinusoidal ripple. At that depth the pattern's own spectrum
matches the target exactly (dB correlation 1.000 inside the blob; the 0.86%
of power outside it is the drawn blob's own tails). A drawn target sets a
shape, not a depth: depth has to be specified separately, and a linear
pattern cannot go past this limit without clipping (C1 shows what clipping
costs). A dB pattern has no limit but distorts the linear spectrum (C1).

**C3. Whether the sound has the drawn spectrum depends as much on the
carrier and the analysis as on the pattern.** [check] The C2 pattern on
four carriers, each analysed with the 12-per-octave bank of C1.

| Carrier | dB correlation with the target, inside the blob | Measured power outside the blob | rms depth of the unmodulated carrier's own envelopes |
|---|---|---|---|
| Random-phase tones, 24 per octave | 0.91 | 72% | 0.45 |
| Noise bands, 24 per octave | 0.83 | 75% | 0.51 |
| Noise bands with flattened envelopes ("low-noise") | 0.91 | 69% | 0.43 |
| Tones on the analysis band centres, 12 per octave | 0.98 | 5.2% | 0.03 |

Every carrier but the last brings its own envelope fluctuations, as large as
the pattern's (0.43–0.51 against 0.28), and most of the measured modulation
power is the carrier's. Tones beat with their neighbours inside a band
(adjacent tones 1/24 octave apart beat at 2.9% of their frequency, 5.8 Hz
at 200 Hz, inferred from the spacing). Even noise bands whose own envelopes
are flat fluctuate once an analysis band sums several of them. Only tones
placed at the analysis bands' centres, where both neighbouring filters are
zero, give each band one tone and so no beats. So "a sound with this
modulation spectrum" means nothing until the analysis is named, and a
one-shot synthesis hits its target only if its carrier is built for that
analysis. sonore's `ripple_sound` defaults (20 tones per octave) and
`ModulationSpectrum.octave` (12 bands per octave) are not aligned in this
way; from this claim they should show a floor of beat modulation under the
ripple (inferred, not measured on sonore's code).

**C4. Editing a measured spectrum and resynthesizing in one step loses
most of the edit.** [check] The measure-and-edit route as Elliott &
Theunissen (2009) describe it: the log-magnitude STFT of the sentence
(32 ms Hann windows, 4 ms hop), a 2-D transform, every temporal modulation
above 4 Hz removed, and back to a target magnitude. Before the edit, 17.5%
of the log spectrogram's modulation power is at 6–40 Hz. Re-analysing the
resynthesized sound:

| Resynthesis | Power left at 6–40 Hz, re the original's | ‖ \|STFT\| − target ‖ / ‖ target ‖ |
|---|---|---|
| Target magnitude with the original phase, one inverse STFT | −5.9 dB | 0.27 |
| Target magnitude with random phase, one inverse STFT | −4.4 dB | 0.73 |

Atlas & Shamma (2003) note that a joint acoustic and modulation
representation needs "added constraints" to be invertible [source]. Here
the edited magnitude is not the magnitude of any STFT: overlapping windows
tie neighbouring frames together, and the inverse STFT returns the nearest
signal, whose own magnitude puts much of the removed modulation back.

**C5. An iterative search keeps more of the edit, and still does not reach
it.** [check] Griffin & Lim's (1984) iteration (alternate between the target
magnitude and the STFT of the nearest signal) from the original phase, on
the C4 target:

| Iterations | Power left at 6–40 Hz, re the original's | ‖ \|STFT\| − target ‖ / ‖ target ‖ |
|---|---|---|
| 10 | −11.6 dB | 0.18 |
| 50 | −14.1 dB | 0.11 |
| 200 | −15.1 dB | 0.09 |

The same iteration on the sentence's own, *unedited* magnitude, a target
that a real sound is known to have, from random phase: mismatch 0.73 after
one step, 0.23 after 10, 0.14 after 50, 0.069 after 200. So even a
consistent target is approached, not reached, and random phase alone (one
step) is far from it.

The mismatch keeps falling, more and more slowly. Each iteration is one
STFT and one inverse (the 200 took 4.0 s for this 2.5 s sentence,
measured in this container, so an estimate elsewhere). A search through
the modulation analysis itself, with gradients as the texture synthesis
does, would aim at the modulation spectrum directly instead of at a
spectrogram; that is not measured here.

**C6. The fine structure must not carry modulations of its own.** [check]
The C4 edit (temporal modulations above 4 Hz removed) made on the
filterbank envelopes of C1 instead of the spectrogram. The envelope array's
2-D transform keeps the sentence's own modulation phase, so the edited
envelopes are exact: their 6–40 Hz share is 21.5 dB below the sentence's
(after clipping the 15% of cells the edit pushed below zero). The envelopes
then go on three fine structures and the sound is re-analysed:

| Fine structure under the edited envelopes | Share at 6–40 Hz, re the sentence's |
|---|---|
| The sentence's own (`tfs()` of its bands) | −3.5 dB |
| Noise's | −4.5 dB |
| Steady tones at the band centres | −14.9 dB |

The first two lose most of the edit, as badly as the one-shot STFT route of
C4. A band's fine structure is not free of envelope: its phase changes as
fast as the band is wide, and re-filtering turns those changes back into
envelope fluctuations (the same mechanism as C3). Only a carrier that is
steady in each band keeps the edit. So the envelope route does not escape
the problem C4 shows for the STFT; it moves it into the choice of carrier.

**C7. A search through the filterbank keeps the edit, on any fine
structure.** [check] Griffin & Lim's iteration with the filterbank in
place of the STFT, on the C6 edit: impose the target magnitudes with the
current modulation phase, put the envelopes on the current fine structure,
re-filter, then take the new sound's own fine structure and modulation
phase.

| Start | Iterations | Share at 6–40 Hz, re the sentence's | ‖ \|MPS\| − target ‖ / ‖ target ‖ |
|---|---|---|---|
| The sentence's own phases | 0 | −3.5 dB | 0.38 |
| | 5 | −11.4 dB | 0.16 |
| | 20 | −16.5 dB | 0.10 |
| Noise's phases | 0 | −4.1 dB | 0.51 |
| | 5 | −10.7 dB | 0.26 |
| | 20 | −14.0 dB | 0.18 |

From the sentence's own phases, 20 iterations do better than steady tones
in one shot (−14.9 dB, C6) and keep the sentence's own fine structure.
Each iteration is one analysis and one synthesis (about 0.7 s for this
2.5 s sentence in this container, an estimate elsewhere). As for the
STFT (C5), the mismatch falls more and more slowly and does not reach zero.

## Ways to specify a target

Each way, what it fixes, what it leaves free, and how it gets to a sound.

**1. Parametric shapes (ripples).** A rate, a density, a depth and a
phase per component; `Ripple`, `RippleSum` and `DynamicRipple` exist. Each
knob has a meaning, which makes this the didactic starting point, and it
fixes more than a modulation spectrum: the phase is a parameter. Moving
ripples are the stimuli of Kowalski, Depireux & Shamma (1996) and Chi et
al. (1999); the dynamic moving ripple of Escabí & Schreiner (2002) is a
random path through the same plane. Route to sound: `ripple_sound`, exact
for the envelope pattern. Free: the carrier (C3).

**2. Drawing the surface.** Paint power over rate × density (item 1),
on a grid with rate signed (positive for downward sweeps, as in
`Ripple`) and density non-negative, since the other half-plane mirrors it.
A drawing also needs, outside the plane: the long-term spectrum, the
depth (C2), the duration, the carrier, and a seed for the phase. Route to
sound: random phase gives an envelope pattern with exactly the drawn
spectrum (C2), put on a carrier (see `to_sound(carrier=...)` below). Free: the phase (C1), so every
seed is a different sound with the same drawing.

The closest published precedent found is Hsu, Woolley, Fremouw &
Theunissen (2004): synthetic log spectrograms built from ripple components
"with modulation phase randomly assigned", with rates and densities drawn
from the modulation spectrum of song, then turned into sound with a
spectrographic inversion [source]. That is ways 1 and 2 combined, with a
search at the end. No paper was found that draws the phase on the 2-D
transform grid directly, as the checker does; the search was not
exhaustive.

Drawing item 2 instead (frequency × rate) is the same route band by band,
with no spectral modulation: each band gets an envelope with that band's
row as its power spectrum, and independent phases make the bands
uncorrelated (inferred). That is what a "modulation-filtered noise" per
band is.

**3. Measure and edit.** Analyse a real sound, change its modulation
spectrum (remove a rate band, a density band, a quadrant, or multiply by a
random mask), and go back. Elliott & Theunissen (2009) did this to speech
to find which modulations intelligibility needs: comprehension was
significantly impaired when temporal modulations below 12 Hz or spectral
modulations below 4 cycles/kHz were removed [source]. Venezia, Hickok &
Richards (2016) are reported to do the same with random masks ("bubbles");
that was not checked. The edit keeps the sound's
own phase, so its timing survives where the edit leaves the spectrum alone.
Two routes back:

- Edit the *envelopes* (item 1 of the sonore analysis, a 2-D transform of
  the `Envelopes` array, which is exactly invertible on that array), then
  put them on a carrier as the vocoder already does,
  `(edited_envelopes * carrier.tfs()).to_sound()`. One step, exact for
  the envelopes, but the edit survives in the sound only on a steady
  carrier (C6).
- Edit a *log spectrogram* and search for a signal (C4, C5). Elliott &
  Theunissen exponentiate the edited log spectrogram and invert it "using
  an iterative spectrogram inversion algorithm" [source]; their text, as
  far as it was read, does not name Griffin & Lim, so C5 stands in for
  their method rather than reproducing it. Chi, Ru & Shamma (2005) outline
  several algorithms to resynthesize sound from their cortical (rate ×
  scale) model output [source], which is the same search one level up.

Filtering a noise carrier in the modulation domain is this route with
noise as the measured sound.

**4. Statistics instead of a spectrum.** The texture route of McDermott &
Simoncelli (2011): per-band modulation power (item 2) plus envelope
moments, correlations across bands and across modulation bands. sonore can
already do this: `TextureStats.replace(mod_power=...)` and
`texture.synthesize`, which searches for a sound by gradient steps. It
pins down more than item 2 (the correlations give some of the phase
structure across bands), and it is slow, about 0.4 s per iteration per
second of sound (from the `synthesize` docstring). Not measured here.

## Turning a target into a sound: `to_sound(carrier=...)`

Cho's proposal (2026-10-03): as envelopes go back to sound on a carrier,
a modulation spectrum should go back on a carrier too, which supplies what
the spectrum lacks. `philosophy.md` already says that `to_sound` takes
as arguments exactly what a view discarded, and `Spectrum.to_sound(carrier=...)`
is the model: the spectrum gives magnitudes, the carrier gives phase. Its
"possible carriers are too many" objection to a modulation spectrum's
`to_sound` is the same for a spectrum, and the answer is the same: make
the carrier an argument.

A modulation spectrum lacks phase at two levels, and the carrier supplies
both:

1. **The modulation phase** (when events happen, how bands line up): the
   phase of the 2-D transform of the carrier's envelope array. The target
   supplies the magnitudes; the inverse transform gives envelopes with
   exactly the target spectrum, clipped at zero where needed (C1, C2).
2. **The fine structure** (the phase within each band): the carrier's
   band phases, or a steady tone per band (C3, C6).
3. **The envelope mean**: the 2-D transform is taken after removing one
   number, the mean over the whole array, so the long-term spectrum is
   kept in the transform's zero-rate column. The spectrum stores that
   number too, and the carrier does not have to supply it (a change made
   while implementing D2).

A person thinks in magnitudes on a time-frequency grid (Cho, 2026-10-03),
so both phases start random and can then be improved by a search that
makes them consistent with the analysis, as Griffin & Lim do for a
spectrogram (C5). The one method then covers the ways above:

- a drawn or parametric target with `carrier="noise"` or a seed: random
  modulation phase (way 2);
- the sound's own, edited spectrum with the sound as carrier: its own
  modulation phase (way 3, envelope route);
- `iterations=n`: a consistency search after the one-shot result (way 3,
  search route).

What the measurements say about the defaults:

- One shot keeps the target exactly on the envelopes (C2) but not in the
  sound: a re-analysis finds it only if the fine structure is steady in each
  band (C3, C6). Steady tones at the analysis band centres are the only
  fine structure measured here that keeps it (−14.9 dB against −3.5 and
  −4.5 dB for speech and noise fine structure).
- A search improves agreement with the analysis it searches through (C5),
  slowly, and never exactly, even for a target a real sound has.
- The result is only defined with respect to one analysis: the filterbank
  and envelope settings the target was measured or drawn on. The
  `ModulationSpectrum` would have to keep those (today it keeps only the dB
  level of half the plane, after a Hann taper), and `to_sound` would refuse
  a carrier analysed differently.

## Fit with the filterbank redesign

`docs/design/frames/filterbanks.md` makes masks views: gains on a frame's
coefficients, applied as `(mask * coefficients).to_sound()`. A
modulation-domain edit is the same idea one level down, gains on the 2-D
transform of an `Envelopes` array, but that transform is not a frame of the
sound, so it cannot be a `Mask` on subbands. With D2, an edit is a change
to the `ModulationSpectrum`'s magnitudes followed by
`to_sound(carrier=the_sound)`; inside, the envelopes are rebuilt and go
back by the existing route for envelopes (a fine structure, then
`Subbands.to_sound()`). Ripple patterns already render on an octave-scale
bank (`bank.scale.name == "octave"` after the redesign), and a drawn
pattern would use the same check.

TrackDraw, a separate future project that may draw tracks or surfaces and
call sonore, would need only a grid in, a sound out. That is context, not
a requirement on this design.

## Decisions

All accepted 2026-10-03.

- **D1. How a target is specified**: in code, as a few blobs on the rate
  (Hz, signed) × density (cycles/octave) plane, each with a centre, two
  widths, a direction and a level; a ripple is a blob of zero width. Or
  as a measured spectrum, edited. A painted grid is accepted too, but a GUI
  for drawing is left to TrackDraw, which would hand over the same blob list
  or grid. Accepted 2026-10-03.
- **D2. `ModulationSpectrum.to_sound(carrier=..., iterations=0, rng=...)`**,
  with the carrier supplying the modulation phase and the fine structure,
  as above (the mean is stored with the spectrum). `iterations` comes
  with D7. `carrier` is a `Sound`, `"noise"` (random
  modulation phase and noise fine structure) or `"tones"` (random
  modulation phase, steady tones at the band centres). Accepted
  2026-10-03 (Cho's proposal). It needs `ModulationSpectrum` to keep the untapered magnitude
  on the full plane and its analysis settings (D4).
- **D3. `philosophy.md` changes**: the modulation spectrum moves from
  "raises `NotInvertibleError`" to the views whose `to_sound` takes what
  they discarded, with the carrier as that argument. Accepted 2026-10-03.
- **D4. `ModulationSpectrum` keeps its analysis**: the filterbank, envelope
  rate, compression and linear or dB scale, and the untapered magnitude, so
  `to_sound` can rebuild envelopes on the same grid and refuse a mismatched
  carrier. The display keeps its Hann taper. Accepted 2026-10-03.
- **D5. Default carrier `"tones"`**: steady tones at the band centres, the
  only fine structure measured here that keeps the target in the sound
  (C3, C6). Accepted 2026-10-03. Whether `ripple_sound`'s own default carrier
  should change for the same reason is separate: measure the beat floor on
  sonore's code first.
- **D6. Depth**: when imposing magnitudes pushes envelopes below zero,
  `to_sound` clips and reports the fraction clipped (15% for the C6 edit,
  33% for the C1 twin), rather than refusing. For drawn targets, an
  `rms_depth` argument scales the spectrum and refuses a depth that cannot
  be reached without clipping (C2). Accepted 2026-10-03.
- **D7. The consistency search** (`iterations > 0`): Griffin–Lim through
  the filterbank (impose the target envelopes, re-analyse, keep the new
  fine structure and modulation phase, repeat). Accepted 2026-10-03,
  as a second step after measuring it; measured in C7 and implemented as
  `to_sound(iterations=...)`.
- **D8. The texture route needs no code**: document
  `TextureStats.replace(mod_power=...)` as the per-band way to specify
  modulation. Accepted 2026-10-03.
- **D9. A gallery page** ("Drawing a modulation spectrum") showing C1 (a
  sentence and its twin, heard), a drawn blob on the three carriers, and an
  edit. Accepted 2026-10-03, once D2 exists.

## References

Checked on 2026-10-03 through publisher pages and Crossref records; PubMed,
PMC and the AIP site were not reachable from the project's environment, so
some rows are partial.

| Source | Status |
|---|---|
| Elliott & Theunissen (2009), "The modulation transfer function for speech intelligibility", PLoS Comput. Biol. 5(3), e1000302. doi:10.1371/journal.pcbi.1000302 | Checked against the full text: MPS of the log-amplitude spectrogram, the 12 Hz and 4 cycles/kHz result, iterative spectrogram inversion |
| Hsu, Woolley, Fremouw & Theunissen (2004), "Modulation power and phase spectrum of natural sounds enhance neural encoding performed by single auditory neurons", J. Neurosci. 24(41), 9201–9211. doi:10.1523/JNEUROSCI.2449-04.2004 | Checked against the journal page: random modulation phase, rates drawn from song's modulation spectrum, spectrographic inversion. DOI not confirmed on Crossref |
| Chi, Ru & Shamma (2005), "Multiresolution spectrotemporal analysis of complex sounds", JASA 118(2), 887–906. doi:10.1121/1.1945807 | Bibliographic details and abstract checked ("several reconstruction algorithms"); full text not read |
| Atlas & Shamma (2003), "Joint acoustic and modulation frequency", EURASIP J. Appl. Signal Process. doi:10.1155/S1110865703305013 | Abstract checked ("added constraints ... to provide invertibility"); volume and pages not confirmed (2003(7), 668–675 from memory) |
| Griffin & Lim (1984), "Signal estimation from modified short-time Fourier transform", IEEE Trans. ASSP 32(2), 236–243. doi:10.1109/TASSP.1984.1164317 | Bibliographic details checked; the paper itself not read. The note uses only the iteration's definition, which C5 implements |
| Kowalski, Depireux & Shamma (1996), J. Neurophysiol. 76(5), 3503–3523. doi:10.1152/jn.1996.76.5.3503 | Checked (abstract: moving ripples) |
| Chi, Gao, Guyton, Ru & Shamma (1999), "Spectro-temporal modulation transfer functions and speech intelligibility", JASA 106, 2719–2732 | Volume and pages checked via a citing record; DOI and the use of ripples not confirmed here (already in the README) |
| Escabí & Schreiner (2002), J. Neurosci. 22(10), 4114–4131. doi:10.1523/JNEUROSCI.22-10-04114.2002 | Checked (abstract: dynamic moving ripple) |
| McDermott & Simoncelli (2011), Neuron 71, 926–940 | Checked for `filterbanks.md`; this note relies only on sonore's own implementation of it |
| Venezia, Hickok & Richards (2016), "Auditory 'bubbles' ...", JASA 140(2), 1072– | Title and venue from search results only; the claim about random masks not checked |
| Singh & Theunissen (2003), JASA 114(6), 3394–3411. doi:10.1121/1.1624067 | Checked (abstract): natural sounds concentrate modulation power at low rates and densities. Not cited above for the log definition, which the abstract does not state |
