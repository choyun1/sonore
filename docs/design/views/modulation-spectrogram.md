# Modulation spectrogram

The design of a time-dependent modulation spectrum for `sonore.analysis`:
for every band of a filterbank, how strongly its envelope is modulated at
each rate, in every time window. An STFT shows how the power spectrum of a sound
changes over time; this shows how its modulation spectrum changes over time.

sonore already has two static views of modulation. `ModulationSpectrum`
(`representations.py`) is the 2-D Fourier transform of a whole envelope
array, temporal rate against spectral density, one number per cell for the
whole sound. The texture statistics (`texture/stats.py`) hold `mod_power`,
the power of each band's envelope in 20 constant-Q modulation bands
(`ConstantQModulationFilterbank`, Q = 2), again averaged over the whole
sound. Neither says *when* a modulation is there. Speech is the obvious case:
its syllabic modulation near 4 Hz comes and goes with the syllables, and the
mobile idea in the README (the modulation spectrum of everyday sounds, live)
needs a version that runs block by block.

Status: accepted 2026-10-01, with decisions D1–D9 as recommended below
(D1 with the kernels as their own modulation filterbank). Patch 2 is
implemented: `HannModulationFilterbank` in `src/sonore/views/modulation.py`
and `ModulationSpectrogram` in `src/sonore/views/modspectrogram.py`,
tested in `tests/views/test_modspectrogram.py`. Patch 3 (display) is
implemented too: `plot`, `pooled_depth`, `slices` and `animate`, with the
plot functions in `src/sonore/plotting.py`. Patch 4, the gallery page, is
`docs/gallery/seeing/modspectrogram.py`. All four patches are merged.

## How the claims are verified

As in `frames/frames.md` and `views/cepstrum.md`, each claim is numbered and tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_modulation_spectrogram_claims.py`.
  The script uses only NumPy, SciPy and soundfile (to read the gallery
  sentence), writes every filter and transform out from its formula, and
  shares no code with sonore. It runs in about 30 s. The numbers below come from NumPy 2.4.6 and SciPy 1.17.1.
- **[source]**: a published result (see References).

## What "modulation spectrogram" has meant

The name covers three different displays in the literature, and the design
should say which one it computes.

1. **Greenberg & Kingsbury (1997); Kingsbury, Morgan & Greenberg (1998).**
   Critical-band envelopes, each passed through a modulation filter tuned to
   slow, syllabic rates, displayed as acoustic frequency against time. The
   1998 abstract (verified) describes "low-frequency (below 16 Hz) amplitude
   modulations in subband channels following critical-band frequency
   analysis", with automatic gain control. It is a spectrogram seen through
   one modulation band, not a modulation spectrum over time. The exact
   filter (from memory, a band centered near 4 Hz) is not verified.
2. **Atlas & Shamma (2003), "joint acoustic and modulation frequency".** For
   a stretch of sound, acoustic frequency against modulation frequency: the
   modulation spectrum of every band at once. Computed over sliding windows
   it becomes time-dependent. The abstract (verified) notes that the
   representation needs added constraints to avoid interference terms and to
   be invertible. Sukittanon, Atlas & Pitton (2004) use it for content
   identification.
3. **Modulation rate against time**, one band or all bands pooled: the
   direct analogue of a spectrogram, and what Cho asked for.

All three are slices of one array: a modulation power for every time window,
acoustic band and modulation band. The proposal computes that array once and
offers each display as a view of it (D7).

## Setting

The input is a set of band envelopes e_b[n] at an envelope rate f_E (sonore's
`Envelopes`, for example from an `ERBFilterbank` resampled to 1000 Hz). For a
modulation band with center f_k, the analysis uses a **complex kernel**: a
Hann window w_k of length L_k samples times a complex exponential,

    h_k[j] = w_k[j] exp(i 2π f_k (j − (L_k − 1)/2) / f_E),   Σ_j w_k[j] = 1,

and for every time window n correlates it with the envelope over the kernel's
support:

    y_bk[n] = Σ_j e_b[n + j − (L_k − 1)/2] conj(h_k[j])     (centered)
    μ_bk[n] = Σ_j e_b[n + j − (L_k − 1)/2] w_k[j]           (local mean)

|y_bk| is the local amplitude of the envelope's component near f_k, μ_bk the
envelope's local mean over the same window. Two derived quantities:

    modulation power   P_bk[n] = |2 y_bk[n]|²
    modulation depth   m_bk[n] = 2 |y_bk[n]| / μ_bk[n]

The factor 2 makes a sinusoidal AM e = A(1 + m cos 2π f_k t) read P = (Am)²
and depth m, so 100% modulation is 0 dB.

There are two ways to choose L_k:

- **Constant-Q** (the recommendation): L_k holds a fixed number of cycles of
  f_k, here 3, so L_k = 3 f_E / f_k. Low rates get long windows and fine
  frequency resolution; high rates get short windows and fine time
  resolution. This is a wavelet transform of the envelope, the shape of the
  auditory modulation filterbank (Dau et al., 1997) and of the texture
  statistics' modulation bands.
- **Fixed window**: every L_k = T f_E, with f_k on a linear grid. This is an
  STFT of each envelope, the "sliding modulation spectrum" in its literal
  form, with the same resolution at every rate.

A kernel defined in time, with finite support, is what makes causal and
streaming versions well defined (C8). sonore's existing modulation banks are
defined in frequency and applied circularly, which is right for textures
(seamless loops) but has no finite latency.

## Claims

**C1. A Hann kernel of length T has a −3 dB bandwidth of 1.44 / T and its
first null at 2 / T; with three cycles, Q = 2.09.** [proof, check] The
Hann window's transform is the classic three-term sum of sincs (Harris,
1978). Checker: full −3 dB bandwidth × T = 1.441 and first null × T = 2.000,
at T = 0.25 and 1 s. A kernel of c cycles at f_k has T = c / f_k, so
Q = f_k / bandwidth = c / 1.44 = 2.09 for c = 3, close to the Q of 2 of the
texture bank and of Dau et al.'s filters above 10 Hz. Kernel lengths: 6 s at
0.5 Hz, 0.75 s at 4 Hz, 47 ms at 64 Hz.

**C2. A kernel holding a whole number (≥ 2) of cycles ignores the envelope's
mean exactly.** [proof, check] The DC response of h_k is the Hann window's
transform at c bins from its center, and the Hann transform is zero at every
integer bin from 2 on. This matters because the mean is the largest part of
any envelope (an envelope is non-negative), so any DC leakage would show up
as modulation. Checker: DC gain 3e-17 for 2, 3 and 4 cycles, against −6.0 dB
at 1 cycle, −15.4 dB at 1.5 and −32.3 dB at 2.5. With lengths rounded to
whole samples at 1000 Hz, the worst DC gain over the half-octave bands
0.5–64 Hz is −65 dB.

**C3. A 4 Hz AM tone shows up in the right audio band, the right modulation
band and the right stretch of time, with the right depth.** [check] A 1 kHz
tone, 4 s long, amplitude-modulated at 4 Hz with depth 0.5 from 1 to 3 s
only. Envelopes from 24 half-cosine ERB-spaced bands (100–7000 Hz) at
1000 Hz; half-octave modulation bands 0.5–64 Hz; 10 ms hop.

- The largest modulation power, averaged over 1.5–2.5 s, is in the audio
  band centered at 1052 Hz and the 4 Hz modulation band.
- The depth there is −6.03 dB (20 log10 0.5 = −6.02), constant to 3e-11 dB.
  The neighboring modulation bands read −10.6 dB (5.66 Hz) and −15.8 dB
  (2.83 Hz): with Q ≈ 2, half-octave bands overlap, so a pure modulation
  spreads into its neighbors, as a pure tone does in a ⅓-octave analysis.
- Outside the modulated stretch, the depth is below −160 dB.
- The depth reaches half its value at 1.01 s and falls back at 3.01 s: the
  centered kernel puts the onset and offset where they are.
- The band below (903 Hz) also passes the tone, through its skirt. It reads
  the same depth (−6.03 dB) at 4.7 dB less power. Depth says how modulated a
  band is; power says how much of the sound that is. Both are needed (D3).

Both ends of the sound are abrupt (the envelope jumps from zero), and within
half a kernel of either end every band reads strong modulation. That is
true of the signal as analyzed, but rarely what one wants to see (D5).

**C4. A gliding modulation rate is tracked to within a few hundredths of an
octave.** [check] The same carrier with an AM rate gliding exponentially from
2 to 32 Hz in 6 s (two thirds of an octave per second), depth 0.5. The rate
estimate is the largest depth over quarter-octave bands, refined by a
parabola on the log axis. Constant-Q (3 cycles): median error 0.021 octave
(max 0.034) while the rate goes from 3.2 to 6.3 Hz, and 0.017 (max 0.031)
from 6.4 to 20 Hz. A fixed 1 s window does as well on this glide (0.014 and
0.013 median, 0.049 and 0.024 max). A fixed 0.25 s window can only look above
8 Hz (its DC lobe reaches 2/T = 8 Hz, C1) and errs by up to 0.17 octave.

**C5. Separating two modulation rates Δ Hz apart needs a window of about
2 / Δ seconds.** [check] One envelope modulated at 4 and 5 Hz at once. With
a fixed window, the depth at 4.5 Hz relative to the peaks is +0.13 dB at
T = 1 s (not resolved), −3.9 dB at 2 s, −13.3 dB at 3 s. The constant-Q
band at 4 Hz is 1.9 Hz wide, so it does not resolve them either; it should
show their 1 Hz beat as a fluctuating depth instead (inferred, not
checked). This is the time-frequency
trade-off of any spectrogram, one level down: to tell 4 from 5 Hz you must
watch for 2 s, longer than a syllable. That is why speech's modulation
spectrum is shown in octave-wide bands.

**C6. A band's envelope holds almost no modulation above the band's width.**
[proof, check] The envelope of a band W Hz wide is built from differences
between its components, all below W. Checker, Gaussian noise in a
rectangular band W wide at 1 kHz: 1.1% of the envelope's AC power lies above
W (25% above W/2), for W = 50 and 200 Hz alike. ERBs are 38 Hz at 125 Hz,
79 Hz at 500 Hz, 133 Hz at 1 kHz and 457 Hz at 4 kHz, so a 64 Hz modulation
is invisible in the lowest auditory bands no matter how strong it is in the
sound. Cells above a band's width should be shown as "not measurable", not
as "unmodulated" (D4).

**C7. Averaged over time, the modulation power is the static band power of
the envelope.** [proof, check] By Parseval, the mean over time of |y_k|²
(circular correlation) equals Σ_f |E(f)|² |H_k(f)|² / N². Checker: the two
agree to 9e-16 relative, over all half-octave bands. So the time average of
the new representation is the per-band modulation power spectrum, the same
kind of quantity as the texture statistics' `mod_power`, with slightly
different filter shapes. The spectrogram unfolds that statistic in time.

**C8. A causal version is the centered one delayed by half a kernel, and
block-by-block processing reproduces it exactly.** [proof, check] The
causal output uses the kernel's support ending at the current sample, which
is the centered one shifted by (L_k − 1) / 2. A block processor keeps the last
L_k − 1 envelope samples and evaluates the kernel at each window time.
Checker: 64-sample blocks against the offline causal result, worst
difference 2e-16; causal against shifted centered, 0. The causal latency is
half the kernel: 3.0 s at 0.5 Hz, 0.37 s at 4 Hz, 23 ms at 64 Hz. Evaluated
directly every 10 ms, the 15 half-octave bands cost 6.1 million
multiply-adds per second per audio band, or about 150 million for 24 bands:
light for a phone, and FFT convolution or decimating the slow bands would cut
it further.

**C9. The modulation spectrogram is not invertible as stored; the complex
coefficients would come close for the envelope, but not for the sound.**
[proof, check] Storing P (or depth) drops the phase of y and divides by the
local mean, as a magnitude spectrogram drops the STFT's phase. Keeping y
instead, the half-octave constant-Q bank is close to a tight frame: the
summed squared response varies by 0.15 dB over 1–32 Hz (0.013 dB with
quarter-octave bands), so a least-squares synthesis would give back the
envelope's 1–32 Hz content, and the fixed window at hop T/4 is an STFT of the
envelope whose squared Hann windows overlap-add to a constant (ripple
1e-15), so it inverts exactly. But an envelope is not a sound: getting a
sound back needs the fine structure as well (sonore already does
`env * subbands.tfs()`), and modulation filtering a sound is already
possible with the existing circular banks. Atlas & Shamma (2003) discuss the
constraints a joint representation needs to be invertible.

**C10. The choice of audio filter barely changes the picture; compression
changes its scale.** [check] There is no standard front end for a modulation
spectrogram, so the checker runs the same analysis through four: the
half-cosine ERB bank (sonore's `ERBFilterbank`) and 4th-order gammatone
filters (b = 1.019 ERB, causal phase) at the same 24 centers, each with
linear Hilbert envelopes and with envelopes raised to the power 0.3 (the
compression McDermott & Simoncelli use).

- The 4 Hz AM tone of C3 (depth 0.5) reads −6.03 dB with either bank. With
  0.3 compression it reads −16.0 dB with either bank, a depth of 0.16: the
  compressed envelope (1 + 0.5 cos)^0.3 is close to 1 + 0.15 cos. With
  linear envelopes, 2 audio bands carry modulation power within 10 dB of
  the loudest for both banks; with compression the gammatone's wider skirts
  bring that to 5 (cosine: still 2).
- The gallery sentence (`bdl_arctic_a0131`, half-octave rates 2–32 Hz,
  window times at least half a window from either end): with all four front ends
  the largest time-averaged pooled depth is in the 5.7 Hz band. The pooled
  depth at 4 Hz is −5.05 dB (cosine) and −5.05 dB (gammatone), and −11.7 and
  −11.6 dB compressed. Over all rate × time cells, the depth images in dB
  correlate 0.98 between the two banks, and 0.95 between linear and
  compressed envelopes. At 4 Hz, the band × time power images of the two
  banks correlate 0.98. Compression multiplies depth by 0.45 (median over
  cells).

So the filter shape is a minor choice for this representation, and the
compression is a major one: it rescales depth by a factor of 2 to 3 and has
to be stated with any number read off the picture.

## Decisions

**D1. A new module and type, and a modulation filterbank. (accepted 2026-10-01)** `sonore/analysis/modspectrogram.py` with a
`ModulationSpectrogram` class, exported as `so.ModulationSpectrogram`, built
from `Envelopes`: `so.ModulationSpectrogram(env, f_lo=0.5, f_hi=64,
per_octave=2, cycles=3, hop=0.010)`. It keeps the envelopes' filterbank (for
the acoustic axis) and stores `power` and `mean` of shape
`(n_channels, n_bands, n_mod, n_windows)`, with `f` the acoustic band centers,
`fm` the modulation band centers and `t` the window times. One way in, as for
`Cepstrum`. Recommended over a method on `Envelopes`
(`env.modulation_spectrogram()`), to keep `envelopes.py` (371 lines) about
envelopes; a thin method can be added later if notebooks want the chain.

Relation to existing types: it is to `Envelopes` what an `STFT` is to a
`Sound`. Its closest relative is `TFPower` (power on a time grid, no
inverse). `ModulationSpectrum` (rate × spectral density, whole sound) is a
sibling with different axes, and the texture statistics' `mod_power` is the
same quantity as its time average (C7). The windowed
kernels live in `modulation.py` as a third bank, `HannModulationFilterbank`,
with a time-domain `filter` that supports both alignments, so the
spectrogram is "envelopes through a modulation filterbank, sampled every
hop", as `Subbands` is a sound through a filterbank, and the kernels can be
used on their own (for example for modulation filtering).

**D2. The analysis window is a parameter, set in one of two ways. (accepted 2026-10-01)** As in
an STFT, the window length decides the trade between time and modulation
resolution (C1, C5), so it is the main knob. `cycles=3` (the default) gives
each modulation band a window of `cycles / f_k` seconds: long for slow rates,
short for fast ones; it must be a whole number of at least 2 so the
window ignores the envelope's mean (C2). `window=T` gives every band the same T seconds. The
spacing of the time windows, `hop` (10 ms), is separate from the window, as in an STFT. By
default, Hann
kernels with 3 cycles (Q ≈ 2.1, C1), whole cycles so the mean is ignored
(C2), half-octave centers from 0.5 to 64 Hz (15 bands). `window=T` [s]
switches to fixed-length kernels on a linear grid from 2/T (the first rate
clear of the mean's lobe, C1) to `f_hi` in steps of 1/T: the STFT of each
envelope, the didactic contrast of C5. Three cycles is the same rule the
cepstrum uses for F0 (three periods), and gives Q ≈ 2 as in the texture bank
and Dau et al. The envelope rate must be at least 3 × `f_hi` (the top band
reaches about 1.25 × `f_hi`); the constructor raises otherwise, and the docs recommend envelopes at 1000 Hz.
The existing `ConstantQModulationFilterbank` is not reused: it is defined in
frequency and circular, so it has no causal form (C8).

**D3. Store power and mean; show depth. (accepted 2026-10-01)** `power` (envelope units squared)
and `mean` are stored; `depth` is a property, 2|y| / mean, with 0 dB = 100%
sinusoidal modulation (C3). Plots default to depth in dB, because it is
comparable across bands and sounds; power is what to use when loud and quiet
bands should count differently (C3's 903 Hz band). Linear envelopes are
analyzed as they are; `Envelopes` already offers dB conversion where wanted.

**D4. Acoustic bands from any filterbank; unmeasurable cells marked. (accepted 2026-10-01)** Any
`Envelopes` works, so the acoustic axis is whatever bank made them (ERB,
octave, gammatone); edge bands are dropped, as in `modulation_spectrum`.
The class does not choose a front end. The default in the docs, the gallery
and the API sketch is `ERBFilterbank` with linear Hilbert envelopes: it is
what the rest of sonore uses, the gammatone gives nearly the same picture
(C10), and with linear envelopes depth keeps its textbook meaning (100% AM
is 0 dB). Compressed envelopes are a documented alternative, with C10's
factor stated, not a default. A
boolean `valid` of shape `(n_bands, n_mod, n_windows)` is False where the
modulation rate exceeds the band's width (C6) and within half a kernel of
either end of the envelopes (C3), and plots gray those cells. The band
width comes from the filterbank's own responses (its −3 dB width), not from
a formula, so it is right for every bank.

**D5. Centered time windows offline; ends not padded away. (accepted 2026-10-01)** Offline analysis is
centered (`align="center"`), so a modulation shows up where it happens (C3)
and bands with different kernel lengths line up. `align="causal"` gives what
a live analysis would see (C8). Outside its extent the envelope is taken to
be zero, so the abrupt start and end of a sound read as modulation; those
time windows are marked invalid (D4) rather than hidden by mirroring or tapering,
which would invent signal. Time windows every 10 ms by default.

**D6. Streaming: designed for, built later. (accepted 2026-10-01)** The block formulation of C8
is the contract: a later `ModulationTracker` would keep the last L_k − 1
envelope samples per band and emit time windows as blocks arrive, matching
`align="causal"` exactly. Not in this step, because the audio filterbanks
in sonore are FFT-based and whole-signal, so a live version also needs a
causal audio stage (gammatone as IIR, envelope by rectify and lowpass). That
belongs with the phone work (`notes/distribution/pypi-browser-phone.md`).

**D7. Display. (accepted 2026-10-01)** `msg.plot()` draws modulation rate (log axis) against time,
depth in dB, pooled over acoustic bands as 2 sqrt(Σ_b |y_b|²) / sqrt(Σ_b μ_b²),
which weights bands by their level. `plot(band=...)` shows one acoustic band.
`plot(rate=4)` shows acoustic band against time at one modulation rate,
Greenberg and Kingsbury's display. `msg.at(t)` returns the acoustic band ×
modulation rate image at one time, Atlas and Shamma's joint display, and
`msg.average()` the time average (C7; named `average` because `mean`
is the stored local mean). All via `sonore.plotting`, like the
other representations.

The data is a cube per channel (time × acoustic band × modulation rate), and
two more views show it as one. `msg.animate(path)` writes a video of the
band × rate image moving with the sound, with the audio track (through
matplotlib's animation writer, which needs ffmpeg), so the joint
display plays like a moving 2-D modulation spectrum. `msg.slices(t)` draws
three linked cuts through the cube with a shared cursor at time t: rate
against time and band against time (each with the cursor as a line), and
band against rate at the cursor. A filled 3-D volume plot is not
recommended: the inner cells hide behind the outer ones. Note that the
band × rate image is not the existing `ModulationSpectrum` (rate × spectral
density, a 2-D Fourier transform); a moving version of that one is the
cortical view left out of scope.

**D8. No inversion. (accepted 2026-10-01)** No `to_sound` or `to_envelopes` (C9). The docstring
says so and points to the circular modulation banks for modulation
filtering. If a later step wants modulation-domain editing, keeping the
complex y is the route, and C9 says it would work for the envelope.

**D9. Docs. (accepted 2026-10-01)** A README module-table row and a reference entry for each
source. A gallery page (the chirped AM of C4, the 4 Hz AM of C3, the gallery
sentence with its syllables showing at 2–8 Hz) comes in a follow-up PR once
the class exists. Its centerpiece is the three linked slices of D7 with the
cursor following the page's audio player, so the band × rate image changes
as the sound plays. The gallery pages are static HTML, so this needs a small
piece of JavaScript that swaps precomputed band × rate frames as the player
moves, the first interactive figure in the gallery.

## API sketch

```python
snd = so.load("docs/speech/bdl_arctic_a0131.flac")
fb = so.ERBFilterbank(n_bands=24, f_lo=100, f_hi=7000)
env = fb.analyze(snd).envelopes(fs=1000)

msg = so.ModulationSpectrogram(env)          # constant-Q, 0.5-64 Hz, 10 ms hop
msg.power.shape                              # (1, 24, 15, n_windows)
msg.plot()                                   # rate vs time, pooled over bands
msg.plot(rate=4)                             # acoustic band vs time at 4 Hz
img = msg.at(1.2)                            # bands x rates at t = 1.2 s
msg.slices(1.2)                              # three linked cuts, cursor at 1.2 s
msg.animate("sentence.mp4")                  # band x rate image playing with the audio

slow = so.ModulationSpectrogram(env, cycles=6)     # longer windows, finer rate resolution
fixed = so.ModulationSpectrogram(env, window=2.0)  # one 2 s window for all rates: STFT of each envelope
live = so.ModulationSpectrogram(env, align="causal")
```

## Tests (target: under 1 s added)

- C2: every default kernel's DC gain below −60 dB.
- C3: a 4 Hz, depth 0.5 AM tone reads −6.0 ± 0.1 dB in the right cell
  between onset and offset, and `valid` is False at the ends.
- C4: on a short glide, the tracked rate is within 0.1 octave.
- C7 holds for circular filtering and is left to the checker; the class
  filters with zero padding, so the tests check `filter` against `response`
  instead.
- C8: `align="causal"` equals `align="center"` delayed by half a kernel.
- D2: an envelope rate below 3 × `f_hi` raises.

## Patch plan

1. This document and `tools/check_modulation_spectrogram_claims.py`.
2. `HannModulationFilterbank` in `modulation.py`, then
   `ModulationSpectrogram` with `power`, `mean`, `depth`, `valid`, `at`,
   `average()` and both alignments; tests (D1–D5).
3. `plot`, `slices` and `animate`, README row and references (D7, D9).
4. Gallery page with the linked slices following the audio, separately.

## Out of scope

- Streaming and a causal audio front end (D6).
- Inversion and modulation-domain editing (D8).
- Spectrotemporal (rate × scale) analysis over time, as in Chi, Ru & Shamma
  (2005): the time-dependent version of `ModulationSpectrum`. A natural next
  step, but a 4-D cortical representation is a bigger design.
- Listener models: these are signal measurements, not models of modulation
  detection.

## References

Verified by lookup on 2026-10-01 unless marked otherwise.

- Atlas, L. & Shamma, S. A. (2003). Joint acoustic and modulation frequency.
  *EURASIP Journal on Applied Signal Processing* 2003(7), 668–675.
  [doi:10.1155/S1110865703305013](https://doi.org/10.1155/S1110865703305013).
  (Volume and DOI verified; pages from memory.)
- Chi, T., Ru, P. & Shamma, S. A. (2005). Multiresolution spectrotemporal
  analysis of complex sounds. *J. Acoust. Soc. Am.* 118(2), 887–906.
  [doi:10.1121/1.1945807](https://doi.org/10.1121/1.1945807). (DOI
  verified; issue and pages from memory.)
- Dau, T., Kollmeier, B. & Kohlrausch, A. (1997). Modeling auditory
  processing of amplitude modulation. I. Detection and masking with
  narrow-band carriers. *J. Acoust. Soc. Am.* 102(5), 2892–2905.
  [PubMed 9373976](https://pubmed.ncbi.nlm.nih.gov/9373976/). Modulation
  filters of constant 5 Hz bandwidth below 10 Hz and Q = 2 from 10 to
  1000 Hz.
- Greenberg, S. & Kingsbury, B. E. D. (1997). The modulation spectrogram: in
  pursuit of an invariant representation of speech. *Proc. ICASSP 1997*,
  vol. 3, 1647–1650.
  [Semantic Scholar](https://www.semanticscholar.org/paper/71c0095d37084b6055a1abc8d4edcde3ef9f130b).
  (Venue and pages verified; no DOI found.)
- Harris, F. J. (1978). On the use of windows for harmonic analysis with the
  discrete Fourier transform. *Proc. IEEE* 66(1), 51–83. (Not verified; C1
  is checked numerically.)
- Kingsbury, B. E. D., Morgan, N. & Greenberg, S. (1998). Robust speech
  recognition using the modulation spectrogram. *Speech Communication*
  25(1–3), 117–132.
  [doi:10.1016/S0167-6393(98)00032-6](https://doi.org/10.1016/S0167-6393(98)00032-6).
- Schimmel, S. & Atlas, L. (2005). Coherent envelope detection for
  modulation filtering of speech. *Proc. ICASSP 2005*.
  [IEEE Xplore 1415090](https://ieeexplore.ieee.org/document/1415090/).
  (Title and venue verified.) Why modulation filtering of Hilbert envelopes
  is subtle, relevant to D8.
- Sukittanon, S., Atlas, L. E. & Pitton, J. W. (2004). Modulation-scale
  analysis for content identification. *IEEE Trans. Signal Processing*
  52(10), 3023–3035. (Title verified; volume and pages from memory.)
- McDermott & Simoncelli (2011) and Singh & Theunissen (2003): already in
  the README, for the static modulation power and modulation spectrum.
