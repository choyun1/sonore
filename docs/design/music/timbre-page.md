# The Timbre page and three timbre descriptors

The Timbre page is the first page of the gallery's Music group, live now as a placeholder
(`docs/gallery/music/timbre.py`). Cho accepted the plan for its full version on 2026-10-04
(T1 to T3 in the plan note), and sent Peeters et al. (2011) and Grey (1977) along with a link
to the Timbre Toolbox repository. This note fixes how the three descriptors are defined and
what the page plays. It adds no code. The decisions at the end come first; they extend D9 and
D10 of `music-pages.md`.

Status: draft for Cho.

## How the claims are verified

As in the other design notes, each claim is tagged:

- **[proof]** is a short argument given here.
- **[check]** is a number printed by `tools/check_timbre_claims.py` (T-C1 to T-C5). It uses
  NumPy and SciPy, shares no code with sonore, and follows the equations in Peeters et al.
  (2011) as printed.
- **[source]** is a published work. The References section says how far each was read.

## Licence of the Timbre Toolbox

The Timbre Toolbox (MATLAB) is licensed for non-profit research only and forbids
sub-licensing (`_licence.txt` in the repository). sonore is MIT-licensed, so it cannot carry
ported toolbox code. As with the texture-synthesis gradients, sonore's descriptors are written
from the paper's equations (clean-room), and the toolbox is read only to see its default
settings. Where its code and the paper's text differ, this note follows the paper and says
so. One such case: the weakest-effort attack search in `Att.m` averages the efforts between
the 30% and 60% thresholds and searches fixed sub-ranges, while the paper averages all the
efforts [source].

## What the papers fix

- **Grey (1977).** 16 resynthesized instrument tones near E♭4 (about 311 Hz), 280 to 400 ms
  long, equalized for pitch, loudness and duration. INDSCAL gave a three-dimensional space.
  Its fit was 0.68 in two dimensions, 0.75 in three and 0.78 in four [check T-C5, copied from
  the paper]. The dimensions were interpreted as (1) spectral energy distribution, (2)
  synchrony of the higher harmonics' onsets together with spectral fluctuation over time, and
  (3) low-amplitude, high-frequency energy in the attack [source, read].
- **McAdams et al. (1995).** The same kind of space for 18 synthesized tones, with
  dimensions correlated with log attack time, spectral centroid and spectral flux (already in
  `music-pages.md` D9) [source, read].
- **Peeters et al. (2011).** These are the definitions most later work uses [source, read:
  sections II and III, and the onset paragraph of IV].
  - **Energy envelope:** the amplitude of the analytic signal, low-passed by a third-order
    Butterworth filter at 5 Hz.
  - **Attack:** the weakest-effort method. Thresholds run from 0.1 to 1 of the maximum, and
    the efforts are the times between successive crossings. The attack starts at the first
    threshold whose effort is below α = 3 times the mean effort and ends at the last one.
    LAT = log10(t_end − t_start).
  - **Spectral centroid:** Σ f_k p_k on each frame of a Hamming STFT (23.2 ms window,
    5.8 ms hop), with p_k the normalized magnitude or power.
  - **Spectral variation (flux):** 1 minus the normalized correlation of successive spectra.
  - **Summaries:** time-varying descriptors are summarized by their median and interquartile
    range, because silent frames make the mean and standard deviation meaningless.

## What the checker found

1. **The 5 Hz envelope hides short attacks [check T-C1].** On a harmonic complex at E♭4 with
   a linear attack, the paper's own setting (5 Hz, one pass, weakest effort) measures:

   | true attack | measured with 5 Hz, one pass | measured with 20 Hz, forward and back |
   |---|---|---|
   | 5 ms | 82 ms | 16 ms |
   | 20 ms | 125 ms | 24 ms |
   | 80 ms | 144 ms | 65 ms |
   | 300 ms | 324 ms | 244 ms |

   The paper itself switches to a 20 Hz forward-backward filter for onset detection (IV A),
   "to achieve a more accurate estimation of onset and offset for rapidly varying signals"
   [source]. A page that plays attacks from 5 to 300 ms needs that faster envelope, or its
   printed LAT would barely change across the shortest steps (D-T2). With 10% and 90%
   thresholds on a 50 Hz forward-backward envelope, a 5 ms ramp measures 8.1 ms against its
   true 4 ms between those levels (0.8 × 5 ms) [check T-C1, proof: a linear ramp spends 80% of
   its length between 10% and 90%].
2. **The magnitude centroid depends on the spectral floor [check T-C2].** For 20 harmonics
   falling as 1/n, the analytic magnitude centroid is 1730 Hz. The STFT median over the
   sustain is 1812 Hz, 4.7% higher, because the window's sidelobes spread small magnitudes
   over every bin up to Nyquist, and a magnitude centroid weights all of them. The power
   centroid of the same tone is 701 Hz. The two scales differ by a factor of 2.5 here, so the
   page must say which one it prints (D-T3).
3. **Brightness steps [check T-C3].** Amplitudes n^−s over 20 harmonics put the magnitude
   centroid at 8.1, 5.6, 3.5, 2.3 and 1.3 times F0 for s = 0.5, 1, 1.5, 2 and 3. Those five
   tones make the brightness section.
4. **Flux at one hop barely moves [check T-C4].** The median variation between frames one
   hop (5.8 ms) apart is 74 × 10⁻⁶ for a steady tone. For a tone whose spectral slope glides
   from 2 to 0.5 over a second (centroid 2.8 to 8.8 times F0), it is 94 × 10⁻⁶. Between
   spectra 100 ms apart the same two tones give 126 and 3205 × 10⁻⁶, which tells them apart
   by a factor of 25. Frame-to-frame flux mainly measures fast change. The slow change that
   McAdams et al.'s flux dimension describes shows up only at a longer spacing (D-T4).

## Page sections

The tones are synthetic, at E♭4, 1 s long, RMS-matched, and made with `so.harmonic_complex`
and an amplitude envelope per harmonic. The page says that RMS matching is not loudness
matching, because sonore has no loudness model.

1. **What timbre is**, kept from the placeholder, with one sentence on Grey (1977) as the
   first timbre space.
2. **Attack time:** one spectrum (s = 1) with linear attacks of 5, 20, 80 and 300 ms, each
   printed with its measured LAT.
3. **Brightness:** the five slopes of T-C3, each printed with its centroid.
4. **Spectral flux:** a steady tone next to the gliding tone of T-C4, with both flux values
   printed and the centroid track plotted.
5. **Three descriptors at once:** every tone of sections 2 to 4 plotted on LAT, centroid and
   flux. The page says the axes are descriptors, not a timbre space built from listeners'
   ratings.
6. **Onsets:** Siedenburg's (2019) three cases on one tone, with his numbers (already read).
7. **Vowels are timbres:** three Klatt vowels at one F0, with their centroids printed.

## Decisions

- **D-T1. Names and placement (accepted as T1).** `so.log_attack_time`,
  `so.spectral_centroid` and `so.spectral_flux` go in a new `views/descriptors.py`, written
  from Peeters et al. (2011), with no toolbox code.
- **D-T2. Energy envelope for the attack.** Recommended: the amplitude of the analytic signal,
  low-passed by a third-order Butterworth filter at 20 Hz, forward and backward, which is the
  paper's onset setting. The attack is found by the weakest-effort method with α = 3, as the
  paper's text gives it. `cutoff` stays a parameter, so `cutoff=5, zero_phase=False`
  reproduces the paper's descriptor setting. Alternative: the 5 Hz default, which T-C1 shows
  cannot tell 5 ms from 20 ms.
- **D-T3. Magnitude or power.** Recommended: the magnitude spectrum by default (the paper's
  STFTmag representation), with `scale="power"` available. Docstrings and the page print which one was used.
- **D-T4. Flux spacing.** Recommended: `spectral_flux(sound, spacing=0.1)`, the variation
  between spectra a set time apart, defaulting to 100 ms, with `spacing=None` meaning one hop,
  as in the paper. The docstring states the T-C4 result. Alternative: one hop only, as
  printed, at the cost of a section 4 whose two tones measure almost alike.
- **D-T5. What the functions return.** `log_attack_time` returns a number plus the start and
  end times (for drawing). The two time-varying ones return a small `DescriptorTrack` view
  (times, values, `median`, `iqr`), a one-way view under philosophy.md's "Views may discard
  information". Recommended: a track rather than a bare array, so the page can plot it and
  print the summaries the paper uses.

## References

- Grey, J. M. (1977). Multidimensional perceptual scaling of musical timbres. *J. Acoust.
  Soc. Am.* 61(5), 1270–1277. doi:10.1121/1.381428 (DOI from memory, not shown in the PDF). Read (PDF from Cho): abstract, stimuli,
  scaling results.
- Krimphoff, J., McAdams, S., & Winsberg, S. (1994). Caractérisation du timbre des sons
  complexes. II. Analyses acoustiques et quantification psychophysique. *J. Phys. IV* 4,
  625–628. Not read; cited through Peeters et al. (2011) and McAdams et al. (1995).
- McAdams et al. (1995), Siedenburg (2019): as in `music-pages.md`.
- Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011). The Timbre
  Toolbox: Extracting audio descriptors from musical signals. *J. Acoust. Soc. Am.* 130(5),
  2902–2916. doi:10.1121/1.3642604. Read (PDF from Cho): sections II and III in full, IV A.
- Timbre Toolbox, github.com/VincentPerreault0/timbretoolbox. Read for its licence and its
  default settings (`TEE.m` cutoff 5 Hz, `Att.m`); no code used.
