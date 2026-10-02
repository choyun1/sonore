# Female voices: a robustness check

Until now every speech example in sonore came from two male speakers of CMU
ARCTIC (`bdl`, `rms`). Female voices sit roughly an octave higher, so their
harmonics sample the vocal tract's response half as densely, and analyses
tuned on male speech can miss formant peaks, lock onto the wrong period, or
read missing harmonics as noise. This note records how sonore's speech
analyses do on female voices, with each analysis run at its defaults. Nothing
was tuned to improve the numbers.

Status: 2026-10-02. The one library fix the findings called for, in
`so.f0_track`, was accepted by Cho and is made here (see "F0"); the
numbers below are with it.

## How the numbers are made

`tools/check_female_voices.py` runs the library itself (`so.f0_track`,
`Cepstrum.f0` and the cepstral envelope, `so.cheaptrick`, `so.d4c`,
`so.harmonic_aperiodicity`) on three sets of sounds:

1. **Synthetic vowels**, where the truth is exact. There are three vowels
   (heed, hod, who'd) with male and with female average formants
   (Hillenbrand et al. 1995), each at a steady F0 from 100 to 350 Hz. They
   are built as `tools/check_world_claims.py` builds its test vowel, so the
   envelope and the share of noise (−30 dB at 0 Hz rising to −5 dB at
   8 kHz) are known at every frequency. The envelope and aperiodicity
   measures are given the true F0.
2. **The FDA database** (Bagshaw, CSTR Edinburgh): 50 sentences each from a
   male (`rl`, median F0 121 Hz) and a female (`sb`, median 253 Hz)
   speaker, with a laryngograph reference F0. It has no stated licence, so
   it is not in the repository.
3. **CMU ARCTIC**: `bdl` (3 sentences) and `slt` (4, including
   `docs/speech/slt_arctic_a0131.flac`). Their single-channel release has
   no EGG channel, so the only reference is WORLD's Harvest, which is not
   ground truth.

```bash
python tools/check_female_voices.py                    # part 1, under a minute
python tools/check_female_voices.py --fda path/to/fda_eval --arctic path/to/arctic
```

The FDA part takes about 9 minutes. The numbers below are from NumPy 2.4.6,
SciPy 1.17.1 and pyworld 0.3.5.

## F0

**On real female speech the tracker holds up.** Against the laryngograph:

| | Voicing error | Gross error (>20%) where both voice | Within 5% | Octave down |
|---|---|---|---|---|
| `so.f0_track`, rl (male) | 5.6% | 0.1% | 98.1% | 0.0% |
| `so.f0_track`, sb (female) | 1.5% | 0.5% | 95.2% | 0.1% |
| `Cepstrum.f0`, rl (male) | 11.7% | 1.2% | 95.1% | 0.0% |
| `Cepstrum.f0`, sb (female) | 6.6% | 2.3% | 91.5% | 1.5% |

The female voice is easier to voice, since its periodicity is strong and
regular. But fewer of its time windows land within 5% (95.2% against
98.1%), and the cepstrum, which judges each 40 ms time window alone, makes
octave-down errors on it that it doesn't make on the male voice. On sb the
tracker's gross errors at 200 to 300 Hz and above 300 Hz are 0.1% to 0.4%.
Where sb's reference F0 drops below 200 Hz (90 time windows, mostly at
voicing edges and in creak) a quarter or more are off, but there are too
few of those windows to support a conclusion.

On ARCTIC `slt` (Harvest median 176 Hz, so lower than sb), the tracker
agrees with Harvest within 5% on 98.6% of the time windows where both
voice. On `bdl` it is 99.3%. Both voices disagree with Harvest on voicing in about
24% of time windows, as expected, since Harvest voices many unvoiced ones
(`f0.md`, C9). The cepstrum disagrees with Harvest by more
than 20% on 53% of the 15 `slt` time windows Harvest puts at 250 to 300 Hz.

**On steady synthetic vowels from 250 Hz up the tracker locked onto a
third of F0, and this is now fixed.** Before the fix, `so.f0_track`
returned F0/3 (99 Hz for a 300 Hz vowel) on nearly every time window at
300 Hz on all six vowels, at 350 Hz on four and at 250 Hz on two. A
perfectly periodic sound has a difference-function minimum at every
multiple of its period, all about equally deep. The tracker kept only the
four deepest, which could all be multiples, and its subharmonic rule (C10
in `f0.md`) checked only the octave above a candidate, which says nothing
against F0/3. Real voices have jitter, so this didn't show on sb, but a
steady synthetic source such as the gallery's Klatt vowels at a female F0
would hit it. Extending the rule to thirds alone moved the error to F0/5,
because the true period was often not among the four. The tracker now
keeps eight candidates and applies the rule to every whole multiple. On all
36 synthetic vowels from 100 to 350 Hz the median error is now under 0.4%,
and the twelve at 400 and 450 Hz are tracked at the right F0 too. On FDA the numbers above barely moved (voicing
error 5.5% to 5.6% male, 1.6% to 1.5% female), and tracking takes about
twice as long. `tests/views/test_f0.py` holds a regression test that
fails on the old code.

## Spectral envelope

**The envelope gets worse as F0 rises, and formant peaks are missed when
they fall between harmonics.** The RMS dB error over 100 to 5000 Hz,
averaged over the six vowels, with overall level removed:

| F0 [Hz] | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|
| CheapTrick | 0.75 | 1.16 | 1.60 | 2.00 | 2.40 | 3.00 |
| Cepstral lifter (40 ms, half a period) | 1.92 | 3.17 | 3.66 | 3.92 | 4.25 | 4.69 |

At the F1 peak itself, CheapTrick's level is off by between +3 and −13 dB.
At 300 and 350 Hz it is 8 to 13 dB low in 8 of the 12 cases, because no harmonic sits
near the peak and the smoothing over F0 flattens it. The cepstral lifter
sits 6 to 15 dB under F1 from 150 Hz up. The male and female formant
sets give much the same numbers, so F0 drives the error, not the vowel.
This is the "sparse harmonics" problem, and no envelope estimator can fully
avoid it: between two harmonics the spectrum holds no information about the
vocal tract.

On real speech CheapTrick barely depends on small F0 errors: swapping the
laryngograph F0 for the tracker's changes the envelope by a median of
0.03 dB (male) and 0.05 dB (female), and 0.21 and 0.32 dB at the 95th
percentile.

## Aperiodicity

On the synthetic vowels, the median error of the noise share over 200 to
6000 Hz:

| F0 [Hz] | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|
| D4C [dB] | 6.4 | 5.4 | 4.3 | 3.0 | 3.1 | 3.2 |
| Harmonic residual [dB] | 1.9 | 2.0 | 2.2 | 2.9 | 2.7 | 3.1 |

The harmonic residual worsens slowly with F0, because a window four periods
long holds fewer samples to fit. D4C does better at higher F0 on this test,
not worse.

On real speech there is no ground truth for aperiodicity, so the noise
share given the laryngograph F0 is only described here, as medians by band:

| | 0–1 kHz | 1–2 kHz | 2–4 kHz | 4–7 kHz |
|---|---|---|---|---|
| D4C, rl (male) | −53.0 | −32.9 | −11.4 | −2.0 |
| D4C, sb (female) | −52.1 | −32.1 | −7.6 | −3.2 |
| Harmonic residual, rl (male) | −17.4 | −5.2 | −2.2 | −1.1 |
| Harmonic residual, sb (female) | −22.6 | −9.9 | −3.4 | −1.8 |

Two things show here. First, the harmonic residual calls almost everything
above 1 kHz noise on both voices. Its docstring explains why: it needs F0
to about 0.1%, and jitter, shimmer and an F0 track interpolated from pitch
marks are all well short of that. That is a limit of the measure on real
speech, not something specific to female voices. Second, neither measure
reads the female voice as clearly noisier. Female voices are often described
as breathier (Klatt & Klatt 1990), and with no reference it can't be said
whether these numbers are right.

## Follow-ups

1. **Gallery pages.** Done: every gallery page that uses recorded speech
   now has a female example next to the male one, with `slt`'s reading of
   the same sentence, and Formant synthesis has the six vowels with female
   formants.

## References

- Bagshaw, P. C., Hiller, S. M. & Jack, M. A. (1993). Enhanced pitch
  tracking and the processing of F0 contours for computer aided intonation
  teaching. *Proc. Eurospeech '93*, 1003–1006. (The FDA database.)
- Hillenbrand, J., Getty, L. A., Clark, M. J. & Wheeler, K. (1995). Acoustic
  characteristics of American English vowels. *J. Acoust. Soc. Am.* 97(5),
  3099–3111.
- Klatt, D. H. & Klatt, L. C. (1990). Analysis, synthesis, and perception
  of voice quality variations among female and male talkers. *J. Acoust.
  Soc. Am.* 87(2), 820–857.
- Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
  *Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224.
