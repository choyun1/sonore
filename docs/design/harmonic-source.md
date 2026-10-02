# Time-varying harmonic source

The design of a harmonic source whose fundamental follows a contour: a
sum of phase-locked harmonics driven by an F0 track, with the harmonics
switched off where the track is unvoiced, faded out before they reach
Nyquist, and optionally weighted by a spectral envelope. It is the
harmonic half of the "pulse-plus-noise synthesis" in roadmap item 1
(speech analysis and synthesis), and the piece that lets an F0 track and a
set of band envelopes be put back together and listened to.

Status: accepted 2026-10-02, with D2–D6 as recommended, D1 as the
pattern-matching form of `harmonic_complex` below, D7 with arbitrary
starting phases, D8 (after the F0 tracker) and D9. Implemented in
`src/sonore/signals/generators.py`, tested in `tests/signals/test_generators.py`.

## Why

Cho's question: with an F0 extraction and an envelope extraction, can
sonore impose the envelopes on harmonics built from the F0 contour, to hear
what the two carry between them? Most of that already exists:

- `so.noise_vocode(sound, n_bands, carrier=...)`, a channel vocoder
  (Shannon et al., 1995), takes any `Sound` as the
  carrier, splits it with the same ERB filterbank as the sound, and
  multiplies each band's fine structure by the sound's band envelope
  (`src/sonore/analysis/filterbank.py`). The envelope half is done.
- `docs/speech/bdl_arctic_a0131_f0.csv` holds a Harvest track (Morise,
  2017) of the gallery sentence (CMU ARCTIC `bdl`; Kominek & Black, 2004), and `so.f0_track` (`f0.md`) now tracks any sound.

What is missing is the carrier. `so.harmonic_complex` takes one fixed F0.
The gallery pages that need a moving F0 (`pv.py`, `resynthesis.py`) each
build one by hand with `np.cumsum`, which is slightly wrong (C2) and
leaves voicing and aliasing to the reader. A time-varying harmonic source
is also what WORLD-style synthesis needs (the voiced excitation, weighted
by a spectral envelope), and what the F0 tracker's gallery page needs to
let a reader *hear* a track, the quickest way to judge one.

## How the claims are verified

As in `cepstrum.md` and `f0.md`, each claim is numbered and tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_harmonic_source_claims.py`.
  The script uses only NumPy, SciPy and soundfile, shares no code with
  sonore, and holds a small prototype of the source described here. It
  runs in about a second. The numbers come from NumPy 2.4.6 and SciPy
  1.17.1.

The listening examples (below) are not claims; they are for Cho's ears.

## The design

Given F0 values f0[j] at window times t[j] (0 meaning unvoiced, the form of
the stored Harvest track and of `F0Track`):

1. **Fill the gaps.** Unvoiced time windows take F0 values interpolated linearly
   between the voiced time windows on either side; leading and trailing ones hold
   the nearest voiced value (D3).
2. **Interpolate to the sample rate.** Linear interpolation in Hz gives
   f(t) at every sample (D2).
3. **Accumulate the phase.** Φ(t) = 2π ∫ f, by the trapezoid rule, which is
   exact for a contour that is linear between samples (C2).
4. **Sum phase-locked harmonics.** x(t) = Σ_k a_k(t) · T(k f(t)) ·
   cos(k Φ(t) + φ_k). Every harmonic's phase is exactly k times the
   fundamental's, so the sound is harmonic at every instant (C3). T is a
   taper that fades harmonic k out as k f(t) approaches f_max (D5), and
   φ_k are starting phases from `harmonic_complex`'s presets (cosine by
   default).
5. **Gate by voicing.** Multiply by a gate that is 1 on voiced time windows and
   0 on unvoiced ones, its steps smoothed by a 5 ms Hann window (D3).
   Optionally fill the unvoiced stretches with white noise (D4).
6. **Normalize** to RMS 1, as every generator does.

## Claims

**C1. A constant contour gives the fixed-F0 harmonic complex.** [check]
A 220 Hz contour sampled every 5 ms, harmonics 1–10, matches
Σ cos(2π k 220 t) to 6.5e-12 (max abs difference, unit-amplitude
harmonics). So the new source is a generalization of `harmonic_complex`,
not a different sound. That is what lets `harmonic_complex` take both (D1).

**C2. The phase is exact for a contour that is linear between samples.**
[proof] If f is linear between t_{i-1} and t_i, its integral over the step
is the step times the mean of the endpoints, which is the trapezoid rule.
Linear interpolation of a contour that is linear between window times is exact,
so a linear glide is reproduced exactly. [check] A 100 → 300 Hz glide
given every 5 ms matches the closed-form chirp cos(2π(100t + 100t²))
to 4.5e-13. The running sum the gallery pages use, 2π Σ_{j≤i} f_j / fs,
differs from it by up to 0.078: it equals the trapezoid plus
π(f_0 + f_i)/fs (checked to 6e-13), a phase lead of about one sample of
the current frequency. Inaudible in those pages, but not exact.

**C3. Every harmonic follows k times the contour.** [proof] Harmonic k's
phase is k Φ, so its instantaneous frequency is k f(t), and the
interpolated f(t) differs from a smooth true contour g by at most
h²/8 · max|g''| between window times spaced h apart. [check] For a 150 Hz vibrato,
±4% at 5.5 Hz, given every 5 ms, that bound is 1.56e-4 (relative).
The instantaneous frequency measured from the analytic signal of
harmonics 1, 10 and 40 alone stays within 1.43e-4, 1.56e-4 and 1.56e-4
of k times the true contour, and harmonic 10's frequency divided by
harmonic 1's is 10 within 5.5e-4 (the Hilbert measurement's own error).

**C4. The phase has no jumps.** [check] On the stored Harvest track of the
gallery sentence (2.53 s, 16 kHz, harmonics to 7.2 kHz), each sample's
phase step equals 2π times the mean frequency over the step to 5e-13 rad.
The obvious shortcut, synthesizing each 5 ms time window with its own F0 against
absolute time, cos(2π k f0[j] t), puts 31 dB-below-total power above
7.6 kHz (clicks at every time window boundary); the accumulated phase puts
76 dB below.

**C5. Gaps must be filled before interpolating.** [check] The track's
lowest voiced F0 is 87.6 Hz. Interpolating straight through the zeros
of unvoiced time windows sweeps the pitch down to 11.5 Hz while the gate is
still open at voicing boundaries, an audible downward chirp at every
onset and offset; filling the gaps first keeps it at or above 87.6 Hz.
Ramping the gate over 5 ms lowers the power above 7.6 kHz from −68 dB
(hard 0/1 gate) to −76 dB.

**C6. The taper keeps every harmonic below f_max.** [check] An
exponential glide from 100 to 1000 Hz at 16 kHz: a fixed set of harmonics
chosen at the start (all 79 below Nyquist at 100 Hz) has 61% of its
power above Nyquist on average over the glide, where it aliases. With the
taper at f_max = 7.2 kHz, the number of harmonics falls from 71 to 7 as
the pitch rises, and the power above 7.6 kHz is 106 dB below the total.

**C7. Speed.** [check] The gallery sentence's track with harmonics to
7.2 kHz takes about 0.06 s per call in a plain Python loop over
harmonics, about 40 times faster than real time. A 60 Hz voice at
44.1 kHz needs about 330 harmonics, and the same 2.53 s then takes about
0.8 s (measured once, not printed by the checker): three times faster
than real time, which is enough for listening; the library version can
batch harmonics if it ever is not.

## Listening examples

`/mnt/project-files/notes/harmonic-source/make_examples.py` (project
files, not the repository) runs the prototype on the stored Harvest track
of the gallery sentence and writes short files beside itself, first:

1. the original sentence;
2. the source alone on the track (the track made audible);
3. a 16-band noise vocoder, for comparison;
4. the same 16 envelopes on the harmonic source, white noise in the
   unvoiced gaps (`so.noise_vocode(sentence, 16, carrier=source)`);
5. the same with 32 bands;
6. 16 bands on a monotone source (the median F0, 120.8 Hz);
7. 16 bands with the contour raised by a fifth (× 1.5);
8. 16 bands with the contour flipped about the median on a log scale
   (rises become falls).

As a rough check that the imposed pitch survives the vocoder, cepstral F0
(`so.Cepstrum.f0`, 50 ms windows) of each output is within 5% of the
imposed track on 89% (16 bands), 91% (32 bands) and 89% (raised a fifth)
of the voiced time windows, against 69% for the original recording and 0% for
the noise vocoder.

Six more files drive the same source from the two F0 estimators sonore
has, instead of the stored Harvest track: 9–11 from cepstral F0
(`so.Cepstrum.f0`, 40 ms Hann time windows, its defaults), the buzz alone and
the 16- and 32-band vocoders; 12–14 the same from `so.f0_track` at its
defaults. On this sentence the cepstral track voices 64% of the time windows
and agrees with Harvest within 5% on 96% of the time windows both voice;
`so.f0_track` voices 68% and agrees on 100%. Harvest voices 85–86%, so
both put noise where Harvest has harmonics in weak voiced stretches.

## Decisions

**D1. One function, `harmonic_complex`, whose `f0` is a number or a
contour (accepted, Cho 2026-10-02).** The pitch is one value that comes in
two shapes, and a `match` statement (Python 3.10, sonore's minimum) picks
the case:

- **A number** runs the existing code unchanged, so every fixed-F0 output
  stays bit-for-bit identical, and harmonics at or above Nyquist are still
  dropped with a warning.
- **A contour** runs the design above: a `(times, values)` pair, or any
  object with `.t` and `.f0` attributes, such as an `F0Track`. The match
  goes by shape, not by class, so the signals layer imports nothing from
  analysis (`tests/test_layers.py`), and the stored Harvest track works as
  a pair.

The signature stays `harmonic_complex(duration, fs, f0, harmonics=None,
...)`: a contour carries its own times and voicing, so nothing else needs
to change shape. `harmonics` becomes optional (all below Nyquist for a
number, all below `f_max` for a contour). The contour-only arguments
(`ramp`, `unvoiced`) do nothing for a number, which has no gaps and does
not move; `f_max` band-limits a number as it does a contour (D5). `typing.overload` stubs show the two call shapes
separately. `noise_vocode`'s `carrier` (`"noise"`, `"tone"` or a Sound)
is the precedent for one argument taking several shapes.

Alternatives considered: a separate `so.harmonic_source` beside an
unchanged `harmonic_complex` (the earlier recommendation; two names for
one idea), and rebuilding the fixed-F0 case on the contour path (one code
path, but every existing output would shift at the 1e-11 level for no
gain to users).

**D2. The contour is window times and F0 values, interpolated linearly in
Hz (accepted).** `harmonic_complex(duration, fs, (t, f0))`, `f0 = 0`
meaning unvoiced; values are held beyond the first and last time windows, so
`duration` may run past the track. A per-sample contour is passed with
`t = sound.t`. Alternatives: interpolation in log F0 (differs from linear
by far less than C3's bound at a 5 ms hop); band-limited (sinc)
interpolation, which overshoots at the steps a real track has.

**D3. Unvoiced time windows: gaps filled, harmonics gated with 5 ms ramps
(accepted).** C5 shows why the filling is needed. The ramp is a
parameter, `ramp=0.005`; 0 gives a hard gate.

**D4. An unvoiced noise option (accepted).** `unvoiced="silence"`
(default) or `"noise"`: white Gaussian noise in the unvoiced stretches at
the same power as the harmonics, crossfaded with the same gate, from a
seeded `rng`. That is enough for the vocoder recipe, where the envelopes
set every band's level anyway. Graded aperiodicity per band (WORLD's D4C; Morise, Yokomori & Ozawa, 2016)
belongs to the later synthesis step and is out of scope.

**D5. Band-limiting by a taper (accepted).** Harmonic k is weighted by
cos² falling from 1 at 0.9 f_max to 0 at f_max, so harmonics fade in and
out as the pitch moves instead of switching. `f_max` defaults to
0.45 fs (7.2 kHz at 16 kHz). A `harmonics` argument selects which
harmonic numbers to include (all by default; `[1]` for a sinusoid,
`range(10, 20)` for an unresolved complex), and the taper still applies to
them, so nothing ever aliases. A fixed F0 keeps its present rule (drop
what is at or above Nyquist, with a warning) and has no taper unless
`f_max` is given; then it takes the same taper (added 2026-10-02, after a
fixed-F0 buzz with `f_max=5000` on the Moving talkers page turned out to
have harmonics up to Nyquist). Alternative: a fixed harmonic count, which
aliases on any upward glide (C6).

**D6. Amplitudes: a per-harmonic array or a spectral envelope
(accepted).** `amplitudes` is either a 1-D array, one value per
harmonic (a fixed spectrum, as in `harmonic_complex`), or a function
`amplitudes(t, f)` returning the gain at time t and frequency f, sampled
at every harmonic's frequency k f(t) (for a fixed F0, at k f0). The
second form keeps formants in
place while the pitch moves, and is how a CheapTrick-style envelope (Morise, 2015) will
drive the source in the WORLD step. Time-varying per-harmonic arrays
(time windows × harmonics) are left out until something needs them.

**D7. Phases and channels (accepted, with arbitrary starting phases).**
Phases are cosine by default. `phases` also takes `harmonic_complex`'s
presets (`"sine"`, `"alternating"`, `"random"`, `"schroeder+"`,
`"schroeder-"`) or an array with one starting phase per harmonic: one per
harmonic number up to the most the contour can ever hold
(f_max / lowest voiced F0), or one per entry of `harmonics` when given. A
wrong length raises an error naming the length expected. Because harmonic
k's phase is k Φ(t) + φ_k, a starting pattern holds for the whole sound: a
random-phase or Schroeder-phase complex stays one while the pitch glides,
its waveform shape stretched with the period. That is what makes phase
experiments with unresolved harmonics (Schroeder, 1970, and the work that
followed) possible with a moving F0. A 2-D `f0` (one row per channel, as
`F0Track.f0` is) gives a multichannel Sound; the phases are shared.

**D8. Where it goes on the roadmap.** Cho, 2026-10-01: wait for the F0
tracker's library code, then land this after it. The tracker has now
landed (`so.f0_track`, PRs #46 and #50), so this is unblocked. It is the
first piece of the speech analysis and synthesis item (now item 1 of the
README roadmap). Once it lands, the Vocoder page can gain a short section
on putting the pitch back (examples 4, 6 and 7 above, on the tracker's
own track).

**D9. The named harmonic waveforms go through `harmonic_complex`
(accepted, Cho 2026-10-02).** `square_wave`, `sawtooth_wave` and
band-limited `pulse_train` are harmonic complexes with fixed amplitudes
(1/n on odd n, ±1/n, and 1) and sine or cosine phases; `schroeder_complex`
already calls `harmonic_complex`. The first three used a separate helper;
now they call `harmonic_complex` too, so all four accept a contour: a
gliding sawtooth, a Schroeder complex that follows a voice. Their names
stay, since they are what papers call them. The cost is that their
fixed-F0 outputs change at the round-off level (at most 6e-11 for unit-RMS
pulse trains, 5e-12 for the others), which does
not touch texture synthesis, the one place outputs must stay bit-for-bit.
The non-band-limited square, sawtooth and pulse train stay as they are and
take only a number.

## API sketch

```python
snd = so.load("docs/speech/bdl_arctic_a0131.flac")
trk = so.f0_track(snd)                                   # f0.md
src = so.harmonic_complex(snd.duration, snd.fs, trk, unvoiced="noise", rng=0)
so.noise_vocode(snd, 16, carrier=src)                    # its envelopes, this pitch

# A number is today's call; anything that is an F0 contour also works:
so.harmonic_complex(1.0, 44100, 220, range(1, 11))
t = np.arange(0, 2.005, 0.005)
glide = so.harmonic_complex(2.0, 44100, (t, 100 * 2**t))  # two octaves, no aliasing
monotone = so.harmonic_complex(snd.duration, snd.fs, (trk.t, np.where(trk.voiced, 120, 0)))
saw = so.sawtooth_wave(2.0, 44100, (t, 100 * 2**t))      # every named waveform too
vowel = so.harmonic_complex(2.0, 16000, (t, 110 + 10 * np.sin(2 * np.pi * 5 * t)),
                            amplitudes=lambda t, f: formant_gain(f))
```

## Tests (target: under 1 s added)

- C1: a constant contour equals the fixed-F0 call with the same
  harmonics and phases, within 1e-10.
- C2: a linear glide matches the closed-form chirp within 1e-9.
- C3: the vibrato's measured instantaneous frequency, harmonics 1 and 10,
  within 2e-4 of k times the contour.
- C5: no F0 below the voiced minimum while the gate is open.
- C6: an upward glide has no component above f_max (power above
  f_max + 400 Hz at least 90 dB below the total).
- `unvoiced="noise"` is reproducible from `rng`, and silent time windows are
  silent with `unvoiced="silence"`; a 2-D `f0` gives one channel per row.
- D1: an `F0Track`, a `(t, f0)` pair and a 2-D pair give the same sound;
  a wrong-length phase array raises.
- D9: the square, sawtooth and pulse train match their previous fixed-F0
  output within 1e-10, and each accepts a contour without aliasing.

## Patch plan

1. This document and `tools/check_harmonic_source_claims.py`.
2. `harmonic_complex` with contours and the D9 waveforms in
   `signals/generators.py`, tests, the README "What it's for" line on
   harmonic sounds and a roadmap note, CHANGELOG.
3. The Vocoder page section, and `pv.py` and `resynthesis.py` switched
   from their hand-rolled running sums to the new source.

## Out of scope

- Glottal pulse shapes (such as KLGLOTT88, Klatt & Klatt, 1990) and
  formant filters: the source-filter vowels and Klatt synthesizer (Klatt,
  1980) listed under roadmap item 1.
- Per-band aperiodicity (WORLD's D4C) and minimum-phase pulse synthesis.
- Jitter and shimmer: easy to add to a contour before it goes in, so not
  parameters of the source.

## References

Taken from the README's References, where each was checked by lookup
(`/mnt/project-files/citations/readme-citations.md` in the project files).

- Klatt, D. H. (1980). Software for a cascade/parallel formant
  synthesizer. *J. Acoust. Soc. Am.* 67(3). doi:10.1121/1.383940.
- Klatt, D. H. & Klatt, L. C. (1990). Analysis, synthesis, and perception
  of voice quality variations among female and male talkers. *J. Acoust.
  Soc. Am.* 87. doi:10.1121/1.398894.
- Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
  *Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224.
- Morise, M. (2015). CheapTrick, a spectral envelope estimator for
  high-quality speech synthesis. *Speech Communication* 67.
  doi:10.1016/j.specom.2014.09.003.
- Morise, M. (2017). Harvest: a high-performance fundamental frequency
  estimator from speech signals. *Proc. Interspeech 2017*.
  doi:10.21437/Interspeech.2017-68.
- Morise, M., Yokomori, F. & Ozawa, K. (2016). WORLD: a vocoder-based
  high-quality speech synthesis system for real-time applications. *IEICE
  Trans. Inf. & Syst.* E99-D(7). doi:10.1587/transinf.2015EDP7457.
- Schroeder, M. R. (1970). Synthesis of low-peak-factor signals and binary
  sequences with low autocorrelation. *IEEE Trans. Inf. Theory* 16.
  doi:10.1109/TIT.1970.1054411.
- Shannon, R. V., Zeng, F.-G., Kamath, V., Wygonski, J. & Ekelid, M.
  (1995). Speech recognition with primarily temporal cues. *Science* 270.
  doi:10.1126/science.270.5234.303.
