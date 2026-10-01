# Time-varying harmonic source

The design of a harmonic source whose fundamental follows a contour: a
sum of phase-locked harmonics driven by an F0 track, with the harmonics
switched off where the track is unvoiced, faded out before they reach
Nyquist, and optionally weighted by a spectral envelope. It is the
harmonic half of the "pulse-plus-noise synthesis" in roadmap item 2
(speech analysis and synthesis), and the piece that lets an F0 track and a
set of band envelopes be put back together and listened to.

Status: proposed 2026-10-01. D8 decided (after the F0 tracker); D1–D7
await Cho's answers; no library code yet.

## Why

Cho's question: with an F0 extraction and an envelope extraction, can
sonore impose the envelopes on harmonics built from the F0 contour, to hear
what the two carry between them? Most of that already exists:

- `so.noise_vocode(sound, n_bands, carrier=...)` takes any `Sound` as the
  carrier, splits it with the same ERB filterbank as the sound, and
  multiplies each band's fine structure by the sound's band envelope
  (`src/sonore/analysis/filterbank.py`). The envelope half is done.
- `docs/speech/bdl_arctic_a0131_f0.csv` holds a Harvest track of the
  gallery sentence, and `so.f0_track` is being designed (`f0.md`).

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

Given F0 values f0[j] at frame times t[j] (0 meaning unvoiced, the form of
the stored Harvest track and of the proposed `F0Track`):

1. **Fill the gaps.** Unvoiced frames take F0 values interpolated linearly
   between the voiced frames on either side; leading and trailing ones hold
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
5. **Gate by voicing.** Multiply by a gate that is 1 on voiced frames and
   0 on unvoiced ones, its steps smoothed by a 5 ms Hann window (D3).
   Optionally fill the unvoiced stretches with white noise (D4).
6. **Normalize** to RMS 1, as every generator does.

## Claims

**C1. A constant contour gives the fixed-F0 harmonic complex.** [check]
A 220 Hz contour sampled every 5 ms, harmonics 1–10, matches
Σ cos(2π k 220 t) to 6.5e-12 (max abs difference, unit-amplitude
harmonics). So the new source is a generalization of `harmonic_complex`,
not a different sound. `harmonic_complex` itself stays as it is.

**C2. The phase is exact for a contour that is linear between samples.**
[proof] If f is linear between t_{i-1} and t_i, its integral over the step
is the step times the mean of the endpoints, which is the trapezoid rule.
Linear interpolation of a contour that is linear between frames is exact,
so a linear glide is reproduced exactly. [check] A 100 → 300 Hz glide
given as 5 ms frames matches the closed-form chirp cos(2π(100t + 100t²))
to 4.5e-13. The running sum the gallery pages use, 2π Σ_{j≤i} f_j / fs,
differs from it by up to 0.078: it equals the trapezoid plus
π(f_0 + f_i)/fs (checked to 6e-13), a phase lead of about one sample of
the current frequency. Inaudible in those pages, but not exact.

**C3. Every harmonic follows k times the contour.** [proof] Harmonic k's
phase is k Φ, so its instantaneous frequency is k f(t), and the
interpolated f(t) differs from a smooth true contour g by at most
h²/8 · max|g''| between frames of period h. [check] For a 150 Hz vibrato,
±4% at 5.5 Hz, given as 5 ms frames, that bound is 1.56e-4 (relative).
The instantaneous frequency measured from the analytic signal of
harmonics 1, 10 and 40 alone stays within 1.43e-4, 1.56e-4 and 1.56e-4
of k times the true contour, and harmonic 10's frequency divided by
harmonic 1's is 10 within 5.5e-4 (the Hilbert measurement's own error).

**C4. The phase has no jumps.** [check] On the stored Harvest track of the
gallery sentence (2.53 s, 16 kHz, harmonics to 7.2 kHz), each sample's
phase step equals 2π times the mean frequency over the step to 5e-13 rad.
The obvious shortcut, synthesizing each 5 ms frame with its own F0 against
absolute time, cos(2π k f0[j] t), puts 31 dB-below-total power above
7.6 kHz (clicks at every frame boundary); the accumulated phase puts
76 dB below.

**C5. Gaps must be filled before interpolating.** [check] The track's
lowest voiced F0 is 87.6 Hz. Interpolating straight through the zeros
of unvoiced frames sweeps the pitch down to 11.5 Hz while the gate is
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
of the gallery sentence and writes eight short files beside itself:

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
of the voiced frames, against 69% for the original recording and 0% for
the noise vocoder.

## Decisions

**D1. Add it, as a new generator (recommended).** A function
`so.harmonic_source` in `sonore/signals/generators.py`, beside
`harmonic_complex`. It takes plain arrays, so it sits in the signals
layer and imports nothing above it (`tests/test_layers.py`); an
`F0Track`'s `t` and `f0` pass straight in, and so does the stored Harvest
track. Alternative: let `harmonic_complex` accept an array for `f0`. Its
signature is built around a duration and one F0, and C1 already shows the
two agree, so a separate function keeps both plain.

**D2. The contour is frame times and F0 values, interpolated linearly in
Hz (recommended).** `harmonic_source(t, f0, fs, duration=None, ...)`,
`f0 = 0` meaning unvoiced; `duration` defaults to the last frame time and
the last value is held beyond it. A per-sample contour is passed with
`t = sound.t`. Alternatives: interpolation in log F0 (differs from linear
by far less than C3's bound at 5 ms frames); band-limited (sinc)
interpolation, which overshoots at the steps a real track has.

**D3. Unvoiced frames: gaps filled, harmonics gated with 5 ms ramps
(recommended).** C5 shows why the filling is needed. The ramp is a
parameter, `ramp=0.005`; 0 gives a hard gate.

**D4. An unvoiced noise option (recommended).** `unvoiced="silence"`
(default) or `"noise"`: white Gaussian noise in the unvoiced stretches at
the same power as the harmonics, crossfaded with the same gate, from a
seeded `rng`. That is enough for the vocoder recipe, where the envelopes
set every band's level anyway. Graded aperiodicity per band (WORLD's D4C)
belongs to the later synthesis step and is out of scope.

**D5. Band-limiting by a taper (recommended).** Harmonic k is weighted by
cos² falling from 1 at 0.9 f_max to 0 at f_max, so harmonics fade in and
out as the pitch moves instead of switching. `f_max` defaults to
0.45 fs (7.2 kHz at 16 kHz). A `harmonics` argument selects which
harmonic numbers to include (all by default; `[1]` for a sinusoid,
`range(10, 20)` for an unresolved complex), and the taper still applies to
them, so nothing ever aliases. Alternative: a fixed harmonic count, which
aliases on any upward glide (C6).

**D6. Amplitudes: a per-harmonic array or a spectral envelope
(recommended).** `amplitudes` is either a 1-D array, one value per
harmonic (a fixed spectrum, as in `harmonic_complex`), or a function
`amplitudes(t, f)` returning the gain at time t and frequency f, sampled
at every harmonic's frequency k f(t). The second form keeps formants in
place while the pitch moves, and is how a CheapTrick-style envelope will
drive the source in the WORLD step. Time-varying per-harmonic arrays
(frames × harmonics) are left out until something needs them.

**D7. Phases and channels.** Starting phases use `harmonic_complex`'s
presets (`"cosine"` by default; `"random"`, `"schroeder+"`, ... and an
array also work, as starting values). A 2-D `f0` (one row per channel,
as `F0Track.f0` is) gives a multichannel Sound.

**D8. Where it goes on the roadmap.** Cho, 2026-10-01: wait for the F0
tracker's library code, then land this after it. The recommendation had
been to land it now, as the first piece of roadmap item 2 and ahead of the F0 tracker's
library code, since it depends on nothing new. The tracker's gallery page
can then play its tracks, and the Vocoder page can gain a short section
on putting the pitch back (examples 4, 6 and 7 above, on the stored
Harvest track until `so.f0_track` lands). Alternative: wait and add it
with the rest of the WORLD-style synthesis.

## API sketch

```python
snd = so.load("docs/speech/bdl_arctic_a0131.flac")
trk = so.f0_track(snd)                                   # f0.md, once it lands
src = so.harmonic_source(trk.t, trk.f0[0], snd.fs, duration=snd.duration,
                         unvoiced="noise", rng=0)
so.noise_vocode(snd, 16, carrier=src)                    # its envelopes, this pitch

# Anything that is an F0 contour:
t = np.arange(0, 2.005, 0.005)
glide = so.harmonic_source(t, 100 * 2**t, 44100)         # two octaves, no aliasing
monotone = so.harmonic_source(trk.t, np.where(trk.voiced[0], 120, 0), snd.fs)
vowel = so.harmonic_source(t, 110 + 10 * np.sin(2 * np.pi * 5 * t), 16000,
                           amplitudes=lambda t, f: formant_gain(f))
```

## Tests (target: under 1 s added)

- C1: a constant contour equals `harmonic_complex` with the same
  harmonics and phases, within 1e-10.
- C2: a linear glide matches the closed-form chirp within 1e-9.
- C3: the vibrato's measured instantaneous frequency, harmonics 1 and 10,
  within 2e-4 of k times the contour.
- C5: no F0 below the voiced minimum while the gate is open.
- C6: an upward glide has no component above f_max (power above
  f_max + 400 Hz at least 90 dB below the total).
- `unvoiced="noise"` is reproducible from `rng`, and silent frames are
  silent with `unvoiced="silence"`; a 2-D `f0` gives one channel per row.

## Patch plan

1. This document and `tools/check_harmonic_source_claims.py`.
2. After Cho's answers: `harmonic_source` in `signals/generators.py`,
   tests, the README "What it's for" line on harmonic sounds, CHANGELOG.
3. The Vocoder page section, and `pv.py` and `resynthesis.py` switched
   from their hand-rolled running sums to the new source.

## Out of scope

- Glottal pulse shapes (Rosenberg, LF) and formant filters: the
  source-filter vowels and Klatt synthesizer listed under roadmap item 2.
- Per-band aperiodicity (WORLD's D4C) and minimum-phase pulse synthesis.
- Jitter and shimmer: easy to add to a contour before it goes in, so not
  parameters of the source.
