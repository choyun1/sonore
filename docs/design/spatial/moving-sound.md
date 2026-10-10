# Moving sound sources

What it would take to make `so.move_sound` render sources that move
convincingly: sources that change distance (level, travel time, Doppler
shift and the balance with a room's reverberation), paths given as
functions of time such as the sinusoidal azimuth swing of Cho & Kidd
(2022), and how the stages compose with measured HRIRs. It sets out what
`move_sound` does today, the physics of a moving point source, what the
current method gets right and wrong (measured), a proposed renderer, and
the decisions for Cho.

Status: D1–D8 accepted by Cho 2026-10-02, all as recommended. Built as
`so.move_sound` and `so.hcc_trajectory` (see "As built" below); the
gallery section is still to come.

"What `move_sound` does today" below describes the renderer before this
work.

## Why

The README roadmap's first Next item: "Revisit `move_sound`, since linear
trajectories sound unconvincing: sources that change distance (level
change, travel-time delay, Doppler shift and room reverberation), a
sinusoidal azimuth trajectory like the one in Cho & Kidd (2022), and
faster rendering via batched frequency-domain filtering."

A source that only changes direction is already rendered well (C5). What
is missing is everything that depends on distance. Today a path that
moves away from the head is filtered by HRIRs at the new positions, but
nothing makes it quieter, later or lower in pitch unless the HRIR set
itself holds several distances, and even then nothing beyond the
outermost measured distance (1.6 m for PKU-IOA) changes at all.

## What `move_sound` does today

`move_sound(sound, trajectory, hrirs)` takes an `(N, 3)` array of
Cartesian points and spreads them evenly over the sound's duration. Each
point gets a raised-cosine window on the input, centered on its moment;
neighboring windows overlap and sum to one. Each windowed piece is
convolved with the HRIR interpolated at that point, and the outputs are
added. The HRIRs are interpolated with their onsets aligned and the onset
delays interpolated separately, so neighboring directions do not
comb-filter when averaged. The Moving talkers gallery page uses 200 points
per second.

This is filter switching on the input side, smoothed (Brandtsegg et al.,
2018, review it with output crossfading and partitioned-convolution
updates). Within one time window the filter is fixed, so a delay that
changes continuously is approximated by a staircase of fixed delays,
cross-faded. That is harmless while neighboring delays differ by a small
fraction of a period, and fails when they do not (C4).

Distance enters only through `distance_gain_db`, which the caller applies
by hand to a static source (the Synthetic reverberation page does this).

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[derived]**: follows from the stated physics in a line or two, shown
  here.
- **[check]**: a number printed by `tools/check_moving_sound_claims.py`.
  The script uses only NumPy and SciPy, shares no code with sonore, holds
  a prototype of the proposed propagation stage, of the current switching
  and of a batched version of it, and runs in about ten seconds. The
  numbers come from NumPy 2.4 and SciPy 1.17.1 at 48 kHz.
- C7 needs the PKU-IOA files, which sonore never bundles; it ran on
  Cho's original `.dat` files with
  `python tools/check_moving_sound_claims.py --pku DIR`.

Nothing new was read for this document. The physics of a moving point
source is textbook acoustics, derived below and checked numerically
against its own closed form; the references are the ones the Moving
talkers page already cites, plus pointers for the standard results.

## Words used here

- **Emission time** t_e: when the source emits a sample, on the source's
  own clock. The input sound is indexed by emission time.
- **Arrival time** t: when that sample reaches the ear. The output is
  indexed by arrival time.
- **Path**: where the source is, as a function of emission time. A
  **trajectory** argument is any of the forms in D1 that describe a path.
- **Time window** and **hop**: as in `philosophy.md`; here, the stretch of
  sound one HRIR shape filters, and their spacing.

## Which cues listeners use

Doppler is not the reason for this design. Read from secondary sources
only (Carlile & Leung's 2016 review, an abstract, and Ghazanfar & Maier's
2009 PDF), the literature says:

- Intensity change and ITD carry most of the information about linear
  motion at moderate speeds; Doppler took over only at the fastest
  speeds tested. Lutfi & Wang (1999), as summarized by Carlile & Leung
  (2016): intensity and ITD "correlated most with displacement
  discrimination, at least at slower velocities (10 m/s), while Doppler
  shifts dominated at the faster velocities (50 m/s)".
- What people report as the "Doppler" pitch rise of an approaching
  source is largely driven by its rising level (Neuhoff & McBeath, 1996,
  "the Doppler illusion"; abstract only). Rising intensity alone makes a
  looming percept (Neuhoff, 1998; Seifritz et al., 2002, via the
  review).
- Rhesus monkeys treat rising frequency as looming, the same bias as
  humans, although a real approaching source falls in frequency as it
  passes (Ghazanfar & Maier, 2009, *Behavioral Neuroscience*).

So the cues for a source changing distance are, in order, level, the
direct-to-reverberant ratio (a distance cue for static sources; not
checked here for moving ones), and only then Doppler. The continuous
delay of stage 2 is still needed, but for a different reason: switching
between fixed delays comb-filters (C4) whether or not listeners use the
pitch change. Doppler comes for free with that delay, so the gallery can
let listeners judge for themselves whether they hear it.

## The physics in brief [derived]

A point source at distance r(t_e) from the head center emits s(t_e). In
free air the sound reaches the head center at

    t = t_e + r(t_e) / c,          c = 343 m/s,

with its amplitude falling as 1/r. So the signal at the head center is

    p(t) = s(t_e(t)) / r(t_e(t)),

where t_e(t) solves the equation above. Doppler shift is not a separate
effect: it is what reading s at t_e(t) does when r changes. The received
frequency is the emitted one times dt_e/dt = 1 / (1 + ṙ/c), up for an
approaching source and down for a receding one.

The equation has one solution whenever the source is slower than sound,
and fixed-point iteration t_e ← t − r(t_e)/c, started from t − r(t)/c,
converges with contraction factor |ṙ|/c (C3).

Two refinements are left out on purpose. A moving monopole's amplitude
also carries a convective factor 1/(1 − M_r)², with M_r = ṙ/c the radial
Mach number toward the listener; it is below 0.8 dB at 15 m/s (C1) and
depends on what one calls "the source signal", so the renderer treats
the input as the sound a still listener at 1 m would record from a still
source. Air absorption is a few tenths of a dB per 10 m below 4 kHz
(inferred from ISO 9613-1 orders of magnitude, not computed here), so it
matters only for distances beyond the scope of this work.

At the ears, the head adds what the HRIR holds: a direction-dependent
filter and the extra delay to each ear. With onset-aligned HRIRs (as
`HRIRSet` already builds them) this splits cleanly into an **interaural
delay** per ear, e_ear(direction), and an aligned **shape**. The delay to
each ear is then r/c + e_ear, and each ear reads the source at its own
emission time. A source moving sideways at constant distance has no
Doppler at the head center, but its two ear delays change in opposite
directions, so each ear hears a slightly different pitch (C2).

## Claims

**C1. Doppler shifts are small but audible at street speeds; the
convective level change is not worth modeling.** [check] Straight toward
the head the pitch rises by 7.1 cents at walking speed (1.4 m/s), 25 cents
running (5 m/s) and 77 cents for a car in town (15 m/s); receding, it falls
by about as much, so a pass-by at 15 m/s glides about 150 cents. The
convective factor changes the level by 0.07 dB at 1.4 m/s and 0.78 dB at
15 m/s.

**C2. The azimuth swing of Cho & Kidd (2022) has no Doppler at the head
center, and about 6 cents between the ears.** [check] A source swinging
30° to either side of straight ahead at 2 Hz on a 1 m circle keeps its
distance exactly (range 1e-16 m), while moving at up to 6.6 m/s along its
arc. With Woodworth's interaural delays (head radius 8.75 cm) the two
ears' delays change in opposite directions, and the pitch difference
between the ears peaks at 5.8 cents as the source crosses the front.

**C3. The propagation stage is exact up to the fractional-delay read,
and a 32-tap windowed sinc makes that read negligible.** [check] For a
source passing 2 m in front at 15 m/s over 3 s, the fixed-point solve
reaches 1e-12 s in 7 iterations (contraction factor 0.044). With t_e
exact, the received tone differs from the closed form p(t) above only by
the interpolation used to read s between samples:

| read | 500 Hz | 2 kHz | 8 kHz |
|---|---|---|---|
| linear | −68 dB | −44 dB | −20 dB |
| cubic Lagrange | −130 dB | −82 dB | −35 dB |
| Kaiser-windowed sinc, 32 taps, β = 8 | −96 dB | −97 dB | −97 dB |

(error energy re signal energy). The sinc's error is flat in frequency;
Lagrange is better at low frequencies and poor near the top. The
received tone glides from +77 to −74 cents. In plain NumPy the 32-tap read
takes about 1.8 s for 3 s of sound and two ears.

**C4. Switching between fixed filters cannot follow a changing
distance.** [check] Rendering the same pass-by the way `move_sound` does
today, with 200 points per second and each point's filter an exact delay
r/c with gain 1/r, neighboring delays differ by up to 218 µs. Two copies
of a sound 218 µs apart cross-fading is a comb filter with its first notch
near 2.3 kHz. Against the closed form the error is −26 dB at 500 Hz,
−9 dB at 2 kHz and +1 dB at 8 kHz (as much error as signal). With the
source still, the same code is exact to −136 dB, so the error is the
switching, not the filters. More points per second shrink the steps but
never remove them.

**C5. For the azimuth swing, the current switching is already good.**
[check] For the 30° swing at 2 Hz, with Woodworth delays as the only
filters, the current switching's error (worse ear) is −54 dB at 500 Hz and
−39 dB at 4 kHz. So the Moving talkers page's stimuli are not wrong; the
renderer is needed for distance, not for direction.

**C6. Batched frequency-domain filtering gives the same output, but
little speed.** [check] Transforming every windowed piece in one batched
FFT and multiplying by every filter's spectrum at once matches the loop
to 7e-16 of the peak (3 s, 200 points per second, 750 taps, the length of
a PKU-IOA HRIR resampled to 48 kHz), and takes about 0.10 s against the
loop's 0.12 s. Measured separately with sonore and the MIT KEMAR set
(1.4 m, 512 taps at 44.1 kHz, shipped with libmysofa), today's
`move_sound` renders 3 s at 48 kHz through 600 points in 0.22 s, about
half of it HRIR interpolation and resampling. Rendering speed is not the
problem the roadmap assumed; the new per-sample delay (C3) will cost more
than the switching does.

**C7. The PKU-IOA responses carry their own travel time and 1/r.**
[check] Run on Cho's original `.dat` files (6456 files, one of them empty,
the azimuth-360 duplicates included), the mean onset over all directions
and both ears, re the 1 m shell, follows r/c to within 0.07 ms at every
distance (−2.27 ms at 20 cm against −2.33 predicted, +1.74 ms at 1.6 m
against +1.75). In absolute terms each shell's mean onset is r/c less
about 12 samples at 65536 Hz (0.18 ms), the same at every distance, so
the files keep the travel time from the source, not a trimmed one. The
mean level follows 1/r to within 0.4 dB from 40 cm out (+7.6 dB at 40 cm
against +8.0, −4.3 dB at 1.6 m against −4.1), and falls short of it near
the head: +9.7 dB at 30 cm against +10.5, and +11.8 dB at 20 cm against
+14.0. That shortfall is a measured near-field effect, not an offset to
remove. So inside 20–160 cm the measured responses already are the
propagation (as Cho expected), and the renderer should take delay and
level from them rather than from geometry (D4).

## Proposed design

`move_sound` keeps its name and signature and gains stages; `spatialize`
stays `move_sound` with one fixed point, so the two always agree.

1. **Path.** The trajectory becomes a function of emission time (D1):
   given as a function, as `(times, points)`, or as today's `(N, 3)`
   array spread evenly over the sound.
2. **Delay, per ear.** Each ear's delay d_ear(t_e) is the onset
   `HRIRSet` already measures and interpolates for that position, which
   for PKU-IOA holds the travel time (C7); beyond the outermost shell it
   is the outermost shell's onset in that direction plus
   (r − r_outer)/c. Interpolated at the hop points and then smoothly in
   time, it gives, for each output sample and each ear, t = t_e +
   d_ear(t_e), solved for t_e by C3's iteration, and the input is read at
   t_e with the 32-tap windowed sinc. Doppler, the travel time and the
   continuously changing interaural delay all come out of this one read.
   The level stays in the HRIR shapes as measured; beyond the outermost
   shell it is scaled by r_outer/r.
3. **Direction, per ear.** The onset-aligned HRIR shapes, interpolated at
   points one hop apart (5 ms by default), filter each ear's propagated
   signal with today's raised-cosine switching on the output timeline.
   The shapes carry no delay that changes with direction, so switching
   them does not comb-filter the way C4 does.
4. **Room** (D6). A two-channel reverberant tail, such as
   `so.synth_ir(rt60, fs, n_channels=2)`, is driven by the source signal
   delayed by the path's travel time but without the 1/r, so the tail's
   level stays put while the direct sound falls 6 dB per doubling of
   distance, and the direct-to-reverberant ratio follows distance as on
   the Synthetic reverberation page. `drr_db` sets it at 1 m.

Distances outside the HRIR set: beyond the outermost measured distance,
the shape and onsets come from the outermost shell in the same direction,
and further distance acts only through the extra (r − r_outer)/c and
r_outer/r (far field). Because PKU-IOA's 1.6 m shell already follows r/c
and 1/r (C7), this continues it without a step. A set measured at one
distance only (the 1 m default) is extended the same way in both
directions, so it still renders distance, without near-field cues. With
several shells, a point inside the innermost one (20 cm for PKU-IOA)
raises an error, as
`HRIRSet.at` does today for points outside the measured region.

API sketch (names provisional until D1, D7, D8):

```python
hrirs = so.load_hrirs(distances="all")

# Cho & Kidd (2022): azimuth swing at 1 m, as a function of time
swing = so.hcc_trajectory(dist=100, elev=0, azim=lambda t: 30 * np.sin(2 * np.pi * 2 * t))
so.move_sound(talker, swing, hrirs)

# a talker walking toward the listener, 3 m to 0.5 m, in a room
approach = so.hcc_trajectory(dist=([0, 2], [300, 50]), elev=0, azim=20)
room = so.synth_ir(0.6, fs, n_channels=2)
so.move_sound(talker, approach, hrirs, room=room, drr_db=0)

# today's call, unchanged in form
so.move_sound(talker, so.circular_trajectory((100, 0, 90), (100, 0, -90), 200), hrirs)
```

## Decisions

- **D1. How a trajectory is given.** (a) Dispatch on the argument's
  type, as `harmonic_complex` does for `f0`: an `(N, 3)` array is spread
  evenly over the sound (today's meaning, so existing calls keep their
  meaning), a `(times, points)` pair is interpolated linearly in time like
  a `Track`, and a callable `t -> (n, 3)` is evaluated wherever the
  renderer needs a position (recommended: one argument, no new class,
  and the callable gives the propagation stage an exact position at every
  sample, which C3's exactness needs). (b) A `Trajectory` class with
  `times`, `points`, an `.at(t)` method and constructors; its best form
  is what (a) reaches without a class, and it would make every caller
  build one. (c) Arrays only, with a `times=` keyword: the smallest
  change, but a swing then has to be sampled by hand finely enough for
  the per-sample delay. Note for (a): a `(times, points)` pair
  interpolated in Cartesian space cuts corners on a sparse arc, which
  shortens the distance between points; `hcc_trajectory` (D8)
  interpolates in distance, elevation and azimuth instead.
- **D2. A continuous delay on by default?** Since C7 the delay and
  level inside the measured range are the HRIRs' own; what is new is
  applying the delay continuously (stage 2) instead of switching it with
  each time window, and extending it past the outermost shell. (a) On
  (recommended: for a source at a fixed distance the output is the same
  as today's up to interpolation error, since a constant delay switches
  cleanly; only moving distance changes). (b) Off by default behind
  `continuous_delay=True`. The convective factor (C1) is left out either
  way.
- **D3. Where time zero is.** (a) The output starts at emission time
  zero, so the sound arrives r/c later, 2.9 ms at 1 m and 29 ms at 10 m
  (recommended: talkers at different distances mixed with `so.mix` then
  arrive in the right order, and it is what the PKU-IOA responses
  already do, C7). (b) Shift each output so its first arrival
  is at zero, which loses the relative timing between sources.
- **D4. HRIR sets that hold several distances.** (a) Use the shells as
  measured, with their own onsets and levels, and extend beyond the
  outermost shell with (r − r_outer)/c and r_outer/r (recommended since
  C7: the PKU-IOA files hold the travel time to within 0.07 ms and 1/r
  to within 0.4 dB from 40 cm out, and their shortfall from 1/r near the
  head is real near-field data worth keeping). (b) Equalize every shell
  to the 1 m shell's mean energy and remove each shell's mean onset, then
  take level and delay from geometry everywhere: right for a database
  that trimmed or normalized its responses, but for PKU-IOA it would
  erase the 2.2 dB near-field shortfall at 20 cm. A database known to be
  trimmed could get (b) through a registry flag, as `azimuth_clockwise`
  marks the mirrored SOFA copy.
- **D5. How the changing filter is applied.** (a) The continuous
  per-ear delay of stage 2 plus switched aligned shapes (recommended:
  exact for distance and timing, C3, and leaves only slowly changing
  shapes to switching). (b) Today's switching of whole HRIRs, extended
  with r/c and 1/r per point: fails for distance (C4). (c) A truly
  time-varying convolution, a fresh interpolated HRIR at every sample:
  the textbook definition. With onset-aligned interpolation it handles
  distance too, since each sample's HRIR carries its own delay. Measured
  with `tools/moving_sound_benchmark.py` (sonore, MIT KEMAR set, 3 s at
  48 kHz, the 30° swing): 37 s, against 0.4 s for today's switching. Of
  the 37 s, 36.6 s go to interpolating and resampling 144,000 HRIRs, and
  only 0.3 s to the filtering itself. (a) is (c) factored: delay
  continuous, shape at the hop rate. Its cost is not measured yet; its
  parts are about 0.4 s of switching plus 1.8 s for the unoptimized read
  in C3, so roughly 15 times faster than (c) as prototyped, not the two
  orders of magnitude this document first claimed. (c) could get close
  to (a) by interpolating shapes per sample between hop points, but the
  delay inside each HRIR then has to be applied by a per-sample
  fractional-delay kernel, which is (a)'s read again. For the read, the 32-tap windowed sinc
  (recommended, flat −97 dB) or cubic Lagrange (cheaper, −35 dB at
  8 kHz).
- **D6. The room.** (a) `room=` a two-channel tail Sound plus `drr_db`
  at 1 m, driven by the delayed source without 1/r (recommended: any
  tail works, measured or `synth_ir`, and the direct path keeps its exact
  rendering). (b) `room=(rt60, drr_db)`, building the tail inside: shorter
  to call but hides `synth_ir`'s options. (c) Early reflections by the
  image-source method, each reflection a moving source of its own: the
  most physical, a later step if (a) sounds too diffuse. `synth_ir`'s
  independent channels make the tail's interaural correlation zero at
  every frequency, unlike a real diffuse field at low frequencies; that
  is a property of the existing tail, unchanged here.
- **D7. One function or a new one.** (a) Extend `move_sound`
  (recommended: one name for one job, `spatialize` stays its special
  case, and D1's dispatch keeps old calls valid). (b) A new
  `so.render_source` beside an unchanged `move_sound`: old outputs stay
  exactly as they are, at the cost of two renderers that disagree on a
  receding source. (c) A `Scene` that holds the HRIR set, the room and
  several sources and renders the mix: matches how an experiment is
  described (three talkers, one room), but `so.mix` already composes, and
  a class with state is more than this needs. And: keep the loop or move
  to the batched FFT of C6? Recommended: whichever reads more simply once
  the shapes run on the output timeline; C6 shows it is not a speed
  decision, and the roadmap's "faster rendering" can be dropped.
- **D8. A helper for paths.** (a) `so.hcc_trajectory(dist, elev, azim)`
  where each coordinate is a number, a `(times, values)` pair or a
  function of time, returning a callable path (recommended: one helper
  covers the swing, an approach, an orbit or a pass-by, in the
  coordinates people think in, with the `Track` convention the Klatt
  parameters use, plus callables). (b) A dedicated
  `so.sinusoidal_trajectory(center, amplitude, rate, phase)`: reads well
  for the one experiment, but every other path needs its own function.
  `linear_trajectory` and `circular_trajectory` stay as they are.

## What a listener hears

The Moving talkers page gains a section on distance, all of it built
locally by Cho (the cloud container cannot download the HRIRs): a talker
walking toward the listener from 3 m to 50 cm, dry and in a room, where
the room makes the approach audible as the direct sound grows out of the
reverberation; and a pass-by at 15 m/s, where the pitch glides about 150
cents as it passes (C1, C3). The existing swing examples stay, re-rendered,
and should sound the same (C5).

## Tests

- A still source matches today's `spatialize` (to float precision).
- A tone on a pass-by, through a set whose shapes are unit impulses,
  matches the closed form p(t) to within C3's −90 dB.
- The interaural delay of a swing follows the HRIR onsets.
- Beyond the outermost shell, level falls as r_outer/r and the delay
  grows as (r − r_outer)/c, with no step at the shell.
- The tail's level does not change with distance; the direct sound's
  falls 6 dB per doubling.
- At build time, measured on the MIT KEMAR set: aligned shapes switched
  at 5 ms against the same shapes switched at 0.5 ms, to show the
  switching left in stage 3 is inaudible.

## As built

- `so.move_sound(sound, trajectory, hrirs, *, room=None, drr_db=0.0,
  hop=5e-3, speed_of_sound=343.0)` and `so.hcc_trajectory(dist, elev,
  azim)`, with `spatialize` still the exact path for a still source.
- Each ear's delay is the interpolated HRIR onset (less the few samples
  the aligned shapes keep before their onset), looked up every 1 ms and
  smoothed with a cubic spline. Each output sample's emission time is
  found to 1e-12 s (see the fast-paths item below for how), and the sound
  is read there with the 32-tap Kaiser sinc. Measured on tones at 48 kHz, its error is
  about −90 dB up to 16 kHz and −78 dB at 20 kHz, rolling off above that,
  so C3's "flat −97 dB" holds only to about a third of the sampling rate.
- Shapes are interpolated every `hop` along the path and cross-faded with
  raised cosines in each ear's own emission time, so the cross-fades
  follow the sound as it arrives at that ear.
- Interpolation between measured distances changed from a 3-D Delaunay
  triangulation to interpolating by direction on the two measured
  distances around the point, then linearly in distance. A Delaunay
  triangulation of points on spheres covers only the polyhedron inside
  the outermost sphere, so a point at exactly the outermost distance
  between two measured directions fell outside it and failed. `HRIRSet.at`
  uses the same interpolation, and `HRIRSet.distances` lists the measured
  distances.
- Beyond the measured distances (and, for a set with one distance, on
  either side of it), `spatialize` and `move_sound` use the HRIR of the
  nearest measured distance in the same direction, delayed by the extra
  distance over the speed of sound and scaled by the ratio of distances.
- The room tail is scaled so that the direct-to-reverberant ratio at 1 m,
  averaged over directions and ears, is `drr_db`. It is driven by the
  sound read at the average of the two ears' emission times and filtered,
  at each ear, by the diffuse-field response: a minimum-phase filter with
  the HRIR power spectrum averaged over the directions of the shell
  nearest 1 m. Without it the tail kept the source's flat spectrum while
  the direct sound took the HRIRs' tilt, and for speech the DRR heard was
  not the one asked for (found building the gallery's walk in a room).
- Tests (`tests/spatial/test_spatialization.py`): a 1 kHz tone passing at
  15 m/s through impulse HRIRs matches the closed form to better than
  −80 dB; the three trajectory forms agree to 1e-12; a still source
  matches `spatialize` to better than −90 dB; past the measured distance,
  doubling the distance gives −6.02 dB and 2 m / 343 m/s more delay; and
  the room tail's level stays put while the direct sound falls 6 dB.
- On Cho's PKU-IOA files (all eight distances), noise approaching from
  3 m to 30 cm straight ahead over 4 s follows 1/r to within 1.4 dB,
  measured in 0.1 s time windows. Crossing the outermost distance, 1.6 m,
  the level rises 0.38 dB in a 0.1 s time window where 1/r predicts 0.37.
  The largest departures, near 1.3 m, are in the data: straight ahead,
  the measured HRIR energy with 1/r removed is −5.2, −4.7 and −5.5 dB at
  1, 1.3 and 1.6 m, and the renderer passes that through.

- Fast paths (built for the gallery's straight paths and its path no
  source could take; numbers from `tools/check_moving_trajectories.py`).
  Emission times are now found by inverting arrival time on a grid eight
  times finer than the delay lookup, then refined by Newton's method; the
  fixed-point iteration before converged ever more slowly as a source
  coming toward the head neared the speed of sound (it gave up at
  300 m/s) and, for one faster than sound, settled outside the sound and
  returned a near-silent result instead of an error. A source whose ear delay shrinks faster than time
  passes is now refused. With PKU-IOA that happens at 300 m/s already
  (0.87 of the speed of sound): crossing 1.6 m, the measured onsets make
  the left ear's delay shrink at 0.99 s per s where the travel time
  shrinks at 0.87. Results for slower paths are unchanged to 1e-10.
  On the gallery's jumping path (top speed 247 m/s, 28° of azimuth in
  5 ms), shapes switched every 5 ms differ from shapes switched every
  0.0625 ms by −13 dB in the worst 20 ms window, and every 0.5 ms by
  −29 dB, so the gallery renders it with `hop=0.5e-3`.
- Speed (numbers from `tools/check_move_sound_speed.py`). A 3 s render
  at 44.1 kHz took 1.3 s, most of it spent outside the convolutions:
  computing the windowed sinc afresh for every output sample, and testing
  every lookup point against every triangle on its shell (about 1,580 on
  PKU-IOA). The sinc is now tabulated at 512 fractions of a sample and
  interpolated linearly between them; the result changes by −117 dB for
  white noise, and the error on tones is the same as before (−95 dB at
  1 kHz, −80 dB at 20 kHz). Each point is now tested only against the
  triangles touching its three nearest measured directions, found with a
  k-d tree, and against all of them only if none holds it; on 20,000
  random directions per shell this picks the same triangles and weights.
  The same render now takes 0.3 s. The convolutions, one per `hop` per
  ear, were never the main cost.

## Order

1. Answer D1–D8 (done).
2. Trajectory forms and `hcc_trajectory` (done).
3. Propagation stage, with the claim tests (done).
4. Extension beyond the outermost shell (D4) (done).
5. Room tail (done).
6. Gallery section, built locally; README roadmap and Done.

## Out of scope

Head rotation (paths are relative to the head, so a turning listener is
a rotating path), ground reflections, air absorption, sources faster than
about 30 m/s, sources inside 20 cm, and listener models.

## References

- Brandtsegg, Saue & Lazzarini (2018). Live convolution with time-varying
  filters. *Applied Sciences* 8(1), 103. The switching and crossfading
  methods.
- Cho & Kidd (2022). Auditory motion as a cue for source segregation and
  selection in a "cocktail party" listening environment. *J. Acoust. Soc.
  Am.* 152(3), 1684–1694. doi:10.1121/10.0013990. The azimuth swing.
- Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related
  transfer functions measured with high spatial resolution using a spark
  gap. *IEEE Trans. Audio, Speech, Lang. Process.* 17(6), 1124–1132. The
  HRIRs at 20–160 cm.
- Traer & McDermott (2016). Statistics of natural reverberation enable
  perceptual separation of sound and space. *PNAS* 113(48), E7856–E7865.
  The synthetic tail.
- Carlile & Leung (2016). The perception of auditory motion. *Trends in
  Hearing* 20. doi:10.1177/2331216516644254. Read for its summary of Lutfi
  & Wang (1999), Neuhoff (1998) and Seifritz et al. (2002), which were not
  read themselves.
- Ghazanfar & Maier (2009). Rhesus monkeys (*Macaca mulatta*) hear rising
  frequency sounds as looming. *Behavioral Neuroscience* (volume and pages
  not checked).
  Read from the authors' PDF, in summary.
- Neuhoff & McBeath (1996). The Doppler illusion: the influence of dynamic
  intensity change on perceived pitch. *J. Exp. Psychol. Hum. Percept.
  Perform.* 22(4), 970–985. Abstract only.
- Pointers for standard results, not read for this document: Morse &
  Ingard (1968), *Theoretical Acoustics*, for the moving point source and
  its convective factor; Laakso, Välimäki, Karjalainen & Laine (1996),
  Splitting the unit delay, *IEEE Signal Processing Magazine* 13(1),
  30–60, for fractional-delay interpolation; Woodworth & Schlosberg
  (1954), *Experimental Psychology*, for the spherical-head interaural
  delay used in C2 and C5.
