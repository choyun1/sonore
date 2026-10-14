# Elevation

What sonore already does with sources above and below the horizontal
plane, what PKU-IOA's HRIRs hold across elevation, one place where
`HRIRSet.at` gives a wrong answer, and what a gallery section on elevation
would show. It ends with the decisions for Cho.

Status: D1–D3 accepted by Cho 2026-10-14, all as recommended, and built
(see "As built" below).

## Why

Cho asked (2026-10-14) whether sonore can do elevation cues. It mostly
can already: every gallery demo so far keeps sources at 0° elevation, so
nothing shows or checks what happens off the horizontal plane.

## What is already there

- Head-centered coordinates carry elevation: `hcc = (dist [cm], elev [°],
  azim [°])`, and `so.hcc_trajectory(dist, elev, azim)` takes each of the
  three as a number, a `(times, values)` pair or a function of time.
- `HRIRSet` interpolates over directions on each measured distance:
  spherical barycentric weights inside the triangles of the measured
  directions' convex hull, on onset-aligned shapes, with the onsets
  interpolated separately (moving-sound.md). Nothing in it treats
  elevation differently from azimuth.
- So `spatialize` and `move_sound` already render any elevation the data
  covers, and paths that change elevation. A 3 s noise swept through the
  median plane from −40° to overhead renders without error.

## What PKU-IOA holds

Numbers from `tools/check_elevation_claims.py`, on Cho's PKU-IOA `.dat`
files at 1 m.

**Coverage.** Rings every 10° from −40° to 90°: 72 directions (5° apart)
on each ring up to 50°, then 36 at 60°, 24 at 70°, 12 at 80° and one
overhead. Nothing below −40°.

**In the median plane only the spectrum changes.** On every ring, front
and back, the largest ITD is 46 µs and the largest broadband ILD 1.8 dB
(KEMAR is not quite symmetric). At 90° to the right on the horizontal
plane they are 763 µs and 15.2 dB. A source that moves straight up in
front of the head changes the two ears' signals in the same way, so what
tells up from down is the spectrum, shaped mostly by the pinna. That
matches the textbook account (Middlebrooks & Green, 1991).

**How large the spectral change is.** RMS difference of the magnitude
spectra over 1–16 kHz, both ears, median over directions on the rings
from −40° to 50°: 4.0 dB between measured neighbors 10° apart in
elevation, 1.8 dB between neighbors 5° apart in azimuth.

**How well interpolation fills in elevation.** With one ring left out,
the interpolation from the rings 10° above and below misses it by
1.5–4.1 dB (median per ring; the larger misses are below the horizontal
plane). Using the measured ring 10° below instead, with no interpolation,
misses by 3.1–4.5 dB. With every other azimuth left out (a 10° gap in
azimuth), interpolation misses by 1.4 dB. So interpolation across
elevation helps, but less than across azimuth, and real gaps on PKU-IOA
are 10° in elevation, half the gap tested here. The cause is probably
that spectral notches move in frequency with elevation and a weighted
sum of two spectra cannot move a notch, only blend two; this was not
measured.

## Not individual ears

PKU-IOA is one KEMAR manikin. Listeners localize with non-individual
HRTFs less well than with their own, mostly in elevation and in front
versus back (Wenzel et al., 1993). A gallery section should say that the
sounds may not seem to rise for every listener, and make no claim that
they do.

## A wrong answer below −40°

The convex hull of the measured directions closes the hole under −40°
with one flat face, which `scipy.spatial.ConvexHull` triangulates as a
fan: all 70 of its triangles share the corner at azimuth 105°. Their
longest sides reach 100°, where every other triangle's longest side is at
most 11.2°.

A direction below −40° falls in one of those triangles and is blended
from corners that can be far apart in azimuth: at (−45°, 0°) 12% of the
weight is on azimuth 105°, at (−50°, 90°) 21% on azimuths 340–345°, at
(−60°, 0°) 34%, and straight below 50% on 105° and 50% on 280–285°. No
warning is given. Positions outside the measured distances already raise
`ValueError`; positions outside the measured directions do not.

## A gallery section

A section "Up and down" on the Binaural cues page, which is about the
cues the two ears share or differ in: after the interaural cues, the
cue that is not interaural.

- A figure: the left ear's magnitude spectrum against elevation in the
  median plane, front and back, as an image (elevation × frequency, dB).
- Sounds: noise at a few fixed elevations in front (−40°, 0°, 40°, 80°),
  and one sweep from −40° in front, over the head, to −40° behind
  (azimuth 0° then 180°, which `hcc_trajectory` does with elevation
  running past 90°, or with two pieces).
- Text that says what the figure shows, that ITD and ILD stay near zero
  (the numbers above), and that KEMAR is not the listener's head.
- Every number on the page comes from a checker, as on the Moving talkers
  page.

The Binaural cues page does not load HRIRs today; this section would make
it download the 1 m PKU-IOA file, as Moving talkers does.

## Decisions for Cho

- **D1. Directions in the hole under the lowest ring.** (a) Raise
  `ValueError`, as for distances outside the measured ones, for any
  direction inside a hull triangle whose longest side is more than twice
  the median longest side on that distance (11.0° on PKU-IOA, so the limit
  is 22°; every triangle of the measured grid passes, the bottom face's
  wide ones fail). (b) Hold the lowest ring's elevation, so a path can dip
  below −40° and sound as if it stays at −40°. (c) Leave it, and document
  it. *Recommended: (a).* The rule names no database, so it also catches
  holes in other sets. A path that dips below fails loudly instead of
  picking up the far side of the head. It changes nothing above −40°;
  it changes, by raising, results that were wrong.
- **D2. Where the gallery section goes.** (a) The Binaural cues page, as
  above. (b) The Moving talkers page, with a talker rising instead of
  noise. *Recommended: (a)*, since the point is the cue, and the Binaural
  cues page holds the other cues one at a time.
- **D3. Better interpolation in elevation.** Not now. The gaps are 10°,
  and the measured shortfall (about 1 dB better than the nearest ring) is
  a reason to say so on the page, not yet a reason to build a different
  interpolation.

## As built

- `HRIRSet` marks, on each measured distance, the hull triangles whose
  longest side is more than twice the median longest side (62 of PKU-IOA's
  1582 at 1 m, all on the bottom face). A direction that lies only in such
  triangles raises `ValueError`; one on the edge of a wide triangle and a
  narrow one is kept, with the triangle it had before, so every direction
  above −40° gives the same HRIR to the bit (compared once with the code
  before this change, on 3000 random directions and the 793 measured ones
  at 1 m).
- The Binaural cues page has an "Up and down" section: noise at −40°, 0°,
  40° and 80° in front, with the left ear's spectrogram and responses, and
  noise moving through the median plane from −40° in front to −40° behind
  over 6 s, with its ITD and ILD drawn to the scale of a source at the right.
  The numbers it states are the median-plane and side numbers above. On the
  rendered sweep, measured in 10 ms windows, the ITD stays within 39 µs and
  the ILD within 2.8 dB.

## References

- Middlebrooks, J. C., & Green, D. M. (1991). Sound localization by human
  listeners. *Annual Review of Psychology* 42, 135–159.
  doi:10.1146/annurev.ps.42.020191.001031.
- Wenzel, E. M., Arruda, M., Kistler, D. J., & Wightman, F. L. (1993).
  Localization using nonindividualized head-related transfer functions.
  *JASA* 94(1), 111–123. doi:10.1121/1.407089.

Bibliographic details checked against Crossref on 2026-10-14; the
papers' content is cited as commonly summarized, not re-read for this note.
