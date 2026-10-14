# Sound quilts

The plan is to add the sound quilts of Overath, McDermott, Zarate & Poeppel
(2015). A quilt cuts a source sound into segments of one length and strings
them together in a new order. Each next segment is the one whose join with
the current segment changes the cochleagram by about as much as the source
changed at the same point. The quilt keeps the source's structure within a
segment and across each join, and scrambles it beyond the segment length.
Varying the segment length (30 to 960 ms) located a region of the superior
temporal sulcus whose response grows with the length of natural speech
structure, up to about 500 ms. This document sets out the method, how it fits
sonore, and the decisions for Cho.

Status: proposed 2026-10-14. Cho agreed to a clean-room implementation from
the paper's methods, not a port of the MATLAB toolbox. All decisions
accepted by Cho 2026-10-14: D1 as two public names, `so.quilt` and
`so.splice`, with the order private (see D1); D2-D7 as recommended.

## How the claims are verified, and what was read

- **[source]**: from the paper, read in full from the PDF Cho supplied (main
  text and Online Methods, 12 pages).
- **[toolbox]**: from reading the Sound Quilting Toolbox v1.3 (Josh
  McDermott; `reorder_segs.m` by Dan Ellis), which Cho supplied. The
  toolbox has no license, so no code is copied or translated; it is
  consulted for behavior only, as was done for texture statistics.
- **[check]**: printed by `tools/check_quilting_claims.py`. It uses NumPy
  only, shares no code with sonore, and runs in about 20 s.
- **[math]**: follows in one line from the formulas shown.
- **[cross]**: printed by `tools/crosscheck_quilting.py`, which runs the
  toolbox in GNU Octave 10.3 (conda-forge, with the signal package) beside
  sonore. Dev-time only, since the toolbox cannot be in the repository.
- **[sound]**: measured with sonore itself by the gallery's Sound quilts
  page (docs/gallery/stimuli/quilts.py), with the seeds given there.

## The method [source]

1. Divide the source into segments of the chosen length.
2. Pick the first segment of the quilt at random, uniformly over the source.
3. From the current segment k, compute the change across the border to
   every other segment n: the summed squared difference between the last
   30 ms of k and the first 30 ms of n in the cochleagram,
   `d(k, n) = sum over t, f of (C_k^R(t, f) - C_n^L(t, f))^2`.
4. Choose as next segment the n whose change is closest to the source's own
   change after k: `argmin over n of |d(k, n) - d(k, k + 1)|`, with
   `n != k + 1` and n not used before.
5. Repeat from step 3 until the quilt is long enough.
6. Join the segments: shift the start of each new segment by up to 15 ms
   either way to maximize the cross-correlation with what precedes it, over
   the 30 ms around the join, then cross-fade with 30 ms raised-cosine ramps
   centered on the join (so each segment carries 15 ms extra at each end).

The cochleagram is the texture model's filterbank: 30 filters equally
spaced on the ERB_N scale from 20 to 10,000 Hz, Hilbert envelopes raised to
the power 0.3. Segment lengths were 30, 60, 120, 240, 480 and 960 ms.

Stimulus preparation outside the algorithm: 20 s sources, resampled to
20 kHz and bandpass filtered from 80 to 8500 Hz (third-order Butterworth,
forward and backward), breaths and pauses removed from speech, quilts 6 s
long with a linear fade over the last second.

The paper's step 5 reads "repeat steps 1-3"; steps 3-4 are meant.

## What the toolbox does differently [toolbox]

- **More exclusions.** Besides used segments, it never picks the current
  segment's two neighbors, segments 1 and 2, or the last two segments. The
  paper excludes only the source's own next segment and used ones. The
  extra exclusions at the ends keep the ±15 ms shift inside the signal.
- **Mean instead of sum.** The distance is a mean over time and bands.
  This divides `d(k, n)` and `d(k, k + 1)` by the same count, so the chosen
  segment is the same [math].
- **All 32 bands.** The distance uses the 30 bandpass filters and the
  lowpass and highpass edge filters.
- **Not pitch-synchronous.** The paper calls the joining PSOLA. What it
  does is align by cross-correlation and overlap-add, the idea of WSOLA
  (Verhelst & Roelands, 1993); no pitch marks are found.
- **Unnormalized cross-correlation.** The correlation is divided by the
  energy of the quilt's overlap region, which is the same for every shift,
  so the best shift is that of the raw cross-correlation [math]. Raw
  correlation favors shifts where the new segment is loud.
- **Level.** The source is scaled to RMS 0.03 first.
- **Length.** The quilt is a whole number of segments; no trim or fade.
- **Bugs, none of which affect the published stimuli** (mono sources, the
  regular filterbank, long sources), each confirmed in Octave [cross]: a
  two-channel source is reduced to two samples and the run fails ("'dist'
  undefined"); the 4x overcomplete filterbank option refers to an
  undefined variable ("'s' undefined"); and when no segment is left, every
  candidate gets the same placeholder distance, the tie returns several
  indices and the loop fails ("nonconformant arguments").
- **Alignment one sample off** [cross]. `reorder_segs` correlates the
  overlap with the source one sample later than the stretch it then reads,
  so each chosen shift is one sample short of the best alignment. Found by
  the cross-check, not by reading.

## Claims

- **Q1 [check].** Segments available and needed, for the paper's set-up
  (20 s at 20 kHz, 6 s quilts) and for a 30 s LibriSpeech passage of the
  gallery's speech (16 kHz):

  | Segment | Source segments (20 s) | Needed for 6 s | Source segments (30 s) |
  |---|---|---|---|
  | 30 ms | 666 | 200 | 1000 |
  | 120 ms | 166 | 50 | 250 |
  | 480 ms | 41 | 13 | 62 |
  | 960 ms | 20 | 7 | 31 |

  At 960 ms a 6 s quilt uses 7 of the 20 segments of the paper's sources,
  so different quilts from one source share material.
- **Q2 [check].** Choosing the order step by step, with each step taking
  differences against every segment directly (as the toolbox does), took
  about 15 s for 30 ms segments on 20 s (666 segments, 600 samples by 32
  bands per border; random envelopes; two runs on the build machine, 14.8
  and 15.8 s). Computing every pair's distance at once, as
  `|e|^2 + |s|^2 - 2 e.s` with one matrix product, took under 0.2 s (0.16
  and 0.18 s), chose the same order, and differed from the direct
  distances by at most 1.1e-15.

## How it fits sonore

- **A classic stimulus, made from a sound.** No listener model, nothing
  new claimed. The paper's controls are already in reach: its modulation
  and co-modulation controls are McDermott-Simoncelli syntheses with
  subsets of statistics (`so.texture`), and its noise-vocoded speech is a
  channel vocoder (`channel_vocode`, though the paper iterates the envelope
  imposition 30 times).
- **Where it lives.** `sources/__init__.py` says it: "Sounds made from
  another sound live with the analysis they go through." The order comes
  from `Envelopes`, so the module goes in `views` (D1).
- **Synthesis.** Rearranging a sound is not synthesis from a view, so the
  "synthesis lives in `to_sound`" rule does not apply: nothing is
  discarded and rebuilt. No new class is needed: the order is an array of
  segment indices, kept private (D1).
- **Pieces sonore already has.** `so.cosine_filterbank(30, 20, 10000)` is
  the paper's bank (the texture model uses it), `Filterbank.analyze` with
  padding is the toolbox's zero-padded filtering, and `.envelopes()` is
  the Hilbert envelope.
- **No silent change.** All of it is new API; nothing existing changes.

## Proposed design

    quilt = so.quilt(speech, 0.12, 6.0, rng=1)          # Sound, 6 s
    joined = so.splice(speech, [1.2, 0.36, 4.8], 0.12)  # any order, aligned joins

1. Cochleagram: `bank.analyze(sound, pad=...).envelopes()` raised to 0.3,
   on the default bank `so.cosine_filterbank(30, 20, 10000)` (a
   `filterbank=` option takes another).
2. Border windows: the first and last 30 ms of each segment
   (`border_dur=0.03`), flattened over bands.
3. Every pair's distance from one matrix product (Q2).
4. The order by the paper's rule (D2), with the first segment from `rng`.
5. Join with `so.splice` (D1, D3), then trim to `duration`.

Rough size, an estimate: 150 to 250 lines in `views/quilting.py` and
`core/processing.py` together, about as many in tests.

## Decisions

- **D1. Where it lives and what is public.** (a) `views/quilting.py`, with
  `so.quilt` (returns a `Sound`), `so.quilt_order` and its result class
  `QuiltOrder`; the joining step stays private. (b) The same in
  `texture/`, as `so.texture.quilt`. The toolbox's README presents
  quilting as the model-free alternative to McDermott-Simoncelli texture
  synthesis, and it shares that cochleagram, but speech quilts are not
  textures and the texture names stay behind `so.texture`. (c) Only
  `so.quilt`, with the order kept internal. One name fewer, but the
  change distributions (the toolbox's own diagnostic, and the gallery's
  plot) would be out of reach. (d) As (a), with the joining step public
  as well, in `core/processing.py` beside `concat`:
  `so.splice(sound, starts, segment_dur, *, crossfade=0.03,
  max_shift=0.015)` joins segments of one sound, taken from `starts` [s]
  in that order, with aligned cross-fades. `quilt` then calls it, so
  there is still one copy of the joining code. It serves stimuli that
  rearrange a sound by another rule (segments in random order as a control
  without the cochleagram rule, a segment repeated, a word spliced into a
  carrier) and lets a reader hear what the alignment does by setting
  `max_shift=0`. It takes source and positions rather than a list of
  sounds, because the shift reads samples beyond each segment's own
  edges. *Recommended: (d) (revised 2026-10-14 after Cho asked; the first
  draft said (a), keeping the joining private until something else needed
  it).* **Decided by Cho (2026-10-14): (e), (d) without `quilt_order`.**
  The public names are `so.quilt` and `so.splice`. The order is not a
  random permutation (choosing it is the quilting rule), but nothing needs
  it outside `quilt`: the change histograms can be measured on the quilt
  itself, since each join sits at a multiple of the segment length (the
  shift moves where a segment is read from, not where it is written). The
  same `rng` and source give the same quilt.
- **D2. Which segments may follow.** (a) The paper's rule: anything not
  used before and not the source's own next segment. Near the ends of the
  source, the ±15 ms shift is limited to what the signal holds. (b) The
  toolbox's rule, with its extra exclusions (above). *Recommended: (a),
  because the paper is the reference and (b) wastes four segments, which
  matters at 960 ms (16 of 20 left, Q1). If the source runs out, raise an
  error that says how many segments the quilt needs and how many the
  source has.* One case the paper leaves open: the source's last segment
  has no next segment, so no source change to match after it. The
  toolbox never picks the last two segments. *Default: the last segment
  may only end the quilt.*
- **D3. Joining.** Cross-correlation alignment within ±`max_shift` (15 ms)
  and raised-cosine cross-fades of `crossfade` (30 ms), as the paper says.
  (a) Raw cross-correlation, as the toolbox. (b) Normalized
  cross-correlation, which does not favor loud shifts. *Recommended: (a),
  so the stimuli are those of the paper, with the docstring saying what it
  favors. It is called "aligned overlap-add", not PSOLA, with the paper's
  word noted once.*
- **D4. Level, length and channels.** Keep the source's level (no RMS
  0.03). Trim to exactly `duration`. A multichannel source raises an
  error for now, because one shift per join cannot align every channel at
  once. Band-limiting, the final fade and pause removal are preparation,
  done in the gallery recipe with sonore's existing calls (`butter_filter`,
  `ramp`). *Recommended: as written.*
- **D5. Breath and pause removal.** The toolbox has a helper (30 ms RMS;
  remove runs of at least 8 windows more than 35 dB below the loudest,
  with 10 ms ramps at the cuts). *Recommended: leave it out; the gallery
  trims a passage by hand, as the moving-sound page already does.*
- **D6. Cross-check.** A dev-time script, `tools/crosscheck_quilting.py`,
  outside the test suite like `tools/crosscheck_texture_stats.py`, running
  a local copy of the toolbox in Octave (from conda-forge): (1) the
  border distances `d(k, n)` on the same source and filterbank should agree
  to rounding (the toolbox takes the Hilbert transform after removing the
  padding, sonore before, so the outermost samples may differ and are
  reported separately); (2) on an order the toolbox chose, sonore's
  joining should give the same shifts and the same waveform to rounding;
  (3) the bugs listed above, run once to confirm them. The toolbox's
  differences sonore keeps are listed in one place, as
  `DIFFERENCES_FROM_TOOLBOX` does for texture. *Recommended: as written.*
- **D7. Gallery.** A section "Sound quilts" on the "Two talkers" page or
  the speech page: a LibriSpeech passage quilted at 30, 120, 480 and
  960 ms, with cochleagrams and the change distributions. The passages are
  English, so meaning survives in long quilts, unlike the paper's German.
  It ends with Cho's demo (2026-10-14): a texture quilted at the same
  lengths, with `TextureStats` measured on the source and on each quilt,
  to show that a texture's quilt is another sample of the same texture
  while slow modulation statistics can drift at short lengths (the paper's
  speech quilts lost power at 3 to 5 Hz at 30 ms, Fig. 4a). The numbers
  shown will be measured when the page is built. *Recommended: the speech
  page, which is about the structure of speech.*

## As built

`so.splice` in `core/processing.py` and `so.quilt` in `views/quilting.py`,
as decided. The differences from the toolbox are listed in this document,
under "What the toolbox does differently" and below, rather than in a
`DIFFERENCES_FROM_TOOLBOX` constant: quilting has no settings object to hang
one on, and the list is short.

- **Cross-check against the toolbox [cross]**, on the toolbox's example
  speech at 20 kHz, 60 ms segments, a 6 s quilt (100 segments):
  - The filters: sonore's `cosine_filterbank(30, 20, 10000)` and the
    toolbox's `make_erb_cos_filters` on the same 400,001-point grid differ
    by at most 1.7e-12.
  - Envelopes made as the toolbox makes them (the Hilbert transform after
    removing the padding) from sonore's subbands differ by at most 7.8e-16,
    the border changes the toolbox records by 6e-15 of the largest, and the
    toolbox's exclusion rule run on sonore's changes reproduces the
    toolbox's order, 100 segments of 100.
  - sonore's own envelopes take the Hilbert transform of the padded
    subbands. They differ from the toolbox's by up to 2.1e-2 within 30 ms of
    the ends and by at most 1.5e-5 elsewhere (the largest envelope is
    0.20), and the border changes by 7e-4 of the largest. That is enough to
    flip one near tie, after which the orders part: 30 of 100 segments
    agree. sonore keeps its own envelopes.
  - Joining without shifts: `so.splice` and `reorder_segs` agree to 1.4e-17
    away from the outer ends (the toolbox fades the first and last half
    cross-fade; sonore starts and ends at full level).
  - Joining with shifts: at the toolbox's 99 joins, sonore's search picks
    the toolbox's shift plus one sample at 95, the same at 3 and a different
    peak at 1. Run one sample later, as the toolbox indexes it, the same
    search gives the toolbox's shift at all 99: the one-sample offset above.
- **Pauses.** The first build of the gallery quilted LibriSpeech passages
  with their pauses, and a 6 s quilt of 30 ms segments held 4 s of silence:
  one silent segment changes little into another, and the passage changes
  little across a silence. The gallery now cuts pauses first with the
  toolbox's thresholds (runs of at least eight 30 ms windows more than 35 dB
  below the loudest), and the cutting is `so.splice` of the kept windows
  with 10 ms cross-fades and no shifts. D5 stands: no helper in the API.
- **Short segments chain.** At 30 ms, quilts of speech show long, steady
  stretches: the rule keeps picking segments with similar spectra. The
  toolbox's rule does the same, since sonore's order matches it.
- **Texture quilts (D7) [sound]**, measured by the gallery: the statistic
  SNR averaged over the paper's classes, against the whole 20 s recording.
  Fire: the four 5 s stretches of the recording score 9.6 to 14.7 dB, and
  three 5 s quilts at each of 30, 120, 480 and 960 ms score 10.6 to 17.0 dB.
  Mud: stretches 13.1 to 16.5 dB, quilts 12.5 to 16.0 dB. One recording each
  and three seeds per length; a quilt matches the recording's statistics
  about as well as the recording's own stretches do.
- **Gallery page.** The section first sat on the Seeing speech page. At
  Cho's request to take it off the crowded Voices pages (2026-10-14), it
  is now a page of its own under Stimuli, since quilting is a way of
  making stimuli.
- **Level [sound].** Quilting keeps the level: over five seeds at each of
  30, 60, 120, 240, 480 and 960 ms, the median level of 100 ms windows of
  a 5 s fire quilt is within 0.7 dB of the whole recording's. The first
  page played each sound on its own (RMS 0.1, then scaled down if its peak
  passed 0.95), and the fire recording has one crackle at 9.98 s peaking
  at 0.90, five times its first 5 s's peak of 0.18. The quilts, which drew
  that crackle or another large pop, were turned down 16 to 17 dB and the
  original 6 dB, so the quilts played about 10 dB quieter. The page now
  plays a recording and its quilts at one gain.

## References

- Overath, T., McDermott, J. H., Zarate, J. M., & Poeppel, D. (2015). The
  cortical analysis of speech-specific temporal structure revealed by
  responses to sound quilts. *Nature Neuroscience*, 18, 903-911.
  doi:10.1038/nn.4021
- McDermott, J. H. Sound Quilting Toolbox v1.3 (MATLAB, 2016), with
  `reorder_segs.m` by D. P. W. Ellis. Consulted for behavior only.
- Verhelst, W., & Roelands, M. (1993). An overlap-add technique based on
  waveform similarity (WSOLA) for high quality time-scale modification of
  speech. *Proc. IEEE ICASSP*, 2, 554-557.
- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception
  via statistics of the auditory periphery: evidence from sound synthesis.
  *Neuron*, 71, 926-940.
  doi:10.1016/j.neuron.2011.06.032
