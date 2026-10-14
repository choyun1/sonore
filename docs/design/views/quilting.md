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
the paper's methods, not a port of the MATLAB toolbox.

## How the claims are verified, and what was read

- **[source]**: from the paper, read in full from the PDF Cho supplied (main
  text and Online Methods, 12 pages).
- **[toolbox]**: from reading the Sound Quilting Toolbox v1.3 (Josh
  McDermott; `reorder_segs.m` by Dan Ellis), which Cho supplied. Read, not
  yet run. The toolbox has no license, so no code is copied or translated;
  it is consulted for behavior only, as was done for texture statistics.
- **[check]**: printed by `tools/check_quilting_claims.py`. It uses NumPy
  only, shares no code with sonore, and runs in about 20 s.
- **[math]**: follows in one line from the formulas shown.

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
  regular filterbank, long sources): a two-channel source is reduced to two
  samples; the 4x overcomplete filterbank option refers to an undefined
  variable; and when no segment is left, every candidate gets the same
  placeholder distance, the tie returns several indices and the loop fails.
  These are from reading and will be confirmed when the cross-check runs
  the toolbox in Octave.

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
  discarded and rebuilt. The quilt's order is not an analysis either, so
  it is not a `View`. The result class joins the tested list of public
  classes that are not analyses (`NOT_ANALYSES` in `tests/views/test_view.py`).
- **Pieces sonore already has.** `so.cosine_filterbank(30, 20, 10000)` is
  the paper's bank (the texture model uses it), `Filterbank.analyze` with
  padding is the toolbox's zero-padded filtering, and `.envelopes()` is
  the Hilbert envelope.
- **No silent change.** All of it is new API; nothing existing changes.

## Proposed design

    quilt = so.quilt(speech, 0.12, 6.0, rng=1)          # Sound, 6 s

    order = so.quilt_order(speech, 0.12, 6.0, rng=1)    # QuiltOrder
    order.segments            # source segment indices, in quilt order
    order.source_changes      # d(k, k + 1) along the source
    order.quilt_changes       # d between neighbors in the quilt
    order.plot()              # the two distributions, as the toolbox's example
    quilt = so.quilt(speech, 0.12, 6.0, order=order)    # the same quilt

1. Cochleagram: `bank.analyze(sound, pad=...).envelopes()` raised to 0.3,
   on the default bank `so.cosine_filterbank(30, 20, 10000)` (a
   `filterbank=` option takes another).
2. Border windows: the first and last 30 ms of each segment
   (`border_dur=0.03`), flattened over bands.
3. Every pair's distance from one matrix product (Q2).
4. The order by the paper's rule (D2), with the first segment from `rng`.
5. Joining (D3), then trim to `duration`.

Rough size, an estimate: 150 to 250 lines in `views/quilting.py`, about as
many in tests.

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
  plot) would be out of reach. (d) The joining step public as well, in
  `core/processing.py` beside `concat`, as a general "splice segments with
  aligned cross-fades". Nothing else needs it yet, and the philosophy adds
  generality when an experiment needs it. *Recommended: (a).*
- **D2. Which segments may follow.** (a) The paper's rule: anything not
  used before and not the source's own next segment. Near the ends of the
  source, the ±15 ms shift is limited to what the signal holds. (b) The
  toolbox's rule, with its extra exclusions (above). *Recommended: (a),
  because the paper is the reference and (b) wastes four segments, which
  matters at 960 ms (16 of 20 left, Q1). If the source runs out, raise an
  error that says how many segments the quilt needs and how many the
  source has.*
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
  via statistics recovered from natural sounds. *Neuron*, 71, 926-940.
  doi:10.1016/j.neuron.2011.06.032
