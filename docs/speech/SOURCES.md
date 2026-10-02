# Speech recordings

The spoken sentences used by the gallery: the "seeing speech" sentence
(docs/design/frames.md, step 3: decisions D12 and D13), which is also the
target on the moving talkers page, and that page's two masker sentences.

## Sources

| File | Recording | Speaker | Source | Licence | Original |
|---|---|---|---|---|---|
| `bdl_arctic_a0131.flac` | CMU ARCTIC utterance `arctic_a0131` | `bdl` (US English, male) | CMU ARCTIC `bdl` database, file `arctic_a0131.wav`, downloaded by Cho from festvox.org/cmu_arctic (2026-10-01) | CMU ARCTIC licence (CMU, permissive; see below) | 2.53 s, 16 kHz, 16-bit, mono |
| `rms_arctic_a0132.flac` | CMU ARCTIC utterance `arctic_a0132` | `rms` (US English, male) | CMU ARCTIC `rms` database, file `arctic_a0132.wav`, supplied by Cho (2026-10-01) | CMU ARCTIC licence (see below) | 2.81 s, 16 kHz, 16-bit, mono |
| `rms_arctic_a0133.flac` | CMU ARCTIC utterance `arctic_a0133` | `rms` (US English, male) | CMU ARCTIC `rms` database, file `arctic_a0133.wav`, supplied by Cho (2026-10-01) | CMU ARCTIC licence (see below) | 4.82 s, 16 kHz, 16-bit, mono |

Citation: Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
*Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224.
https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html

**Licence.** The CMU ARCTIC notice is reproduced verbatim in
[`COPYING_CMU_ARCTIC`](COPYING_CMU_ARCTIC), copied from the `COPYING` file
of the `bdl` release Cho downloaded (supplied 2026-10-01). It grants use,
copying and modification for any purpose, without fee, on three conditions:
keep the copyright notice, conditions and disclaimer; mark modifications
clearly; keep the original authors' names. This folder meets them by
shipping that file next to the recordings, by the "Modifications" note
below, and by the citation above.

**The sentence:** "Providence had delivered him through the maelstrom."
From the release's prompt list, `etc/txt.done.data`, line 131.

## Processing

**The `rms` files.** Their release's notice, as supplied by Cho, is in
[`COPYING_CMU_ARCTIC_rms`](COPYING_CMU_ARCTIC_rms), with Windows line endings
converted to Unix ones. It is the same notice as the `bdl` one except for the
copyright year (2004 rather than 2003). The prompt text of these two
utterances has not been copied here.

**Modifications** (marked as the licence requires):
`bdl_arctic_a0131.flac` is the original `arctic_a0131.wav` re-encoded
losslessly as FLAC. The samples are identical (checked sample for sample);
there is no trimming, resampling or level change, and the gallery keeps
it at its native 16 kHz (D12).
`bdl_arctic_a0131.pm` is the release's pitch-mark file for this utterance,
unmodified.
`rms_arctic_a0132.flac` and `rms_arctic_a0133.flac` are the original
`.wav` files re-encoded losslessly as FLAC, identical sample for sample, with
no trimming, resampling or level change.

## F0 track

`bdl_arctic_a0131_f0.csv`: F0 every 5 ms (`time` [s], `f0` [Hz], 0 where
unvoiced), from WORLD's Harvest estimator through `pyworld` 0.3.5 with its
default search range (71–800 Hz). Regenerate it with

```bash
pip install pyworld   # development only; not a sonore dependency
python tools/make_speech_f0.py docs/speech/bdl_arctic_a0131.flac
```

- 506 time windows, 85% voiced, F0 88–193 Hz (median about 120 Hz).
- Why Harvest: D13 preferred an F0 track from the corpus's EGG channel.
  These files are the single-channel `wav/` versions, which carry no EGG
  channel, so D13's fallback applies.
- Sanity check (not a claim about Harvest's accuracy): on the 343 voiced
  time windows where a plain autocorrelation (40 ms Hann window, peak > 0.5) finds a
  clear period, 96% agree with Harvest to within 5%. On 9 time windows the
  autocorrelation gives twice Harvest's F0; none gives half. The time windows
  above 160 Hz are two brief runs (at 0.82 s and 1.71 s, 15–20 ms each).
- Cross-check against the corpus's own pitch marks
  (`bdl_arctic_a0131.pm`: 285 marks, about one per glottal cycle in voiced
  speech, and marks through unvoiced stretches too, such as a uniform
  93 Hz run at 1.46–1.60 s where Harvest finds no voicing). Where Harvest says voiced, the F0
  implied by consecutive marks agrees with Harvest to within 5% at 79% of
  the 251 intervals (median deviation 1.4%). The disagreements sit where the
  marks themselves alternate between short and long cycles (around 1.6–1.76
  s, cycle-to-cycle jumps such as 154, 89, 154 Hz), which Harvest's smooth
  track averages over. How the release made its pitch marks (from the
  speech or from an EGG channel) has not been checked.
- Licences: WORLD is modified BSD; `pyworld` is MIT (from its repository's
  LICENSE file). Neither is redistributed here, and only the track they
  produced is.
