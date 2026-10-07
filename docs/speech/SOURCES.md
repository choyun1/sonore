# Speech recordings

The spoken sentences used by the gallery: the "seeing speech" sentence
(docs/design/frames/frames.md, step 3: decisions D12 and D13), which is also the
target on the moving talkers page, and that page's two masker sentences;
and the same sentence read by a female speaker, for checking the analyses on
a voice an octave higher (tools/check_female_voices.py); and four passages
from LibriSpeech for the cocktail-party scenes on the moving talkers page.

## Sources

| File | Recording | Speaker | Source | License | Original |
|---|---|---|---|---|---|
| `bdl_arctic_a0131.flac` | CMU ARCTIC utterance `arctic_a0131` | `bdl` (US English, male) | CMU ARCTIC `bdl` database, file `arctic_a0131.wav`, downloaded by Cho from festvox.org/cmu_arctic (2026-10-01) | CMU ARCTIC license (CMU, permissive; see below) | 2.53 s, 16 kHz, 16-bit, mono |
| `rms_arctic_a0132.flac` | CMU ARCTIC utterance `arctic_a0132` | `rms` (US English, male) | CMU ARCTIC `rms` database, file `arctic_a0132.wav`, supplied by Cho (2026-10-01) | CMU ARCTIC license (see below) | 2.81 s, 16 kHz, 16-bit, mono |
| `rms_arctic_a0133.flac` | CMU ARCTIC utterance `arctic_a0133` | `rms` (US English, male) | CMU ARCTIC `rms` database, file `arctic_a0133.wav`, supplied by Cho (2026-10-01) | CMU ARCTIC license (see below) | 4.82 s, 16 kHz, 16-bit, mono |
| `slt_arctic_a0131.flac` | CMU ARCTIC utterance `arctic_a0131` | `slt` (US English, female) | CMU ARCTIC `slt` database, file `arctic_a0131.wav`, supplied by Cho (2026-10-02) | CMU ARCTIC license (see below) | 2.64 s, 16 kHz, 16-bit, mono |

Citation: Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
*Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224.
https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html

**License.** The CMU ARCTIC notice is reproduced verbatim in
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

**The `slt` file.** Its release's notice, as supplied by Cho (2026-10-02),
is byte for byte the `bdl` one, so [`COPYING_CMU_ARCTIC`](COPYING_CMU_ARCTIC)
covers it too. It reads the same sentence as `bdl_arctic_a0131.flac`.

**Modifications** (marked as the license requires):
`bdl_arctic_a0131.flac` is the original `arctic_a0131.wav` re-encoded
losslessly as FLAC. The samples are identical (checked sample for sample);
there is no trimming, resampling or level change, and the gallery keeps
it at its native 16 kHz (D12).
`bdl_arctic_a0131.pm` is the release's pitch-mark file for this utterance,
unmodified.
`slt_arctic_a0131.flac` is likewise the original `arctic_a0131.wav` of the
`slt` release re-encoded losslessly, identical sample for sample.
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
- Licenses: WORLD is modified BSD; `pyworld` is MIT (from its repository's
  LICENSE file). Neither is redistributed here, and only the track they
  produced is.

## LibriSpeech passages

| File | Reader (LibriVox ID, name) | Sex | Book, chapter | Utterances | Length |
|---|---|---|---|---|---|
| `librispeech_1462_170142.flac` | 1462, E. Tavano | female | *Alexander's Bridge* (version 2), Chapter 6 | `1462-170142-0000` to `-0004` | 31.6 s |
| `librispeech_1993_147149.flac` | 1993, Wendy Belcher | female | *Mary Barton*, "Jem Wilson's Repulse." | `1993-147149-0000` to `-0003` | 37.2 s |
| `librispeech_3000_15664.flac` | 3000, Brian von Dedenroth | male | *Steep Trails*, "05 - Shasta Rambles and Modoc Memories" | `3000-15664-0000` to `-0002` | 31.8 s |
| `librispeech_1272_141231.flac` | 1272, John Rose | male | *Planet of the Damned*, Chapter 01 | `1272-141231-0000` to `-0003` | 30.8 s |
| `librispeech_2035_152373.flac` | 2035, Sharon Bautista | female | *Popular History of Ireland, Book 01*, "06 - Kings of the Seventh Century" | `2035-152373-0000` to `-0002` | 34.5 s |
| `librispeech_6345_93306.flac` | 6345, Jean Bascom | female | *Literary Sense*, "05 The Girl With The Guitar" | `6345-93306-0000` to `-0001` | 35.8 s |
| `librispeech_5694_64029.flac` | 5694, Winston Tharp | male | *'Co. Aytch,' Maury Grays, First Tennessee Regiment*, "06 - Murfreesboro" | `5694-64029-0000` to `-0006` | 32.6 s |
| `librispeech_251_136532.flac` | 251, Mark Nelson | male | *Omnilingual*, Part 4 | `251-136532-0000` to `-0003` | 40.5 s |
| `librispeech_2428_83705.flac` | 2428, Stephen Kinford | male | *Amusement Only*, "29 - Mr. Whitings and Mary Ann" | `2428-83705-0000` to `-0003` | 30.2 s |
| `librispeech_5338_284437.flac` | 5338, S R Colon | female | *Sky Island* (version 2), "14 - Tourmaline the Poverty Queen" | `5338-284437-0000` to `-0005` | 32.0 s |
| `librispeech_8842_304647.flac` | 8842, Mary J | female | *Sonnets of Michael Angelo Buonarroti and Tommaso Campanella*, "Campanella XVI-XXX" | `8842-304647-0000` to `-0002` | 44.7 s |
| `librispeech_2035_147961.flac` | 2035, Sharon Bautista | female | *My Antonia*, "Book 1 (The Shimerdas), Chapter 8" | `2035-147961-0000` to `-0003` | 30.1 s |
| `librispeech_8297_275155.flac` | 8297, David Mecionis | male | *Evil Genius*, "Fifth Book - Chapter XL - Keep Your Temper" | `8297-275155-0000` to `-0003` | 30.4 s |
| `librispeech_5694_64025.flac` | 5694, Winston Tharp | male | *'Co. Aytch,' Maury Grays, First Tennessee Regiment*, "02 - Shiloh" | `5694-64025-0000` to `-0005` | 35.3 s |
| `librispeech_2428_83699.flac` | 2428, Stephen Kinford | male | *Amusement Only*, "23 - An Old-Fashioned Christmas, Chapter 1" | `2428-83699-0000` to `-0005` | 38.4 s |

Source: the `dev-clean` subset of LibriSpeech (`dev-clean.tar.gz` from
https://www.openslr.org/12/, supplied by Cho, 2026-10-03). Reader names, sex,
book and chapter titles are from the corpus's `SPEAKERS.TXT` and
`CHAPTERS.TXT`. The recordings are LibriVox readings of public-domain books.

Citation: Panayotov, V., Chen, G., Povey, D. & Khudanpur, S. (2015).
LibriSpeech: an ASR corpus based on public domain audio books. *Proc. IEEE
ICASSP 2015*, 5206–5210. https://doi.org/10.1109/ICASSP.2015.7178964

**License.** "LibriSpeech (c) 2014 by Vassil Panayotov. LibriSpeech ASR
corpus is licensed under a Creative Commons Attribution 4.0 International
License" (the corpus's `LICENSE.TXT`;
https://creativecommons.org/licenses/by/4.0/). Attribution is the citation
and this table; the changes are listed below.

**Why these readers.** All 40 `dev-clean` readers were measured (speech-to-pause
level range, median F0, chapter length), and the 12 with the cleanest
recordings and a spread of F0 were auditioned by Cho, who found all of them
usable. All 12 are used, with median F0 of about 173, 180, 195, 202, 203 and
237 Hz (female) and 94, 111, 118, 118, 120 and 136 Hz (male), from
`so.f0_track` on each reader's first minute. Each cocktail-party scene has its
own cast; the three readers who also appear in the six-talker scene read
there from their second-longest chapter.

**Changes** (as CC BY 4.0 asks): each file joins the first consecutive
utterances of one chapter of the reader in `dev-clean` (the longest, or for
the second file of a reader the second longest), in order, with
0.3 s of silence between them, until at least 30 s; the samples of each
utterance are unchanged (16 kHz, 16-bit). No trimming inside utterances,
resampling or level change. The transcripts are not copied here.
