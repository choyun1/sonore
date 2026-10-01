# Speech recordings

The spoken sentence used by the "seeing speech" gallery section
(docs/design/frames.md, step 3: decisions D12 and D13).

## Sources

| File | Recording | Speaker | Source | Licence | Original |
|---|---|---|---|---|---|
| `bdl_arctic_a0131.flac` | CMU ARCTIC utterance `arctic_a0131` | `bdl` (US English, male) | CMU ARCTIC `bdl` database, file `arctic_a0131.wav`, downloaded by Cho from festvox.org/cmu_arctic (2026-10-01) | CMU ARCTIC licence: **to be confirmed** (see below) | 2.53 s, 16 kHz, 16-bit, mono |

Citation: Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
*Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224.
https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html

**Status of the licence.** The CMU ARCTIC databases are distributed under a
permissive CMU notice (use, copy and modify for any purpose, keeping the
copyright notice, conditions and disclaimer, marking modifications, and
keeping the authors' names). That summary comes from a copy redistributed by
another project, not from the corpus itself. The release's own `COPYING`
file has not yet been read, so this file does not yet reproduce it. That must
be done before the recording is merged.

**The sentence text** has not been checked against the corpus prompt list
(`etc/txt.done.data` in the release); it will be added with the licence.

## Processing

`bdl_arctic_a0131.flac` holds exactly the samples of the original WAV file,
re-encoded losslessly as FLAC (checked sample-for-sample). There is no
trimming, resampling or level change. The gallery keeps it at its native
16 kHz (D12).

## F0 track

`bdl_arctic_a0131_f0.csv`: F0 every 5 ms (`time` [s], `f0` [Hz], 0 where
unvoiced), from WORLD's Harvest estimator through `pyworld` 0.3.5 with its
default search range (71–800 Hz). Regenerate it with

```bash
pip install pyworld   # development only; not a sonore dependency
python tools/make_speech_f0.py docs/speech/bdl_arctic_a0131.flac
```

- 506 frames, 85% voiced, F0 88–193 Hz (median about 120 Hz).
- Why Harvest: D13 preferred an F0 track from the corpus's EGG channel.
  These files are the single-channel `wav/` versions, which carry no EGG
  channel, so D13's fallback applies.
- Sanity check (not a claim about Harvest's accuracy): on the 343 voiced
  frames where a plain autocorrelation (40 ms Hann frame, peak > 0.5) finds a
  clear period, 96% agree with Harvest to within 5%. On 9 frames the
  autocorrelation gives twice Harvest's F0; none gives half. The frames
  above 160 Hz are two brief runs (at 0.82 s and 1.71 s, 15–20 ms each).
- Licences: WORLD is modified BSD; `pyworld` is MIT (from its repository's
  LICENSE file). Neither is redistributed here, and only the track they
  produced is.
