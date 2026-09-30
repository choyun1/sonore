# Texture recordings

Short excerpts of real recordings, used by the texture-synthesis gallery
section and benchmark (McDermott & Simoncelli, 2011). All are CC0 or
public domain, so they are redistributed here. Regenerate them from the
original downloads with

```bash
python tools/make_texture_excerpts.py DIR   # DIR holds the files named below
```

## Sources

| Excerpt | Recording | Author | Source | License | Original | Excerpt (s) |
|---|---|---|---|---|---|---|
| `rain.flac` | Rain Ambience | nick121087 | [Freesound 234317](https://freesound.org/people/nick121087/sounds/234317/) | CC0 | 42.0 s, 48 kHz, 24-bit, stereo | 17.50–24.50 |
| `stream.flac` | water in creek burbling.wav | cognito perceptu | [Freesound 370870](https://freesound.org/people/cognito%20perceptu/sounds/370870/) | CC0 | 30.8 s, 44.1 kHz, 16-bit, stereo | 11.88–18.88 |
| `crickets.flac` | AMBIENCE NIGHT FIELD CRICKET 01.wav | sengjinn | [Freesound 175020](https://freesound.org/people/sengjinn/sounds/175020/) | CC0 | 60.4 s, 48 kHz, 24-bit, stereo | 26.72–33.72 |
| `applause.flac` | Small applause | Breviceps | [Freesound 462362](https://freesound.org/people/Breviceps/sounds/462362/) | CC0 | 4.5 s, 44.1 kHz, 16-bit, stereo | 0.49–3.99 |
| `fire.flac` | Crackling Fire | Sauron974 | [Freesound 204348](https://freesound.org/people/Sauron974/sounds/204348/) | CC0 | 95.5 s, 48 kHz, 24-bit, stereo | 37.75–57.75 |
| `mud.flac` | Fountain Paint Pot | NPS / Jennifer Jerrett | [Yellowstone Sound Library](https://www.nps.gov/yell/learn/photosmultimedia/sounds-fountainpaintpots.htm) ([MP3](https://www.nps.gov/nps-audiovideo/legacy/mp3/imr/avElement/yell-FountainPaintPot.mp3)) | Public domain (credit: National Park Service) | 120.5 s, 44.1 kHz, MP3, mono | 50.25–70.25 |
| `wind_rain.flac` | Wind Gusts and Rain.WAV | sonicwars | [Freesound 213872](https://freesound.org/people/sonicwars/sounds/213872/) | CC0 | 44.0 s, 44.1 kHz, 16-bit, stereo | 18.50–25.50 |

Licenses are as shown on each source page when downloaded (September 2026).

## Processing

Every excerpt is the **middle** of its recording (centered on the file's
midpoint), mixed to mono, resampled to 44.1 kHz, peak-normalized to 0.9, and
saved as 16-bit FLAC. Level is irrelevant to the texture model, which
normalizes RMS before measuring; the normalization only keeps quiet
recordings (crickets are at about −46 dBFS) well above 16-bit quantization.
The model itself works at 20 kHz, so everything above 10 kHz is ignored.

## Decisions

**Why these recordings.** Candidates were chosen for being textures in the
paper's sense (many similar events, statistically stationary over seconds)
and for being lossless where possible. Rejected along the way: "Rain Loop" by
qubodup (CC-BY 3.0, not CC0, though built from CC0 sources); "Crickets At
Night – Clean sound" by Defelozedd94 (noise-reduced in Audacity, which would
leave processing artifacts in the envelope statistics); "wind.wav" by
nithala (16 kHz sample rate, below the model's 20 kHz, leaving its top
filters empty). No clean public-domain recording of wind alone was found, so
wind appears only mixed with rain.

**Why the middle.** Recordings tend to start and end with handling noise,
fades, or, for applause, the swell and decay of the clapping. The middle is
the most likely stretch to be stationary.

**Why 7 s.** The MATLAB toolbox measures originals of up to 7 s. Excerpt
length trades stationarity against estimation noise. A test on synthetic
textures (two independent samples of the same texture, per-class SNR of
their statistics) found that dense textures lose only about 3–5 dB of
agreement going from 7 s to 3 s, but sparse ones collapse: with bursts at
3/s, kurtosis agreement fell to 8 dB at 2 s. What matters is the number of
events, not seconds. The slowest modulation band (0.5 Hz) is noisy at any
length under several seconds.

**Exceptions.**

- *Applause: 3.5 s.* The whole recording is 4.5 s; the first 0.5 s (onset)
  and last 0.5 s (decay) are trimmed.
- *Fire and mud: 20 s.* At 7 s both were too sparse to characterize. The
  check is split-half reliability: measure the first and second halves of an
  excerpt separately and compare (per-class SNR in dB; higher means the
  excerpt is long enough to pin down its own statistics).

  | Excerpt | Length | env var | env kurt | mod power | C1 |
  |---|---|---|---|---|---|
  | fire | 7 s | 5 | 6 | 6 | 11 |
  | fire | 20 s | 8 | 6 | 12 | 15 |
  | mud | 7 s | 11 | 3 | 8 | 10 |
  | mud | 20 s | 13 | 8 | 13 | 13 |

  Both improved, but fire's envelope kurtosis stays poorly determined: it is
  dominated by a handful of loud pops (about 2/s over a low-frequency
  rumble), and heavy-tailed statistics converge slowly. Expect fire's
  kurtosis SNR to be low in synthesis for this reason, not only because of
  the algorithm.

  For comparison, the 7 s excerpts of the dense textures give: rain 19 / 15 /
  16 / 8, stream 18 / 9 / 13 / 10, crickets 23 / 16 / 15 / 5, applause
  (3.5 s) 18 / 12 / 9 / 14, wind + rain 21 / 14 / 15 / 6. Low C1 values for
  dense noise-like textures reflect near-zero targets (see the SNR note in
  `sonore.texture.TextureStats.snr`), not unreliability.

## Notes on individual recordings

- **Rain**: stationary and broadband; a narrow spectral notch near 5 kHz and a
  steady rumble below about 200 Hz, both presumably from the recording
  environment. They are part of the texture.
- **Stream**: a faint steady tone around 500–600 Hz, probably the resonance of
  the hollow in the stones that the recordist describes.
- **Crickets**: nearly all energy above 1 kHz, with tonal lines at 3–7 kHz.
  The low channels are more than 30 dB below the loudest and are excluded
  from SNR, as intended.
- **Fire**: the source description mentions crickets in the background; in
  this excerpt a low-frequency rumble dominates the level and the pops are
  sparse.
- **Mud**: MP3 (the NPS monitoring systems record MP3), but its spectrum is
  clean well past the model's 10 kHz ceiling.
- **Wind + rain**: the recording loses about 8 dB of level over its 44 s; the
  middle excerpt is steadier than the whole and sounds more like rain on wind
  than distinct gusts.
