# Random sounds from a Gaussian spectrogram

The plan is to add the synthetic sources of McDermott, Wrobleski & Oxenham
(2011). Each is white noise, shaped by a spectrogram drawn from a
multivariate Gaussian whose correlations fall off exponentially in time and
frequency, like those of natural sounds. The sounds have the coarse
statistics of real sources without being any recognizable one. That made
them the stimuli for the source-repetition experiments, and later for
source-and-room separation (Traer & McDermott, 2016). This document sets out
the model, how it fits sonore, and the decisions for Cho.

Status: D1-D6 accepted by Cho 2026-10-10, all as recommended. Built as
`so.gaussian_spectrogram` (see "As built"). D7 decided 2026-10-10: the
section goes on the "Hearing a modulation spectrum" page, beside the drawn
blobs; the "Classic stimuli" page is renamed "Synthetic sounds" and links
to it. D8 accepted by Cho 2026-10-10 and built (see "As built").

## How the claims are verified, and what was read

- **[source]**: from the paper, read in full from the PDF Cho supplied
  (main text and SI Materials and Methods, 11 pages).
- **[check]**: printed by `tools/check_gaussian_spectrogram_claims.py`. It
  uses NumPy only, shares no code with sonore, and runs in under a second.
- **[sound]**: printed by `tools/check_gaussian_spectrogram_sound.py`,
  which uses sonore itself to measure what the built function does.
- **[blob]**: printed by `tools/check_db_blob_claims.py` (uses sonore and a
  prototype dB draw).
- **[math]**: follows in one line from the formulas shown.
- **[prior]**: measured on Cho's 2017 prototype, outside this repository
  (the archive is private; the scripts are kept with the project's notes).

## The model [source]

- **Analysis grid.** 39 filters equally spaced on the ERB_N scale from 20
  to 4000 Hz, with half-cosine frequency responses. The time windows are
  20 ms raised cosines. Adjacent filters and windows overlap by 50%. A cell
  of the spectrogram is the log RMS amplitude of one filter's output in one
  window.
- **Generating distribution.** The spectrogram is a multivariate Gaussian.
  The covariance of two cells is a constant variance times a temporal
  correlation times a spectral correlation (separable). Both correlations
  decay exponentially, with decay constants 0.075 per filter and 0.065 per
  time window. These approximate correlation functions measured on 350
  spoken words and 30 animal vocalizations. The paper describes no formal
  fit. The mean of each cell makes the stimuli's spectrum flat on average
  ("proportional to the corresponding filter bandwidth"). The variance is
  not given.
- **Synthesis.** Decompose a sample of white noise with the same filters
  and windows. Scale the signal in each window so that its log amplitude
  matches the sampled cell. Pass the result through the filter bank again
  and sum. Overlap means the resulting sound's spectrogram differs subtly
  from the sampled one, but the correlations survive (their Fig. 1 C, D).
- Onset and offset ramps are 10 ms half-Hanning windows.

## Claims

- **G1 [check].** The grid spacing is 26.33 / 40 = 0.658 ERB per filter
  (39 filters plus two edges make 40 equal steps, the same tiling as
  `so.cosine_filterbank`). The window hop is 10 ms. So the paper's
  constants are 0.114 per ERB (a correlation length of 8.78 ERB) and
  6.50 per second (154 ms). Lag-1 correlations are 0.928 across filters
  and 0.937 across windows. This uses Glasberg & Moore's ERB_N. sonore's
  ERB scale may differ slightly; the build prints its own spacing.
- **G2 [check].** On a regular grid, an exponential correlation is exactly
  a first-order autoregression. Run it along time, then along frequency,
  starting each run in its stationary distribution, and the field's
  covariance matches the Kronecker covariance to 1e-15. For 400 ms (39 x
  81 = 3159 cells), a Cholesky factorization of the full covariance took
  288 ms. The recursion took 0.6 ms (one run on the build machine).
- **G3 [check].** If the 2017 prototype's variance of 0.5 is read on a
  log10-amplitude grid, the cells have a standard deviation of 14.1 dB.
- **G4 [prior].** Two bugs in the 2017 prototype, which this design avoids.
  (1) The correlated vector was reshaped in the wrong order, scrambling the
  correlations. With the paper's constants, lag-1 correlation across
  channels came out 0.33 (wanted 0.93). (2) Each windowed noise segment was
  multiplied by its RMS instead of divided. Cell levels measured on the
  output correlated 0.72 with the intended grid; dividing gives 0.97. The
  prototype also used other constants (0.065 for frequency, 0.109 for
  time), of unknown origin.

- **G5 [sound].** How well a sound keeps the drawn cells, for 20 seeds:
  re-analyze it with the same filterbank and windows and compare cell
  levels in dB with those of the drawn envelopes. On the fine structure of
  a noise (D5 a): correlation 0.970, slope 0.947, rms difference 3.1 dB.
  On `to_sound("noise")` (D5 b), whose bands keep their own fluctuations:
  0.941, 0.937 and 4.4 dB.
- **G6 [sound].** Over 200 draws, the drawn cells' lag-1 correlations are
  0.931 across bands (intended 0.928 on sonore's ERB scale, whose spacing
  is 0.656 ERB) and 0.939 across windows (intended 0.937). The standard
  deviation is held to 14.1 dB within 5% by a test.
- **G7 [math].** Whether "flat on average" means flat in expected power or
  in mean level (D4) makes no difference to the shape: with the same
  standard deviation in every band, the lognormal correction
  `exp((sd ln10 / 10)^2 / 2)` is the same factor for all of them, so only
  the overall level changes, and that is set by the carrier anyway.

- **G8 [blob].** Drawing a blob target in dB instead of linear amplitude
  (D8). One blob at 4 Hz and 1 cyc/oct, 5 seeds, on `to_sound("noise")`:
  the linear draw at rms depth 0.2 (near its cap of about 0.28, from
  `views/modulation-targets.md`, C2) gives drawn envelopes of depth 0.20.
  Only 12% of the sound's modulation power (rates of at least 0.5 Hz)
  falls near the blob: within an octave of 4 Hz and 0.5 cyc/oct of 1. dB
  draws with standard deviations of 3, 6 and 10 dB give depths of 0.35,
  0.76 and 1.50, with 24%, 42% and 41% of the power near the blob. In the
  dB modulation spectrum the shares are 20%, 46% and 65%. The peak stays
  at 4 Hz in every case (5 Hz once), and at a density of 0.85 or 1.06,
  the grid's two nearest bins. One blob and five seeds; a single
  measurement, not a survey.

## Proposed design

A sampler builds the view, and the view's existing `to_sound` makes the
sound. That follows the philosophy's "synthesis lives in `to_sound`":

    env = so.gaussian_spectrogram(0.4, fs, rng=1)     # Envelopes
    sound = env.to_sound(so.gaussian_noise(0.4, fs, rng=2))

1. Build the filterbank (default `so.cosine_filterbank(39, 20, 4000)`) and
   the window centers (default 20 ms windows, 10 ms hop).
2. Draw the field over bandpass bands by windows with the AR(1) recursion
   (G2), scale it by the standard deviation, and add the mean.
3. Turn cells into envelopes. Each band's amplitude envelope is the sum
   over windows of the cell amplitude times that window's raised cosine.
   The raised cosines sum to 1, so this is the paper's scaling written as
   an envelope.
4. Return `Envelopes` on the filterbank. The edge bands get zero envelopes;
   say so in the docstring.

Tests: with a fixed seed, the field's lag-1 correlations match
`exp(-step / length)` within a stated tolerance over many draws; the field
is reproducible from `rng`; and the correlations measured on the
synthesized sound (re-analyzed with the same grid) are close to the
intended ones, as in the paper's Fig. 1 C, D. Texture synthesis is
untouched, so its output stays bit-for-bit identical.

Gallery: a short section with the paper's description, one sampled field
plotted next to the sound's measured envelopes, and audio. It goes on the
Hearing a modulation spectrum page (D7).

## Decisions

- **D1. Shape of the API.** (a) A sampler returning `Envelopes`, with the
  sound made by `to_sound` (above). (b) One function returning a `Sound`.
  It is simpler to call, but it holds synthesis code of its own, which the
  philosophy rules out. (c) A random pattern for `ripple_sound`, like
  `DynamicRipple`. Patterns live on an octave axis, though, while the
  paper's grid and constants are in ERB. *Recommended: (a).*
- **D2. Name.** `gaussian_spectrogram` (the paper's word), or
  `gaussian_envelopes` (sonore's word for the returned view, which the
  vocabulary also calls a cochleagram). *Recommended:
  `gaussian_spectrogram`, because readers of the paper will look for it;
  the docstring says it returns `Envelopes`.*
- **D3. Units of the correlations.** (a) Lengths in ERB and seconds
  (8.78 ERB and 154 ms by default, G1). These stay meaningful when the
  filterbank or window changes. (b) The paper's constants per filter and
  per window, which silently change meaning with the grid. *Recommended:
  (a), with the paper's values as defaults and the conversion in the
  docstring.*
- **D4. Level statistics.** The paper gives no variance. Options: a
  `sd_db` argument defaulting to 14.1 dB (G3; Cho's 2017 choice), or with
  no default. The mean is set so the long-term spectrum is flat on average
  (equal power per Hz). Should "on average" mean in expected power (which
  corrects for the lognormal) or in mean log level? *Recommended: `sd_db`
  with a default Cho confirms or replaces, and flat in expected power.*
- **D5. Carrier.** (a) `to_sound(noise_sound)`, which keeps only the
  noise's fine structure, so the noise adds no envelope fluctuations of its
  own. (b) `to_sound("noise")`, whose bands keep their own random
  fluctuations within each window. (c) The paper's exact step: divide each
  window by the noise's own RMS there. That is not a route `to_sound` has,
  and it would need a new carrier option. *Recommended: (a) in the
  docstring example, then measure how closely each of (a) and (b)
  reproduces the sampled cells before settling. (c) only if both fall
  short.*
- **D6. Out of scope.** The paper's conditional samples (probes that share
  part of a target), the mixtures, and the energetic-masking thresholds of
  Experiment 6 are experiment-specific. So is the 2017 "rove by up to 10x".
  They can be gallery recipes later if wanted. *Recommended: leave out.*
- **D7. Gallery page.** The Spectrotemporal ripples page (sounds defined
  by their spectrotemporal envelope) or the Classic stimuli page.
  *Recommended: the ripples page.* Decided by Cho: rename the Classic
  stimuli page "Synthetic sounds" (stimuli made from parameters alone),
  and put the section on "Hearing a modulation spectrum", since a random
  spectrogram is a random draw from a modulation spectrum, as the blobs
  are; Synthetic sounds links to it.

- **D8. Blobs drawn in dB.** This came up while building: "can the way
  of imposing statistics carry over to blobs?" A random-phase draw from a
  modulation spectrum is a Gaussian field. A field with exponential
  correlations has a Lorentzian modulation spectrum at zero rate and
  density, so this function is one particular "blob", drawn in dB. Linear
  draws cannot go deep, because envelopes can't go negative. dB draws can
  go as deep as asked and put three to five times more of the modulation
  power where the target is (G8). Proposal:
  `ModulationSpectrum.from_blobs(..., scale="db", sd_db=...)`, which uses
  the dB envelope analysis the class already has. `rms_depth` stays the
  linear option, and the default stays linear, so nothing changes for
  existing code. What a dB draw gives up: its envelopes are lognormal, so
  the blob shapes the spectrum of the log envelope, and the linear
  spectrum is that shape plus its spread, as G8's linear shares show.
  *Recommended: add it in this PR.* Accepted by Cho.

## As built

`src/sonore/sources/gaussian_spectrogram.py`:
`gaussian_spectrogram(duration, fs, band_correlation_erb=8.78,
time_correlation=0.154, sd_db=14.1, n_bands=39, f_lo=20, f_hi=4000,
window=0.020, rng=None)` returns `Envelopes` on
`so.cosine_filterbank(39, 20, 4000)`, edge bands zero. The field is drawn
by the recursion of G2. Each band's mean level is `10 log10` of its width
in Hz, relative to the mean width. Windows are centered every `window / 2`
from t = 0 until they cover the sound. Tests are in
`tests/sources/test_gaussian_spectrogram.py`: the recursion's covariance
against the Kronecker one, reproducibility, cell statistics (G6), the
flat-on-average slope, and a sound that keeps its drawn levels.

D8: `ModulationSpectrum.from_blobs(..., scale="db", sd_db=6)` draws the
blobs as the spectrum of envelopes in dB, with mean 0 dB and the given rms
in dB, using the dB envelope analysis. `rms_depth` (default 0.2) stays the
linear option and the default stays linear; giving the other scale's depth
raises `TypeError`. `tools/check_db_blob_claims.py` now calls this API and
prints the same G8 numbers as the prototype did. Tests are in
`tests/views/test_modulation.py`: the drawn spread and magnitudes, a depth
past the linear cap without clipping, the target found again in the sound's
dB spectrum, and the parameter checks.

## Prior code

Cho's 2017-18 SourceReverbInference prototype (with Maddie Cusimano)
implemented this generator in `sri/sigtools.py` for source-and-room
inference. G4 lists its bugs. The inference itself moves to the separate
sonore-inference project; only the stimulus belongs here.

## References

- McDermott, J. H., Wrobleski, D., & Oxenham, A. J. (2011). Recovering
  sound sources from embedded repetition. *PNAS*, 108(3), 1188–1193.
  doi:10.1073/pnas.1004765108
- Traer, J., & McDermott, J. H. (2016). Statistics of natural reverberation
  enable perceptual separation of sound and space. *PNAS*, 113(48),
  E7856–E7865. doi:10.1073/pnas.1612524113
- Glasberg, B. R., & Moore, B. C. J. (1990). Derivation of auditory filter
  shapes from notched-noise data. *Hearing Research*, 47, 103–138.
