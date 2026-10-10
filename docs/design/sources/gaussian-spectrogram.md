# Random sounds from a Gaussian spectrogram

The plan is to add the synthetic sources of McDermott, Wrobleski & Oxenham
(2011). Each is white noise, shaped by a spectrogram drawn from a
multivariate Gaussian whose correlations fall off exponentially in time and
frequency, like those of natural sounds. The sounds have the coarse
statistics of real sources without being any recognizable one. That made
them the stimuli for the source-repetition experiments, and later for
source-and-room separation (Traer & McDermott, 2016). This document sets out
the model, how it fits sonore, and the decisions for Cho.

Status: proposed 2026-10-10, nothing built. Cho asked for the stimulus on
2026-10-10, after reading a 2017 prototype of his own (see "Prior code").

## How the claims are verified, and what was read

- **[source]**: from the paper, read in full from the PDF Cho supplied
  (main text and SI Materials and Methods, 11 pages).
- **[check]**: printed by `tools/check_gaussian_spectrogram_claims.py`. It
  uses NumPy only, shares no code with sonore, and runs in under a second.
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
Spectrotemporal ripples page or the Classic stimuli page (D7).

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
  *Recommended: the ripples page.*

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
