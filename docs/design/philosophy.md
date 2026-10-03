# Design philosophy

The principles behind sonore's design, in plain words. The design documents
in this folder (`frames/frames.md`, one section per step) give the derivations and the
numbered claims and decisions behind them; the code and its docstrings are
meant to be readable without them.

## Sound first

- **`Sound` is the bedrock, so analysis comes first.** What a user holds is
  a sound or an analysis of one. A way back to sound belongs on the
  analysis it goes back from. Only sounds made from parameters alone (tones,
  noises, chirps, the glottal source, Klatt) stand as plain functions.
- **Folders follow meaning.** A module lives where what it means puts it,
  so a reader finds it where they expect it. Import order is still enforced,
  but between modules (no cycles), not by folders. A voice is not a
  different kind of sound, so there is no `voice` subpackage: synthesizers
  live in `sources` and analyses of a voice in `views`.
- **Heavy dependencies are optional.** They go in optional extras. A
  component becomes its own distribution only when it needs a heavy
  dependency, a different release cadence, or a separate audience.

## Sounds and units

- **A sound is a value.** `Sound` is an immutable array plus a sampling rate,
  and every operation returns a new one. Code is written as pure array
  functions.
- **Quantities are written in the units people think in.** Levels in dB
  (`snd + 6*dB`), times in seconds (`snd[0.1:0.5]`), hops and windows
  specified in seconds and rounded to samples per sampling rate. Adding a bare
  number to a sound is an error rather than a silent DC offset.
- **No global side effects.** Plotting functions take an `ax` and never
  change matplotlib settings; randomness always comes from an explicit `rng`.

## Transforms

- **A transform is an operator with an exact inverse.** Every frame sonore
  ships (filterbanks, STFTs, time-varying Gabor) reconstructs unmodified
  coefficients to floating-point precision, on the whole signal, not just on
  the band it was designed for. Non-tight banks therefore get edge filters by
  default, so that they cover DC to Nyquist.
- **Edited coefficients have a defined meaning.** Synthesis of modified
  coefficients (masks, gains, imposed statistics) returns the least-squares
  signal: the one whose own coefficients are nearest. "Nearest" is measured
  in a stated norm, the one `Frame.energy` uses, and frame bounds, SNRs and
  tests all use that same norm.
- **Analysis always works; synthesis refuses what it cannot invert.** A bank
  with coverage gaps still gives a fine cochleagram, so `analyze` never
  refuses. `synthesize` raises when the frame bounds say there is no stable
  inverse, and the error reports the bounds. `frame_bounds` is exposed
  because it also says how much coefficient errors can be amplified.
- **Views may discard information.** Not every analysis is a frame.
  Magnitudes, cepstra, F0 tracks, modulation spectra and reassigned
  spectrograms drop something (phase, fine structure, everything but a
  pitch). They are welcome as displays and features. Every analysis is a
  frame's coefficients or a `View`, in the branches too (`InterauralCues`,
  `TextureStats`, `PVAnalysis`), and a test sorts every public class. A
  plain number such as `rms` is a measurement, not a view.
- **`synthesize` is exact; `to_sound` is a stated route back.** Only frames
  synthesize. Each view's `discards` sentence says what it dropped, and its
  `to_sound` takes exactly that as arguments: a spectrum takes a carrier
  for its phase, envelopes a carrier for their fine structure, each of
  WORLD's three views the other two. A view gets `to_sound` only where a
  canonical route exists. Elsewhere (a modulation spectrum, whose possible
  carriers are too many) it raises `NotInvertibleError` with the reason,
  and searching for a matching sound stays an experiment until a canonical
  method is found. Today `Spectrum`, `Cepstrum` and `PVAnalysis` have
  `to_sound`; envelopes and WORLD's views go back through `noise_vocode`
  and `world_synthesize`, which their refusal names (`layout/sound-first.md`).
- **Keep the simplest representation that loses nothing.** Subbands stay
  real; the complex analytic signal is computed from them exactly when
  envelopes or phase are needed.
- **Implement the thing itself, not an approximation of it.** The gammatone
  is the exact Fourier transform of its impulse response, causal by default,
  rather than an IIR approximation.
- **Add generality when an experiment needs it.** Iterative exact duals,
  complex-coefficient synthesis and unions of frames are all deferred until a
  use for them appears, and the simpler behavior is documented where it
  differs.

## Code

- **Name every variable for what it holds,** locals included: `n_ramp`,
  `start_phases`, `impulse_response`, not `n`, `ph`, `h`. A reader auditing a
  line should not have to work out what a name stands for. There are three
  exceptions, and only these: the field's own short names (`fs`, `f0`, `cfs`,
  `n_fft`, `itd` and the others listed under "Names and conventions" in the
  API reference); names taken from a cited paper or from a reference
  implementation that sonore ports, where matching the source matters more,
  so a reader can check one against the other (Klatt's synthesis parameters
  such as `AV` and `F1`, and a resonator's coefficients `a`, `b`, `c`; the LF
  model's `rd`, `ra`, `rg`, `rk`, `te`, `tp`, `alpha`, `epsilon`; McDermott and
  Simoncelli's texture statistics C, C1, C2; WORLD's `q1` and `f0_floor`); and
  symbols a docstring defines as the notation of a formula, used in the code
  that implements it. A rename sweep leaves all of these alone. A new
  exception is written down here or in the docstring that uses it.
- **"Frame" means only the mathematical frame.** A frame is an analysis
  with frame bounds 0 < A ≤ B and so a stable exact inverse: `Frame`,
  `GaborFrame`, `TVGaborFrame`, a filterbank seen as a frame, the frame
  operator. The speech and STFT sense, one point of an analysis's time grid
  and the stretch of sound under the window there, is a **time window**:
  "per time window", "voiced time windows", shapes `(n_channels, n_windows)`.
  Write "time window" in full, because a bare "window" is the window shape
  (Hann, `win_dur`). The spacing of time windows is the **hop**, also where
  WORLD says "frame period". Outside names keep their own words (pyworld's
  `frame_period=`, paper titles), and an animation's frames are "video
  frames". In prose, the classes are always set as code (`Frame`,
  `GaborFrame`, `TVGaborFrame`), so a capital-F "Frame" in plain text never
  stands for the class.

## Verification and records

- **Implement from the papers, and check every mathematical claim
  numerically.** Design documents number their claims, and standalone scripts
  in `tools/` check them using only NumPy and SciPy, with every operator built
  as an explicit dense matrix, so the checks share no code with the
  implementation. The tests build a different dense oracle: they analyze
  unit impulses with the frame's own `analyze`, so they check the bounds,
  energy and exact synthesis of what the code computes, while the `tools/`
  checkers and specific tests check that analysis against the formulas.
- **Checking and measuring are kept apart.** A `check_*_claims.py` script
  in `tools/` is independent of sonore. A script that runs sonore to measure
  what it does (`measure_docstring_numbers.py`, and the older
  `check_voice_change_claims.py`, `check_female_voices.py`,
  `check_moving_trajectories.py`, `check_pku_ioa_sofa.py` and
  `check_reorganization_walkthrough.py`) says so in its docstring. A number
  a docstring quotes about sonore's behavior names the test or script that
  measures it.
- **Cross-check against independent implementations** where one exists (see
  "Reference implementations" in the README).
- **Document every deviation and data choice.** Differences from reference
  toolboxes, texture recordings and design decisions are written down
  (`so.texture.DIFFERENCES_FROM_TOOLBOX`, `docs/textures/SOURCES.md`, this
  folder).
- **Write a design document before a large feature,** and agree on it before
  implementing.
- **Do not silently change existing output.** *(Inferred from how changes
  have been made, not stated as a rule in the documents.)* A change that
  touches shared code is checked to leave existing results unchanged; texture
  synthesis, in particular, is kept bit-for-bit identical, verified against
  the previous commit.
