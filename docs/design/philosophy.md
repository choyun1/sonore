# Design philosophy

The principles behind sonore's design, in plain words. The design documents
in this folder (`frames.md`, one section per step) give the derivations and the
numbered claims and decisions behind them; the code and its docstrings are
meant to be readable without them.

## Sounds and units

- **A sound is a value.** `Sound` is an immutable array plus a sampling rate,
  and every operation returns a new one. Code is written as pure array
  functions, which also keeps a JAX port mechanical.
- **Quantities are written in the units people think in.** Levels in dB
  (`snd + 6*dB`), times in seconds (`snd[0.1:0.5]`), frames and windows
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
  spectrograms are built on a frame and drop something (phase, fine
  structure, everything but a pitch). They are welcome as displays and
  features, and each says what it drops and whether a sound can be
  recovered from it: exactly, approximately (Griffin-Lim, texture synthesis,
  which search for a sound whose view matches through the exact frame
  underneath), or not at all.
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

## Verification and records

- **Implement from the papers, and check every mathematical claim
  numerically.** Design documents number their claims, and standalone scripts
  in `tools/` check them using only NumPy and SciPy, with every operator built
  as an explicit dense matrix, so the checks share no code with the
  implementation. Tests compare each frame against such a dense oracle.
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
