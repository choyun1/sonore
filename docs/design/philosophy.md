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
  ships (filterbanks, decimated filterbanks, STFTs, time-varying Gabor) reconstructs unmodified
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
  WORLD's three views the other two, a modulation spectrum a carrier for
  its modulation phase and fine structure (`views/modulation-targets.md`).
  A view gets `to_sound` only where a canonical route exists. Elsewhere it
  raises `NotInvertibleError` with the reason, and searching for a matching
  sound stays an experiment until a canonical method is found. Today
  `Spectrum`, `Cepstrum`, `PVAnalysis`, `ModulationSpectrum` and
  `TemporalEnvelope` have `to_sound`; the voice views (F0 track, spectral
  envelope, aperiodicity) go back together through `source_filter_synthesize`,
  which their refusal names (`layout/sound-first.md`).
- **"Envelope" means two things, and each has one class.** A *temporal*
  envelope is how a sound's level changes over time, in one band or in every
  band of a filterbank (`TemporalEnvelope`, which holds that filterbank or
  none). A *spectral* envelope is how power is spread over frequency once the
  harmonics are smoothed away, held on a grid of time windows and frequencies
  (`SpectralEnvelope`, whatever estimator made it, named by its `method`).
  Neither takes a plural class: one envelope per band is the banded case of
  the first, and one spectral envelope per time window is already the grid.
  What the views on a time-frequency grid share (`t`, `f`, `data`, reading
  as `view(t, f)`) is the `TimeFrequencyGrid` Protocol, not a base class
  (`views/spectral-envelope-classes.md`).
- **Synthesis lives in `to_sound`.** A function next to a view either
  builds the view from a sound (`f0_track`, `pv_analyze`) or is a recipe
  that analyzes, changes the view and calls its `to_sound` (`time_stretch`,
  `channel_vocode`); it holds no synthesis code of its own. Where two views
  share a route, the bigger one is the smaller one applied piece by piece:
  a banded `TemporalEnvelope.to_sound` is the single envelope's `to_sound`
  in every band, then the filterbank's synthesis, and a test holds them to
  that.
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
- **Say what a call makes by position, and every option by name.** The
  arguments without a default (the sound, the duration and rate, the F0) come
  first and positionally; everything after them is keyword-only, behind a
  bare `*`: `so.harmonic_complex(1.0, fs, 200, harmonics=range(1, 11))`,
  `so.STFT(snd, win_dur=0.025)`. A parameter with a default stays positional
  only when it names the thing itself rather than tuning it: `ax` of every
  `.plot`, `n_bands, f_lo, f_hi` of a filterbank or `channel_vocode`, the LF
  model's `rd`, an LPC's `order`. A new exception is argued in its design
  document, and the same kind of parameter gets the same treatment
  everywhere.
- **Build a thing with its class; a factory only picks arguments or names
  a result.** Values and analyses that have their own class are built by
  calling it: `so.Ripple(4, 1)`, `so.STFT(snd, win_dur=0.025)`,
  `so.MFCC(snd)`, `so.Sound(data, fs)` for an existing array. A function
  stands in for the constructor only in three cases: it chooses the
  arguments of one general class (`so.gammatone_filterbank` returns a
  `Filterbank`, as `so.pure_tone` returns a `Sound`); the class is just
  what an analysis returns (`so.f0_track` gives an `F0Track`); or the
  input is of another kind, which takes a classmethod (`Sound.load`,
  `ModulationSpectrum.from_blobs`). No subclass per variant, and no
  private class in a public signature: an argument that accepts "anything
  with this method" is typed by a public `Protocol` (`F0Contour`,
  `Distribution`, `EnvelopePattern`), so a user's own class works without
  inheriting anything. A private base class may share code, but it never
  appears in a signature.
- **Type every public signature, and type an option by the values it
  accepts.** A string option is a `Literal` (`Literal["linear", "db"]`), not
  `str`, so an editor offers its values and a checker catches a typo. The
  package passes mypy with the settings in `pyproject.toml`.
- **American spelling throughout:** "behavior", "color", "center",
  "analyze", "normalize", "gray", "modeling", "labeled", "license". This
  covers prose, docstrings, comments and names. Outside text keeps its own
  spelling: paper titles and quotations (Cuevas-Rodríguez et al.'s "binaural
  spatialisation"), authors' names (Grey, 1977), third-party license notices,
  file names in other repositories, and outside API names such as
  matplotlib's `Greys_r` colormap and HTML's `aria-labelledby`.

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
  "Reference implementations" in docs/references.md).
- **Document every deviation and data choice.** Differences from reference
  toolboxes, texture recordings and design decisions are written down
  (`so.texture.DIFFERENCES_FROM_TOOLBOX`, `docs/textures/SOURCES.md`, this
  folder).
- **Say when something is sonore's own.** sonore implements published
  methods and claims no new ones. Where a page combines them in a way no
  cited source describes (the background plus pops recipe on the Sound
  textures page), it says so, says what tested it and how (by ear, by
  whom, on which sounds), and claims nothing beyond that.
- **Write a design document before a large feature,** and agree on it before
  implementing.
- **Do not silently change existing output.** *(Inferred from how changes
  have been made, not stated as a rule in the documents.)* A change that
  touches shared code is checked to leave existing results unchanged; texture
  synthesis, in particular, is kept bit-for-bit identical, verified against
  the previous commit.
