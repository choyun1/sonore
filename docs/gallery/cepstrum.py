"""Cepstral analysis: separating a voice's pitch from its timbre with the cepstrum.

This script is the gallery page https://choyun1.github.io/sonore/gallery/cepstrum.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/cepstrum.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Cepstral analysis
#
# A voiced sound is, to a first approximation, a train of glottal pulses filtered by the vocal
# tract. In the spectrum the two multiply: the pulses give a comb of harmonics at multiples of
# $F_0$, and the tract shapes their heights into formants. Take the logarithm and the product
# becomes a sum. The *cepstrum* is the Fourier transform of that log spectrum (Bogert et al.,
# 1963):
#
# $$c(q) = \mathcal{F}^{-1}\{\ln |X(f)|\}(q)$$
#
# Its variable $q$, the *quefrency*, is a time. The formants vary slowly along frequency, so
# they land at low quefrencies, the first millisecond or two. The harmonics ripple the log
# spectrum once every $F_0$ hertz, so they land at one period, $q = 1/F_0$, as a single peak.
# The two parts of the voice that were tangled in the spectrum come apart, and can be measured
# or edited separately.
#
# This page applies `so.Cepstrum` to the sentence from [Seeing speech](speech.html), and ends by
# comparing it with reference implementations.

# %% [markdown]
# ## The sentence, and code the examples share
#
# Every cepstrum below is taken from an STFT with Hann windows 40 ms long, every 5 ms. A window
# must hold about three periods for the harmonics to show as a ripple, and 40 ms is three periods
# at 75 Hz, below this speaker's lowest pitch.

# %%
import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


# The sentence at its native 16 kHz, and its F0 track (WORLD Harvest, 0 where unvoiced).
# Sources: docs/speech/SOURCES.md.
sentence = finish(so.load("docs/speech/bdl_arctic_a0131.flac"))
f0_times, f0_harvest = np.loadtxt("docs/speech/bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2).T
fs = sentence.fs

stft = so.STFT(sentence, win_dur=0.040, hop_dur=0.005)
cep = so.Cepstrum(stft)
FMAX = 5000

# %% [markdown]
# ## One frame
#
# Take one frame in the middle of a vowel. Its log spectrum is a comb of harmonics riding on the
# formant envelope. Its cepstrum has most of its weight in the first couple of milliseconds, and
# one clear peak at the pitch period. Keeping only the quefrencies below half a period (a
# *lifter*, the cepstral counterpart of a filter) and transforming back gives the smooth
# envelope.

# %% [about]
# Top: the frame's log spectrum and the liftered envelope. Bottom: its cepstrum, with the peak
# at one period. The envelope runs a few dB under the harmonic peaks because the lifter averages
# the peaks with the dips between them; spectral-envelope estimators such as WORLD's CheapTrick
# correct for this (Morise, 2015).

# %% [figure c1] One frame
i = int(np.argmin(np.abs(cep.t - 0.50)))
_, f0_all, peak_all = cep.f0()
t_frame, f0_frame, peak_frame = cep.t[i], f0_all[0, i], peak_all[0, i]
cutoff = 0.5 / f0_frame  # half a period
envelope = cep.lifter(cutoff).envelope()[0, :, i]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 5.6), layout="constrained")
ax0.plot(stft.f, 20 * np.log10(np.abs(stft.data[0, :, i])), color="0.4", lw=0.8, label="spectrum")
ax0.plot(stft.f, 20 * np.log10(envelope), color="tab:red", lw=1.6, label="liftered below half a period")
ax0.set(
    xlim=(0, FMAX),
    xlabel="Frequency [Hz]",
    ylabel="Level [dB]",
    title=f"Frame at {t_frame:.3f} s: log spectrum and envelope",
)
ax0.legend(loc="upper right", fontsize=8)
q_ms = cep.q * 1e3
ax1.plot(q_ms, cep.data[0, :, i], color="k", lw=0.8)
ax1.axvline(cutoff * 1e3, color="tab:red", ls="--", lw=1, label=f"lifter cutoff, {cutoff * 1e3:.1f} ms")
peak_label = f"peak at {1e3 / f0_frame:.2f} ms: F0 = {f0_frame:.0f} Hz"
ax1.plot(1e3 / f0_frame, peak_frame, "o", color="tab:blue", label=peak_label)
ax1.set(xlim=(0, 15), ylim=(-0.2, 0.6), xlabel="Quefrency [ms]", ylabel="Cepstrum", title="Its cepstrum")
ax1.legend(loc="upper right", fontsize=8)
for ax in (ax0, ax1):
    ax.grid(ls=":")

# %% [markdown]
# ## Pitch from the cepstrum
#
# Doing this for every frame gives a *cepstrogram*: time across, quefrency up. Wherever the
# voice is voiced, a bright line runs at one period, falling as the pitch rises. Picking the
# largest peak between 2.5 and 13.3 ms (400 and 75 Hz) in each frame is classic cepstral pitch
# estimation (Noll, 1967). A frame counts as voiced when its peak is taller than 0.1.

# %% [about]
# Top: the cepstrogram, with Harvest's pitch period drawn over it. Bottom: the cepstral F0 next
# to Harvest's. Each frame is judged on its own, with no continuity from one frame to the next,
# so the occasional frame jumps an octave.

# %% [demo c2] The cepstrogram and its pitch
t, f0_cep, peak = cep.f0(f_lo=75, f_hi=400)
voiced = f0_harvest > 0
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, layout="constrained")
cep.plot(ax0, colorbar=False)
ax0.set_xlabel("")
ax0.plot(f0_times[voiced], 1e3 / f0_harvest[voiced], ".", ms=1.5, color="c", label="Harvest period")
ax0.legend(loc="upper right", fontsize=8, markerscale=4)
ax1.plot(f0_times[voiced], f0_harvest[voiced], ".", ms=3, color="0.6", label="Harvest")
ax1.plot(t[f0_cep[0] > 0], f0_cep[0][f0_cep[0] > 0], ".", ms=3, color="tab:blue", label="cepstral peak > 0.1")
ax1.set(ylim=(60, 260), xlabel="Time [s]", ylabel="F0 [Hz]", title="Pitch")
ax1.legend(loc="upper right", fontsize=8, markerscale=2)
ax1.grid(ls=":")
for ax in (ax0, ax1):
    ax.set_xlim(0, sentence.duration)
playhead = [ax0, ax1]
sound = sentence

# %%
harvest_at = np.interp(t, f0_times, f0_harvest)
both = (harvest_at > 0) & (f0_cep[0] > 0)
agree = np.mean(np.abs(f0_cep[0][both] / harvest_at[both] - 1) < 0.05)
print(f"frames both call voiced: {both.sum()}; cepstral F0 within 5% of Harvest on {agree:.0%}")

# %% [markdown]
# ## Splitting the voice in two
#
# Lifter the other way and the two parts can be heard separately. Each frame keeps its own
# lifter cutoff, half of its pitch period, following Harvest's track (bridged across unvoiced
# stretches). Low quefrencies keep the vocal tract; high quefrencies keep the source.

# %%
voiced_t, voiced_f0 = f0_times[voiced], f0_harvest[voiced]
cutoffs = 0.5 / np.exp(np.interp(cep.t, voiced_t, np.log(voiced_f0)))  # half a period, per frame
tract = cep.lifter(cutoffs)  # low quefrencies: the envelope
source = cep.lifter(cutoffs, keep="high")  # high quefrencies: the harmonics
source.data[:, 0] = cep.data[:, 0]  # but keep each frame's mean log level, so pauses stay quiet


def show(snd, title, win_dur=0.005):
    fig, ax = plt.subplots(figsize=(10, 3.0), layout="constrained")
    so.STFT(snd, win_dur=win_dur, hop_dur=0.001).plot(ax, db_range=60, colorbar=False, fmax=FMAX)
    ax.set(xlim=(0, sentence.duration), title=f"{title} (Hann {win_dur * 1e3:g} ms spectrogram)")
    return fig, [ax]


# %% [about]
# The vocal tract alone: every frame's envelope given its minimum phase, so that each frame
# becomes one short pulse at the frame's center. The frames are 5 ms apart, so the pulses make
# a steady 200 Hz buzz. The words survive; the intonation does not.

# %% [demo c3] Envelope only, on a 200 Hz pulse train
robot = finish(tract.to_sound(phase="minimum"))
fig, playhead = show(robot, "Envelope only, minimum phase")
sound = robot

# %% [about]
# The source alone: the high quefrencies, plus each frame's overall level, with the original
# phase. The formants are flattened away, leaving the harmonics at roughly equal level, a buzzy
# voice that still carries the talker's intonation. The narrowband spectrogram shows the
# harmonics running flat across frequency.

# %% [demo c4] Harmonics only, the envelope flattened
flat = finish(source.to_sound(phase="original"))
fig, playhead = show(flat, "Harmonics only, original phase", win_dur=0.0333)
sound = flat

# %% [about]
# For comparison, both parts put back together: an unliftered cepstrum resynthesizes the
# sentence to rounding error.

# %% [demo c5] Both parts together
whole = cep.to_sound()
print(f"largest difference from the sentence: {np.abs(whole.data - sentence.data).max():.1e}")
fig, playhead = show(whole, "Unliftered, original phase")
sound = finish(whole)

# %% [markdown]
# ## Reference implementations
#
# `so.Cepstrum` is written from the definitions. `tools/crosscheck_cepstrum.py` compares it with
# three independent implementations; the numbers below are from SciPy 1.17.1 and Praat 6.1.38.
#
# - **MATLAB `rceps`.** Its documented definition, `real(ifft(log(abs(fft(x)))))`, written out in
#   NumPy for one 40 ms frame, matches sonore's real cepstrum to 4e-16.
# - **SciPy `scipy.signal.minimum_phase`** (`method="homomorphic", half=False`) builds a
#   minimum-phase filter by the same folding of the real cepstrum. On a 17-tap mixed-phase filter
#   it matches sonore's minimum phase to 4e-8 of the peak; SciPy adds a tiny constant to the
#   magnitude before the log, which is the whole difference.
# - **Praat's PowerCepstrogram** (Boersma & Weenink), through `parselmouth`, measures the same
#   peak by a different route: a Gaussian window, the power spectrum in dB, and the sound
#   resampled to 10 kHz. On the frames of this sentence that sonore calls voiced, the two peaks
#   agree within 5% on 98% of frames; on every frame Harvest calls voiced, on 81%.
#
# See also the design and its numerical checks, `docs/design/cepstrum.md`.

# %% [markdown]
# ## What this page leaves out
#
# - **Tracking.** Cepstral F0 judges each frame alone. Real pitch trackers, such as WORLD's
#   Harvest used above, generate several candidates per frame and choose a smooth path through
#   them, which removes the octave jumps.
# - **Better envelopes.** A plain low lifter sits under the harmonic peaks; WORLD's CheapTrick
#   smooths the spectrum over one $F_0$ first and corrects the lifter. Mel-cepstra, the basis of
#   MFCCs, warp the frequency axis first.
# - **The complex cepstrum.** It keeps the phase as well, but needs phase unwrapping, which is
#   fragile on real sounds; sonore provides the real cepstrum and takes phase from the STFT or
#   from the minimum-phase fold.

# %% [markdown]
# ## References
#
# - Boersma & Weenink. Praat: doing phonetics by computer. [praat.org](https://www.praat.org).
# - Bogert, Healy & Tukey (1963). The quefrency alanysis of time series for echoes: cepstrum,
#   pseudo-autocovariance, cross-cepstrum and saphe cracking. In M. Rosenblatt (Ed.), *Time Series
#   Analysis*. Wiley.
#   [`cepstrum.Cepstrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/cepstrum.py#L16)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence.
# - Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis.
#   *Speech Communication* 67, 1–7.
#   [doi:10.1016/j.specom.2014.09.003](https://doi.org/10.1016/j.specom.2014.09.003).
#   [`frames.TVGaborFrame.pitch_adaptive`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py#L494)
# - Morise, Yokomori & Ozawa (2016). WORLD: a vocoder-based high-quality speech synthesis system for
#   real-time applications. *IEICE Trans. Inf. & Syst.* E99-D(7), 1877–1884.
#   [doi:10.1587/transinf.2015EDP7457](https://doi.org/10.1587/transinf.2015EDP7457). Harvest.
# - Noll (1967). Cepstrum pitch determination. *J. Acoust. Soc. Am.* 41(2), 293–309.
#   [PubMed](https://pubmed.ncbi.nlm.nih.gov/6040805/).
#   [`cepstrum.Cepstrum.f0`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/cepstrum.py#L142)
# - Oppenheim & Schafer (2010). *Discrete-Time Signal Processing*, 3rd ed., ch. 13. Pearson.
#   [Pearson](https://www.pearson.com/en-us/subject-catalog/p/Oppenheim-Discrete-Time-Signal-Processing-3rd-Edition/P200000003226).
#   [`cepstrum.Cepstrum.to_stft`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/cepstrum.py#L109)
