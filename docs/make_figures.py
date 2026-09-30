"""Regenerate the README figures: ``python docs/make_figures.py``."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sonore as so  # noqa: E402
from sonore import dB  # noqa: E402

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)
FS = 44100
rng = np.random.default_rng(1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# 1. overview of iterated rippled noise: pitch at 1/delay, spectral ripple at 8 cyc/kHz
irn = so.iterated_ripple_noise(1.0, FS, delay=8e-3, iterations=16, rng=rng)
fig = so.overview(irn, win_dur=50e-3, figsize=(11, 6.5), fmax=4000)
fig.suptitle("so.overview(so.iterated_ripple_noise(1.0, 44100, delay=8e-3))", family="monospace")
save(fig, "overview_irn.png")

# 2. ideal binary mask: a gliding harmonic target in noise at -5 dB SNR
t = np.arange(FS) / FS
f0 = 150 + 100 * t
phase = 2 * np.pi * np.cumsum(f0) / FS
target = so.Sound(sum(np.cos(k * phase) / k for k in range(1, 30)), FS).normalize()
target = target * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
masker = so.gaussian_noise(1.0, FS, tilt=-3, rng=rng)
mixture = target + (masker + 5 * dB)
S_t, S_m, S_x = (so.STFT(s, 25e-3) for s in (target, masker, mixture))
separated = (S_x * so.ideal_binary_mask(S_t, S_m)).to_sound()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, layout="constrained")
S_x.plot(axes[0], fmax=5000, colorbar=False)
axes[0].set_title("Mixture (target at -5 dB SNR)")
so.ideal_binary_mask(S_t, S_m).plot(axes[1])
axes[1].set_title("Ideal binary mask")
so.STFT(separated, 25e-3).plot(axes[2], fmax=5000, colorbar=False)
axes[2].set_title("Masked resynthesis")
save(fig, "ibm.png")

# 3. binaural cues: Oscor and Phasewarp
fig, axes = plt.subplots(2, 2, figsize=(12, 5.5), sharex=True, layout="constrained")
for col, (name, snd) in enumerate(
    [
        ("so.oscor(3, fs, f_mod=2)", so.oscor(3, FS, 2, rng=rng)),
        ("so.phasewarp(3, fs, f_mod=2)", so.phasewarp(3, FS, 2, rng=rng)),
    ]
):
    c = so.interaural_cues(snd, 10e-3)
    axes[0, col].plot(c.t, 1e6 * c.itd, color="m", lw=0.8)
    axes[0, col].set(title=name, ylabel="ITD [µs]")
    axes[1, col].plot(c.t, c.corr0, color="tab:orange", label="zero-lag correlation")
    axes[1, col].plot(c.t, c.iac, color="k", alpha=0.6, label="coherence (peak)")
    axes[1, col].set(xlabel="Time [s]", ylabel="Interaural corr.", ylim=(-1.05, 1.05))
    for ax in axes[:, col]:
        ax.grid(ls=":")
axes[1, 0].legend(loc="lower left", fontsize=8)
save(fig, "binaural_cues.png")

# 4. noise vocoder: subband envelopes of the target drive noise carriers
syllables = target * np.sin(2 * np.pi * 4 * t).clip(0) ** 2  # syllable-like 4 Hz on/off
voc = so.noise_vocode(syllables, n_bands=8, rng=rng)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True, layout="constrained")
so.STFT(syllables, 25e-3).plot(axes[0], fmax=6000, colorbar=False)
axes[0].set_title("Original")
so.STFT(voc, 25e-3).plot(axes[1], fmax=6000, colorbar=False)
axes[1].set_title("so.noise_vocode(snd, n_bands=8)")
save(fig, "vocoder.png")

# 5. phase vocoder: time stretch, pitch shift, and an oscillator-bank frequency remap
t2 = np.arange(int(1.0 * FS)) / FS
vib = 2 * np.pi * np.cumsum(220 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t2))) / FS
sung = so.Sound(sum(np.cos(k * vib) / k for k in range(1, 20)), FS).normalize().ramp(30e-3)
panels = [
    ("original (220 Hz, 5 Hz vibrato)", sung),
    ("so.time_stretch(snd, 2)", so.time_stretch(sung, 2)),
    ("so.pitch_shift(snd, 7)", so.pitch_shift(sung, 7)),
    (
        "pv_analyze(snd).resynthesize(freq_map=lambda f: f + 70)",
        so.pv_analyze(sung).resynthesize(freq_map=lambda f: f + 70),
    ),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True, layout="constrained")
for ax, (title, s) in zip(axes, panels, strict=True):
    so.STFT(s, 46e-3).plot(ax, fmax=3000, colorbar=False, db_range=70)
    ax.set_title(title, fontsize=9, family="monospace")
save(fig, "phase_vocoder.png")

# 6. filterbank: decompose an exponential sweep into 6 ERB-spaced bands and reconstruct it
sweep = so.exponential_chirp(1.0, FS, 100, 6000).ramp(20e-3)
sb = so.subbands(sweep, n_bands=6, f_lo=100, f_hi=6000)  # 6 bandpass + lowpass/highpass edges
recon = sb.synthesize()
err = recon - sweep

fig = plt.figure(figsize=(13, 5.2), layout="constrained")
left, right = fig.subfigures(1, 2, width_ratios=[1.35, 1])
band_axes = left.subplots(len(sb), 1, sharex=True)
sb.plot(band_axes)  # the built-in stacked-waveform plot
band_axes[0].set_title(
    "so.subbands(sweep, n_bands=6, f_lo=100, f_hi=6000).plot()", family="monospace", fontsize=9
)

r_axes = right.subplots(3, 1, sharex=True)
sweep.plot(r_axes[0], color="k")
r_axes[0].set_title("Original: exponential sweep, 100 Hz to 6 kHz")
recon.plot(r_axes[1], color="tab:blue")
r_axes[1].set_title("Reconstruction: sb.synthesize()")
r_axes[2].plot(err.t, 1e15 * err.data[:, 0], color="tab:red", lw=0.6)
r_axes[2].set(title=f"Difference (max |error| = {err.peak:.1e})", ylabel="× 1e-15", xlabel="Time [s]")
r_axes[2].grid(ls=":")
for ax in r_axes[:2]:
    ax.set_xlabel("")
save(fig, "filterbank.png")

# 7. spectrotemporal ripples: pattern as designed -> sound -> measured modulation spectrum
patterns = [
    ("so.Ripple(4, 1)", so.Ripple(4, 1)),
    (
        "so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45)",
        so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45),
    ),
    ("so.DynamicRipple(rate_range=(-40, 40), seed=3)", so.DynamicRipple(rate_range=(-40, 40), seed=3)),
]
fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), layout="constrained")
for col, (label, pattern) in enumerate(patterns):
    pattern.plot(duration=1.0, f_lo=250, f_hi=8000, ax=axes[0, col], colorbar=False)
    axes[0, col].set_title(label, family="monospace", fontsize=9)
    snd = so.ripple_sound(pattern, 1.0, FS, rng=rng)
    fb = so.OctaveFilterbank.per_octave(24, 250, 8000)
    fb.analyze(snd).envelopes(lowpass=200, fs=1000).plot(axes[1, col], db_range=30, colorbar=False)
    axes[1, col].set_title("the synthesized sound's .envelopes()")
    dmr = isinstance(pattern, so.DynamicRipple)
    ms = so.ModulationSpectrum.octave(
        snd, bands_per_octave=12, f_lo=250, f_hi=8000, scale="db" if dmr else "linear"
    )
    ms.plot(axes[2, col], db_range=30, wt_max=50, wf_max=4, colorbar=False)
    axes[2, col].set_title(
        'ModulationSpectrum.octave(sound, scale="db")' if dmr else "ModulationSpectrum.octave(sound)"
    )
save(fig, "ripples.png")

# 8. waveforms of ripple sounds: flat overall, the pattern lives across bands
fig = plt.figure(figsize=(15, 8.5), layout="constrained")
columns = fig.subfigures(1, 3)
for sub, (label, pattern) in zip(columns, patterns, strict=True):
    snd = so.ripple_sound(pattern, 1.0, FS, rng=rng).ramp(20e-3)
    sb = so.OctaveFilterbank.per_octave(8, 250, 8000).analyze(snd)  # narrow bands: 1/4 octave wide
    show = range(3, len(sb) - 1, 5)  # every 5th band, ~0.6 octave apart
    top, bottom = sub.subfigures(2, 1, height_ratios=[1, 3.3])
    snd.plot(top.subplots(), lw=0.4)
    top.suptitle(label, family="monospace", fontsize=9)
    sb.plot(bottom.subplots(len(show), 1, sharex=True), bands=show, color="tab:purple")
save(fig, "ripple_waveforms.png")
