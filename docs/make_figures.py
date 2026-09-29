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
        "pv_analyze(snd).resynthesize(freq_map=lambda f: f + 110)",
        so.pv_analyze(sung).resynthesize(freq_map=lambda f: f + 110),
    ),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True, layout="constrained")
for ax, (title, s) in zip(axes, panels, strict=True):
    so.STFT(s, 46e-3).plot(ax, fmax=3000, colorbar=False, db_range=70)
    ax.set_title(title, fontsize=9, family="monospace")
save(fig, "phase_vocoder.png")
