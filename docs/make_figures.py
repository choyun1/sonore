"""Regenerate the README figures: ``python docs/make_figures.py``.

Every README example is a subset of the listening gallery: the sounds are taken
from the gallery's own demo list (``docs/gallery/build.py``) by key, so the
README can't drift from what the gallery plays. Each README figure links to
its gallery entries (``gallery/#d-<key>``).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sonore as so  # noqa: E402
from sonore.texture import TextureStats  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)

_rc = dict(plt.rcParams)
sys.path.insert(0, str(HERE / "gallery"))
import build as gallery  # noqa: E402

plt.rcParams.update(_rc)  # keep the README's own figure style
FS = gallery.FS
DEMOS = {d.key: d for _, _, items in gallery.demos() for d in items}


def demo(key):
    return DEMOS[key]


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# 1. overview of iterated rippled noise: pitch at 1/delay, spectral ripple at 8 cyc/kHz
irn = demo("09").sound
fig = so.overview(irn, win_dur=50e-3, figsize=(11, 6.5), fmax=4000)
fig.suptitle("so.overview(so.iterated_ripple_noise(2, 44100, delay=8e-3, iterations=16))", family="monospace")
save(fig, "overview_irn.png")

# 2. ideal binary mask: gallery 24 (mixture) and 25 (masked resynthesis)
mixture, separated = demo("24").sound, demo("25").sound
mask = demo("24").extra["mask"]
S_x = so.STFT(mixture, 25e-3)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, layout="constrained")
S_x.plot(axes[0], fmax=5000, colorbar=False)
axes[0].set_title("Mixture (target at -5 dB SNR)")
mask.plot(axes[1])
axes[1].set_ylim(0, 5)
axes[1].set_title("Ideal binary mask")
so.STFT(separated, 25e-3).plot(axes[2], fmax=5000, colorbar=False)
axes[2].set_title("Masked resynthesis")
save(fig, "ibm.png")

# 3. binaural cues: Oscor and Phasewarp
fig, axes = plt.subplots(2, 2, figsize=(12, 5.5), sharex=True, layout="constrained")
for col, (name, snd) in enumerate(
    [
        ("so.oscor(4, fs, f_mod=3)", demo("07").sound),
        ("so.phasewarp(4, fs, f_mod=2)", demo("08").sound),
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
voc, syllables = demo("15").sound, demo("15").extra["source"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True, layout="constrained")
so.STFT(syllables, 25e-3).plot(axes[0], fmax=6000, colorbar=False)
axes[0].set_title("Original")
so.STFT(voc, 25e-3).plot(axes[1], fmax=6000, colorbar=False)
axes[1].set_title("so.noise_vocode(snd, n_bands=8)")
save(fig, "vocoder.png")

# 5. phase vocoder: time stretch, pitch shift, and an oscillator-bank frequency remap
panels = [
    ("original (220 Hz, 5 Hz vibrato)", demo("11").sound),
    ("so.time_stretch(snd, 2)", demo("12").sound),
    ("so.pitch_shift(snd, 7)", demo("13").sound),
    ("pv_analyze(snd).resynthesize(freq_map=lambda f: f + 70)", demo("14").sound),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True, layout="constrained")
for ax, (title, s) in zip(axes, panels, strict=True):
    so.STFT(s, 46e-3).plot(ax, fmax=3000, colorbar=False, db_range=70)
    ax.set_title(title, fontsize=9, family="monospace")
save(fig, "phase_vocoder.png")

# 6. filterbank: decompose an exponential sweep into 6 ERB-spaced bands and reconstruct it
sweep = demo("26").extra["original"]
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
    ("so.Ripple(4, 1)", demo("01")),
    ("so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45)", demo("03")),
    ("so.DynamicRipple(rate_range=(-40, 40), seed=3)", demo("06")),
]
fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), layout="constrained")
for col, (label, d) in enumerate(patterns):
    pattern, snd = d.extra["pattern"], d.sound
    pattern.plot(duration=snd.duration, f_lo=250, f_hi=8000, ax=axes[0, col], colorbar=False)
    axes[0, col].set_title(label, family="monospace", fontsize=9)
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
for sub, (label, d) in zip(columns, patterns, strict=True):
    snd = d.sound[0:1.0].ramp(20e-3)  # the first second, so the band waveforms are legible
    sb = so.OctaveFilterbank.per_octave(8, 250, 8000).analyze(snd)  # narrow bands: 1/4 octave wide
    show = range(3, len(sb) - 1, 5)  # every 5th band, ~0.6 octave apart
    top, bottom = sub.subfigures(2, 1, height_ratios=[1, 3.3])
    snd.plot(top.subplots(), lw=0.4)
    top.suptitle(label, family="monospace", fontsize=9)
    sb.plot(bottom.subplots(len(show), 1, sharex=True), bands=show, color="tab:purple")
save(fig, "ripple_waveforms.png")

# 9. sound texture: a stream recording and a synthesis from its statistics (gallery t01a / t01b)
# The texture page (docs/gallery/stimuli/textures.py) runs as a script, so its files are loaded here
# as it loads them.
TEXTURES = gallery.ROOT / "docs" / "textures"
original, synth = so.load(TEXTURES / "stream.flac"), so.load(TEXTURES / "synth" / "stream.flac").resample(FS)
info = json.loads((TEXTURES / "synth" / "stream.json").read_text())
snr = np.mean(list(info["snr_all_classes"].values()))
target = TextureStats.measure(original)
this = TextureStats.measure(synth, window="uniform")  # syntheses are circular
cfs, ok, mcf = target.model.filterbank.cfs[1:-1], target.channel_mask(), target.model.mod_bank.cfs

fig = plt.figure(figsize=(13, 7), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
ax_o, ax_s = top.subplots(1, 2, sharey=True)
so.STFT(original[0 : synth.duration], 20e-3).plot(ax_o, fmax=10000, db_range=70, colorbar=False)
ax_o.set_title("Original: water over rocks in a creek (first 5 s)")
so.STFT(synth, 20e-3).plot(ax_s, fmax=10000, db_range=70, colorbar=False)
ax_s.set_title(
    f"Synthesized from noise: {info['best_iteration']} iterations, average statistic SNR {snr:.0f} dB"
)
ax_l, ax_v, ax_p = bottom.subplots(1, 3)
so.long_term_spectrum(synth).plot(ax_l, lw=1.2, label="synthesized")
so.long_term_spectrum(original).plot(ax_l, color="k", ls="--", lw=1, label="original")
ax_l.set(xlim=(20, 10000), ylim=(-40, 3), title="Long-term spectrum")
ax_v.semilogx(cfs, this.env_var[1:-1], lw=1.2)
ax_v.semilogx(cfs, target.env_var[1:-1], "k--", lw=1)
ax_v.set(xlabel="Band center [Hz]", ylabel="var / mean²", title="Envelope sparsity by band")
ax_p.loglog(mcf, this.mod_power[ok].mean(0), lw=1.2)
ax_p.loglog(mcf, target.mod_power[ok].mean(0), "k--", lw=1)
ax_p.set(xlabel="Modulation rate [Hz]", ylabel="Power / variance", title="Modulation power (band average)")
for ax in (ax_v, ax_p):
    ax.grid(ls=":", which="both", lw=0.5)
ax_l.legend(fontsize=8)
save(fig, "texture_stream.png")
