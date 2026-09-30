"""Build the listening gallery: every demo sound, its plots, and the page.

    python docs/gallery/build.py                 # -> docs/gallery/ (for GitHub Pages)
    python docs/gallery/build.py --single out.html   # also a self-contained page

Everything is generated from sonore itself with fixed seeds, so the gallery is
reproducible. Audio is FLAC (lossless: lossy codecs would alter the interaural
phase and correlation that the binaural demos are about).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

import sonore as so  # noqa: E402
from sonore import dB  # noqa: E402

HERE = Path(__file__).parent
FS = 44100
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


# --------------------------------------------------------------------- demos
@dataclass
class Demo:
    key: str
    title: str
    text: str
    sound: so.Sound
    plot: str  # which figure recipe
    extra: dict = field(default_factory=dict)
    headphones: bool = False


def finish(snd: so.Sound) -> so.Sound:
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def vibrato_complex(dur=2.0):
    t = np.arange(int(dur * FS)) / FS
    phase = 2 * np.pi * np.cumsum(220 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / FS
    return so.Sound(sum(np.cos(k * phase) / k for k in range(1, 20)), FS).normalize().ramp(30e-3)


def starter_pistol():
    """A sharp broadband crack: a shock-like pulse (0.15 ms exponential), highpassed.
    Its spectrum is smooth; a short burst of noise would have random deep notches."""
    t = np.arange(int(0.03 * FS)) / FS
    x = sosfilt(butter(2, 300, "highpass", fs=FS, output="sos"), np.exp(-t / 0.15e-3))
    return so.Sound(x, FS).pad(before=0.05, after=0.05)


def gliding_target(dur=2.0):
    """A harmonic complex gliding 150 -> 250 Hz with a 3 Hz level fluctuation: a stand-in for speech."""
    t = np.arange(int(dur * FS)) / FS
    phase = 2 * np.pi * np.cumsum(150 + 50 * t) / FS
    target = so.Sound(sum(np.cos(k * phase) / k for k in range(1, 30)), FS).normalize()
    return (target * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).ramp(20e-3)


def demos() -> list[tuple[str, str, list[Demo]]]:
    rip = {
        "01": so.Ripple(4, 1),
        "02": so.Ripple(-4, 1),
        "03": so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45),
        "06": so.DynamicRipple(rate_range=(-40, 40), seed=3),
    }
    sung = vibrato_complex()
    pv = so.pv_analyze(sung)
    syllables = sung * np.sin(2 * np.pi * 3 * sung.t).clip(0) ** 2
    noise = so.gaussian_noise(0.8, FS, rng=0).ramp(20e-3)
    noise2 = so.gaussian_noise(0.8, FS, rng=1).ramp(20e-3)

    ripples = [
        Demo(
            "01",
            "Downward ripple",
            "Rate 4 Hz, density 1 cycle/octave, on log-spaced tones. Listen for a "
            "continuous downward sweep, four times a second. The diagonal bands in the cochleagram are what you "
            "are hearing; the waveform alone shows almost none of it.",
            so.ripple_sound(rip["01"], 3, FS, rng=0),
            "ripple",
            {"pattern": rip["01"]},
        ),
        Demo(
            "02",
            "Upward ripple",
            "The same ripple at −4 Hz, so the sweep rises. Its modulation-spectrum peak "
            "moves to negative rate.",
            so.ripple_sound(rip["02"], 3, FS, rng=0),
            "ripple",
            {"pattern": rip["02"]},
        ),
        Demo(
            "03",
            "Two ripples at once",
            "4 Hz at 1 cycle/octave plus −12 Hz at 2.5 cycles/octave: two motions "
            "in opposite directions, superimposed. The modulation spectrum separates them into two clean peaks.",
            so.ripple_sound(rip["03"], 3, FS, rng=0),
            "ripple",
            {"pattern": rip["03"]},
        ),
        Demo(
            "04",
            "Same ripple, harmonic carrier",
            "The first ripple on harmonics of 60 Hz. The motion is the "
            "same, now over a low pitch; resolved low harmonics show as horizontal lines in the cochleagram.",
            so.ripple_sound(rip["01"], 3, FS, carrier="harmonic", f0=60, rng=0),
            "ripple",
            {"pattern": rip["01"]},
        ),
        Demo(
            "05",
            "Same ripple, noise carrier",
            "The same pattern on narrowband noise, which brings its own random "
            "fluctuations, so the sweep sounds rougher.",
            so.ripple_sound(rip["01"], 3, FS, carrier="noise", rng=0),
            "ripple",
            {"pattern": rip["01"]},
        ),
        Demo(
            "06",
            "Dynamic moving ripple",
            "Rate and density wander randomly (rates within ±40 Hz at the lowest "
            "frequency). Where the density passes through zero, every band pulses together, and only there does "
            "the waveform show the modulation.",
            so.ripple_sound(rip["06"], 4, FS, rng=0),
            "ripple",
            {"pattern": rip["06"], "dmr": True},
        ),
    ]
    binaural = [
        Demo(
            "07",
            "Oscor",
            "Interaural correlation swings between +1 and −1 three times a second. The image in "
            "your head alternates between focused and diffuse; only the correlation panel shows what changes.",
            so.oscor(4, FS, f_mod=3, rng=0),
            "binaural",
            headphones=True,
        ),
        Demo(
            "08",
            "Phasewarp",
            "The interaural phase of every component rotates through 360° twice a second, "
            "so the zero-lag correlation follows a 2 Hz cosine.",
            so.phasewarp(4, FS, f_mod=2, rng=0),
            "binaural",
            headphones=True,
        ),
        Demo(
            "10",
            "Timing alone",
            "Noise with a 500 µs interaural time difference, leading in the left ear and "
            "then in the right. The level difference is zero throughout; the sideways shift comes from timing.",
            so.concat(
                [
                    so.apply_itd_ild(noise, itd=-500e-6),
                    so.silence(0.3, FS),
                    so.apply_itd_ild(noise2, itd=500e-6),
                ]
            ),
            "binaural",
            headphones=True,
        ),
    ]
    pitch = [
        Demo(
            "09",
            "Iterated rippled noise",
            "Noise added to an 8 ms delayed copy of itself, sixteen times over. "
            "A 125 Hz pitch rises out of the hiss; the spectrum ripples at multiples of 125 Hz and the modulation "
            "spectrum peaks at 8 cycles/kHz, the delay in milliseconds.",
            so.iterated_ripple_noise(2, FS, delay=8e-3, iterations=16, rng=0),
            "overview",
        ),
    ]
    vocoder = [
        Demo("11", "Reference", "A 220 Hz harmonic complex with a 5 Hz, ±3% vibrato.", sung, "pv"),
        Demo(
            "12",
            "Twice as long",
            "Same pitch, double duration. The vibrato slows to 2.5 Hz as well: time "
            "stretching stretches every temporal feature.",
            so.time_stretch(sung, 2),
            "pv",
        ),
        Demo(
            "13",
            "Up a fifth",
            "Seven semitones higher, same duration, same 5 Hz vibrato.",
            so.pitch_shift(sung, 7),
            "pv",
        ),
        Demo(
            "14",
            "Partials shifted up 70 Hz",
            "Oscillator-bank resynthesis with every partial moved up 70 Hz, "
            "to 290, 510, 730 Hz and on: still 220 Hz apart, but no longer harmonics of anything nearby, so the "
            "tone turns metallic and its pitch less certain.",
            pv.resynthesize(freq_map=lambda f: f + 70),
            "pv",
        ),
        Demo(
            "14b",
            "Partials shifted up 110 Hz",
            "The same with half the spacing: 330, 550, 770 Hz are exactly "
            "the odd harmonics of 110 Hz. The result is harmonic again, a hollow, clarinet-like tone an octave "
            "below the reference.",
            pv.resynthesize(freq_map=lambda f: f + 110),
            "pv",
        ),
        Demo(
            "15",
            "Noise-vocoded",
            "A syllabic 3 Hz harmonic sound through an 8-band noise vocoder: the rhythm "
            "remains and the pitch is gone. The original isn't played; it is shown for comparison.",
            so.noise_vocode(syllables, 8, rng=0),
            "vocoder",
            {"source": syllables},
        ),
    ]

    target = gliding_target()
    masker = so.gaussian_noise(target.duration, FS, tilt=-3, rng=0)
    mixture = target + (masker + 5 * dB)
    S_t, S_m, S_x = (so.STFT(x, 25e-3) for x in (target, masker, mixture))
    ibm = {"target": target, "masker": masker, "mask": so.ideal_binary_mask(S_t, S_m, lc_db=0)}
    speech_in_noise = [
        Demo(
            "24",
            "Target in noise",
            "A gliding harmonic target (a stand-in for a voice) in pink noise at −5 dB SNR.",
            mixture,
            "ibm",
            ibm,
        ),
        Demo(
            "25",
            "Ideal binary mask",
            "The same mixture with every time-frequency cell where the noise dominates switched off. "
            "The mask is computed from the separate target and noise, which is what makes it ideal "
            "(Wang, 2005); the resynthesis is exact.",
            (S_x * ibm["mask"]).to_sound(),
            "ibm",
            ibm,
        ),
    ]
    sweep = so.exponential_chirp(2.0, FS, 100, 6000).ramp(20e-3)
    filterbanks = [
        Demo(
            "26",
            "Perfect reconstruction",
            "An exponential sweep split into 6 ERB-spaced bands plus the lowpass and highpass edge filters, "
            "then summed back. What you hear is the reconstruction; it differs from the original only at the "
            "level of floating-point rounding.",
            so.subbands(sweep, n_bands=6, f_lo=100, f_hi=6000).synthesize(),
            "filterbank",
            {"original": sweep},
        ),
    ]

    pistol = starter_pistol()
    rooms = [
        Demo(
            "16",
            "Starter pistol, dry",
            "The source for the room demos: a sharp, broadband crack lasting a "
            "fraction of a millisecond. Its cochleagram shows the filterbank's own response to a click: the "
            "filters are zero-phase, so they ring symmetrically before and after it (visible at this 60 dB "
            "range), whereas a real cochlea rings only afterwards.",
            pistol,
            "reverb",
            {"ir": None},
        )
    ]
    variants = [
        (
            "17",
            "In a natural room",
            "A synthetic room with RT60 = 1 s whose decay follows the statistics of 271 "
            "real rooms: exponential, with mid frequencies ringing longest. Listeners can't tell such IRs from real "
            "ones.",
            {},
        ),
        (
            "18",
            "Time-reversed decay",
            "The same decay run backwards: the reverberation swells up to the shot "
            "instead of dying away after it.",
            {"decay_shape": "time_reversed"},
        ),
        (
            "19",
            "Linear decay, matched start",
            "Starts at the natural level but falls linearly (in amplitude) "
            "rather than exponentially, with the same energy per band; it has to end early to do so.",
            {"decay_shape": "linear_matched_start"},
        ),
        (
            "20",
            "Linear decay, matched end",
            "Linear decay that reaches zero where the natural decay is 60 dB "
            "down, again with the same energy per band. On a dB scale the decay bows outward instead of falling "
            "in a straight line.",
            {"decay_shape": "linear_matched_end"},
        ),
        (
            "21",
            "Inverted frequency dependence",
            "Exponential, but low and high frequencies ring longest and the "
            "middle dies fastest, the reverse of real rooms. In the paper listeners often heard a separate "
            "high-frequency hiss rather than a room.",
            {"rt60_profile": "inverted"},
        ),
        (
            "22",
            "Exaggerated frequency dependence",
            "The profile of a room twice as reverberant, scaled down: "
            "more sharply peaked than real rooms of this size. The subtlest variant; in the paper it was detected "
            "with impulses but not with speech.",
            {"rt60_profile": "exaggerated"},
        ),
        (
            "23",
            "Reduced frequency dependence",
            "The profile of a room half as reverberant, scaled up: flatter "
            "than real rooms of this size. Also subtle.",
            {"rt60_profile": "reduced"},
        ),
    ]
    for key, title, text, kw in variants:
        ir = so.synth_ir(1.0, FS, drr_db=-3, rng=5, **kw)
        rooms.append(Demo(key, title, text, pistol.convolve(ir), "reverb", {"ir": ir, "kw": kw}))

    return [
        (
            "Spectrotemporal ripples",
            "Each ripple is a sinusoidal pattern in time and log-frequency: a rate in Hz "
            "and a density in cycles per octave. The top-left panel is the pattern as specified; the others are "
            "measured from the sound itself.",
            ripples,
        ),
        (
            "Binaural",
            "These need headphones. Over speakers the two ears' signals mix in the room and the effects "
            "disappear.",
            binaural,
        ),
        ("Pitch from delay", "", pitch),
        (
            "Filterbanks",
            "Cosine filterbanks whose squared responses sum to 1, so analysis followed by synthesis is exact.",
            filterbanks,
        ),
        (
            "Speech in noise",
            "Time-frequency masking: the STFT is invertible, so a masked spectrogram is a sound.",
            speech_in_noise,
        ),
        (
            "Phase vocoder",
            "The first sound is the reference; the others change its duration, pitch, or partials.",
            vocoder,
        ),
        (
            "Rooms, natural and not",
            "After Traer & McDermott (2016), who measured 271 real rooms and found their "
            "reverberation tightly constrained: an exponential decay, fastest at low and high frequencies. Synthetic "
            "rooms that break these regularities sound wrong. All rooms here have the same median RT60 (1 s) and "
            "direct-to-reverberant ratio (−3 dB); unlike the paper, they are not further equated for distortion, "
            "so some differences in loudness and length remain. Presented diotically, as in the paper.",
            rooms,
        ),
    ]


# ------------------------------------------------------------------- figures
def ripple_fig(snd, pattern, dmr=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    pattern.plot(duration=snd.duration, f_lo=250, f_hi=8000, ax=axes[0, 0], colorbar=False)
    axes[0, 0].set_title("Pattern as specified (envelope, dB)")
    so.OctaveFilterbank.per_octave(24, 250, 8000).analyze(snd).envelopes(lowpass=200, fs=1000).plot(
        axes[0, 1], db_range=30, colorbar=False
    )
    axes[0, 1].set_title("Measured envelopes of the sound (cochleagram)")
    snd.plot(axes[1, 0], lw=0.4)
    axes[1, 0].set_title("Waveform (the pattern barely shows here)")
    ms = so.ModulationSpectrum.octave(snd, f_lo=250, f_hi=8000, scale="db" if dmr else "linear")
    ms.plot(axes[1, 1], db_range=30, wt_max=50 if dmr else 20, wf_max=4, colorbar=False)
    axes[1, 1].set_title("Measured modulation spectrum")
    return fig, [axes[0, 0], axes[0, 1], axes[1, 0]]


def binaural_fig(snd):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True, layout="constrained")
    snd.plot(axes[0], lw=0.4)
    axes[0].set_title("Waveform, left and right")
    cues = so.interaural_cues(snd, win_dur=10e-3)
    cues.plot(axes[1])
    axes[1].set_xlabel("")
    axes[2].plot(cues.t, cues.corr0, color="tab:orange", lw=1, label="correlation at zero lag")
    axes[2].plot(cues.t, cues.iac, color="k", alpha=0.6, lw=1, label="coherence (peak over lags)")
    axes[2].axhline(0, color="k", alpha=0.25, lw=0.8)
    axes[2].set(ylim=(-1.05, 1.05), ylabel="Interaural corr.", xlabel="Time [s]", xlim=(0, snd.duration))
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].grid(ls=":")
    return fig, list(axes)


def overview_fig(snd):
    fig = so.overview(snd, win_dur=50e-3, figsize=(10, 6.2), fmax=4000)
    return fig, [ax for ax in fig.axes if ax.get_title() in ("Waveform", "Spectrogram")]


def pv_fig(snd):
    fig = plt.figure(figsize=(10, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], width_ratios=[1.6, 1])
    ax_w, ax_s, ax_f = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    snd.plot(ax_w, lw=0.5)
    so.STFT(snd, 46e-3).plot(ax_s, fmax=3000, colorbar=False, db_range=70)
    spec = so.long_term_spectrum(snd, nperseg=16384)
    ax_f.plot(spec.f, spec.level - spec.level.max(), lw=0.8)
    ax_f.set(
        xlim=(0, 3000),
        ylim=(-70, 3),
        xlabel="Frequency [Hz]",
        ylabel="Level [dB]",
        title="Long-term spectrum",
    )
    ax_f.grid(ls=":")
    return fig, [ax_w, ax_s]


def vocoder_fig(snd, source):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    snd.plot(axes[0, 0], lw=0.4)
    axes[0, 0].set_title("Waveform of the vocoded sound")
    so.subbands(snd, 8, 80, 8000).envelopes(lowpass=50, fs=1000).plot(axes[0, 1], colorbar=False, db_range=40)
    axes[0, 1].set_title("Its 8 band envelopes")
    so.STFT(source, 25e-3).plot(axes[1, 0], fmax=8000, colorbar=False)
    axes[1, 0].set_title("Original (not played): harmonic, syllabic")
    so.STFT(snd, 25e-3).plot(axes[1, 1], fmax=8000, colorbar=False)
    axes[1, 1].set_title("Vocoded: envelopes kept, harmonics gone")
    return fig, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]


def reverb_fig(snd, ir, kw=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    snd.plot(axes[0, 0], lw=0.4)
    axes[0, 0].set_title("Waveform")
    # no extra lowpass: resampling to 1 kHz is already band-limited, and a
    # lowpass would ring visibly around the sharp onset of the shot
    so.subbands(snd, 30, 50, 8000).envelopes(fs=1000).plot(axes[0, 1], colorbar=False, db_range=60)
    axes[0, 1].set_title("Cochleagram (60 dB range)")
    time_axes = [axes[0, 0], axes[0, 1]]
    if ir is None:
        for ax in axes[1]:
            ax.axis("off")
        axes[1, 0].text(0.0, 0.5, "No room: the dry source.", transform=axes[1, 0].transAxes, fontsize=11)
        return fig, time_axes
    # band decays of the impulse response itself (dB), a few bands
    tail = so.Sound(ir.data[1:], ir.fs)
    env = so.subbands(tail, 30, 50, 8000).envelopes(lowpass=30, fs=1000)
    for f, color in zip(
        (125, 500, 2000, 6000), ("tab:blue", "tab:green", "tab:orange", "tab:red"), strict=True
    ):
        k = int(np.argmin(np.abs(env.cfs - f)))
        db = env.db[:, k, 0]
        axes[1, 0].plot(env.t, db - db.max(), color=color, lw=0.8, label=f"{env.cfs[k]:.0f} Hz")
    axes[1, 0].set(
        ylim=(-70, 3), xlabel="Time [s]", ylabel="Band envelope [dB]", title="The IR's decay in four bands"
    )
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].grid(ls=":")
    # measured RT60 profile vs the ecological one
    cfs, rt = so.measure_rt60(tail)
    eco = so.band_rt60s(1.0, cfs, "ecological")
    axes[1, 1].semilogx(cfs, eco, color="k", lw=1.2, ls="--", label="natural rooms (RT60 = 1 s)")
    profile = (kw or {}).get("rt60_profile")
    if profile:
        # requested profile, computed over synth_ir's own bands (which run to 16 kHz)
        synth_cfs = so.ERBFilterbank(32, 20, min(16000, 0.95 * FS / 2)).cfs
        requested = np.interp(cfs, synth_cfs, so.band_rt60s(1.0, synth_cfs, profile))
        axes[1, 1].semilogx(cfs, requested, color="tab:purple", lw=1, ls=":", label=f"requested ({profile})")
    if not kw or "decay_shape" not in kw:
        axes[1, 1].semilogx(cfs, rt, color="tab:purple", lw=1.2, marker=".", label="this IR (measured)")
    else:
        axes[1, 1].text(
            0.03,
            0.06,
            "decay isn't exponential, so an RT60\nisn't meaningful for this IR",
            transform=axes[1, 1].transAxes,
            fontsize=8,
        )
    axes[1, 1].set(
        xlabel="Frequency [Hz]", ylabel="RT60 [s]", title="Decay time by frequency", ylim=(0, None)
    )
    axes[1, 1].legend(fontsize=8, loc="upper right")
    axes[1, 1].grid(ls=":", which="both")
    return fig, time_axes


def ibm_fig(snd, target, masker, mask):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, layout="constrained")
    snd.plot(axes[0, 0], lw=0.4)
    axes[0, 0].set_title("Waveform")
    mask.plot(axes[0, 1])
    axes[0, 1].set(title="Ideal binary mask (target > noise)", ylim=(0, 5))
    so.STFT(snd, 25e-3).plot(axes[1, 0], fmax=5000, colorbar=False)
    axes[1, 0].set_title("Spectrogram of this sound")
    so.STFT(target, 25e-3).plot(axes[1, 1], fmax=5000, colorbar=False)
    axes[1, 1].set_title("The target alone (not played)")
    return fig, [axes[0, 0], axes[1, 0]]


def filterbank_fig(snd, original):
    fig = plt.figure(figsize=(10, 6.2), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.35, 1])
    sb = so.subbands(original, n_bands=6, f_lo=100, f_hi=6000)
    band_axes = left.subplots(len(sb), 1, sharex=True)
    sb.plot(band_axes)
    band_axes[0].set_title(
        "so.subbands(sweep, n_bands=6, f_lo=100, f_hi=6000)", family="monospace", fontsize=8
    )
    r = right.subplots(3, 1, sharex=True)
    original.plot(r[0], color="k", lw=0.4)
    r[0].set(title="Original sweep, 100 Hz to 6 kHz", xlabel="")
    recon = sb.synthesize()  # recomputed: the played sound has been level-normalized
    recon.plot(r[1], color="tab:blue", lw=0.4)
    r[1].set(title="Reconstruction: sb.synthesize()", xlabel="")
    err = recon - original
    r[2].plot(err.t, 1e15 * err.data[:, 0], color="tab:red", lw=0.5)
    r[2].set(title=f"Difference (max |error| = {err.peak:.1e})", ylabel="× 1e-15", xlabel="Time [s]")
    r[2].grid(ls=":")
    return fig, list(band_axes) + list(r)


FIGURES = {
    "ibm": lambda d: ibm_fig(d.sound, **d.extra),
    "filterbank": lambda d: filterbank_fig(d.sound, **d.extra),
    "ripple": lambda d: ripple_fig(d.sound, d.extra["pattern"], d.extra.get("dmr", False)),
    "binaural": lambda d: binaural_fig(d.sound),
    "overview": lambda d: overview_fig(d.sound),
    "pv": lambda d: pv_fig(d.sound),
    "vocoder": lambda d: vocoder_fig(d.sound, d.extra["source"]),
    "reverb": lambda d: reverb_fig(d.sound, d.extra["ir"], d.extra.get("kw")),
}


def render_figure(demo) -> tuple[bytes, list[dict], tuple[int, int]]:
    fig, time_axes = FIGURES[demo.plot](demo)
    fig.canvas.draw()
    regions = []
    for ax in time_axes:
        box, (t0, t1) = ax.get_position(), ax.get_xlim()
        regions.append(
            {"x0": box.x0, "x1": box.x1, "top": 1 - box.y1, "bottom": 1 - box.y0, "t0": t0, "t1": t1}
        )
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100)  # no bbox_inches: keeps figure fractions valid for the playhead
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    q.save(out, "PNG", optimize=True)
    return out.getvalue(), regions, im.size


def flac_bytes(snd: so.Sound) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, snd.data, int(snd.fs), format="FLAC", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------- page
ICON = (
    '<svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/>'
    '<path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3'
    'a2 2 0 0 0-2-2H3z"/></svg>'
)

CSS = """
:root { --paper: #ECEFF1; --plate: #FFFFFF; --ink: #16202A; --muted: #566573; --accent: #235B7C;
  --rule: #CDD4DA; --head: #235B7C;
  --serif: "Spectral", Georgia, "Times New Roman", serif;
  --sans: "Atkinson Hyperlegible", system-ui, -apple-system, "Segoe UI", sans-serif;
  box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --paper: #121A21; --ink: #E4E9ED;
  --muted: #9AA8B4; --accent: #7FB6D6; --rule: #2C3A46; } }
:root[data-theme="dark"] { --paper: #121A21; --ink: #E4E9ED; --muted: #9AA8B4; --accent: #7FB6D6; --rule: #2C3A46; }
*, *::before, *::after { box-sizing: inherit; }
html { scroll-padding-top: env(safe-area-inset-top, 0px); }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--serif); font-size: 1.0625rem; line-height: 1.6; }
main { max-width: 76rem; margin: 0 auto; padding: 3.5rem clamp(1rem, 4vw, 2.5rem) 5rem; }
header { max-width: 40rem; margin-bottom: 3.5rem; }
h1 { font-weight: 500; font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.05; letter-spacing: -0.015em; margin: 0 0 1.25rem; }
header p { margin: 0 0 0.9rem; }
header .how { font-family: var(--sans); font-size: 0.95rem; color: var(--muted); line-height: 1.55; }
section { border-top: 1px solid var(--rule); padding-top: 2.25rem; margin-top: 3rem; }
h2 { font-weight: 500; font-size: 1.75rem; line-height: 1.2; margin: 0 0 0.5rem; }
.section-intro { max-width: 44rem; color: var(--muted); margin: 0 0 1rem; }
.sound { display: grid; grid-template-columns: minmax(15rem, 19rem) minmax(0, 1fr); gap: 2rem; align-items: start; padding: 1.75rem 0; }
.sound + .sound { border-top: 1px dotted var(--rule); }
article.sound { scroll-margin-top: 1.5rem; }
article.sound:target h3 { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 4px; }
h3 { font-weight: 600; font-size: 1.25rem; line-height: 1.25; margin: 0 0 0.5rem; }
.desc { margin: 0 0 1rem; }
.headphones { display: inline-flex; align-items: center; gap: 0.4rem; margin: 0 0 0.6rem; font-family: var(--sans);
  font-size: 0.9rem; font-weight: 700; color: var(--accent); }
audio { width: 100%; max-width: 19rem; display: block; }
audio:focus-visible, .plate:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
.file { font-family: var(--sans); font-size: 0.8rem; color: var(--muted); margin: 0.5rem 0 0; overflow-wrap: anywhere; }
figure { margin: 0; }
.plate { position: relative; background: var(--plate); border-radius: 2px; cursor: crosshair; }
.plate img { display: block; width: 100%; height: auto; }
.heads { position: absolute; inset: 0; pointer-events: none; }
.head { position: absolute; width: 2px; margin-left: -1px; background: var(--head); box-shadow: 0 0 0 1px rgba(255,255,255,0.7); display: none; }
footer { margin-top: 4rem; font-family: var(--sans); font-size: 0.85rem; color: var(--muted); max-width: 40rem; }
footer a { color: var(--accent); }
@media (max-width: 54rem) { .sound { grid-template-columns: minmax(0, 1fr); gap: 1rem; } audio { max-width: none; } }
"""

JS = """
(() => {
  const items = [...document.querySelectorAll(".sound")].map((el) => {
    const regions = JSON.parse(el.dataset.regions);
    const heads = el.querySelector(".heads");
    const lines = regions.map((r) => {
      const d = document.createElement("div");
      d.className = "head";
      d.style.top = (r.top * 100) + "%";
      d.style.height = ((r.bottom - r.top) * 100) + "%";
      heads.appendChild(d);
      return d;
    });
    return { el, audio: el.querySelector("audio"), plate: el.querySelector(".plate"), regions, lines };
  });
  function draw(item) {
    const t = item.audio.currentTime;
    item.regions.forEach((r, i) => {
      const line = item.lines[i];
      if (t < r.t0 || t > r.t1) { line.style.display = "none"; return; }
      line.style.left = ((r.x0 + (t - r.t0) / (r.t1 - r.t0) * (r.x1 - r.x0)) * 100) + "%";
      line.style.display = "block";
    });
  }
  let active = null;
  function loop() { if (!active) return; draw(active); if (!active.audio.paused) requestAnimationFrame(loop); }
  items.forEach((item) => {
    item.audio.addEventListener("play", () => {
      items.forEach((o) => { if (o !== item && !o.audio.paused) o.audio.pause(); });
      active = item; requestAnimationFrame(loop);
    });
    item.audio.addEventListener("seeked", () => draw(item));
    item.audio.addEventListener("ended", () => item.lines.forEach((l) => (l.style.display = "none")));
    item.plate.tabIndex = 0;
    item.plate.setAttribute("role", "button");
    item.plate.setAttribute("aria-label", "Play or pause; click a time axis to play from that point");
    item.plate.addEventListener("click", (e) => {
      const box = item.plate.getBoundingClientRect();
      const fx = (e.clientX - box.left) / box.width, fy = (e.clientY - box.top) / box.height;
      const r = item.regions.find((r) => fx >= r.x0 && fx <= r.x1 && fy >= r.top && fy <= r.bottom);
      if (r) { item.audio.currentTime = Math.max(0, r.t0 + (fx - r.x0) / (r.x1 - r.x0) * (r.t1 - r.t0)); item.audio.play(); }
    });
    item.plate.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); item.audio.paused ? item.audio.play() : item.audio.pause(); }
    });
  });
})();
"""


def page(sections_html: str, n_sounds: int) -> str:
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
        'href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?'
        'family=Atkinson+Hyperlegible:wght@400;700&family=Spectral:wght@400;500;600&display=swap" '
        'rel="stylesheet">'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Listening to sonore</title>
{fonts}
<style>{CSS}</style>
</head>
<body>
<main>
<header>
  <h1>Listening to sonore</h1>
  <p>{n_sounds} sounds made with <a href="https://github.com/choyun1/sonore">sonore</a>, each beside plots of
  the same audio you hear.</p>
  <p class="how">Press play and a line follows the sound across every time axis in its plots. Click any time
  axis to play from that point. The audio is lossless, since compression would alter the binaural sounds.
  Start with your volume low.</p>
</header>
{sections_html}
<footer>Every sound and plot here is generated by <code>docs/gallery/build.py</code> in the
<a href="https://github.com/choyun1/sonore">sonore repository</a>, using the library's own functions.</footer>
</main>
<script>{JS}</script>
</body>
</html>
"""


def build(out_dir: Path | None, single: Path | None) -> None:
    sections, n = [], 0
    if out_dir:
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
        (out_dir / "img").mkdir(parents=True, exist_ok=True)
    single_sections = []
    for title, intro, items in demos():
        slug = "h-" + "".join(c if c.isalnum() else "-" for c in title.lower())
        head = [f'<section aria-labelledby="{slug}">', f'<h2 id="{slug}">{html.escape(title)}</h2>']
        if intro:
            head.append(f'<p class="section-intro">{html.escape(intro)}</p>')
        body_rel, body_inline = list(head), list(head)
        for d in items:
            snd = finish(d.sound)
            audio, (png, regions, (w, h)) = (
                flac_bytes(snd),
                render_figure(Demo(**{**d.__dict__, "sound": snd})),
            )
            name = f"{d.key}_{d.title.lower().replace(' ', '_').replace(',', '')}"
            chan = "stereo" if snd.n_channels == 2 else "mono"
            hp = f'\n    <p class="headphones">{ICON}<span>Headphones</span></p>' if d.headphones else ""

            def article(
                audio_src, img_src, d=d, hp=hp, name=name, snd=snd, chan=chan, regions=regions, w=w, h=h
            ):
                return f"""
<article class="sound" id="d-{d.key}" data-regions="{html.escape(json.dumps(regions))}">
  <div class="about">
    <h3>{html.escape(d.title)}</h3>{hp}
    <p class="desc">{html.escape(d.text)}</p>
    <audio controls preload="metadata" src="{audio_src}"></audio>
    <p class="file">{html.escape(name)}.flac, {snd.duration:.1f} s, {chan}</p>
  </div>
  <figure class="plot"><div class="plate">
    <img src="{img_src}" width="{w}" height="{h}" alt="Plots of the {html.escape(d.title.lower())} sound" loading="lazy">
    <div class="heads" aria-hidden="true"></div>
  </div></figure>
</article>"""

            if out_dir:
                (out_dir / "audio" / f"{name}.flac").write_bytes(audio)
                (out_dir / "img" / f"{name}.png").write_bytes(png)
                body_rel.append(article(f"audio/{name}.flac", f"img/{name}.png"))
            if single:
                body_inline.append(
                    article(
                        "data:audio/flac;base64," + base64.b64encode(audio).decode(),
                        "data:image/png;base64," + base64.b64encode(png).decode(),
                    )
                )
            n += 1
            print(f"  {name}")
        sections.append("\n".join(body_rel + ["</section>"]))
        single_sections.append("\n".join(body_inline + ["</section>"]))
    if out_dir:
        (out_dir / "index.html").write_text(page("\n".join(sections), n))
    if single:
        single.write_text(page("\n".join(single_sections), n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE, help="directory for the Pages site")
    ap.add_argument("--single", type=Path, default=None, help="also write a self-contained HTML file")
    ap.add_argument("--no-site", action="store_true")
    args = ap.parse_args()
    build(None if args.no_site else args.out, args.single)
