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
import contextlib
import html
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

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
}


def encode_figure(fig, time_axes) -> tuple[bytes, list[dict], tuple[int, int]]:
    """The figure as a 256-colour PNG, and where each time axis sits (for the playhead)."""
    fig.canvas.draw()
    regions = []
    for ax in time_axes:
        # ax.bbox is in display units; converting through the *figure* transform gives true figure
        # fractions even for axes inside subfigures (whose get_position() is relative to the subfigure).
        box, (t0, t1) = ax.bbox.transformed(fig.transFigure.inverted()), ax.get_xlim()
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


def render_figure(demo) -> tuple[bytes, list[dict], tuple[int, int]]:
    return encode_figure(*FIGURES[demo.plot](demo))


def flac_bytes(snd: so.Sound) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, snd.data, int(snd.fs), format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def file_name(key: str, title: str) -> str:
    return f"{key}_{title.lower().replace(' ', '_').replace(',', '')}"


# ------------------------------------------------------------- example pages
# The example pages (speech, textures, moving, vocoder, cepstrum, reverb) are runnable scripts in percent format,
# where "# %%" starts a cell:
#
#   # %% [markdown]            prose: "# Title" names the page, "## Heading" starts a section
#   # %% [about]               the description of the example that follows
#   # %% [demo KEY] Title      code that leaves `sound`, `fig` and `playhead` (the axes the playhead follows),
#                              and optionally `scene`, sources to draw from above as the sound plays
#   # %% [figure KEY] Title    code that leaves `fig`: a figure without sound
#   # %%                       any other code; what it prints is shown under it
#
# Every code cell is shown on the page exactly as it ran. Prose may use $TeX$, $$display TeX$$,
# `code`, **bold**, *italic*, [links](url), "- " lists, and {{ expression }}, which is evaluated
# where the cell stands. The scripts run from the repository root.
EXAMPLE_PAGES = ["speech", "textures", "moving", "vocoder", "cepstrum", "reverb"]
ROOT = HERE.parent.parent
CELL = re.compile(r"# %%(?: \[(\w+)(?: (\w+))?\])?(?: (.*))?")


@dataclass
class Cell:
    kind: str  # markdown, about, demo, figure or code
    source: str
    key: str = ""
    title: str = ""


def read_cells(path: Path) -> list[Cell]:
    cells = []
    for chunk in re.split(r"(?m)^(?=# %%)", path.read_text())[
        1:
    ]:  # what precedes the first cell is not shown
        head, _, body = chunk.partition("\n")
        m = CELL.fullmatch(head.rstrip())
        if not m:
            raise ValueError(f"{path.name}: bad cell header {head!r}")
        kind = m.group(1) or "code"
        body = body.strip("\n")
        if kind in ("markdown", "about"):
            body = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
        cells.append(Cell(kind, body, m.group(2) or "", (m.group(3) or "").strip()))
    return cells


INLINE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|`([^`]+)`", re.S)


def _marks(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)


def inline(text: str) -> str:
    out, pos = [], 0
    for m in INLINE.finditer(text):
        out.append(_marks(html.escape(text[pos : m.start()], quote=False)))
        display, tex, code = m.groups()
        if code is not None:
            out.append(f"<code>{html.escape(code)}</code>")
        else:
            cls = "tex display" if display is not None else "tex"
            out.append(f'<span class="{cls}">{html.escape((display or tex).strip())}</span>')
        pos = m.end()
    out.append(_marks(html.escape(text[pos:], quote=False)))
    return "".join(out)


def markdown(text: str) -> list[tuple[str, str]]:
    """Blocks of prose as (kind, html), kind one of h1, h2, h3 or p (anything else)."""
    blocks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if m := re.match(r"(#{1,3}) (.*)", block):
            blocks.append((f"h{len(m.group(1))}", inline(m.group(2).strip())))
        elif block.startswith("$$") and block.endswith("$$"):
            blocks.append(("p", f'<div class="tex display">{html.escape(block[2:-2].strip())}</div>'))
        elif lines[0].startswith("- "):
            items = re.split(r"(?m)^- ", block)[1:]
            lis = "".join(f"<li>{inline(' '.join(i.split()))}</li>" for i in items)
            blocks.append(("p", f"<ul>{lis}</ul>"))
        else:
            blocks.append(("p", f"<p>{inline(block)}</p>"))
    return blocks


def substitute(text: str, ns: dict) -> str:
    return re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1), ns)), text)  # noqa: S307


def code_block(source: str) -> str:
    return f'<pre><code class="language-python">{html.escape(source)}</code></pre>'


def example_page(path: Path) -> dict:
    """Run a page script cell by cell. Returns its title, intro and sections, where each
    part is either HTML or an example (a dict with the media to write)."""
    ns: dict = {"__name__": "__gallery__", "__file__": str(path)}
    title, intro, sections, about = "", [], [], ""
    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for cell in read_cells(path):
            if cell.kind in ("markdown", "about"):
                blocks = markdown(substitute(cell.source, ns))
                if cell.kind == "about":
                    about = "".join(h for _, h in blocks)
                    continue
                for kind, h in blocks:
                    if kind == "h1":
                        title = html.unescape(re.sub("<[^>]+>", "", h))
                    elif kind == "h2":
                        sections.append({"title": h, "parts": []})
                    else:
                        (sections[-1]["parts"] if sections else intro).append(h)
                continue
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(cell.source, f"{path.name}, cell {cell.key or cell.title}", "exec"), ns)  # noqa: S102
            part = {"key": cell.key, "title": cell.title, "about": about, "code": code_block(cell.source)}
            if cell.kind == "demo":
                snd = ns.pop("sound")
                png, regions, size = encode_figure(ns.pop("fig"), ns.pop("playhead"))
                part.update(sound=snd, audio=flac_bytes(snd), png=png, regions=regions, size=size)
                if "scene" in ns:
                    part["scene"] = scene_json(ns.pop("scene"))
            elif cell.kind == "figure":
                png, _, size = encode_figure(ns.pop("fig"), [])
                part.update(png=png, size=size)
            else:
                printed = out.getvalue().rstrip()
                part = part["code"] + (f'<pre class="out">{html.escape(printed)}</pre>' if printed else "")
                part = f'<div class="cell">{part}</div>'
            (sections[-1]["parts"] if sections else intro).append(part)
            about = ""
    finally:
        os.chdir(cwd)
    return {"title": title, "intro": intro, "sections": sections}


# ---------------------------------------------------------------------- page
ICON = (
    '<svg aria-hidden="true" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/>'
    '<path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3'
    'a2 2 0 0 0-2-2H3z"/></svg>'
)

CSS = """
:root { --paper: #ECEFF1; --plate: #FFFFFF; --ink: #16202A; --muted: #566573; --accent: #235B7C;
  --rule: #CDD4DA; --head: #235B7C; --code: #F7F9FA; --string: #2F6B3A; --number: #8A4B12;
  --serif: "Spectral", Georgia, "Times New Roman", serif;
  --sans: "Atkinson Hyperlegible", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --paper: #121A21; --ink: #E4E9ED;
  --muted: #9AA8B4; --accent: #7FB6D6; --rule: #2C3A46; --code: #0C1318; --string: #9CCB8E; --number: #E0A76B; } }
:root[data-theme="dark"] { --paper: #121A21; --ink: #E4E9ED; --muted: #9AA8B4; --accent: #7FB6D6; --rule: #2C3A46;
  --code: #0C1318; --string: #9CCB8E; --number: #E0A76B; }
*, *::before, *::after { box-sizing: inherit; }
html { scroll-padding-top: env(safe-area-inset-top, 0px); }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--serif); font-size: 1.0625rem; line-height: 1.6; }
main { max-width: 76rem; margin: 0 auto; padding: 3.5rem clamp(1rem, 4vw, 2.5rem) 5rem; }
header { max-width: 44rem; margin-bottom: 3.5rem; }
h1 { font-weight: 500; font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.05; letter-spacing: -0.015em; margin: 0 0 1.25rem; }
header p { margin: 0 0 0.9rem; }
header ul { margin: 0 0 0.9rem; padding-left: 1.25rem; }
header li { margin: 0 0 0.35rem; }
header .how { font-family: var(--sans); font-size: 0.95rem; color: var(--muted); line-height: 1.55; }
nav.pages { display: flex; flex-wrap: wrap; gap: 0.4rem 1.5rem; font-family: var(--sans); font-size: 0.95rem; margin: 0 0 2rem; }
nav.pages a { color: var(--accent); }
nav.pages a[aria-current] { color: var(--ink); font-weight: 700; text-decoration: none; }
a { color: var(--accent); }
section { border-top: 1px solid var(--rule); padding-top: 2.25rem; margin-top: 3rem; }
h2 { font-weight: 500; font-size: 1.75rem; line-height: 1.2; margin: 0 0 0.5rem; }
.section-intro { max-width: 44rem; color: var(--muted); margin: 0 0 1rem; }
section > p, section > ul, section > .tex.display { max-width: 44rem; }
section > p, header > p { margin: 0 0 1rem; }
.sound, .still { display: grid; grid-template-columns: minmax(15rem, 19rem) minmax(0, 1fr); gap: 1.25rem 2rem; align-items: start; padding: 1.75rem 0; }
.sound > *, .still > * { min-width: 0; }
article + article, .cell + article, article + .cell { border-top: 1px dotted var(--rule); }
article.sound, article.still { scroll-margin-top: 1.5rem; }
article:target h3 { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 4px; }
h3 { font-weight: 600; font-size: 1.25rem; line-height: 1.25; margin: 0 0 0.5rem; }
.desc { margin: 0 0 1rem; }
.desc p { margin: 0 0 0.75rem; }
.headphones { display: inline-flex; align-items: center; gap: 0.4rem; margin: 0 0 0.6rem; font-family: var(--sans);
  font-size: 0.9rem; font-weight: 700; color: var(--accent); }
audio { width: 100%; max-width: 19rem; display: block; }
.scene { display: block; width: 100%; max-width: 19rem; aspect-ratio: 1; margin-top: 1rem; }
audio:focus-visible, .plate:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
.file { font-family: var(--sans); font-size: 0.8rem; color: var(--muted); margin: 0.5rem 0 0; overflow-wrap: anywhere; }
figure { margin: 0; }
.plate { position: relative; background: var(--plate); border-radius: 2px; cursor: crosshair; }
.still .plate { cursor: auto; }
.plate img { display: block; width: 100%; height: auto; }
.heads { position: absolute; inset: 0; pointer-events: none; }
.head { position: absolute; width: 2px; margin-left: -1px; background: var(--head); box-shadow: 0 0 0 1px rgba(255,255,255,0.7); display: none; }
code { font-family: var(--mono); font-size: 0.86em; }
pre { font-family: var(--mono); font-size: 0.8rem; line-height: 1.5; background: var(--code); border: 1px solid var(--rule);
  border-radius: 3px; padding: 0.85rem 1rem; margin: 0; overflow-x: auto; }
pre code { font-size: inherit; }
pre.out { background: transparent; border-style: dashed; margin-top: 0.5rem; }
.cell { padding: 1.25rem 0; }
details.code { grid-column: 1 / -1; }
details.code summary { font-family: var(--sans); font-size: 0.9rem; color: var(--muted); cursor: pointer; margin: 0 0 0.5rem; }
.hljs-keyword, .hljs-built_in, .hljs-literal { color: var(--accent); }
.hljs-string { color: var(--string); }
.hljs-number { color: var(--number); }
.hljs-comment { color: var(--muted); font-style: italic; }
.tex.display { display: block; margin: 1rem 0 1.25rem; overflow-x: auto; overflow-y: hidden; }
footer { margin-top: 4rem; font-family: var(--sans); font-size: 0.85rem; color: var(--muted); max-width: 40rem; }
@media (max-width: 54rem) { .sound, .still { grid-template-columns: minmax(0, 1fr); gap: 1rem; } audio { max-width: none; } }
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
    const canvas = el.querySelector(".scene");
    const scene = canvas ? JSON.parse(canvas.dataset.scene) : null;
    return { el, audio: el.querySelector("audio"), plate: el.querySelector(".plate"), regions, lines, canvas, scene };
  });
  // A top-down view: the listener's head in the middle, nose up (the front), and each source on a
  // circle at its azimuth (clockwise from straight ahead) at the current time.
  function drawScene(item) {
    const c = item.canvas;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1, size = c.clientWidth;
    if (c.width !== Math.round(size * dpr)) { c.width = c.height = Math.round(size * dpr); }
    const g = c.getContext("2d"), css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    const cx = size / 2, cy = size / 2, R = size * 0.38, head = size * 0.07, t = item.audio.currentTime;
    g.strokeStyle = muted; g.lineWidth = 1; g.setLineDash([3, 4]);
    g.beginPath(); g.arc(cx, cy, R, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    g.fillStyle = muted; g.font = "11px system-ui, sans-serif"; g.textAlign = "center";
    g.fillText("1 m", cx + R * 0.72, cy + R * 0.72 + 14);
    g.strokeStyle = ink; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(cx - head * 0.45, cy - head * 0.9); g.lineTo(cx, cy - head * 1.45); g.lineTo(cx + head * 0.45, cy - head * 0.9); g.stroke();
    g.beginPath(); g.arc(cx, cy, head, 0, 2 * Math.PI); g.stroke();
    g.beginPath(); g.ellipse(cx - head, cy, head * 0.18, head * 0.4, 0, 0, 2 * Math.PI); g.stroke();
    g.beginPath(); g.ellipse(cx + head, cy, head * 0.18, head * 0.4, 0, 0, 2 * Math.PI); g.stroke();
    const at = (src, time) => {
      if (!src.t) return src.az[0];
      const k = Math.min(src.az.length - 1, Math.max(0, time / src.t)), i = Math.floor(k), f = k - i;
      return i + 1 < src.az.length ? src.az[i] * (1 - f) + src.az[i + 1] * f : src.az[i];
    };
    const xy = (az) => [cx + R * Math.sin(az * Math.PI / 180), cy - R * Math.cos(az * Math.PI / 180)];
    item.scene.forEach((src) => {
      if (src.t) {  // the path it travels, faintly
        g.strokeStyle = src.color; g.globalAlpha = 0.25; g.lineWidth = 6; g.lineCap = "round";
        const lo = Math.min(...src.az), hi = Math.max(...src.az);
        g.beginPath(); g.arc(cx, cy, R, (lo - 90) * Math.PI / 180, (hi - 90) * Math.PI / 180); g.stroke();
        g.globalAlpha = 1;
      }
      const [x, y] = xy(at(src, t));
      g.fillStyle = src.color; g.beginPath(); g.arc(x, y, size * 0.035, 0, 2 * Math.PI); g.fill();
      g.fillStyle = ink; g.fillText(src.label, x, y + size * 0.035 + 13);
    });
  }
  function draw(item) {
    drawScene(item);
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
    drawScene(item);
    if (item.canvas) {
      new ResizeObserver(() => drawScene(item)).observe(item.canvas);
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawScene(item));
    }
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
      if (r) { item.audio.currentTime = Math.max(0, r.t0 + (fx - r.x0) / (r.t1 - r.t0) * (r.t1 - r.t0)); item.audio.play(); }
    });
    item.plate.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); item.audio.paused ? item.audio.play() : item.audio.pause(); }
    });
  });
})();
"""

# TeX and code highlighting are drawn in the browser; without scripts the TeX source and plain code show.
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/"
EXAMPLE_HEAD = f"""<link rel="stylesheet" href="{KATEX}katex.min.css" crossorigin="anonymous">
<script>function renderTex() {{ document.querySelectorAll(".tex").forEach((el) => katex.render(el.textContent, el,
  {{ displayMode: el.classList.contains("display"), throwOnError: false }})); }}</script>
<script defer src="{KATEX}katex.min.js" crossorigin="anonymous" onload="renderTex()"></script>
<script defer src="{HLJS}highlight.min.js" onload="hljs.highlightAll()"></script>"""

NAV = [
    ("index.html", "Listening gallery"),
    ("speech.html", "Seeing speech"),
    ("textures.html", "Sound textures"),
    ("moving.html", "Moving talkers"),
    ("vocoder.html", "Hearing through a vocoder"),
    ("cepstrum.html", "Cepstral analysis"),
    ("reverb.html", "Synthetic reverberation"),
]


def nav(current: str) -> str:
    current_attr = ' aria-current="page"'
    links = [f'<a href="{href}"{current_attr if href == current else ""}>{label}</a>' for href, label in NAV]
    return f'<nav class="pages" aria-label="Gallery pages">{"".join(links)}</nav>'


HOW = """<p class="how">Press play and a line follows the sound across every time axis in its plots. Click any time
  axis to play from that point. The audio is lossless, since compression would alter the binaural sounds.
  Start with your volume low.</p>"""


def page(title: str, current: str, header: str, sections_html: str, head: str = "", footer: str = "") -> str:
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
        'href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?'
        'family=Atkinson+Hyperlegible:wght@400;700&family=Spectral:wght@400;500;600&display=swap" '
        'rel="stylesheet">'
    )
    footer = footer or (
        "Every sound and plot here is generated by <code>docs/gallery/build.py</code> in the "
        '<a href="https://github.com/choyun1/sonore">sonore repository</a>, using the library\'s own functions.'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
{fonts}
{head}
<style>{CSS}</style>
</head>
<body>
<main>
{nav(current)}
<header>
  <h1>{html.escape(title)}</h1>
{header}
</header>
{sections_html}
<footer>{footer}</footer>
</main>
<script>{JS}</script>
</body>
</html>
"""


def scene_json(scene: list[dict], rate: float = 50.0) -> str:
    """A top-down scene for the page to animate: each source's azimuth [deg, clockwise
    from straight ahead] over time, resampled to ``rate`` Hz to keep the page small."""
    out = []
    for src in scene:
        t = np.asarray(src["t"], float)
        grid = np.arange(0, t[-1] + 0.5 / rate, 1 / rate) if len(t) > 1 else t
        az = np.interp(grid, t, np.asarray(src["azimuth"], float))
        out.append(
            {
                "label": src["label"],
                "color": src["color"],
                "t": round(1 / rate, 6) if len(t) > 1 else 0,
                "az": [round(float(a), 2) for a in az],
            }
        )
    return json.dumps(out, separators=(",", ":"))


def sound_article(
    key, title, desc_html, snd, audio_src, img_src, regions, size, extra="", code="", scene=None
) -> str:
    name, (w, h) = file_name(key, title), size
    scene = (
        f'\n    <canvas class="scene" data-scene="{html.escape(scene)}" role="img" '
        f'aria-label="The sources seen from above, moving as the sound plays"></canvas>'
        if scene
        else ""
    )
    chan = "stereo" if snd.n_channels == 2 else "mono"
    code = f'\n  <details class="code" open><summary>Code</summary>{code}</details>' if code else ""
    return f"""
<article class="sound" id="d-{key}" data-regions="{html.escape(json.dumps(regions))}">
  <div class="about">
    <h3>{html.escape(title)}</h3>{extra}
    {desc_html}
    <audio controls preload="metadata" src="{audio_src}"></audio>
    <p class="file">{html.escape(name)}.flac, {snd.duration:.1f} s, {chan}</p>{scene}
  </div>
  <figure class="plot"><div class="plate">
    <img src="{img_src}" width="{w}" height="{h}" alt="Plots of the {html.escape(title.lower())} sound" loading="lazy">
    <div class="heads" aria-hidden="true"></div>
  </div></figure>{code}
</article>"""


def still_article(key, title, desc_html, img_src, size, code) -> str:
    w, h = size
    return f"""
<article class="still" id="d-{key}">
  <div class="about">
    <h3>{html.escape(title)}</h3>
    <div class="desc">{desc_html}</div>
  </div>
  <figure class="plot"><div class="plate">
    <img src="{img_src}" width="{w}" height="{h}" alt="{html.escape(title)}" loading="lazy">
  </div></figure>
  <details class="code" open><summary>Code</summary>{code}</details>
</article>"""


class Site:
    """Writes media next to the pages (relative links) and/or inlines it (one self-contained file per page)."""

    def __init__(self, out_dir: Path | None, single: Path | None):
        self.out_dir, self.single = out_dir, single
        if out_dir:
            (out_dir / "audio").mkdir(parents=True, exist_ok=True)
            (out_dir / "img").mkdir(parents=True, exist_ok=True)

    def media(self, name: str, audio: bytes | None, png: bytes) -> tuple[list, list]:
        """(audio, image) sources for the site and for the single file."""
        rel, inl = [], []
        if self.out_dir:
            if audio is not None:
                (self.out_dir / "audio" / f"{name}.flac").write_bytes(audio)
                rel.append(f"audio/{name}.flac")
            (self.out_dir / "img" / f"{name}.png").write_bytes(png)
            rel.append(f"img/{name}.png")
        if self.single:
            if audio is not None:
                inl.append("data:audio/flac;base64," + base64.b64encode(audio).decode())
            inl.append("data:image/png;base64," + base64.b64encode(png).decode())
        return rel, inl

    def write(self, file: str, make) -> None:
        """``make(variant)`` renders a page; variant 0 links media, 1 inlines it."""
        if self.out_dir:
            (self.out_dir / file).write_text(make(0))
        if self.single:
            stem = self.single.stem if file == "index.html" else f"{self.single.stem}-{Path(file).stem}"
            self.single.with_name(stem + self.single.suffix).write_text(make(1))


def section_html(title_html: str, parts: list[str], intro: str = "") -> str:
    slug = "h-" + "".join(
        c if c.isalnum() else "-" for c in html.unescape(re.sub("<[^>]+>", "", title_html)).lower()
    )
    out = [f'<section aria-labelledby="{slug}">', f'<h2 id="{slug}">{title_html}</h2>']
    if intro:
        out.append(f'<p class="section-intro">{html.escape(intro)}</p>')
    return "\n".join(out + parts + ["</section>"])


def build_example_page(site: Site, name: str) -> list[str]:
    """Build one example page; returns the keys of its examples."""
    content = example_page(HERE / f"{name}.py")
    keys, rendered = [], {}  # part id -> (linked, inline) HTML
    for part in [p for s in content["sections"] for p in s["parts"]] + content["intro"]:
        if not isinstance(part, dict):
            continue
        key, title = part["key"], part["title"]
        keys.append(key)
        srcs = site.media(file_name(key, title), part.get("audio"), part["png"])
        variants = []
        for src in srcs:
            if not src:
                variants.append("")
            elif "sound" in part:
                desc = f'<div class="desc">{part["about"]}</div>'
                variants.append(
                    sound_article(
                        key,
                        title,
                        desc,
                        part["sound"],
                        src[0],
                        src[1],
                        part["regions"],
                        part["size"],
                        code=part["code"],
                        scene=part.get("scene"),
                    )
                )
            else:
                variants.append(still_article(key, title, part["about"], src[0], part["size"], part["code"]))
        rendered[id(part)] = variants
        print(f"  {name}: {file_name(key, title)}")

    def make(variant):
        def show(p):
            return rendered[id(p)][variant] if isinstance(p, dict) else p

        header = "\n".join(show(p) for p in content["intro"]) + "\n" + HOW
        sections = "\n".join(
            section_html(s["title"], [show(p) for p in s["parts"]]) for s in content["sections"]
        )
        footer = (
            f"This page is the script <code>docs/gallery/{name}.py</code> in the "
            '<a href="https://github.com/choyun1/sonore">sonore repository</a>, run cell by cell by '
            "<code>docs/gallery/build.py</code>: each block of code is shown exactly as it ran. Run it yourself "
            f"from the repository root with <code>python docs/gallery/{name}.py</code>, or a cell at a time."
        )
        return page(content["title"], f"{name}.html", header, sections, EXAMPLE_HEAD, footer)

    site.write(f"{name}.html", make)
    return keys


def build(out_dir: Path | None, single: Path | None) -> None:
    site = Site(out_dir, single)
    moved = {}
    for name in EXAMPLE_PAGES:
        moved.update({key: f"{name}.html" for key in build_example_page(site, name)})

    sections, n = ([], []), 0
    for title, intro, items in demos():
        bodies = ([], [])
        for d in items:
            snd = finish(d.sound)
            png, regions, size = render_figure(Demo(**{**d.__dict__, "sound": snd}))
            name = file_name(d.key, d.title)
            hp = f'\n    <p class="headphones">{ICON}<span>Headphones</span></p>' if d.headphones else ""
            desc = f'<p class="desc">{html.escape(d.text)}</p>'
            for body, src in zip(bodies, site.media(name, flac_bytes(snd), png), strict=True):
                if src:
                    body.append(sound_article(d.key, d.title, desc, snd, src[0], src[1], regions, size, hp))
            n += 1
            print(f"  {name}")
        for out, body in zip(sections, bodies, strict=True):
            out.append(section_html(html.escape(title), body, intro))

    # Links to examples that moved to their own pages still land on them.
    redirect = (
        f"<script>(() => {{ const moved = {json.dumps(moved)}; const k = location.hash.slice(3);"
        ' if (location.hash.startsWith("#d-") && moved[k]) location.replace(moved[k] + location.hash); })();</script>'
    )
    header = f"""  <p>{n} sounds made with <a href="https://github.com/choyun1/sonore">sonore</a>, each beside plots of
  the same audio you hear. Six topics have pages of their own, with the code for every example beside it:</p>
  <ul>
    <li><a href="speech.html">Seeing speech</a>: a short course in time-frequency analysis on one spoken sentence.</li>
    <li><a href="textures.html">Sound textures</a>: recordings and their syntheses from statistics.</li>
    <li><a href="moving.html">Moving talkers</a>: three talkers rendered through measured HRIRs, one of them moving.</li>
    <li><a href="vocoder.html">Hearing through a vocoder</a>: a simulation of cochlear-implant hearing.</li>
    <li><a href="cepstrum.html">Cepstral analysis</a>: separating a voice's pitch from its timbre.</li>
    <li><a href="reverb.html">Synthetic reverberation</a>: rooms built from the statistics of real ones, and rooms
      that break them.</li>
  </ul>
  {HOW}"""
    site.write(
        "index.html",
        lambda v: page("Listening to sonore", "index.html", header, "\n".join(sections[v]), redirect),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE, help="directory for the Pages site")
    ap.add_argument("--single", type=Path, default=None, help="also write self-contained HTML files")
    ap.add_argument("--no-site", action="store_true")
    args = ap.parse_args()
    build(None if args.no_site else args.out, args.single)
