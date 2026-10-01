"""Moving talkers: three talkers in front of a listener, one of them moving.

This script is the gallery page https://choyun1.github.io/sonore/gallery/moving.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/moving.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext). It downloads
the PKU-IOA head-related impulse responses at 1 m (about 13 MB) the first time it runs.
"""

# %% [markdown]
# # Moving talkers
#
# Three men talk at once, one straight ahead and one 40° to each side. Which one do you follow?
# If one of them moves, does that help? Cho & Kidd (2022) asked this with stimuli like the ones
# on this page; the experiment's code, written with sonore's predecessor sigtools, is archived at
# [github.com/choyun1/MSM](https://github.com/choyun1/MSM).
#
# - [The talkers and the trajectories](#h-the-talkers-and-the-trajectories): three sentences, and
#   the paths the target takes.
# - [One talker, moving](#h-one-talker-moving): what motion alone sounds like.
# - [Three talkers](#h-three-talkers): the target swinging back and forth in azimuth while the
#   other two stay still.
# - [How the motion is rendered](#h-how-the-motion-is-rendered): filtering with HRIRs that change
#   over time, without clicks.
#
# For the interaural cues one at a time, timing alone and correlation that changes, see [Binaural
# cues](binaural.html).
#
# Every sound here is rendered through measured head-related impulse responses (HRIRs) of a
# KEMAR manikin at 1 m (Qu et al., 2009), so listen with headphones.

# %% [markdown]
# ## The talkers and the trajectories
#
# The target is the sentence from [Seeing speech](speech.html); the two maskers are sentences by
# another man in the CMU ARCTIC corpus (sources in docs/speech/SOURCES.md). Each is scaled to the
# same RMS before rendering, as in the experiment.
#
# Azimuth is measured clockwise from straight ahead, so $+40°$ is to the right. A talker at
# center azimuth $\theta_0$ that oscillates with amplitude $A$ at rate $f$ follows
#
# $$\theta(t) = \theta_0 + A \sin\!\big(2\pi (f t + \phi)\big),$$
#
# so it swings $A$ to either side of $\theta_0$, $2A$ from one end to the other. Here
# $f = 2$ Hz, the rate in the experiment, and the start phase $\phi$ decides whether it first
# moves right ($\phi = 0$) or left ($\phi = 1/2$).

# %%
import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


hrirs = so.load_hrirs()  # PKU-IOA KEMAR at 1 m; downloaded on first use
print(hrirs)

# Three male talkers at 16 kHz, equal in RMS, centered in time on the longest.
files = ["bdl_arctic_a0131", "rms_arctic_a0132", "rms_arctic_a0133"]
target, *maskers = so.normalize(so.pad([so.load(f"docs/speech/{f}.flac") for f in files], align="center"))
fs, duration = target.fs, target.duration
spoken = so.load(f"docs/speech/{files[0]}.flac").duration
TALKING = ((duration - spoken) / 2, (duration + spoken) / 2)  # when the target is talking [s]
CENTERS = (0.0, -40.0, 40.0)  # target ahead, maskers to the left and right [deg]
RATE = 2.0  # oscillations per second


def trajectory(center, amplitude=0.0, phase=0.0, points_per_second=200):
    """Azimuth over the whole sentence [deg], and the Cartesian points at 1 m that
    so.move_sound passes through at a constant rate."""
    t = np.linspace(0, duration, int(points_per_second * duration))
    azimuth = center + amplitude * np.sin(2 * np.pi * (RATE * t + phase))
    return t, azimuth, np.column_stack(so.hcc_to_rect(100.0, 0.0 * azimuth, azimuth))


LABELS = ("target", "left masker", "right masker")
COLORS = ("#d62728", "#7f7f7f", "#b0b0b0")  # one per talker, in plots and in the view from above


def render(amplitude, phase=0.0, alone=False):
    """The target oscillating with ``amplitude`` [deg] about straight ahead, with the
    maskers standing still unless ``alone``. Returns the mix, the target as rendered
    on its own, the trajectories' times and each talker's azimuth."""
    t, az_target, points = trajectory(CENTERS[0], amplitude, phase)
    parts = [so.move_sound(target, points, hrirs)]
    azimuths = [az_target]
    if not alone:
        for snd, center in zip(maskers, CENTERS[1:], strict=True):
            _, az, points = trajectory(center)
            parts.append(so.move_sound(snd, points, hrirs))
            azimuths.append(az)
    return so.mix(parts), parts[0], t, azimuths


def scene_of(t, azimuths):
    """The talkers for the gallery's view from above, which moves as the sound plays."""
    return [
        {"label": label, "color": color, "t": t, "azimuth": az}
        for az, label, color in zip(azimuths, LABELS, COLORS, strict=False)
    ]


def show(mix, target_alone, t, azimuths, title):
    """Each talker's azimuth, and the interaural time and level differences at the ears,
    measured in 20 ms windows: of the mix (black) and of the target on its own (red)."""
    fig = plt.figure(figsize=(10, 6.0), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    for az, label, color in zip(azimuths, LABELS, COLORS, strict=False):
        axes[0].plot(t, az, color=color, lw=1.5, label=label)
    axes[0].axvspan(*TALKING, color=COLORS[0], alpha=0.08, lw=0, label="target talking")
    axes[0].set(ylim=(-75, 75), yticks=[-60, -40, -20, 0, 20, 40, 60], ylabel="Azimuth [deg]", title=title)
    axes[0].legend(loc="upper right", fontsize=8, ncols=4)
    sources = [(mix, "k", "mix")] if len(azimuths) > 1 else []
    for snd, color, label in sources + [(target_alone, COLORS[0], "target alone")]:
        cues = so.interaural_cues(snd, win_dur=20e-3)
        clear = cues.iac > 0.8  # frames where the two ears are well correlated
        axes[1].plot(cues.t[clear], 1e6 * cues.itd[clear], ".", color=color, ms=2.5, label=label)
        axes[2].plot(cues.t, cues.ild, ".", color=color, ms=2.5, label=label)
    axes[1].set(
        ylim=(-500, 500), ylabel="ITD [µs]", title="Interaural time difference (> 0: right ear first)"
    )
    axes[2].set(
        ylim=(-12, 12), ylabel="ILD [dB]", title="Interaural level difference (> 0: right ear louder)"
    )
    for ax in axes:
        ax.grid(ls=":")
        ax.set_xlim(0, duration)
    for ax in axes[1:]:
        ax.legend(loc="upper right", fontsize=8, ncols=2, markerscale=3)
    axes[-1].set_xlabel("Time [s]")
    return fig, list(axes)


# %% [markdown]
# ## One talker, moving
#
# Alone, a talker swinging 30° to each side twice a second is easy to hear moving. The motion
# reaches the ears as interaural differences that change over time: the sound arrives earlier
# and louder at the ear it moves toward. The view from above, beside each example, follows the
# talkers as the sound plays.

# %% [about]
# The target alone, swinging 30° to either side of straight ahead at 2 Hz. Its interaural time
# and level differences, measured from the rendered sound in 20 ms windows, follow the azimuth.
# Time differences are drawn only where the two ears are well correlated.

# %% [demo m1] One talker, moving
mix, target_alone, t, azimuths = render(30.0, alone=True)
fig, playhead = show(mix, target_alone, t, azimuths, "The target alone, swinging 30° to either side")
scene = scene_of(t, azimuths)
sound = finish(mix)

# %% [markdown]
# ## Three talkers
#
# With all three talking, the target is hard to pick out when everyone stands still, since the
# voices are similar and they say similar things. Set the target moving and listen for whether
# it stands out from the others, and how far it has to move before it does. In the mix, the
# interaural differences at any moment come from whichever talker is loudest, so the target's
# motion shows only in the stretches it dominates.

# %% [about]
# All three still: the target straight ahead, the maskers 40° to the left and right.

# %% [demo m2] Three talkers, standing still
mix, target_alone, t, azimuths = render(0.0)
fig, playhead = show(mix, target_alone, t, azimuths, "Nobody moves")
scene = scene_of(t, azimuths)
sound = finish(mix)

# %% [about]
# The target swings 10° to either side of straight ahead.

# %% [demo m3] The target swings 10 degrees
mix, target_alone, t, azimuths = render(10.0)
fig, playhead = show(mix, target_alone, t, azimuths, "The target swings 10° to either side")
scene = scene_of(t, azimuths)
sound = finish(mix)

# %% [about]
# The target swings 30° to either side, three quarters of the way to the maskers, starting to
# the left this time.

# %% [demo m4] The target swings 30 degrees
mix, target_alone, t, azimuths = render(30.0, phase=0.5)
fig, playhead = show(mix, target_alone, t, azimuths, "The target swings 30° to either side")
scene = scene_of(t, azimuths)
sound = finish(mix)

# %% [markdown]
# ## How the motion is rendered
#
# `so.move_sound` spaces the trajectory's points evenly over the sound, here 200 a second. Each
# point gets a raised-cosine window centered on its moment, overlapping its neighbors so that the
# windows sum to one, and each windowed piece of the sound is convolved with the HRIR for that
# point. HRIRs between the measured directions are interpolated with their onsets aligned, and
# the onset delays interpolated separately, which avoids the comb filtering of averaging two
# responses that arrive at different times.
#
# This is a smoothed form of a long-standing way to filter with a filter that changes: when the
# filter changes, the input switches to the new filter while the old one rings out, and the two
# outputs add. The other common way is to crossfade between the outputs of the old and the new
# filter; real-time renderers also swap parts of the filter inside a partitioned convolution
# (Brandtsegg et al., 2018, review all three). Switching with hard edges clicks, which the
# raised-cosine windows avoid. For the same reason the 3D Tune-In toolkit removes each HRIR's
# interaural delay before interpolating and puts it back afterwards (Cuevas-Rodríguez et al.,
# 2019).

# %% [markdown]
# ## References
#
# - Brandtsegg, Saue & Lazzarini (2018). Live convolution with time-varying filters. *Applied
#   Sciences* 8(1), 103. [MDPI](https://www.mdpi.com/2076-3417/8/1/103).
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L273)
# - Cho & Kidd (2022). Auditory motion as a cue for source segregation and selection in a "cocktail
#   party" listening environment. *J. Acoust. Soc. Am.* 152(3), 1684–1694.
#   [doi:10.1121/10.0013990](https://doi.org/10.1121/10.0013990). Experiment code:
#   [choyun1/MSM](https://github.com/choyun1/MSM).
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L273)
#   [`binaural.interaural_cues`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/binaural.py#L95)
# - Cuevas-Rodríguez, Picinali, González-Toledo et al. (2019). 3D Tune-In Toolkit: an open-source
#   library for real-time binaural spatialisation. *PLOS ONE* 14(3), e0211899.
#   [doi:10.1371/journal.pone.0211899](https://doi.org/10.1371/journal.pone.0211899).
#   [`spatialization.HRIRSet`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L93)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentences.
# - Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related transfer functions
#   measured with high spatial resolution using a spark gap. *IEEE Trans. Audio, Speech, Lang.
#   Process.* 17(6), 1124–1132. [PKU
#   Scholar](http://scholar.pku.edu.cn/qutianshu/publications/distance-dependent-head-related-transfer-functions-measured-high-spatial).
#   The HRIRs.
#   [`hrir_data.load_hrirs`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/hrir_data.py#L122)
