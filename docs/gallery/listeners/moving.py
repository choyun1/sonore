"""Moving talkers: three talkers in front of a listener, one of them moving.

This script is the gallery page https://choyun1.github.io/sonore/gallery/moving.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/listeners/moving.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext). It downloads
the PKU-IOA head-related impulse responses at all eight distances (about 104 MB) the first
time it runs.
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
# - [Coming closer](#h-coming-closer): the target walking up to the listener, without and with
#   a room.
# - [Passing by](#h-passing-by): a buzz going past at 15 m/s, and the Doppler glide of its pitch.
# - [How the motion is rendered](#h-how-the-motion-is-rendered): filtering with HRIRs that change
#   over time, without clicks or comb filtering.
#
# For the interaural cues one at a time, timing alone and correlation that changes, see [Binaural
# cues](binaural.html).
#
# Every sound here is rendered through measured head-related impulse responses (HRIRs) of a
# KEMAR manikin at eight distances from 20 cm to 1.6 m (Qu et al., 2009), so listen with
# headphones. Farther than 1.6 m, distance changes only the travel time and the level.

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


hrirs = so.load_hrirs(distances="all")  # PKU-IOA KEMAR, 20 cm to 1.6 m; downloaded on first use
print(hrirs)

# Three male talkers at 16 kHz, equal in RMS, centered in time on the longest.
files = ["bdl_arctic_a0131", "rms_arctic_a0132", "rms_arctic_a0133"]
target, *maskers = so.normalize(so.pad([so.load(f"docs/speech/{f}.flac") for f in files], align="center"))
fs, duration = target.fs, target.duration
spoken = so.load(f"docs/speech/{files[0]}.flac").duration
TALKING = ((duration - spoken) / 2, (duration + spoken) / 2)  # when the target is talking [s]
CENTERS = (0.0, -40.0, 40.0)  # target ahead, maskers to the left and right [deg]
RATE = 2.0  # oscillations per second


def trajectory(center, amplitude=0.0, phase=0.0):
    """Azimuth over the whole sentence [deg] every 20 ms, for the plots, and the path at 1 m
    that so.move_sound follows, as a function of time."""

    def azimuth(t):
        return center + amplitude * np.sin(2 * np.pi * (RATE * t + phase))

    t = np.arange(0, duration, 0.02)
    return t, azimuth(t), so.hcc_trajectory(100.0, 0.0, azimuth)


LABELS = ("target", "left masker", "right masker")
COLORS = ("#d62728", "#7f7f7f", "#b0b0b0")  # one per talker, in plots and in the view from above


def render(amplitude, phase=0.0, alone=False):
    """The target oscillating with ``amplitude`` [deg] about straight ahead, with the
    maskers standing still unless ``alone``. Returns the mix, the target as rendered
    on its own, the trajectories' times and each talker's azimuth."""
    t, az_target, path = trajectory(CENTERS[0], amplitude, phase)
    parts = [so.move_sound(target, path, hrirs)]
    azimuths = [az_target]
    if not alone:
        for snd, center in zip(maskers, CENTERS[1:], strict=True):
            _, az, path = trajectory(center)
            parts.append(so.move_sound(snd, path, hrirs))
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
        clear = cues.iac > 0.8  # time windows where the two ears are well correlated
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
# ## Coming closer
#
# Now the target walks toward the listener while it says its sentence, from 3 m to 30 cm,
# 30° to the right, about 1 m/s. Its direct sound grows as 1/r, by 20 dB in all, and arrives
# 8 ms sooner at the end than at the start. Closer than 1.6 m the measured responses take over,
# and the ear on the talker's side gains more than the other: the level difference between the
# ears grows as the talker comes near, which is what Qu et al. (2009) measured distances for.
#
# A room changes the picture. The reverberation, built up from reflections off every wall, has
# about the same level wherever the talker stands, while the direct sound grows as it comes
# closer, so a talker far away is mostly room and one close by mostly direct sound. That ratio
# is one of the cues to distance; see [Synthetic reverberation](reverb.html) for still talkers.

# %%
WALK = (300.0, 30.0)  # cm: from 3 m to 30 cm
WALK_AZIMUTH = 30.0  # deg, to the right
talker = so.load(f"docs/speech/{files[0]}.flac").normalize()  # the target sentence, without padding
room = so.synth_ir(0.6, fs, n_channels=2, rng=1)  # RT60 0.6 s, a different tail at each ear


def walk_in(room=None):
    """The target walking in while it says its sentence, through all eight measured distances.
    The room, if any, is set so direct and reverberant sound are equal 1 m away."""
    path = so.hcc_trajectory(dist=([0, talker.duration], WALK), elev=0, azim=WALK_AZIMUTH)
    return so.move_sound(talker, path, hrirs, room=room, drr_db=0.0)


def levels_db(data, window=0.2):
    """Level [dB] of each channel of ``data`` (samples, channels) in time windows."""
    n = int(window * fs)
    count = len(data) // n
    power = np.mean(data[: count * n].reshape(count, n, -1) ** 2, axis=1)
    return (np.arange(count) + 0.5) * window, 10 * np.log10(power + 1e-20)


def show_walk(rendered, title):
    """Distance, and the level at each ear re the talker, in 0.2 s time windows."""
    t_walk = np.linspace(0, talker.duration, 200)
    distance = np.interp(t_walk, [0, talker.duration], WALK) / 100
    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    axes[0].plot(t_walk, distance, color=COLORS[0], lw=1.5)
    axes[0].set(ylim=(0, 3.2), ylabel="Distance [m]", title=title)
    t_level, talker_db = levels_db(talker.data)
    ear_db = levels_db(rendered.data)[1][: len(t_level)]
    gain = ear_db - talker_db  # (windows, ears)
    for ear, (label, color) in enumerate([("left ear", "#7f7f7f"), ("right ear", COLORS[0])]):
        axes[1].plot(t_level, gain[:, ear], "o-", color=color, ms=3, lw=1, label=label)
    # 1/r, placed at the right ear's median
    one_over_r = -20 * np.log10(np.interp(t_level, t_walk, distance))
    axes[1].plot(t_level, one_over_r + np.median(gain[:, 1] - one_over_r), "--", color="k", lw=1, label="1/r")
    axes[1].set(ylabel="Level [dB]", title="Level at each ear, re the talker, in 0.2 s windows")
    axes[1].legend(loc="upper left", fontsize=8, ncols=3)
    for ax in axes:
        ax.grid(ls=":")
        ax.set_xlim(0, rendered.duration)
    axes[-1].set_xlabel("Time [s]")
    return fig, list(axes)


def walk_scene():
    t_walk = np.linspace(0, talker.duration, 100)
    return [
        {
            "label": "target",
            "color": COLORS[0],
            "t": t_walk,
            "azimuth": np.full_like(t_walk, WALK_AZIMUTH),
            "distance": np.interp(t_walk, [0, talker.duration], WALK) / 100,
        }
    ]


# %% [about]
# The target walking in from 3 m to 30 cm, with no room. The level at each ear follows 1/r
# (dashed), and the right ear, nearer the talker, pulls ahead of the left in the last meter.

# %% [demo m5] Walking in
rendered = walk_in()
fig, playhead = show_walk(rendered, "The target walks in, 30° to the right, with no room")
scene = walk_scene()
sound = finish(rendered)

# %% [about]
# The same walk in a room with an RT60 of 0.6 s, where direct and reverberant sound are equal at
# 1 m. Far away the level changes little, because the room sets it; it rises with 1/r only once
# the direct sound takes over.

# %% [demo m6] Walking in, in a room
rendered = walk_in(room)
fig, playhead = show_walk(rendered, "The same walk in a room (RT60 0.6 s)")
scene = walk_scene()
sound = finish(rendered)

# %% [markdown]
# ## Passing by
#
# A source that changes distance also changes pitch: approaching, each wave starts a little
# closer than the last and arrives sooner, so the waves are squeezed together, and receding they
# are stretched apart. Here a 120 Hz buzz passes 3 m in front of the listener, left to right, at
# 15 m/s (54 km/h). Its pitch is highest while it approaches and falls as it passes, about 150
# cents (a semitone and a half) in all, most of it in the second around the moment of passing.
# Nothing makes the shift happen but the delay changing from sample to sample.
#
# Doppler is a weak cue for people: at speeds like these, listeners judge motion mostly from the
# change in level and in interaural differences, and the rising pitch people report as a source
# approaches comes largely from its rising loudness (Carlile & Leung, 2016, review this).

# %%
SPEED, CLOSEST, BUZZ_F0 = 15.0, 3.0, 120.0  # m/s, m, Hz
PASS_DURATION = 2.5  # s
buzz = so.harmonic_complex(PASS_DURATION, fs, BUZZ_F0, f_max=5000).normalize()
emitted = (len(buzz) - 1) / fs


def passing(t):
    """Left to right in front of the listener, closest at the middle of the sound."""
    t = np.atleast_1d(t)
    return np.column_stack([SPEED * (t - emitted / 2), np.full_like(t, CLOSEST), np.zeros_like(t)])


t_pass = np.linspace(0, emitted, 250)
distance_pass = np.linalg.norm(passing(t_pass), axis=1)
azimuth_pass = np.mod(so.rect_to_hcc(*passing(t_pass).T)[2] + 180, 360) - 180
# received / emitted frequency = 1 / (1 + (rate of change of distance) / speed of sound)
shift_cents = -1200 * np.log2(1 + np.gradient(distance_pass, t_pass) / so.SPEED_OF_SOUND)
arrival = t_pass + distance_pass / so.SPEED_OF_SOUND
print(f"approaching: {shift_cents[0]:+.0f} cents; receding: {shift_cents[-1]:+.0f} cents")


# %% [about]
# A 120 Hz buzz passing 3 m in front at 15 m/s. The pitch measured at the left ear (dots, by
# so.f0_track) follows the Doppler shift computed from the path (line).

# %% [demo m7] Passing by
rendered = so.move_sound(buzz, passing, hrirs)
fig = plt.figure(figsize=(10, 4.5), layout="constrained")
axes = fig.subplots(2, 1, sharex=True)
axes[0].plot(arrival, distance_pass, color=COLORS[0], lw=1.5)
axes[0].set(ylabel="Distance [m]", title="A buzz passing 3 m in front at 15 m/s, as heard")
track = so.f0_track(rendered.left, f_lo=80, f_hi=200)
f0_left = track.f0[0]  # one channel
voiced = f0_left > 0
axes[1].plot(
    track.t[voiced],
    1200 * np.log2(f0_left[voiced] / BUZZ_F0),
    ".",
    color="k",
    ms=3,
    label="measured, left ear",
)
axes[1].plot(arrival, shift_cents, color=COLORS[0], lw=1.5, label="Doppler shift from the path")
axes[1].set(ylim=(-120, 120), ylabel="Pitch re 120 Hz [cents]", title="Pitch at the ear")
axes[1].legend(loc="upper right", fontsize=8)
for ax in axes:
    ax.grid(ls=":")
    ax.set_xlim(0, rendered.duration)
axes[-1].set_xlabel("Time [s]")
playhead = list(axes)
scene = [
    {"label": "buzz", "color": COLORS[0], "t": arrival, "azimuth": azimuth_pass, "distance": distance_pass}
]
sound = finish(rendered)

# %% [markdown]
# ## How the motion is rendered
#
# A sample the source emits reaches each ear after a delay: the onset of the head-related impulse
# response at the source's position. PKU-IOA's responses keep the travel time from the source, so
# at 1.6 m the onset comes 1.7 ms later than at 1 m. `so.move_sound` follows the path sample by
# sample: for every output sample and each ear it finds when the sound now arriving was emitted,
# and reads the source's waveform at that moment, between samples. The delay therefore changes
# smoothly, and the Doppler shift above comes out of it. What is left of each response once its
# onset is removed, its shape, is interpolated every 5 ms and cross-faded with raised-cosine
# windows. Responses between measured directions and distances are interpolated with their onsets
# aligned and the onsets interpolated separately, which avoids the comb filtering of averaging two
# responses that arrive at different times; the 3D Tune-In toolkit removes each HRIR's
# interaural delay before interpolating for the same reason (Cuevas-Rodríguez et al., 2019).
#
# Before, `so.move_sound` switched whole responses every 5 ms, a smoothed form of a long-standing
# way to filter with a filter that changes; crossfading the outputs of the old and new filter and
# swapping parts of a partitioned convolution are the others (Brandtsegg et al., 2018, review all
# three). That is accurate while the source keeps its distance, as in the swings above. When the
# distance changes, neighbouring responses hold different travel times, and cross-fading two
# copies of a sound a fraction of a millisecond apart comb-filters it: passing at 15 m/s, copies
# 5 ms apart in the path differ by 0.2 ms, with the first notch near 2.3 kHz.

# %% [markdown]
# ## References
#
# - Brandtsegg, Saue & Lazzarini (2018). Live convolution with time-varying filters. *Applied
#   Sciences* 8(1), 103. [MDPI](https://www.mdpi.com/2076-3417/8/1/103).
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L498)
# - Carlile & Leung (2016). The perception of auditory motion. *Trends in Hearing* 20.
#   [doi:10.1177/2331216516644254](https://doi.org/10.1177/2331216516644254).
# - Cho & Kidd (2022). Auditory motion as a cue for source segregation and selection in a "cocktail
#   party" listening environment. *J. Acoust. Soc. Am.* 152(3), 1684–1694.
#   [doi:10.1121/10.0013990](https://doi.org/10.1121/10.0013990). Experiment code:
#   [choyun1/MSM](https://github.com/choyun1/MSM).
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L498)
#   [`binaural.interaural_cues`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/binaural.py#L98)
# - Cuevas-Rodríguez, Picinali, González-Toledo et al. (2019). 3D Tune-In Toolkit: an open-source
#   library for real-time binaural spatialisation. *PLOS ONE* 14(3), e0211899.
#   [doi:10.1371/journal.pone.0211899](https://doi.org/10.1371/journal.pone.0211899).
#   [`spatialization.HRIRSet`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L178)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentences.
# - Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related transfer functions
#   measured with high spatial resolution using a spark gap. *IEEE Trans. Audio, Speech, Lang.
#   Process.* 17(6), 1124–1132. [PKU
#   Scholar](http://scholar.pku.edu.cn/qutianshu/publications/distance-dependent-head-related-transfer-functions-measured-high-spatial).
#   The HRIRs.
#   [`hrir_data.load_hrirs`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/hrir_data.py#L122)
