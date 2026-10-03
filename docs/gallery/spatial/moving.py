"""Moving talkers: three talkers in front of a listener, one of them moving.

This script is the gallery page https://choyun1.github.io/sonore/gallery/moving.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/spatial/moving.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext). It downloads
the PKU-IOA head-related impulse responses at all eight distances (about 104 MB) the first
time it runs.
"""

# %% [markdown]
# # Moving talkers
#
# Three male talkers speak at once, one straight ahead and one 40° to each side. Which one do you
# follow? If one of them moves, does that help? Cho & Kidd (2022) asked this with stimuli like the
# ones on this page; the experiment's code, written with sonore's predecessor sigtools, is archived
# at [github.com/choyun1/MSM](https://github.com/choyun1/MSM).
#
# - [The talkers and the trajectories](#h-the-talkers-and-the-trajectories): three sentences, and
#   the paths the target takes.
# - [One talker, moving](#h-one-talker-moving): what motion alone sounds like.
# - [Three talkers](#h-three-talkers): the target swinging back and forth in azimuth while the
#   other two stay still.
# - [A different voice](#h-a-different-voice): a female talker as the target, still and moving.
# - [Coming closer](#h-coming-closer): the target walking up to the listener, without and with
#   a room.
# - [A cocktail party](#h-a-cocktail-party): two to six talkers walking around the listener for
#   20 to 30 s.
# - [Passing by](#h-passing-by): a buzz going past at 15 m/s, and the Doppler glide of its pitch.
# - [Straight paths across the plane](#h-straight-paths-across-the-plane): the buzz in front,
#   close by, down the side, behind, crossing at an angle and coming straight at the listener.
# - [A path no source could take](#h-a-path-no-source-could-take): the buzz jumping around the
#   head far faster than anything can move, and what the renderer makes of it.
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
# another male talker in the CMU ARCTIC corpus (sources in docs/speech/SOURCES.md). Each is scaled
# to the same RMS before rendering, as in the experiment.
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
from matplotlib.collections import LineCollection

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


def render(amplitude, phase=0.0, alone=False, talker=None):
    """The target (``talker``, by default the sentence above) oscillating with ``amplitude``
    [deg] about straight ahead, with the maskers standing still unless ``alone``. Returns the
    mix, the target as rendered on its own, the trajectories' times and each talker's azimuth."""
    t, az_target, path = trajectory(CENTERS[0], amplitude, phase)
    parts = [so.move_sound(target if talker is None else talker, path, hrirs)]
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


def show(mix, target_alone, t, azimuths, title, talking=TALKING):
    """Each talker's azimuth, and the interaural time and level differences at the ears,
    measured in 20 ms windows: of the mix (black) and of the target on its own (red).
    ``talking`` is when the target speaks [s]."""
    fig = plt.figure(figsize=(10, 6.0), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    for az, label, color in zip(azimuths, LABELS, COLORS, strict=False):
        axes[0].plot(t, az, color=color, lw=1.5, label=label)
    axes[0].axvspan(*talking, color=COLORS[0], alpha=0.08, lw=0, label="target talking")
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
# ## A different voice
#
# Motion is one way to set the target apart. A different voice is another, and a strong one: with
# two talkers at once, Brungart (2001) found a masker of the other sex far easier to ignore than one
# of the same sex. Here the target is the same sentence read by a female talker (slt), among the
# same two male talkers.

# %%
target_female = so.normalize(
    so.pad([so.load("docs/speech/slt_arctic_a0131.flac"), maskers[0]], align="center")
)[0]
spoken_female = so.load("docs/speech/slt_arctic_a0131.flac").duration
TALKING_FEMALE = ((duration - spoken_female) / 2, (duration + spoken_female) / 2)

# %% [about]
# The female talker straight ahead and the male talkers 40° to either side, nobody moving. Compare
# it with the three male talkers standing still above.

# %% [demo m8] A female talker among two male talkers, standing still
mix, target_alone, t, azimuths = render(0.0, talker=target_female)
fig, playhead = show(mix, target_alone, t, azimuths, "A female talker ahead, nobody moves", TALKING_FEMALE)
scene = scene_of(t, azimuths)
sound = finish(mix)

# %% [about]
# The female talker swings 10° to either side, as the male talker did above: both cues at once.

# %% [demo m9] A female talker among two male talkers, swinging 10 degrees
mix, target_alone, t, azimuths = render(10.0, talker=target_female)
fig, playhead = show(
    mix, target_alone, t, azimuths, "A female talker ahead swings 10° to either side", TALKING_FEMALE
)
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
# ## A cocktail party
#
# Up to six talkers walk around the listener for 20 to 30 s, each reading a passage from an
# audiobook. Each talker keeps a speed of its own between 0.9 and 1.3 m/s, inside the range of
# mean normal walking speeds reported for adults (0.94 to 1.43 m/s across 41 studies; Bohannon
# & Andrews, 2011), that drifts by up to 10 % over several seconds, and some talkers hurry
# once, 25 to 50 % faster for 2 to 3 s. A talker walks straight ahead while its heading drifts
# slowly at random, and bends away from the walls of a 16 m by 16 m room, from the listener in
# the middle (no talker comes nearer than 1.5 m) and from the other talkers, slowing down when
# someone is close ahead. These rules were chosen for this page; they are not fitted to how
# people move at a party.
#
# A talker 3 m away walking across the line of sight at 1.1 m/s changes azimuth by at most 21°
# per second (1.1/3 radians per second), and less when it walks toward or away from the
# listener; the swing of 30° at 2 Hz above peaks at 377° per second ($2\pi \cdot 2 \cdot 30$).
#
# The voices are LibriSpeech readers (Panayotov et al., 2015), a different cast in each scene,
# each scaled to the same RMS before rendering, so a talker's level at the ears depends on its
# distance alone. The room is drier than the one above: RT60 0.3 s, with direct and reverberant
# sound equal 5.6 m from a talker (15 dB more direct sound at 1 m than the walk above). Each
# scene fades in and out over 2 s. The four-talker scene is also played without the room. Not
# rendered: talkers sound the same whichever way they face (real voices radiate unevenly, and
# differently at high frequencies; Monson et al., 2012), and every voice keeps the same level
# however many others talk.

# %%
# Each scene's cast: LibriSpeech reader and chapter, label in the plots, color. A reader in two
# casts reads from a different chapter.
PAIR = [("2035_152373", "female talker", "#d62728"), ("5694_64029", "male talker", "#1f77b4")]
TRIO = [
    ("6345_93306", "female talker", "#d62728"),
    ("251_136532", "male talker A", "#1f77b4"),
    ("2428_83705", "male talker B", "#2ca02c"),
]
FOURSOME = [
    ("1462_170142", "female talker A", "#d62728"),
    ("3000_15664", "male talker A", "#1f77b4"),
    ("1993_147149", "female talker B", "#ff7f0e"),
    ("1272_141231", "male talker B", "#2ca02c"),
]
SIXSOME = [
    ("5338_284437", "female talker A", "#d62728"),
    ("8297_275155", "male talker A", "#1f77b4"),
    ("8842_304647", "female talker B", "#ff7f0e"),
    ("5694_64025", "male talker B", "#2ca02c"),
    ("2035_147961", "female talker C", "#9467bd"),
    ("2428_83699", "male talker C", "#8c564b"),
]
ROOM_HALF_WIDTH = 8.0  # m: the room is 16 m by 16 m, with the listener in the middle
PERSONAL_SPACE = 1.5  # m: no talker comes nearer the listener than this
PATH_STEP = 0.05  # s between points on a path


def walks(n_talkers, duration, seed):
    """One walk across the room per talker: a list of (times [s], points [m], shape (n, 3)).
    Each talker walks straight ahead, but its heading drifts slowly (a few degrees per second,
    changing over about 3 s), and it bends away from what is near: a wall within 2 m, the
    listener within 3 m (and never nearer than 1.5 m), another talker within 1.5 m (stepping to
    the right to pass), harder the closer it gets, turning at most 90 degrees per second. Its
    speed drifts by up to 10 % around its own, and it slows down when another
    talker is close ahead, easing into the new pace over about 0.3 s rather than braking at once.
    Half the talkers (at least one) also hurry once, for 2 to 3 s, 25 to 50 % faster, easing
    in and out over 0.5 s. Returns the walks and the hurries as (talker, start [s], length
    [s], speed-up)."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration + PATH_STEP, PATH_STEP)
    own_speed = rng.uniform(0.9, 1.3, n_talkers)
    period, phase = rng.uniform(5, 10, n_talkers), rng.uniform(0, 2 * np.pi, n_talkers)
    speed = own_speed * (1 + 0.1 * np.sin(2 * np.pi * times[:, None] / period + phase))  # (times, talkers)
    hurries = []
    for talker in rng.choice(n_talkers, size=max(1, n_talkers // 2), replace=False):
        start, length, speed_up = rng.uniform(3, duration - 6), rng.uniform(2, 3), rng.uniform(1.25, 1.5)
        ease = np.clip(np.minimum(times - start, start + length - times) / 0.5, 0, 1)
        speed[:, talker] *= 1 + (speed_up - 1) * np.sin(np.pi / 2 * ease) ** 2
        hurries.append((int(talker), start, length, speed_up))
    starts = []
    while len(starts) < n_talkers:  # apart, at least 1.5 m from the walls and 3 m from the listener
        candidate = rng.uniform(-ROOM_HALF_WIDTH + 1.5, ROOM_HALF_WIDTH - 1.5, 2)
        clear = all(np.linalg.norm(candidate - other) > 1.5 for other in starts)
        if np.linalg.norm(candidate) > PERSONAL_SPACE + 1.5 and clear:
            starts.append(candidate)
    position = np.array(starts)  # (talkers, 2)
    heading = rng.uniform(-np.pi, np.pi, n_talkers)  # direction of travel, angle from +x
    wobble = np.zeros(n_talkers)  # rad/s
    pace = np.ones(n_talkers)  # the slowing, eased
    others = 1 - np.eye(n_talkers)
    points = np.zeros((n_talkers, len(times), 3))
    for step in range(len(times)):
        points[:, step, :2] = position
        facing = np.column_stack([np.cos(heading), np.sin(heading)])
        wanted = facing.copy()
        gap = ROOM_HALF_WIDTH - np.abs(position)  # to the nearer wall along each axis
        wanted -= 4.0 * np.sign(position) * np.clip(1 - gap / 2.0, 0, None)
        distance = np.linalg.norm(position, axis=1, keepdims=True)
        wanted += 3.0 * position / distance * np.clip(1 - (distance - PERSONAL_SPACE) / 1.5, 0, None)
        between = position[:, None] - position[None, :]  # (talker, other, xy): from the other to the talker
        apart = np.linalg.norm(between, axis=2) + np.eye(n_talkers)
        push = (np.clip(1 - apart / 1.5, 0, None) * others)[:, :, None] * between / apart[:, :, None]
        away = push.sum(axis=1)
        wanted += 3.0 * (away + 0.7 * np.column_stack([away[:, 1], -away[:, 0]]))  # keep to the right
        # others within 60 degrees of straight ahead
        ahead = (np.einsum("tox,tx->to", -between, facing) > 0.5 * apart) & (others > 0)
        nearest_ahead = np.where(ahead, apart, np.inf).min(axis=1)
        slowing = np.clip((nearest_ahead - 0.5) / 1.0, 0.15, 1.0)
        pace += (slowing - pace) * PATH_STEP / 0.3
        wobble += -wobble * PATH_STEP / 3.0 + np.radians(4) * np.sqrt(PATH_STEP) * rng.normal(size=n_talkers)
        off_course = np.angle(np.exp(1j * (np.arctan2(wanted[:, 1], wanted[:, 0]) - heading)))
        heading += (np.clip(2.0 * off_course, -np.radians(90), np.radians(90)) + wobble) * PATH_STEP
        step_length = speed[step] * pace * PATH_STEP
        position = position + step_length[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
        distance = np.linalg.norm(position, axis=1, keepdims=True)  # step back out of the listener's space
        position = np.where(distance < PERSONAL_SPACE, position * PERSONAL_SPACE / distance, position)
        position = np.clip(position, -ROOM_HALF_WIDTH, ROOM_HALF_WIDTH)
    return [(times, talker_points) for talker_points in points], hurries


PARTY_ROOM = so.synth_ir(0.3, fs, n_channels=2, rng=2)  # RT60 0.3 s
PARTY_DRR = 15.0  # dB at 1 m: direct and reverberant sound are equal 5.6 m away
PARTY_FADE = 2.0  # s, at the start and end of each scene


def party(cast, duration, seed, room=PARTY_ROOM):
    """The cast's voices, each from the start of its passage for ``duration`` [s], each on its
    own walk. Returns the mix, each talker's path and the hurries."""
    voices = so.normalize([so.load(f"docs/speech/librispeech_{reader}.flac") for reader, _, _ in cast])
    paths, hurries = walks(len(cast), duration, seed)
    rendered = []
    for voice, (times, points) in zip(voices, paths, strict=False):
        excerpt = so.Sound(voice.data[: int(duration * fs)], fs).ramp(20e-3)
        rendered.append(so.move_sound(excerpt, (times, points), hrirs, room=room, drr_db=PARTY_DRR))
    return so.mix(rendered).ramp(PARTY_FADE), paths, hurries


def close_passes(paths, within=1.0):
    """The pairs of talkers that, at some moment, are less than ``within`` [m] apart."""
    pairs = []
    for first in range(len(paths)):
        for second in range(first + 1, len(paths)):
            apart = np.linalg.norm(paths[first][1] - paths[second][1], axis=1)
            if apart.min() < within:
                pairs.append((first, second, float(apart.min())))
    return pairs


def passes_title(cast, duration, paths, hurries):
    pairs = close_passes(paths)
    if pairs:
        closest = min(apart for _, _, apart in pairs)
        passing = f"{len(pairs)} {'pair passes' if len(pairs) == 1 else 'pairs pass'} within 1 m"
        passing += f" (closest {closest:.1f} m)"
    else:
        passing = "no two pass within 1 m"
    hurrying = ", ".join(
        f"{cast[talker][1].replace(' talker', '')} +{100 * (speed_up - 1):.0f} %"
        for talker, _, _, speed_up in hurries
    )
    return f"{len(cast)} talkers, {duration:.0f} s: {passing}\nhurrying (shaded): {hurrying}"


def show_party(cast, paths, hurries, title):
    """The paths from above (a dot where each talker starts, thick where it hurries), and each
    talker's distance (shaded where it hurries). Both lines thicken with the talker's speed
    above its usual pace, so a hurry shows as a line that swells and thins again."""
    fig = plt.figure(figsize=(10, 4.6), layout="constrained")
    top, distance_axes = fig.subplots(1, 2, width_ratios=(1, 1.6))
    for (times, points), (_, label, color) in zip(paths, cast, strict=False):
        speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(times)
        above_usual = np.clip(speed / np.median(speed) - 1.1, 0, None)  # the drift stays within 10 %
        widths = 0.7 + 8 * above_usual
        distance = np.linalg.norm(points, axis=1)
        for x, y, ax in ((points[:, 0], points[:, 1], top), (times, distance, distance_axes)):
            segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], 1)
            ax.add_collection(LineCollection(segments, linewidths=widths, color=color, capstyle="round"))
        top.plot(*points[0, :2], "o", color=color, ms=5)
        distance_axes.plot([], [], color=color, lw=1.5, label=label)
    for talker, start, length, _ in hurries:
        distance_axes.axvspan(start, start + length, color=cast[talker][2], alpha=0.12, lw=0)
    top.add_patch(plt.Circle((0, 0), 0.09, fill=False, color="k"))
    top.plot([0], [0.16], "^", color="k", ms=4)  # the listener faces +y
    top.add_patch(plt.Circle((0, 0), PERSONAL_SPACE, fill=False, color="0.6", ls=":"))
    edge = ROOM_HALF_WIDTH
    top.plot([-edge, edge, edge, -edge, -edge], [-edge, -edge, edge, edge, -edge], color="0.4", lw=1)
    top.set(
        xlim=(-edge - 0.2, edge + 0.2),
        ylim=(-edge - 0.2, edge + 0.2),
        aspect="equal",
        xlabel="Right [m]",
        ylabel="Front [m]",
        title="From above (dot: start; thicker: faster)",
    )
    distance_axes.set(
        ylim=(0, 10), xlim=(0, paths[0][0][-1]), xlabel="Time [s]", ylabel="Distance [m]", title=title
    )
    distance_axes.legend(loc="upper right", fontsize=8, ncols=3)
    for ax in (top, distance_axes):
        ax.grid(ls=":")
    return fig, [distance_axes]


def party_scene(cast, paths):
    """The talkers for the view from above, every 50 ms: azimuth (unwrapped, so a talker
    crossing straight ahead does not jump) and distance."""
    scene = []
    for (times, points), (_, label, color) in zip(paths, cast, strict=False):
        keep = slice(None, None, 2)
        _, _, azimuth = so.rect_to_hcc(*points[keep].T)
        scene.append(
            {
                "label": label,
                "color": color,
                "t": times[keep],
                "azimuth": np.degrees(np.unwrap(np.radians(azimuth))),
                "distance": np.linalg.norm(points[keep], axis=1),
            }
        )
    return scene


# %% [about]
# Two talkers, one female and one male, for 20 s.

# %% [demo cp1] A party of two
mix, paths, hurries = party(PAIR, duration=20.0, seed=0)
fig, playhead = show_party(PAIR, paths, hurries, passes_title(PAIR, 20, paths, hurries))
scene = party_scene(PAIR, paths)
sound = finish(mix)

# %% [about]
# Three new talkers, one female and two male, for 25 s.

# %% [demo cp2] A party of three
mix, paths, hurries = party(TRIO, duration=25.0, seed=0)
fig, playhead = show_party(TRIO, paths, hurries, passes_title(TRIO, 25, paths, hurries))
scene = party_scene(TRIO, paths)
sound = finish(mix)

# %% [about]
# Four more talkers, two female and two male, for 30 s.

# %% [demo cp3] A party of four
mix, paths, hurries = party(FOURSOME, duration=30.0, seed=0)
fig, playhead = show_party(FOURSOME, paths, hurries, passes_title(FOURSOME, 30, paths, hurries))
scene = party_scene(FOURSOME, paths)
sound = finish(mix)

# %% [about]
# The same four talkers on the same paths, without the room: only the direct sound, so each
# voice is sharper and its direction and distance easier to follow.

# %% [demo cp4] A party of four, without the room
mix, paths, hurries = party(FOURSOME, duration=30.0, seed=0, room=None)
fig, playhead = show_party(FOURSOME, paths, hurries, "The same four talkers without the room")
scene = party_scene(FOURSOME, paths)
sound = finish(mix)

# %% [about]
# Six talkers, three female and three male, for 30 s. Three of them also read in the scenes
# above, here from another chapter.

# %% [demo cp5] A party of six
mix, paths, hurries = party(SIXSOME, duration=30.0, seed=0)
fig, playhead = show_party(SIXSOME, paths, hurries, passes_title(SIXSOME, 30, paths, hurries))
scene = party_scene(SIXSOME, paths)
sound = finish(mix)

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
# ## Straight paths across the plane
#
# The same buzz, at the same 15 m/s, now along six straight lines through different parts of the
# horizontal plane. The Doppler shift depends only on how fast the distance changes, so every
# path that passes the listener starts 65 to 77 cents sharp and ends about as flat. What differs
# is how quickly the change comes and what the ears hear meanwhile. The glide takes longer the
# farther away the path passes, in proportion at this speed: from 95% to 5% of its range, 0.14 s
# at 50 cm, 0.27 s at 1 m and 0.53 s at 2 m. The level rises and falls as 1/r, by 6.5 dB on the
# path 10 m away and 31 dB on the one 50 cm away.
#
# The buzz stops at 5 kHz (`f_max`), so that a shift of several semitones up still keeps every
# harmonic below 8 kHz, half the sampling rate (see [A path no source could
# take](#h-a-path-no-source-could-take)).

# %%
TOWARD_30 = np.array([np.sin(np.radians(30)), np.cos(np.radians(30))])  # 30° to the right


def straight(point, heading, at=None):
    """A source at ``point`` (x right, y ahead) [m] at time ``at`` (by default the middle of the
    sound), moving along ``heading`` at SPEED: a path as a function of the time it emits."""
    point = np.array([*point, 0.0])
    velocity = SPEED * np.array([*heading, 0.0]) / np.hypot(*heading)
    at = emitted / 2 if at is None else at
    return lambda t: point + np.outer(np.atleast_1d(t) - at, velocity)


PATHS = {
    "Far in front": straight((0, 10), (1, 0)),
    "Close in front": straight((0, 0.5), (1, 0)),
    "Down the right side": straight((1, 0), (0, -1)),
    "Behind": straight((0, -2), (-1, 0)),
    "Crossing at an angle": straight((np.sqrt(0.5), np.sqrt(0.5)), (1, -1)),
    "Straight at the listener": straight(tuple(0.4 * TOWARD_30), tuple(-TOWARD_30), at=emitted),
}
PATH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def as_heard(path, n_points=1000):
    """When each point of the path is heard at the head's center [s], and the source's
    distance [m], azimuth [deg, unwrapped] and Doppler shift [cents] then."""
    t = np.linspace(0, emitted, n_points)
    position = path(t)
    distance = np.linalg.norm(position, axis=1)
    azimuth = np.degrees(np.unwrap(np.arctan2(position[:, 0], position[:, 1])))
    cents = -1200 * np.log2(1 + np.gradient(distance, t) / so.SPEED_OF_SOUND)
    return t + distance / so.SPEED_OF_SOUND, distance, azimuth, cents


def show_path(rendered, path, title, color, f0_range=(80, 200), cents_limit=120):
    """Azimuth, the level at each ear with 1/r, and the pitch at the ears against the Doppler
    shift computed from the path, all on the time the sound is heard."""
    arrival, distance, azimuth, cents = as_heard(path)
    fig = plt.figure(figsize=(10, 6.0), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    wrapped = np.mod(azimuth + 180, 360) - 180
    wrapped[1:][np.abs(np.diff(wrapped)) > 180] = np.nan  # no line across ±180°
    axes[0].plot(arrival, wrapped, color=color, lw=1.5)
    axes[0].set(ylim=(-180, 180), yticks=[-180, -90, 0, 90, 180], ylabel="Azimuth [deg]", title=title)
    t_level, ear_db = levels_db(rendered.data, window=0.05)
    for ear, (label, ear_color) in enumerate([("left ear", "#7f7f7f"), ("right ear", "k")]):
        axes[1].plot(t_level, ear_db[:, ear], ".-", color=ear_color, ms=3, lw=0.8, label=label)
    one_over_r = -20 * np.log10(np.interp(t_level, arrival, distance))
    sounding = (t_level > arrival[0] + 0.05) & (t_level < arrival[-1] - 0.05)
    offset = np.median(np.max(ear_db, axis=1)[sounding] - one_over_r[sounding])
    axes[1].plot(
        t_level, np.where(sounding, one_over_r + offset, np.nan), "--", color=color, lw=1.2, label="1/r"
    )
    peak = np.max(ear_db)
    axes[1].set(ylim=(peak - 45, peak + 5), ylabel="Level [dB]", title="Level at each ear, in 50 ms windows")
    axes[1].legend(loc="upper left", fontsize=8, ncols=3)
    for ear, (channel, marker_color) in enumerate([(rendered.left, "#7f7f7f"), (rendered.right, "k")]):
        track = so.f0_track(channel, f_lo=f0_range[0], f_hi=f0_range[1])
        f0_ear = track.f0[0]
        voiced = f0_ear > 0
        axes[2].plot(
            track.t[voiced],
            1200 * np.log2(f0_ear[voiced] / BUZZ_F0),
            ".",
            color=marker_color,
            ms=2.5,
            label=["measured, left ear", "measured, right ear"][ear],
        )
    axes[2].plot(arrival, cents, color=color, lw=1.2, label="Doppler shift from the path")
    axes[2].set(ylim=(-cents_limit, cents_limit), ylabel="Pitch re 120 Hz [cents]", title="Pitch at the ears")
    axes[2].legend(loc="upper right", fontsize=8, ncols=3, markerscale=3)
    for ax in axes:
        ax.grid(ls=":")
        ax.set_xlim(0, rendered.duration)
    axes[-1].set_xlabel("Time [s]")
    return fig, list(axes)


def path_scene(path, label, color):
    arrival, distance, azimuth, _ = as_heard(path, 250)
    return [{"label": label, "color": color, "t": arrival, "azimuth": azimuth, "distance": distance}]


def render_path(number):
    title = list(PATHS)[number]
    path = PATHS[title]
    rendered = so.move_sound(buzz, path, hrirs)
    fig, playhead = show_path(rendered, path, f"{title}, at 15 m/s", PATH_COLORS[number])
    return fig, playhead, path_scene(path, "buzz", PATH_COLORS[number]), finish(rendered)


# %% [about]
# The six paths seen from above, the listener at the center facing up. Each covers 37.5 m in
# 2.5 s; the dot marks where it starts.

# %% [figure ml0] Six straight paths
fig = plt.figure(figsize=(10, 5.0), layout="constrained")
axes = fig.subplots(1, 2)
for ax, half_width in zip(axes, (20, 3), strict=True):
    for (title, path), color in zip(PATHS.items(), PATH_COLORS, strict=True):
        position = path(np.linspace(0, emitted, 500))
        ax.plot(position[:, 0], position[:, 1], color=color, lw=2, label=title)
        ax.plot(*position[0, :2], "o", color=color, ms=5)
        middle = len(position) // 2 if title != "Straight at the listener" else -60
        ax.annotate(
            "",
            xy=position[middle + 10, :2],
            xytext=position[middle, :2],
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
        )
    ax.add_patch(plt.Circle((0, 0), 0.0875, fill=False, color="k"))
    for radius in (1, 1.6, 10):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="0.7", ls=":"))
    ax.set(
        xlim=(-half_width, half_width), ylim=(-half_width, half_width), aspect="equal", xlabel="x, right [m]"
    )
    ax.grid(ls=":", alpha=0.5)
axes[0].set(ylabel="y, ahead [m]", title="From above, 40 m across (circles at 1, 1.6 and 10 m)")
axes[1].set(title="The middle 6 m (the head drawn to scale)")
axes[0].legend(loc="lower left", fontsize=8)

# %% [about]
# 10 m in front, left to right. Far away, the direction changes slowly and the level only by
# 6.5 dB, and the pitch glides gently over most of two seconds.

# %% [demo ml1] Far in front
fig, playhead, scene, sound = render_path(0)

# %% [about]
# 50 cm in front, left to right, inside the measured distances. The level jumps by about 30 dB,
# the azimuth swings from one side to the other in about 0.1 s, and the pitch drops almost at once.

# %% [demo ml2] Close in front
fig, playhead, scene, sound = render_path(1)

# %% [about]
# 1 m to the right, from front to back. The interaural differences grow as the buzz comes
# alongside and shrink as it goes behind, just as they would if it turned back the way it came;
# front and back differ only in the spectral cues of the outer ear, here a KEMAR's, not yours.
# Once behind, both ears fall a few dB below 1/r, in the shadow of the outer ear.

# %% [demo ml3] Down the right side
fig, playhead, scene, sound = render_path(2)

# %% [about]
# 2 m behind, right to left: the front pass in a mirror. Interaural time and level differences
# are almost the same for a source in front and behind, so with someone else's ears it may well
# sound in front.

# %% [demo ml4] Behind
fig, playhead, scene, sound = render_path(3)

# %% [about]
# From ahead on the left to behind on the right, passing 1 m away 45° to the front right. It
# crosses the line through the ears 0.07 s after it passes closest.

# %% [demo ml5] Crossing at an angle
fig, playhead, scene, sound = render_path(4)

# %% [about]
# Straight at the listener along 30° to the right, from 38 m to 40 cm, where it stops with the
# sound. The distance shrinks at a steady 15 m/s, so the pitch stays put, 77 cents sharp
# throughout, while the level climbs by 40 dB. If the pitch seems to rise, that comes from the
# loudness, as described under Passing by.

# %% [demo ml6] Straight at the listener
fig, playhead, scene, sound = render_path(5)

# %% [markdown]
# ## A path no source could take
#
# Last, a path that is continuous but that nothing in the world could follow. The buzz stays put
# for 0.1 s, then jumps 60° to 150° around the head and to a new distance between 40 cm and 4 m,
# in 50 ms, sixteen times over. Each jump follows a quintic "smoothstep" in azimuth and log
# distance, so position, velocity and acceleration all change without a break; the path is
# smooth, only absurdly fast.

# %%
HOLD, MOVE = 0.10, 0.05  # s: still, then a jump
rng = np.random.default_rng(7)
n_jumps = int(np.ceil(PASS_DURATION / (HOLD + MOVE))) + 1
stops_azimuth, stops_distance = [0.0], [1.0]  # deg, m
for _ in range(n_jumps):
    stops_azimuth.append(stops_azimuth[-1] + rng.choice([-1, 1]) * rng.uniform(60, 150))
    stops_distance.append(np.exp(rng.uniform(np.log(0.4), np.log(4.0))))
stops_azimuth, stops_distance = np.array(stops_azimuth), np.array(stops_distance)


def jumping(t):
    """Distance [m] and azimuth [deg, unwrapped] at times ``t`` [s]."""
    t = np.atleast_1d(np.asarray(t, float))
    jump = np.clip(np.floor(t / (HOLD + MOVE)).astype(int), 0, n_jumps - 1)
    progress = np.clip((t - jump * (HOLD + MOVE) - HOLD) / MOVE, 0, 1)
    weight = progress**3 * (10 - 15 * progress + 6 * progress**2)  # quintic smoothstep
    distance = np.exp(np.log(stops_distance[jump]) * (1 - weight) + np.log(stops_distance[jump + 1]) * weight)
    return distance, stops_azimuth[jump] * (1 - weight) + stops_azimuth[jump + 1] * weight


jumping_path = so.hcc_trajectory(lambda t: 100 * jumping(t)[0], 0.0, lambda t: jumping(t)[1])
t_fine = np.linspace(0, PASS_DURATION, 250001)
velocity = np.gradient(jumping_path(t_fine), t_fine, axis=0)
top_speed = np.linalg.norm(velocity, axis=1).max()
top_acceleration = np.linalg.norm(np.gradient(velocity, t_fine, axis=0), axis=1).max()
jump_starts = np.arange(HOLD, PASS_DURATION, HOLD + MOVE)
longest_jump = np.linalg.norm(jumping_path(jump_starts + MOVE) - jumping_path(jump_starts), axis=1).max()
fastest_turn = np.abs(np.gradient(jumping(t_fine)[1], t_fine)).max() * 5e-3
doppler_range = np.percentile(as_heard(jumping_path, 250001)[3], [0, 100])


# %% [markdown]
# Measured on the path:
#
# - its top speed is {{ f"{top_speed:.0f}" }} m/s, {{ f"{top_speed / so.SPEED_OF_SOUND:.2f}" }}
#   of the speed of sound, reached from rest within 25 ms (the buzz passing by above moves at a
#   steady 15 m/s);
# - its acceleration peaks at {{ f"{top_acceleration / 9.81:.0f}" }} g;
# - the longest jump covers {{ f"{longest_jump:.1f}" }} m in 50 ms, and the fastest turns sweep
#   {{ f"{fastest_turn:.0f}" }}° of azimuth in 5 ms.
#
# The Doppler shift follows the path as faithfully as before: {{ f"{doppler_range[1]:+.0f}" }}
# cents (more than four semitones) as the buzz dives in, {{ f"{doppler_range[0]:+.0f}" }} cents
# as it leaps out, so here the pitch jumps around as much as the place.

# %% [about]
# The buzz jumping around the head. HRIR shapes are switched every 0.5 ms instead of the usual
# 5 ms, since a turn of 28° in 5 ms is too coarse a step. The pitch tracker finds the pitch only
# while the buzz holds still; during the jumps it changes too fast for the tracker's time windows,
# and the dots stop.

# %% [demo mw1] A path no source could take
rendered = so.move_sound(buzz, jumping_path, hrirs, hop=0.5e-3)
fig, playhead = show_path(
    rendered,
    jumping_path,
    "Still for 0.1 s, then a jump in 50 ms",
    "#d62728",
    f0_range=(60, 250),
    cents_limit=700,
)
scene = path_scene(jumping_path, "buzz", "#d62728")
sound = finish(rendered)

# %% [markdown]
# What the renderer does with this, measured by `tools/check_moving_trajectories.py` (nobody has
# listened for these artifacts yet):
#
# - While the buzz holds still, the pitch the tracker finds is the buzz's own, to within 3 cents
#   (95th percentile). During the jumps, where the tracker finds none, the shift was checked on a
#   1 kHz tone instead: its frequency at each ear follows the shift computed from the distance to
#   that ear to within 9 cents (median) and 48 cents (95th percentile), while the shift itself
#   spans more than ±500 cents. The rest of the difference is not yet explained; the HRIRs'
#   phase, changing with direction, is one candidate.
# - Switching HRIR shapes at the usual 5 ms leaves an error 13 dB below the signal in the worst
#   20 ms window, against switching every 0.0625 ms; every 0.5 ms, as here, 29 dB below.
# - Nothing aliases, by design: the largest upward shift, a factor of 1.35, takes the top harmonic
#   to 6.6 kHz, below the 8 kHz limit. A buzz with every harmonic up to 8 kHz, which is what
#   `so.harmonic_complex` makes without `f_max`, does alias: coming straight at the listener at
#   15 m/s, its top three harmonics are pushed over the limit and fold back near 7.7–8 kHz, 25 dB
#   below the harmonics.
# - The source comes toward the head at up to 130 m/s here. `so.move_sound` refuses a source
#   coming at the head faster than sound, whose sound would arrive in reverse order. It also
#   refuses one at 300 m/s, 0.87 of the speed of sound, because between the measured distances
#   the HRIR onsets change a little faster than the travel time does.
# - Not modelled at all is the noise a body moving through air that fast would make itself.

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
# - Bohannon & Andrews (2011). Normal walking speed: a descriptive meta-analysis. *Physiotherapy*
#   97(3), 182–189. [doi:10.1016/j.physio.2010.12.004](https://doi.org/10.1016/j.physio.2010.12.004).
# - Brandtsegg, Saue & Lazzarini (2018). Live convolution with time-varying filters. *Applied
#   Sciences* 8(1), 103. [MDPI](https://www.mdpi.com/2076-3417/8/1/103).
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L520)
# - Brungart (2001). Informational and energetic masking effects in the perception of two
#   simultaneous talkers. *J. Acoust. Soc. Am.* 109(3), 1101–1109.
#   [doi:10.1121/1.1345696](https://doi.org/10.1121/1.1345696).
# - Carlile & Leung (2016). The perception of auditory motion. *Trends in Hearing* 20.
#   [doi:10.1177/2331216516644254](https://doi.org/10.1177/2331216516644254).
# - Cho & Kidd (2022). Auditory motion as a cue for source segregation and selection in a "cocktail
#   party" listening environment. *J. Acoust. Soc. Am.* 152(3), 1684–1694.
#   [doi:10.1121/10.0013990](https://doi.org/10.1121/10.0013990). Experiment code:
#   [choyun1/MSM](https://github.com/choyun1/MSM).
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L520)
#   [`binaural.interaural_cues`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py#L105)
# - Cuevas-Rodríguez, Picinali, González-Toledo et al. (2019). 3D Tune-In Toolkit: an open-source
#   library for real-time binaural spatialisation. *PLOS ONE* 14(3), e0211899.
#   [doi:10.1371/journal.pone.0211899](https://doi.org/10.1371/journal.pone.0211899).
#   [`spatialization.HRIRSet`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L175)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentences, by speakers bdl, rms and slt.
# - Monson, Hunter & Story (2012). Horizontal directivity of low- and high-frequency energy in
#   speech and singing. *J. Acoust. Soc. Am.* 132(1), 433–441.
#   [doi:10.1121/1.4725963](https://doi.org/10.1121/1.4725963).
# - Panayotov, Chen, Povey & Khudanpur (2015). LibriSpeech: an ASR corpus based on public domain
#   audio books. *Proc. ICASSP 2015*, 5206–5210.
#   [doi:10.1109/ICASSP.2015.7178964](https://doi.org/10.1109/ICASSP.2015.7178964). The
#   cocktail-party passages, by 12 LibriVox readers (CC BY 4.0; sources in
#   docs/speech/SOURCES.md).
# - Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related transfer functions
#   measured with high spatial resolution using a spark gap. *IEEE Trans. Audio, Speech, Lang.
#   Process.* 17(6), 1124–1132. [PKU
#   Scholar](http://scholar.pku.edu.cn/qutianshu/publications/distance-dependent-head-related-transfer-functions-measured-high-spatial).
#   The HRIRs.
#   [`hrir_data.load_hrirs`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/hrir_data.py#L122)
