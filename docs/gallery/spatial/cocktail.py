"""A cocktail party: two to six talkers walking around a listener.

This script is the gallery page https://choyun1.github.io/sonore/gallery/cocktail.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/spatial/cocktail.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext). It downloads
the PKU-IOA head-related impulse responses at all eight distances (about 104 MB) the first
time it runs.
"""

# %% [markdown]
# # A "cocktail party"
#
# Two to six talkers walk around the listener for 20 to 30 s, each reading a passage from an
# audiobook.
#
# - [How the talkers walk](#h-how-the-talkers-walk): the rules the paths follow, and how fast
#   a walking talker moves in azimuth.
# - [The voices and the room](#h-the-voices-and-the-room): who reads, and what the rendering
#   leaves out.
# - [Code the examples share](#h-code-the-examples-share): the casts, the walks and the
#   rendering.
# - [Five scenes](#h-five-scenes): two, three, four and six talkers, and four without the room.
#
# Every sound here is rendered through measured head-related impulse responses (HRIRs) of a
# KEMAR manikin (Qu et al., 2009) with `so.move_sound`, as on [Moving talkers](moving.html),
# which shows how the motion is rendered. Listen with headphones.

# %% [markdown]
# ## How the talkers walk
#
# Each talker keeps a speed of its own between 0.9 and 1.3 m/s, inside the range of mean normal
# walking speeds reported for adults (0.94 to 1.43 m/s across 41 studies; Bohannon & Andrews,
# 2011), that drifts by up to 10 % over several seconds, and some talkers hurry once, 25 to 50 %
# faster for 2 to 3 s. A talker walks straight ahead while its heading drifts slowly at random,
# and bends away from the walls of a 16 m by 16 m room, from the listener in the middle (no talker
# comes nearer than 1.5 m) and from the other talkers, slowing down when someone is close ahead.
# These rules were chosen for this page; they are not fitted to how people move at a party.
#
# A talker 3 m away walking across the line of sight at 1.1 m/s changes azimuth by at most 21°
# per second (1.1/3 radians per second), and less when it walks toward or away from the
# listener; the swing of 30° at 2 Hz on [Moving talkers](moving.html#d-m4) peaks at 377° per
# second ($2\pi \cdot 2 \cdot 30$).

# %% [markdown]
# ## The voices and the room
#
# The voices are LibriSpeech readers (Panayotov et al., 2015), a different cast in each scene,
# each scaled to the same RMS before rendering, so a talker's level at the ears depends on its
# distance alone. The room is drier than the one in [Coming closer](moving.html#h-coming-closer):
# RT60 0.3 s, with direct and reverberant sound equal 5.6 m from a talker (15 dB more direct sound
# at 1 m than that walk). Each scene fades in and out over 2 s. The four-talker scene is also
# played without the room. Not rendered: talkers sound the same whichever way they face (real
# voices radiate unevenly, and differently at high frequencies; Monson et al., 2012), and every
# voice keeps the same level however many others talk.

# %% [markdown]
# ## Code the examples share

# %% [setup]
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


def fetch(path):
    """A file from the sonore repository, by its path there: the local copy when this runs from
    the repository root, otherwise downloaded from GitHub to the same relative path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/choyun1/sonore/main/" + path, path)
    return path


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


# %%
from matplotlib.collections import LineCollection

hrirs = so.load_hrirs(distances="all")  # PKU-IOA KEMAR, 20 cm to 1.6 m; downloaded on first use
print(hrirs)
fs = 16000

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
    voices = so.normalize([so.load(fetch(f"docs/speech/librispeech_{reader}.flac")) for reader, _, _ in cast])
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
        ylim=(0, 12), xlim=(0, paths[0][0][-1]), xlabel="Time [s]", ylabel="Distance [m]", title=title
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


# %% [markdown]
# ## Five scenes
#
# Each scene is drawn as the talkers' paths seen from above and each talker's distance over
# time, and a view from above follows the talkers as the scene plays.

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
# ## References
#
# - Bohannon & Andrews (2011). Normal walking speed: a descriptive meta-analysis. *Physiotherapy*
#   97(3), 182–189. [doi:10.1016/j.physio.2010.12.004](https://doi.org/10.1016/j.physio.2010.12.004).
# - Monson, Hunter & Story (2012). Horizontal directivity of low- and high-frequency energy in
#   speech and singing. *J. Acoust. Soc. Am.* 132(1), 433–441.
#   [doi:10.1121/1.4725963](https://doi.org/10.1121/1.4725963).
# - Panayotov, Chen, Povey & Khudanpur (2015). LibriSpeech: an ASR corpus based on public domain
#   audio books. *Proc. ICASSP 2015*, 5206–5210.
#   [doi:10.1109/ICASSP.2015.7178964](https://doi.org/10.1109/ICASSP.2015.7178964). The
#   passages, by 12 LibriVox readers (CC BY 4.0; sources in docs/speech/SOURCES.md).
# - Qu, Xiao, Gong, Huang, Li & Wu (2009). Distance-dependent head-related transfer functions
#   measured with high spatial resolution using a spark gap. *IEEE Trans. Audio, Speech, Lang.
#   Process.* 17(6), 1124–1132. [PKU
#   Scholar](http://scholar.pku.edu.cn/qutianshu/publications/distance-dependent-head-related-transfer-functions-measured-high-spatial).
#   The HRIRs.
#   [`hrir_data.load_hrirs`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/hrir_data.py#L124)
#   [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L609)
