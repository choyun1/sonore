"""Source and aperiodicity: how much of a voice is noise, frequency by frequency.

This script is the gallery page https://choyun1.github.io/sonore/gallery/aperiodicity.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/voice/aperiodicity.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Source and aperiodicity
#
# In the first-order source-filter picture ([Formant synthesis](formants.html)) a voice has a
# source that is either a buzz (voiced) or a hiss (unvoiced), and everything else is the filter:
# the vocal tract's formants. A vocoder such as WORLD (Morise, Yokomori & Ozawa, 2016) keeps three
# things instead of two: the pitch, a smooth spectral envelope, and an **aperiodicity**. The
# aperiodicity replaces the voicing switch. At every frequency it says what share of the power is
# noise rather than harmonics,
#
# $$A(f) = \frac{N(f)}{N(f) + P(f)},$$
#
# with $N$ the power of the noise and $P$ that of the harmonics at $f$. The buzz is $A = 0$
# everywhere and the hiss is $A = 1$; a voice is in between, and usually more so at high
# frequencies, so it can be clearly periodic at 500 Hz and mostly noise at 5 kHz.
#
# Both talkers are used throughout ([Two talkers](talkers.html)): their recordings in the
# sentence sections, and their median pitches for the synthetic vowel, since a higher pitch
# leaves fewer harmonics to measure the noise between.
#
# - [Buzz, hiss, and both](#h-buzz-hiss-and-both): a breathy vowel whose aperiodicity is known,
#   and why the filter does not change it.
# - [Fit the harmonics, keep the rest](#h-fit-the-harmonics-keep-the-rest): measuring the share
#   of noise directly, at each talker's pitch.
# - [What WORLD reports](#h-what-world-reports): D4C, WORLD's measure, and why it differs.
# - [A sentence](#h-a-sentence): both measures on the two recordings.
# - [Listening](#h-listening): each sentence rebuilt by WORLD's synthesis with each
#   aperiodicity, with none, and with nothing else.
# - [What this page leaves out](#h-what-this-page-leaves-out).

# %% [markdown]
# ## The vowel, and code the examples share
#
# The vowel of "hod", made from Klatt's (1980) glottal source, the radiation from the lips and
# five formants, as [Formant synthesis](formants.html#h-a-vowel-piece-by-piece) explains; the code
# is repeated here so that the page runs on its own. Here the source is a sum of harmonics at a
# steady pitch plus white noise, both put through the same formants. The source's harmonics fall
# about 6 dB per octave and the noise is flat, so the share of noise rises about 6 dB per octave.
# The noise is set as strong as the harmonics at 4 kHz. Everything about the vowel is known, so
# its aperiodicity can be written down.
#
# The vowel is made at two pitches: the median pitch of each talker's recording, measured with
# `so.f0_track` as on [Two talkers](talkers.html#h-pitch). At the male talker's pitch it has
# Peterson and Barney's (1952) male formants for "hod", and at the female talker's pitch their
# female ones ([Formant synthesis](formants.html#h-six-vowels)).

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
# The two talkers, by the CMU ARCTIC speaker names. Sources: docs/speech/SOURCES.md.
SPEAKERS = {"Male talker": "bdl", "Female talker": "slt"}
talkers = {
    label: finish(so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac")))
    for label, speaker in SPEAKERS.items()
}
tracks = {label: so.f0_track(snd) for label, snd in talkers.items()}
medians = {label: float(np.median(track.f0[0][track.voiced[0]])) for label, track in tracks.items()}
for label, f0 in medians.items():
    print(f"{label}: median F0 {f0:.1f} Hz")

FS = 16000
DUR = 1.0
# "Hod" for each talker's pitch, (frequency, bandwidth) in Hz: Peterson and Barney's male and
# female F1 to F3, and F4 and F5 as on the Formant synthesis page.
HOD = {
    "Male talker": [(730, 60), (1090, 90), (2440, 150), (3500, 200), (4500, 250)],
    "Female talker": [(850, 60), (1220, 90), (2810, 150), (4100, 200), (4900, 250)],
}


def draw(container, snd, title, fmax):
    """Waveform and narrowband spectrogram (Hann 33 ms, harmonics resolved), on one time axis, in
    a figure or subfigure. Returns the two panels, for the playhead."""
    ax0, ax1 = container.subplots(2, 1, sharex=True, height_ratios=[0.45, 1])
    snd.plot(ax0, color="k", lw=0.4)
    ax0.set(title=title, xlabel="")
    so.STFT(snd, win_dur=0.0333, hop_dur=0.002).plot(ax1, db_range=70, colorbar=False, fmax=fmax)
    ax1.set_title("Spectrogram (Hann 33 ms)")
    for ax in (ax0, ax1):
        ax.set_xlim(0, snd.duration)
    return [ax0, ax1]


def show(snd, title, fmax=5000):
    fig = plt.figure(figsize=(10, 4.2), layout="constrained")
    return fig, draw(fig, snd, f"Waveform: {title}", fmax)


def show_pair(sounds, titles, fmax=5000):
    """One column per sound; `titles` maps each sound's label to its column's title."""
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    columns = fig.subfigures(1, 2)
    playhead = {
        label: draw(column, snd, titles[label], fmax)
        for column, (label, snd) in zip(columns, sounds.items(), strict=True)
    }
    return fig, playhead


def formants(snd, table):
    for freq, bandwidth in table:
        snd = so.resonator(snd, freq, bandwidth)
    return snd


# Klatt's glottal source and the radiation from the lips, as a gain at any frequency.
impulse = so.Sound(np.r_[1.0, np.zeros(2**14 - 1)], FS)
glottal_gain = np.abs(np.fft.rfft(so.resonator(impulse, 0, 100).data[:, 0]))
gain_freqs = np.fft.rfftfreq(2**14, 1 / FS)


def source_gain(f):
    return np.interp(f, gain_freqs, glottal_gain) * np.abs(2 * np.sin(np.pi * np.asarray(f) / FS))


def breathy_source(f0):
    """Harmonics of f0 with Klatt's source spectrum, up to 7.2 kHz, and white noise with the
    harmonics' power per hertz at 4 kHz."""
    harmonic_numbers = np.arange(1, int(0.45 * FS / f0) + 1)
    buzz = so.harmonic_complex(
        DUR, FS, f0, harmonics=harmonic_numbers, amplitudes=lambda t, f: source_gain(f)
    )
    # harmonic_complex scales its output to RMS 1; the harmonics' power per hertz near f is then
    # (scale * gain(f))^2 / 2 per harmonic, one harmonic every f0 hertz
    scale = 1 / np.sqrt(np.sum(source_gain(harmonic_numbers * f0) ** 2) / 2)
    noise_density = (scale * source_gain(4000.0)) ** 2 / 2 / f0
    hiss = so.gaussian_noise(DUR, FS, rng=1) * np.sqrt(noise_density * FS / 2)  # white: flat density
    return buzz, hiss


def true_aperiodicity(f):
    """The share of noise at f, from the source alone. With the noise as strong as the harmonics
    at 4 kHz it is the same at any pitch."""
    return 1 / (1 + (source_gain(f) / source_gain(4000.0)) ** 2)


sources = {label: breathy_source(f0) for label, f0 in medians.items()}
vowels = {label: formants(buzz + hiss, HOD[label]) for label, (buzz, hiss) in sources.items()}
window_times = np.arange(0, DUR, 0.005)
exact = {label: (window_times, np.full(len(window_times), f0)) for label, f0 in medians.items()}  # every 5 ms
inside = (window_times > 0.1) & (window_times < 0.9)  # time windows away from the ends
pitch_names = {label: f"the {label.lower()}'s pitch, {f0:.0f} Hz" for label, f0 in medians.items()}

# %% [markdown]
# ## Buzz, hiss, and both
#
# The first-order picture has two settings of the source, and the voice in between. The first
# two sounds are at the male talker's pitch.

# %% [about]
# The buzz alone: harmonics through the formants, $A = 0$ everywhere. Every frequency is
# periodic: the spectrogram shows harmonics all the way up, with nothing between them.

# %% [demo ap1] The buzz alone
buzz, hiss = sources["Male talker"]
sound = finish(formants(buzz, HOD["Male talker"]))
fig, playhead = show(sound, "harmonics through the formants (A = 0)")

# %% [about]
# The hiss alone: white noise through the same formants, $A = 1$ everywhere. The vowel is still
# there, whispered, because the formants are.

# %% [demo ap2] The hiss alone
sound = finish(formants(hiss, HOD["Male talker"]))
fig, playhead = show(sound, "noise through the formants (A = 1)")

# %% [about]
# Both together, at each talker's pitch. Below about 1 kHz the harmonics stand well above the
# noise; above 3 kHz the noise fills in between them and then covers them. This is what makes a
# voice breathy: neither switch setting, but a mixture whose proportion changes with frequency.
# At the female talker's pitch the harmonics are farther apart, and the noise shows in wider gaps
# between them (see [Two talkers](talkers.html#h-harmonics-sample-the-envelope)). [Formant
# synthesis](formants.html#d-fn3) makes a breathy vowel with Klatt's aspiration noise.

# %% [demo ap3] A breathy vowel
sounds = {label: finish(vowel) for label, vowel in vowels.items()}
fig, playhead = show_pair(sounds, {label: f"Harmonics and noise, {pitch_names[label]}" for label in sounds})

# %% [markdown]
# Its aperiodicity is the noise's share of the source at each frequency. The harmonics and the
# noise both pass through the same filter $H$, so the share after it is
#
# $$\frac{|H(f)|^2 N(f)}{|H(f)|^2 N(f) + |H(f)|^2 P(f)} = \frac{N(f)}{N(f) + P(f)}:$$
#
# the filter cancels. The aperiodicity is a property of the source, measured on the output,
# and it can be changed without touching the formants, and the formants without touching it.
# Here it is also the same at both pitches, since the noise is set against the harmonics at
# 4 kHz in both.

# %% [figure ap4] Aperiodicity of the vowel, from its source
freqs = np.linspace(50, 7200, 1000)
fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
ax.plot(freqs, 10 * np.log10(true_aperiodicity(freqs)), color="k", label="the breathy vowel, at either pitch")
ax.axhline(0, color="C3", ls="--", lw=1, label="the hiss (A = 1)")
ax.axhline(-40, color="C0", ls="--", lw=1, label="the buzz (A = 0, at the bottom of the plot)")
ax.set(xlabel="Frequency (Hz)", ylabel="Share of noise (dB)", xlim=(0, 7200), ylim=(-40, 3))
ax.set_title("Aperiodicity: noise power over total power at each frequency")
ax.legend(loc="lower right", fontsize=8)

# %% [markdown]
# ## Fit the harmonics, keep the rest
#
# The definition suggests the measurement: find the best periodic sound, take it away, and see
# how much is left. If the pitch is known, the periodic part of a short stretch is a sum of
# harmonics whose phases follow the running phase $\Phi(t) = 2\pi \int F_0$, each with an amplitude
# and a phase to be found. That is a linear least-squares fit. Here is one time window of the vowel
# at the male talker's pitch, under a Hann window four periods long, with each harmonic also allowed
# an amplitude that changes linearly across the window.

# %%
vowel, F0 = vowels["Male talker"], medians["Male talker"]
center = int(0.5 * FS)  # the time window at 0.5 s
half = int(round(2 * FS / F0))  # half of four periods
offsets = np.arange(-half, half + 1)
window = 0.5 + 0.5 * np.cos(np.pi * offsets / (half + 1))
phase = 2 * np.pi * F0 * offsets / FS  # the running phase over the time window
harmonic_phases = np.outer(phase, np.arange(1, int(FS / 2 / F0) + 1))
ramp = (offsets / half)[:, None]
columns = np.column_stack(
    [np.ones(len(offsets)), np.cos(harmonic_phases), np.sin(harmonic_phases)]
    + [ramp * np.cos(harmonic_phases), ramp * np.sin(harmonic_phases)]
)
segment = vowel.data[center + offsets, 0]
root_window = np.sqrt(window)  # weighted least squares: minimize the windowed residual
weights, *_ = np.linalg.lstsq(columns * root_window[:, None], segment * root_window, rcond=None)
residual = segment - columns @ weights
print(f"{columns.shape[1]} columns fitted to {len(offsets)} samples")
residual_power, signal_power = np.sum((window * residual) ** 2), np.sum((window * segment) ** 2)
print(f"windowed residual power over windowed signal power: {residual_power / signal_power:.3f}")

# %% [about]
# The time window's spectrum, and the spectrum of what is left after the fit. The harmonic peaks
# are gone from the residual, and what remains is the noise between and under them. Where the
# noise was below the harmonics (low frequencies) the residual lies far below the time window's
# spectrum; above 4 kHz the two are close. Their ratio, read across frequency, is the
# aperiodicity.

# %% [figure ap5] One time window, before and after the harmonics are taken away
n_fft = 4096
segment_freqs = np.fft.rfftfreq(n_fft, 1 / FS)
segment_db = 20 * np.log10(np.abs(np.fft.rfft(window * segment, n_fft)) + 1e-12)
residual_db = 20 * np.log10(np.abs(np.fft.rfft(window * residual, n_fft)) + 1e-12)
fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
ax.plot(segment_freqs, segment_db, color="0.6", lw=0.7, label="the time window")
ax.plot(segment_freqs, residual_db, color="C3", lw=0.7, label="the time window less the fitted harmonics")
top = segment_db.max() + 5
ax.set(xlabel="Frequency (Hz)", ylabel="Level (dB)", xlim=(0, 6000), ylim=(top - 95, top))
ax.set_title(f"One time window of the breathy vowel at {F0:.0f} Hz (Hann, four periods)")
ax.legend(loc="upper right", fontsize=8)

# %% [markdown]
# `so.harmonic_aperiodicity` does this at every time window. It adds one correction: the fit also
# absorbs a little of the noise, the part that happens to look like the harmonics near each
# harmonic frequency, so the residual is a little too small there. How much is known exactly from
# the fit itself (it is what the fit would do to white noise), and is divided out. The shares are
# then summed over cells two harmonics wide and put on the frequency grid of WORLD's envelope.
#
# At the female talker's pitch the same window, four periods long, is shorter, and the vowel has
# fewer harmonics to fit:

# %%
residual_shares = {label: so.harmonic_aperiodicity(vowels[label], exact[label]) for label in vowels}
for label, f0 in medians.items():
    share = residual_shares[label]
    error_db = 10 * np.log10(share.share[0][:, inside].mean(axis=1) / true_aperiodicity(share.f))
    low, high = np.percentile(error_db[(share.f >= 300) & (share.f <= 7000)], [0, 100])
    print(f"{pitch_names[label]}: {int(0.45 * FS / f0)} harmonics below 7.2 kHz, window {4000 / f0:.1f} ms")
    print(f"  measured minus truth, 300 to 7000 Hz: {low:+.1f} to {high:+.1f} dB")

# %% [about]
# The share of noise measured at each talker's pitch, averaged over the time windows, against the
# truth, which is the same at both.

# %% [figure ap6] The measured share of noise against the truth
fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
ax.plot(freqs, 10 * np.log10(true_aperiodicity(freqs)), color="k", lw=2.5, alpha=0.35, label="truth")
for (label, share), ls in zip(residual_shares.items(), ["-", "--"], strict=True):
    mean_share = share.share[0][:, inside].mean(axis=1)
    ax.plot(
        share.f,
        10 * np.log10(mean_share),
        color="C3",
        ls=ls,
        label=f"so.harmonic_aperiodicity, {pitch_names[label]}",
    )
ax.set(xlabel="Frequency (Hz)", ylabel="Share of noise (dB)", xlim=(0, 7200), ylim=(-40, 3))
ax.set_title("The harmonic residual reads the vowel's aperiodicity (mean over time windows)")
ax.legend(loc="lower right", fontsize=8)

# %% [markdown]
# At both pitches the measure follows the truth to within a few decibels above 300 Hz (the
# printout gives the range). At the female talker's pitch the cells, two harmonics wide, are
# wider in hertz, so the curve is coarser: it reads too much noise around 1.2 kHz, where at the
# male talker's pitch it reads too little.
#
# The measure is the definition, which makes it easy to explain and to check. It has one
# weakness: it needs the pitch to about a tenth of a percent. A harmonic whose frequency is
# slightly wrong drifts out of phase with the fit over the window, and what the fit misses reads
# as noise, more so the higher the harmonic. For the same reason it reads cycle-to-cycle
# irregularity (jitter and shimmer) as noise.

# %% [markdown]
# ## What WORLD reports
#
# WORLD's own measure is D4C (Morise, 2016), which `so.d4c` reproduces exactly. It works from a
# "group delay" of the time window, smoothed over the harmonics, whose spectrum is sorted to see how
# much of its power lies outside the strongest components, in bands 3 kHz wide around each
# multiple of 3 kHz. A correction for F0 follows, and the curve is drawn as straight lines (in dB)
# from −60 dB at 0 Hz through the bands' values to 0 dB at the top. At 16 kHz there is only one
# band, at 3 kHz. D4C's paper says it was tuned with listening tests so that WORLD's resynthesis
# sounds natural; the −60 dB at 0 Hz says "the lowest frequencies are periodic", which is usually
# true of speech.

# %% [about]
# Both measures at both pitches, against the truth: solid lines at the male talker's pitch, dashed
# at the female talker's.

# %% [figure ap7] D4C and the harmonic residual against the truth
d4c_shares = {label: so.d4c(vowels[label], exact[label]) for label in vowels}
fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
ax.plot(freqs, 10 * np.log10(true_aperiodicity(freqs)), color="k", lw=2.5, alpha=0.35, label="truth")
for name, shares, color in [
    ("so.harmonic_aperiodicity", residual_shares, "C3"),
    ("so.d4c (WORLD)", d4c_shares, "C0"),
]:
    for (label, share), ls in zip(shares.items(), ["-", "--"], strict=True):
        mean_share = share.share[0][:, inside].mean(axis=1)
        ax.plot(
            share.f, 10 * np.log10(mean_share), color=color, ls=ls, label=f"{name}, {medians[label]:.0f} Hz"
        )
ax.set(xlabel="Frequency (Hz)", ylabel="Share of noise (dB)", xlim=(0, 7200), ylim=(-62, 3))
ax.set_title("D4C at 16 kHz: one measured value at 3 kHz, joined to −60 dB at 0 Hz and 0 dB at 8 kHz")
ax.legend(loc="lower right", fontsize=8, ncol=2)

# %% [markdown]
# So the two answer different questions. The harmonic residual reports the share of noise. D4C
# reports a value that makes WORLD's synthesis sound natural, and below 3 kHz it is far from
# the share of noise on this vowel, at both pitches, and its curve hardly changes with the pitch.
# In exchange, D4C hardly cares whether the pitch is exact.
# Averaged over bands and time windows, at each talker's pitch, with the exact pitch track and
# with one 1% too high:

# %%
band_edges = [0, 1000, 2000, 4000, 7000]
bands = "".join(f"{f'{low}-{high}':>11}" for low, high in zip(band_edges[:-1], band_edges[1:], strict=True))
print(f"{'Share of noise [dB], band [Hz]':<34}{bands}")
for label, vowel in vowels.items():
    track = exact[label]
    envelope = so.cheaptrick(vowel, track)
    # the truth in each band: noise power over total power, the envelope weighting each frequency
    # as .bands() weights it
    weights = envelope.data[0][:, inside].mean(axis=1)
    truth_db = []
    for low, high in zip(band_edges[:-1], band_edges[1:], strict=True):
        in_band = (envelope.f >= low) & (envelope.f < high)
        share = np.sum(true_aperiodicity(envelope.f[in_band]) * weights[in_band]) / np.sum(weights[in_band])
        truth_db.append(10 * np.log10(share))
    print(f"{label.lower()}'s pitch, {medians[label]:.0f} Hz")
    print(f"{'  truth':<34}" + "".join(f"{value:>11.1f}" for value in truth_db))
    wrong_track = (window_times, track[1] * 1.01)
    for name, measure in [("harmonic residual", so.harmonic_aperiodicity), ("D4C", so.d4c)]:
        for f0_name, f0 in [("exact F0", track), ("F0 1% high", wrong_track)]:
            shares = measure(vowel, f0).bands(band_edges, envelope)[0][:, inside].mean(axis=1)
            row = f"  {name}, {f0_name}"
            print(f"{row:<34}" + "".join(f"{value:>11.1f}" for value in 10 * np.log10(shares)))

# %% [markdown]
# With the wrong pitch, the harmonic residual's upper bands move toward 0 dB (all noise), as the
# high harmonics drift out of phase with the fit; D4C's numbers do not change. At the female
# talker's pitch the same error matters less. Over a window four periods long, a harmonic drifts
# out of phase in proportion to its number, and at a higher pitch a given frequency belongs to a
# lower harmonic.

# %% [markdown]
# ## A sentence
#
# The sentence both talkers read ([Two talkers](talkers.html)), with the F0 track WORLD's Harvest
# (Morise, 2017) measured on each recording, stored with it. On recorded speech there is no truth
# to compare with, but the two maps can be set side by side. Both are 1 (0 dB, all noise) where
# the track says unvoiced. The cell prints, for each talker, the median share of noise over the
# voiced time windows in each band.

# %%
harvest = {}
for label, speaker in SPEAKERS.items():
    harvest_times, harvest_f0 = np.loadtxt(
        fetch(f"docs/speech/{speaker}_arctic_a0131_f0.csv"), delimiter=",", skiprows=2
    ).T
    harvest[label] = (harvest_times, harvest_f0)
envelopes = {label: so.cheaptrick(talkers[label], harvest[label]) for label in talkers}
d4c_maps = {label: so.d4c(talkers[label], harvest[label]) for label in talkers}
residual_maps = {label: so.harmonic_aperiodicity(talkers[label], harvest[label]) for label in talkers}

print(f"{'Median share of noise [dB], band [Hz]':<44}{bands}")
for label in talkers:
    voiced = np.interp(d4c_maps[label].t, *harvest[label]) > 0
    for name, maps in [("D4C", d4c_maps), ("harmonic residual", residual_maps)]:
        shares = maps[label].bands(band_edges, envelopes[label])[0][:, voiced]
        row = f"{label}, {name}"
        print(f"{row:<44}" + "".join(f"{value:>11.1f}" for value in 10 * np.log10(np.median(shares, axis=1))))

# %% [markdown]
# For both talkers, in voiced time windows the harmonic residual is lowest below 1 kHz, where the
# harmonics are strong, and within a few decibels of 0 dB (all noise) above 2 kHz. D4C has the same
# shape in every voiced time window, a single bend at 3 kHz, and reads far less noise below it.

# %% [figure ap8] Envelope and aperiodicities of the sentence
fig = plt.figure(figsize=(10, 7.2), layout="constrained")
for column, label in zip(fig.subfigures(1, 2), talkers, strict=True):
    axes = column.subplots(3, 1, sharex=True)
    envelopes[label].plot(axes[0], db_range=70)
    axes[0].set_title(f"{label}: spectral envelope (CheapTrick)")
    for ax, aperiodicity, title in [
        (axes[1], d4c_maps[label], "Aperiodicity: so.d4c (WORLD)"),
        (axes[2], residual_maps[label], "Aperiodicity: so.harmonic_aperiodicity"),
    ]:
        aperiodicity.plot(ax, db_range=40)
        ax.set_title(title)
    for ax in axes:
        ax.set_xlim(0, talkers[label].duration)
        ax.set_xlabel("")
    axes[2].set_xlabel("Time (s)")

# %% [markdown]
# ## Listening
#
# `so.world_synthesize` is WORLD's synthesis, reproduced sample for sample. At every pitch period
# it adds two pieces: the harmonics' share of the envelope, $S(1 - A)$, as one pulse, and the
# noise's share, $S A$, as a short burst of filtered noise. Below, each talker's own pitch track
# and envelope with four aperiodicities. (WORLD keeps $A$ between −60 dB and just under 0 dB, and
# time windows the track calls unvoiced are always noise.)


# %%
def titled(what):
    return {label: f"{label}: {what}" for label in talkers}


# %% [about]
# The sentence, for reference.

# %% [demo ap9] The sentence
sounds = talkers
fig, playhead = show_pair(sounds, titled("the sentence"), fmax=8000)

# %% [about]
# WORLD's resynthesis, with D4C's aperiodicity: what WORLD itself makes from these pitch tracks,
# sample for sample.

# %% [demo ap10] Resynthesis with D4C
sounds = {
    label: finish(so.world_synthesize(harvest[label], envelopes[label], d4c_maps[label])) for label in talkers
}
fig, playhead = show_pair(sounds, titled("with so.d4c"), fmax=8000)

# %% [about]
# The same with the harmonic residual's aperiodicity, which puts more of each voice into noise,
# mostly at high frequencies.

# %% [demo ap11] With the harmonic residual
sounds = {
    label: finish(so.world_synthesize(harvest[label], envelopes[label], residual_maps[label]))
    for label in talkers
}
fig, playhead = show_pair(sounds, titled("with so.harmonic_aperiodicity"), fmax=8000)

# %% [about]
# With $A = 0$ in every voiced time window: harmonics only, the voicing switch of the first-order
# picture. The voiced parts are harmonics all the way up.

# %% [demo ap12] No noise in the voice
sounds = {}
for label in talkers:
    periodic = so.Aperiodicity(np.zeros_like(d4c_maps[label].data), d4c_maps[label].t, FS, "none")
    sounds[label] = finish(so.world_synthesize(harvest[label], envelopes[label], periodic))
fig, playhead = show_pair(sounds, titled("with A = 0"), fmax=8000)

# %% [about]
# With $A = 1$ everywhere: noise only, through the same envelope. The pitch is gone and the
# sentence is whispered; it is still intelligible, because the envelope carries the words.

# %% [demo ap13] Nothing but noise
sounds = {}
for label in talkers:
    noise_only = so.Aperiodicity(np.ones_like(d4c_maps[label].data), d4c_maps[label].t, FS, "all noise")
    sounds[label] = finish(so.world_synthesize(harvest[label], envelopes[label], noise_only))
fig, playhead = show_pair(sounds, titled("with A = 1"), fmax=8000)

# %% [markdown]
# ## What this page leaves out
#
# - **WORLD's pitch tracker.** Harvest is not part of sonore; its tracks are stored with the
#   recordings. With `so.f0_track` instead ([Pitch tracking](pitch.html)), the analysis is still
#   WORLD's but the numbers are not the ones WORLD would give, which `so.DIFFERENCES_FROM_WORLD`
#   notes.
# - **Phase.** WORLD's synthesis gives every pulse a minimum phase, so the waveform within each
#   period is not the original's. Its authors call minimum phase inappropriate for low-pitched
#   speech, where differences of phase are easier to hear.
# - **Jitter and shimmer.** A pitch track smoothed over 5 ms time windows cannot follow
#   cycle-to-cycle irregularity, so the harmonic residual counts it as noise, and a resynthesis
#   can only carry it as noise.

# %% [markdown]
# ## References
#
# - Klatt (1980). Software for a cascade/parallel formant synthesizer. *J. Acoust. Soc. Am.*
#   67(3), 971–995. [doi:10.1121/1.383940](https://doi.org/10.1121/1.383940). The glottal source.
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, by speakers bdl and slt.
# - Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis.
#   *Speech Communication* 67, 1–7.
#   [doi:10.1016/j.specom.2014.09.003](https://doi.org/10.1016/j.specom.2014.09.003).
#   [`spectral_envelope.cheaptrick`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectral_envelope.py#L217)
# - Morise (2016). D4C, a band-aperiodicity estimator for high-quality speech synthesis. *Speech
#   Communication* 84, 57–65. [doi:10.1016/j.specom.2016.09.001](https://doi.org/10.1016/j.specom.2016.09.001).
#   [`aperiodicity.d4c`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/aperiodicity.py#L119)
# - Morise (2017). Harvest: a high-performance fundamental frequency estimator from speech
#   signals. *Proc. Interspeech 2017*, 2321–2325.
#   [doi:10.21437/Interspeech.2017-68](https://doi.org/10.21437/Interspeech.2017-68). The stored
#   F0 tracks.
# - Morise, Yokomori & Ozawa (2016). WORLD: a vocoder-based high-quality speech synthesis system for
#   real-time applications. *IEICE Trans. Inf. & Syst.* E99-D(7), 1877–1884.
#   [doi:10.1587/transinf.2015EDP7457](https://doi.org/10.1587/transinf.2015EDP7457).
#   [`world.world_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/world.py#L233)
# - Peterson & Barney (1952). Control methods used in a study of the vowels. *J. Acoust. Soc. Am.*
#   24(2), 175–184. [ASA](https://pubs.aip.org/asa/jasa/article/24/2/175/722376/Control-Methods-Used-in-a-Study-of-the-Vowels).
#   The formants of "hod".
