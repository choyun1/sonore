"""How much of a modulation edit survives on each carrier of
ModulationSpectrum.to_sound, the numbers quoted in its docstring. Uses sonore.

Each talker's sentence (docs/speech/*_arctic_a0131.flac, as on the gallery
page "Hearing a modulation spectrum") has every modulation faster than 4 Hz
removed with with_gain, and is put back on three carriers:

- the sentence itself (its own modulation phase and fine structure);
- carrier="tone", with 5 random modulation phases;
- a Gaussian noise as a Sound carrier (its fine structure only), with 5
  random modulation phases.

Printed: the share of the measured modulation power at 6-40 Hz, in dB
relative to the sentence's own share. Run from the repository root.
"""

import warnings

import numpy as np

import sonore as so

warnings.simplefilter("ignore")  # random modulation phases clip a few envelope values


def finish(snd):
    """As the gallery plays every sound: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def measure(snd):
    return so.ModulationSpectrum.octave(snd, f_lo=125, f_hi=8000)


def fast_share(snd):
    measured = measure(snd)
    power = 10 ** (measured.level / 10)
    fast = (np.abs(measured.w_t) >= 6) & (np.abs(measured.w_t) <= 40)
    return power[:, fast].sum() / power.sum()


for speaker in ("bdl", "slt"):
    sentence = finish(so.load(f"docs/speech/{speaker}_arctic_a0131.flac"))
    slow = measure(sentence).with_gain(lambda rate, density: np.abs(rate) <= 4)
    reference = fast_share(sentence)

    def left(snd, reference=reference):
        return f"{10 * np.log10(fast_share(snd) / reference):.1f}"

    print(speaker, "sentence carrier:", left(slow.to_sound(carrier=sentence)))
    tones = [slow.to_sound(carrier="tone", fs=sentence.fs, rng=seed) for seed in range(5)]
    print(speaker, "tone carrier:", " ".join(left(snd) for snd in tones))
    noises = [
        slow.to_sound(carrier=so.gaussian_noise(sentence.duration, sentence.fs, rng=50 + seed), rng=seed)
        for seed in range(5)
    ]
    print(speaker, "noise's fine structure:", " ".join(left(snd) for snd in noises))
