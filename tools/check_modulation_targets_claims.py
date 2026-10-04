"""Numerical checks for the claims in docs/design/views/modulation-targets.md (C1-C6).

Like the other design checkers, this is deliberately independent of sonore:
only NumPy, SciPy and soundfile (to read the gallery sentence), with every
filter and transform written out from its formula. Each line prints the
claim number and the number that supports it.

    python tools/check_modulation_targets_claims.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import istft, resample_poly, stft

FS = 16000  # audio rate [Hz]
FE = 400  # envelope rate [Hz]
F_LO = 200.0  # lowest band centre [Hz]
N_OCTAVES = 5  # bands span F_LO .. F_LO * 2**N_OCTAVES = 6400 Hz
SPEECH = Path(__file__).resolve().parent.parent / "docs" / "speech" / "bdl_arctic_a0131.flac"


def report(claim, text, value):
    print(f"{claim:4s} {text:<78s} {value:.4g}")


# ------------------------------------------------------------- front end
def band_centres(per_octave, offset=0.0):
    """Centres [octaves above F_LO], ``per_octave`` to the octave."""
    return (np.arange(int(N_OCTAVES * per_octave) + 1) + offset) / per_octave


def cosine_responses(n_samples, centres, per_octave):
    """Half-cosine filters on log2 frequency, one spacing wide on each side:
    filter k is cos(pi/2 * (x - c_k) * per_octave) for |x - c_k| < 1/per_octave.
    Their squares sum to 1 between the outermost centres. Shape (F, K)."""
    freqs = np.fft.rfftfreq(n_samples, 1 / FS)
    with np.errstate(divide="ignore"):
        octaves = np.log2(freqs / F_LO)
    distance = (octaves[:, None] - centres[None, :]) * per_octave
    return np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)


def envelopes(x, centres, per_octave):
    """Hilbert envelopes of each band at FE, shape (K, T)."""
    n_samples = len(x)
    transfer = cosine_responses(n_samples, centres, per_octave)
    spectrum = np.fft.rfft(x)[:, None] * transfer
    full = np.zeros((n_samples, len(centres)), complex)  # analytic signal: positive frequencies doubled
    full[: spectrum.shape[0]] = spectrum
    full[1 : (n_samples + 1) // 2] *= 2
    env = np.abs(np.fft.ifft(full, axis=0))
    return np.maximum(resample_poly(env, FE, FS, axis=0), 0).T  # resampling can undershoot zero


def mps(env):
    """Modulation power spectrum: |2-D DFT|^2 of the envelope array with its
    overall mean removed (circular, no taper, so phase randomization below is
    exact). Rows: spectral modulation [cyc/oct]; columns: rate [Hz]."""
    return np.abs(np.fft.fft2(env - env.mean())) ** 2


def randomize_phase(spectrum, rng, keep_long_term=False):
    """Same magnitudes, the phases of white noise's 2-D DFT (which are
    Hermitian-symmetric, so the inverse is real). With keep_long_term, the
    zero-rate column (axis 1 index 0) keeps its own phase: its magnitudes
    alone don't say which band is loud, so this keeps each band's mean."""
    noise_phase = np.angle(np.fft.fft2(rng.standard_normal(spectrum.shape)))
    if keep_long_term:
        noise_phase[:, 0] = np.angle(spectrum[:, 0])
    return np.fft.ifft2(np.abs(spectrum) * np.exp(1j * noise_phase)).real


def db(p, floor=1e-300):
    return 10 * np.log10(np.maximum(p, floor))


def speech():
    x, fs = sf.read(SPEECH)
    x = resample_poly(x, FS, fs) if fs != FS else x
    return x / np.sqrt(np.mean(x**2))


# ------------------------------------------------------------- C1: phase is free
def c1():
    """Two envelope patterns with the same MPS: speech and its phase-randomized twin."""
    rng = np.random.default_rng(1)
    per_octave = 12
    centres = band_centres(per_octave)
    env = envelopes(speech(), centres, per_octave)
    spectrum = np.fft.fft2(env - env.mean())
    twin = env.mean() + randomize_phase(spectrum, rng)

    report(
        "C1",
        "max relative difference between the two MPSs",
        np.max(np.abs(mps(twin) - mps(env))) / mps(env).max(),
    )
    report(
        "C1",
        "correlation of speech and twin envelopes over all band x time cells",
        np.corrcoef(env.ravel(), twin.ravel())[0, 1],
    )
    report("C1", "fraction of the twin's cells below zero (no envelope can do that)", np.mean(twin < 0))
    clipped = np.maximum(twin, 0)
    in_target = mps(env) > mps(env).max() * 1e-3
    change = db(mps(clipped)[in_target]) - db(mps(env)[in_target])
    report(
        "C1",
        "clipping at zero: median |MPS change| [dB] where MPS > -30 dB re peak",
        np.median(np.abs(change)),
    )
    report(
        "C1",
        "clipping at zero: power added where MPS < -30 dB, re total",
        mps(clipped)[~in_target].sum() / mps(clipped).sum(),
    )

    pooled_speech = env.sum(axis=0)
    pooled_twin = clipped.sum(axis=0)
    for name, pooled in (("speech", pooled_speech), ("twin (clipped)", pooled_twin)):
        quiet = np.mean(pooled < pooled.max() * 10 ** (-30 / 20))
        report("C1", f"{name}: fraction of time the pooled envelope is 30 dB below its peak", quiet)
    for name, pooled in (("speech", pooled_speech), ("twin (clipped)", pooled_twin)):
        deviation = pooled - pooled.mean()
        report(
            "C1",
            f"{name}: kurtosis of the pooled envelope over time",
            np.mean(deviation**4) / np.mean(deviation**2) ** 2,
        )

    # The twin above also scrambles the long-term spectrum, which is set by the
    # zero-rate column's phase. ModulationSpectrum.to_sound keeps that phase.
    # The same noise as the first twin (a fresh generator, so the dB twin below is unchanged).
    kept = np.maximum(
        env.mean() + randomize_phase(spectrum, np.random.default_rng(1), keep_long_term=True), 0
    )
    for name, pattern in (("twin (clipped)", clipped), ("twin keeping the zero-rate phase (clipped)", kept)):
        level_error = db(pattern.mean(axis=1)) - db(env.mean(axis=1))
        report(
            "C1",
            f"{name}: rms error of the band means (long-term spectrum) [dB], level offset removed",
            np.sqrt(np.mean((level_error - level_error.mean()) ** 2)),
        )
    report("C1", "twin keeping the zero-rate phase: fraction of cells clipped", np.mean(kept == 0))
    pooled_kept = kept.sum(axis=0)
    report(
        "C1",
        "twin keeping the zero-rate phase: fraction of time the pooled envelope is 30 dB below its peak",
        np.mean(pooled_kept < pooled_kept.max() * 10 ** (-30 / 20)),
    )
    deviation = pooled_kept - pooled_kept.mean()
    report(
        "C1",
        "twin keeping the zero-rate phase: kurtosis of the pooled envelope over time",
        np.mean(deviation**4) / np.mean(deviation**2) ** 2,
    )

    # The same in dB, as Singh & Theunissen define the MPS: the twin of a log
    # envelope is always a valid envelope, but its linear MPS is not the original's.
    log_env = 20 * np.log10(env + 1e-6 * env.max())
    log_twin = log_env.mean() + randomize_phase(np.fft.fft2(log_env - log_env.mean()), rng)
    linear_twin = 10 ** (log_twin / 20)
    change = db(mps(linear_twin)[in_target]) - db(mps(env)[in_target])
    report(
        "C1", "dB twin: median |linear MPS change| [dB] where MPS > -30 dB re peak", np.median(np.abs(change))
    )


# ------------------------------------------------------------- C2, C3: one shot from a drawn target
def drawn_target(n_bands, n_times, per_octave):
    """A target one might draw: a blob at 4 Hz, 0.5 cyc/oct, downward sweeps
    only (positive rate and density), one octave wide in rate (s.d. 0.5 oct)
    and 0.25 cyc/oct (s.d.) wide in density. Amplitude, on the 2-D DFT grid."""
    density = np.fft.fftfreq(n_bands, 1 / per_octave)[:, None]
    rate = np.fft.fftfreq(n_times, 1 / FE)[None, :]

    # A sound's MPS is symmetric under (rate, density) -> (-rate, -density),
    # so the blob and its mirror are drawn together.
    def blob(r, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.exp(-(np.log2(r / 4.0) ** 2) / (2 * 0.5**2) - (d - 0.5) ** 2 / (2 * 0.25**2))
        return np.where((r > 0) & (d > 0), g, 0.0)

    power = blob(rate, density) + blob(-rate, -density)
    return np.sqrt(power)


def pattern_from_target(amplitude, rng, depth):
    """Random-phase envelope pattern 1 + depth * P / max|P|."""
    p = randomize_phase(amplitude, rng)
    return 1 + depth * p / np.max(np.abs(p))


def tone_carrier(pattern, centres_oct, rng, duration):
    """One random-phase tone at each centre, its amplitude the pattern's row
    (linearly interpolated to FS), weighted for equal energy per octave."""
    t = np.arange(int(duration * FS)) / FS
    t_env = np.arange(pattern.shape[1]) / FE
    out = np.zeros_like(t)
    for row, x in zip(pattern, centres_oct, strict=True):
        f = F_LO * 2**x
        out += np.interp(t, t_env, row) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out


def noise_carrier(pattern, centres_oct, per_octave, rng, duration, flatten):
    """Noise bands (filtered by the squared responses, which sum to one, so the
    bands add to flat noise) times the pattern's rows. ``flatten`` divides each
    noise band by its own Hilbert envelope first, so the carrier adds no
    envelope fluctuation of its own (sonore's "low-noise")."""
    n_samples = int(duration * FS)
    t = np.arange(n_samples) / FS
    t_env = np.arange(pattern.shape[1]) / FE
    transfer = cosine_responses(n_samples, centres_oct, per_octave)
    noise_spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    out = np.zeros(n_samples)
    for k, row in enumerate(pattern):
        band = np.fft.irfft(noise_spectrum * transfer[:, k] ** 2, n=n_samples)
        if flatten:
            band_spectrum = np.fft.fft(band)
            band_spectrum[(n_samples + 1) // 2 :] = 0
            band_spectrum[1 : (n_samples + 1) // 2] *= 2
            band = np.real(np.fft.ifft(band_spectrum) / (np.abs(np.fft.ifft(band_spectrum)) + 1e-12))
        band /= np.sqrt(np.mean(band**2))
        out += np.interp(t, t_env, row) * band
    return out


def compare(measured, target_power, mask):
    """dB correlation inside the drawn region, and the share of the measured
    modulation power outside it."""
    inside = np.corrcoef(db(measured[mask]), db(target_power[mask]))[0, 1]
    outside = measured[~mask].sum() / measured.sum()
    return inside, outside


def c2_c3():
    rng = np.random.default_rng(2)
    duration = 4.0
    per_octave = 12
    centres = band_centres(per_octave)
    n_times = int(duration * FE)
    amplitude = drawn_target(len(centres), n_times, per_octave)
    target_power = amplitude**2
    mask = target_power > target_power.max() * 1e-2  # the drawn blob, down to -20 dB

    # C2: non-negativity limits depth.
    depths = []
    for _ in range(20):
        p = randomize_phase(amplitude, rng)
        depths.append(np.std(p) / np.max(np.abs(p)))
    report(
        "C2", "random-phase pattern scaled to just touch zero: rms depth (median of 20)", np.median(depths)
    )
    report("C2", "the same, lowest of 20 draws", np.min(depths))
    report("C2", "a single sinusoidal ripple at full depth: rms depth", 1 / np.sqrt(2))

    pattern = pattern_from_target(amplitude, rng, depth=1.0)
    report(
        "C2",
        "pattern's own MPS vs target: dB correlation inside the blob",
        compare(mps(pattern), target_power, mask)[0],
    )
    report(
        "C2",
        "pattern's own MPS: power outside the blob, re total",
        compare(mps(pattern), target_power, mask)[1],
    )
    report(
        "C2",
        "the target itself: power outside the blob (its tails), re total",
        target_power[~mask].sum() / target_power.sum(),
    )

    # C3: does one-shot synthesis put the target's MPS into the sound?
    # Carriers on a 24 per octave grid (finer than the analysis), analysed at 12 per octave.
    # The pattern is drawn at 12 per octave and interpolated in frequency to the carrier grid.
    carrier_per_octave = 24
    carrier_centres = band_centres(carrier_per_octave)
    fine_pattern = np.array([np.interp(carrier_centres, centres, column) for column in pattern.T]).T
    carriers = {
        "tones, 24 per octave": tone_carrier(fine_pattern, carrier_centres, rng, duration),
        "noise bands": noise_carrier(fine_pattern, carrier_centres, carrier_per_octave, rng, duration, False),
        "noise bands, envelopes flattened": noise_carrier(
            fine_pattern, carrier_centres, carrier_per_octave, rng, duration, True
        ),
    }
    # Tones on the analysis grid itself: each tone sits at its band's centre,
    # where both neighbouring filters are zero, so no band hears two tones.
    carriers["tones on the analysis band centres"] = tone_carrier(pattern, centres, rng, duration)
    flat = np.ones_like(fine_pattern)
    for name, sound in carriers.items():
        measured = mps(envelopes(sound, centres, per_octave))
        inside, outside = compare(measured, target_power, mask)
        report("C3", f"{name}: dB correlation with the target inside the blob", inside)
        report("C3", f"{name}: power outside the blob, re total", outside)
    for name, sound in (
        ("tones", tone_carrier(flat, carrier_centres, rng, duration)),
        ("tones on the band centres", tone_carrier(np.ones_like(pattern), centres, rng, duration)),
        ("noise bands", noise_carrier(flat, carrier_centres, carrier_per_octave, rng, duration, False)),
        ("noise, flattened", noise_carrier(flat, carrier_centres, carrier_per_octave, rng, duration, True)),
    ):
        env = envelopes(sound, centres, per_octave)
        report(
            "C3", f"unmodulated {name} carrier: rms depth of its own envelopes", np.std(env) / np.mean(env)
        )
    env = envelopes(carriers["tones, 24 per octave"], centres, per_octave)
    report("C3", "modulated tone carrier: rms depth of measured envelopes", np.std(env) / np.mean(env))


# ------------------------------------------------------------- C4, C5: measure, edit, resynthesize
N_FFT, HOP_STFT = 512, 64  # 32 ms Hann windows, 4 ms hop: 250 frames per second


def spectrogram(x):
    return stft(x, FS, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP_STFT)[2]


def to_sound(z, n_samples):
    return istft(z, FS, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP_STFT)[1][:n_samples]


def log_mps(log_mag):
    return np.abs(np.fft.fft2(log_mag - log_mag.mean())) ** 2


def c4_c5():
    rng = np.random.default_rng(3)
    x = speech()
    z = spectrogram(x)
    frame_rate = FS / HOP_STFT
    floor = 1e-4 * np.abs(z).max()  # -80 dB floor before the log
    log_mag = np.log(np.abs(z) + floor)

    # The edit: remove temporal modulations above 4 Hz from the log spectrogram
    # (a modulation lowpass in rate only, all spectral modulations kept).
    rate = np.abs(np.fft.fftfreq(log_mag.shape[1], 1 / frame_rate))[None, :]
    removed = (rate > 6.0) & (rate < 40.0)  # where the attenuation is scored
    gain = (rate <= 4.0).astype(float)
    target_log = np.real(np.fft.ifft2(np.fft.fft2(log_mag) * gain))
    target_mag = np.exp(target_log)

    original_share = log_mps(log_mag)[np.broadcast_to(removed, log_mag.shape)].sum() / log_mps(log_mag).sum()
    report("C4", "speech: share of log-spectrogram MPS power at 6-40 Hz", original_share)

    def scored(y):
        zz = spectrogram(y)
        lm = np.log(np.abs(zz) + floor)
        share = log_mps(lm)[np.broadcast_to(removed, lm.shape)].sum() / log_mps(lm).sum()
        inconsistency = np.linalg.norm(np.abs(zz) - target_mag) / np.linalg.norm(target_mag)
        return share, inconsistency

    tries = {
        "one shot, the original phase": target_mag * np.exp(1j * np.angle(z)),
        "one shot, random phase": target_mag * np.exp(1j * rng.uniform(0, 2 * np.pi, z.shape)),
    }
    for name, coefs in tries.items():
        share, inconsistency = scored(to_sound(coefs, len(x)))
        report("C4", f"{name}: share of MPS power left at 6-40 Hz", share)
        report("C4", f"{name}: that share re speech's [dB]", 10 * np.log10(share / original_share))
        report("C4", f"{name}: | |STFT| - target | / | target |", inconsistency)

    # Griffin & Lim (1984): alternate between the target magnitude and the
    # nearest consistent STFT, from the original phase.
    phase = np.angle(z)
    for iteration in range(1, 201):
        y = to_sound(target_mag * np.exp(1j * phase), len(x))
        phase = np.angle(spectrogram(y))
        if iteration in (10, 50, 200):
            share, inconsistency = scored(y)
            report("C5", f"Griffin-Lim, {iteration} iterations: share of MPS power left at 6-40 Hz", share)
            report(
                "C5",
                f"Griffin-Lim, {iteration} iterations: that share re speech's [dB]",
                10 * np.log10(share / original_share),
            )
            report(
                "C5", f"Griffin-Lim, {iteration} iterations: | |STFT| - target | / | target |", inconsistency
            )

    # The same iteration on the sentence's own, unedited magnitude (a target a
    # real sound has), from random phase.
    magnitude = np.abs(z)
    phase = rng.uniform(0, 2 * np.pi, z.shape)
    for iteration in range(1, 201):
        y = to_sound(magnitude * np.exp(1j * phase), len(x))
        phase = np.angle(spectrogram(y))
        if iteration in (1, 10, 50, 200):
            mismatch = np.linalg.norm(np.abs(spectrogram(y)) - magnitude) / np.linalg.norm(magnitude)
            report("C5", f"unedited magnitude, random start, {iteration} iterations: mismatch", mismatch)


# ------------------------------------------------------------- C6: what the carrier must supply
def band_share(env, removed_rate_lo=6.0, removed_rate_hi=40.0):
    """Share of the envelope array's modulation power at removed_rate_lo..hi Hz."""
    rate = np.abs(np.fft.fftfreq(env.shape[1], 1 / FE))[None, :]
    removed = np.broadcast_to((rate > removed_rate_lo) & (rate < removed_rate_hi), env.shape)
    power = mps(env)
    return power[removed].sum() / power.sum()


def analytic_bands(x, transfer):
    """Analytic signal of every band, shape (n_samples, K)."""
    n_samples = len(x)
    spectrum = np.fft.rfft(x)[:, None] * transfer
    full = np.zeros((n_samples, transfer.shape[1]), complex)
    full[: spectrum.shape[0]] = spectrum
    full[1 : (n_samples + 1) // 2] *= 2
    return np.fft.ifft(full, axis=0)


def c6():
    """The edit of C4 (temporal modulations above 4 Hz removed), made on the
    filterbank envelopes instead: the envelope array's 2-D transform keeps
    the sentence's own modulation phase, so the edited envelopes are exact.
    They then go on three fine structures, and the sound is re-analysed."""
    rng = np.random.default_rng(5)
    per_octave = 12
    centres = band_centres(per_octave)
    x = speech()
    n_samples = len(x)
    transfer = cosine_responses(n_samples, centres, per_octave)
    env = envelopes(x, centres, per_octave)
    original = band_share(env)
    report("C6", "speech: share of envelope modulation power at 6-40 Hz", original)

    rate = np.abs(np.fft.fftfreq(env.shape[1], 1 / FE))[None, :]
    edited = env.mean() + np.real(np.fft.ifft2(np.fft.fft2(env - env.mean()) * (rate <= 4.0)))
    report("C6", "edited envelopes: fraction of cells below zero (clipped)", np.mean(edited < 0))
    edited = np.maximum(edited, 0)
    report(
        "C6",
        "edited envelopes themselves: share re speech's [dB]",
        10 * np.log10(band_share(edited) / original),
    )
    edited_full = np.maximum(resample_poly(edited.T, FS, FE, axis=0)[:n_samples], 0)

    t = np.arange(n_samples) / FS
    fine_structures = {
        "speech's own fine structure": np.cos(np.angle(analytic_bands(x, transfer))),
        "noise fine structure": np.cos(np.angle(analytic_bands(rng.standard_normal(n_samples), transfer))),
        "steady tones at the band centres": np.cos(
            2 * np.pi * (F_LO * 2**centres)[None, :] * t[:, None] + rng.uniform(0, 2 * np.pi, len(centres))
        ),
    }
    for name, fine in fine_structures.items():
        bands = edited_full * fine
        if name.startswith("steady tones"):
            y = bands.sum(axis=1)  # each tone sits where only its own band responds
        else:  # back through the bank (tight: squared responses sum to one)
            y = np.fft.irfft(np.sum(np.fft.rfft(bands, axis=0) * transfer, axis=1), n=n_samples)
        share = band_share(envelopes(y, centres, per_octave))
        report(
            "C6", f"{name}: re-analysed share at 6-40 Hz re speech's [dB]", 10 * np.log10(share / original)
        )


# ------------------------------------------------------------- C7: a search through the filterbank
def c7():
    """Griffin & Lim's idea through the filterbank instead of the STFT, on
    the C6 edit: impose the target magnitudes with the current modulation
    phase, put the envelopes on the current fine structure, re-filter, then
    take the new sound's own fine structure and modulation phase. Started
    from the sentence's own phases, or from noise's."""
    rng = np.random.default_rng(5)
    per_octave = 12
    centres = band_centres(per_octave)
    x = speech()
    n_samples = len(x)
    transfer = cosine_responses(n_samples, centres, per_octave)
    env = envelopes(x, centres, per_octave)
    original = band_share(env)
    rate = np.abs(np.fft.fftfreq(env.shape[1], 1 / FE))[None, :]
    target = np.abs(np.fft.fft2(env - env.mean())) * (rate <= 4.0)
    starts = {
        "the sentence's own phases": (
            np.cos(np.angle(analytic_bands(x, transfer))),
            np.angle(np.fft.fft2(env - env.mean())),
        ),
        "noise's phases": (
            np.cos(np.angle(analytic_bands(rng.standard_normal(n_samples), transfer))),
            np.angle(np.fft.fft2(rng.standard_normal(env.shape))),
        ),
    }
    for name, (fine, phase) in starts.items():
        for iteration in range(21):
            rebuilt = np.maximum(env.mean() + np.real(np.fft.ifft2(target * np.exp(1j * phase))), 0)
            rebuilt_full = np.maximum(resample_poly(rebuilt.T, FS, FE, axis=0)[:n_samples], 0)
            bands = rebuilt_full * fine
            y = np.fft.irfft(np.sum(np.fft.rfft(bands, axis=0) * transfer, axis=1), n=n_samples)
            fine = np.cos(np.angle(analytic_bands(y, transfer)))
            measured = envelopes(y, centres, per_octave)
            phase = np.angle(np.fft.fft2(measured - measured.mean()))
            if iteration in (0, 5, 20):
                share = band_share(measured)
                mismatch = np.linalg.norm(
                    np.abs(np.fft.fft2(measured - measured.mean())) - target
                ) / np.linalg.norm(target)
                report(
                    "C7",
                    f"{name}, {iteration} iterations: share at 6-40 Hz re speech's [dB]",
                    10 * np.log10(share / original),
                )
                report("C7", f"{name}, {iteration} iterations: | |MPS| - target | / | target |", mismatch)


if __name__ == "__main__":
    c1()
    c2_c3()
    c4_c5()
    c6()
    c7()
