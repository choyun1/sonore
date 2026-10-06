# Migrating from sigtools

sonore was previously `sigtools`, renamed to avoid a clash with an unrelated
PyPI package of that name. Version 0.2 also redesigned the API:

| sigtools 0.1 | sonore |
|---|---|
| `from sigtools.sounds import *` etc. | `import sonore as so` |
| `PureTone(dur, fs, f)`, `GaussianNoise(...)`, ... | `so.pure_tone(dur, fs, f)`, `so.gaussian_noise(...)`, ... |
| `GaussianNoise(dur, fs, lo, hi, tilt)` | `so.gaussian_noise(dur, fs, band=(lo, hi), tilt=...)`; tilt is now dB/octave |
| `SchroederPhase(dur, fs, f0, n)` | `so.schroeder_complex(dur, fs, f0, n)` |
| `SoundLoader(path)`, `Silence(dur, fs)` | `so.load(path)`, `so.silence(dur, fs)` |
| `snd + 6` (dB gain) | `snd + 6*dB` |
| `snd.make_binaural()`, `snd.extract_envelope()` | `snd.to_stereo()`, `snd.envelope()` (now returns an `Envelope`, not a Sound) |
| `ramp_edges(snd, d)` | `snd.ramp(d)` |
| `butter_bandpass_filter(snd, lo, hi)` | `so.bandpass(snd, lo, hi)` (no longer RMS-normalizes) |
| `equalize_fs`, `zeropad_sounds`, `center_sounds`, `truncate_sounds` | `so.match_fs`, `so.match_lengths(align="start"/"center")`, `so.match_lengths(mode="truncate")` |
| `normalize_rms`, `zero_mean`, `concat_sounds`, `compare_relative_db` | `so.normalize`, `snd.zero_mean()`, `so.concat`, `so.relative_db` |
| `sum(zeropad_sounds([a, b]))` | `so.mix([a, b])` |
| `MagnitudeSpectrum(s).to_Noise(dur, fs)` | `so.long_term_spectrum(s).to_sound(dur, fs)` |
| `STFT(snd, win)`, `S.to_Sound()`, `method="GLA"` | `so.STFT(snd, win)`, `S.to_sound()`, `S.griffin_lim()` |
| `IBM = S_t > S_m + lc`; `IBM * S_mix` | `so.ideal_binary_mask(S_t, S_m, lc_db=lc)`; `S_mix * mask` |
| `Subbands(snd, n)`, `.extract_envelopes()`, `.to_Sound()` | `so.cosine_filterbank(n).analyze(snd)`, `.envelopes()`, `.to_sound()` |
| `InterauralCues(snd, win)` | `so.interaural_cues(snd, win)` |
| `SimpleBIR(fs, itd, ild)` | `so.simple_bir(fs, itd, ild)` or `so.apply_itd_ild(snd, itd, ild)` |
| `SynthIR(drr, rt60, dB_thresh, fs)` | `so.synth_ir(rt60, fs, drr_db=..., decay_db=-dB_thresh)` |
| `move_sound(traj, snd)` | `so.move_sound(snd, traj, hrirs)` with `so.load_hrirs()` (downloads PKU-IOA), `so.HRIRSet.from_pku_ioa(dir)` or `.from_sofa(path)` |
| `display_STFT(x, S)`, `AudioControl(snd).display()` | `so.overview(x)`; put `snd` at the end of a cell |

Results computed with 0.1 can differ, because these 0.1 bugs were fixed:
spectrum and STFT "dB" were half the true value; the bandpass filter filtered
stereo across channels; `SimpleBIR` was a sample short and got louder with
larger ITDs; `SynthIR`'s DRR had no effect and its resynthesis filters were
shifted in frequency; tone frequencies were off by a factor of `(n-1)/n`;
`move_sound` summed ~100 unwindowed overlapping convolutions per sample; and IAC
was never computed. The ILD in `apply_itd_ild` is now split ±ILD/2 across the
ears (0.1 applied it to the right ear only).
