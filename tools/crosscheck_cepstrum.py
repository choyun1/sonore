"""Cross-check so.Cepstrum against reference implementations.

- Minimum phase: SciPy's ``scipy.signal.minimum_phase(method="homomorphic",
  half=False)``, which computes a minimum-phase filter from the folded real
  cepstrum, on a mixed-phase FIR placed alone in one frame.
- Real cepstrum: the definition MATLAB's ``rceps`` documents,
  ``real(ifft(log(abs(fft(x)))))``, written out with NumPy for one frame.
- Cepstral peak: Praat's PowerCepstrogram (through parselmouth), whose peak
  quefrency on the gallery sentence is compared with ``Cepstrum.f0`` and
  with the Harvest track.

Praat is a development-time dependency only:

    pip install praat-parselmouth
    python tools/crosscheck_cepstrum.py
"""

import numpy as np
import parselmouth
import scipy
from parselmouth.praat import call
from scipy.signal import minimum_phase

import sonore as so

FS = 16000.0


def report(text, value):
    print(f"{text:<78s} {value:.4g}")


# ------------------------------------------------------------ minimum phase
rng = np.random.default_rng(0)
roots = 0.85 * np.sqrt(rng.random(8)) * np.exp(1j * np.pi * rng.random(8))
roots[:3] = 1 / roots[:3].conj()  # three conjugate pairs outside the unit circle
h = np.real(np.poly(np.concatenate([roots, roots.conj()])))
n_fft = 4096
x = np.zeros(3 * n_fft)
x[n_fft : n_fft + len(h)] = h  # starts at the phase reference of the frame centered at n_fft
frame = so.GaborFrame(n_fft / FS, n_fft / 4 / FS, window="boxcar")
cep = so.Cepstrum(frame.analyze(so.Sound(x, FS)))
i = int(np.argmin(np.abs(cep.t - n_fft / FS)))
ours = np.fft.irfft(cep.to_stft("minimum").data[0, :, i], n_fft)[: len(h)]
ref = minimum_phase(h, method="homomorphic", n_fft=n_fft, half=False)
print(
    f"SciPy {scipy.__version__} minimum_phase(homomorphic, half=False), 17-tap mixed-phase FIR, n_fft {n_fft}"
)
report("  max |sonore - SciPy| / max |SciPy|", np.abs(ours - ref).max() / np.abs(ref).max())
roots_ref = np.roots(ref)
report("  largest root magnitude of SciPy's result (< 1: minimum phase)", np.abs(roots_ref).max())

# ------------------------------------------------------------- rceps formula
seg = rng.standard_normal(640) * np.hanning(640)
rceps = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(seg, 1024)))))
c = so.Cepstrum(
    so.GaborFrame(640 / FS, 160 / FS, window="boxcar", n_fft=1024).analyze(
        so.Sound(np.r_[np.zeros(320), seg, np.zeros(320)], FS)
    )
)
j = int(np.argmin(np.abs(c.t - 640 / FS)))
print("MATLAB rceps definition, real(ifft(log(abs(fft(x))))), one 40 ms frame")
report("  max |sonore - rceps| over quefrencies 0..n_fft/2", np.abs(c.data[0, :, j] - rceps[:513]).max())

# --------------------------------------------------------------- Praat peak
path = "docs/speech/bdl_arctic_a0131.flac"
tab = np.loadtxt("docs/speech/bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
times, harvest = tab[:, 0], tab[:, 1]
snd = so.load(path)
ours_t, ours_f0, ours_peak = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.005)).f0(75, 400, threshold=0)
ours_at = np.interp(times, ours_t, ours_f0[0])
peak_at = np.interp(times, ours_t, ours_peak[0])
praat_sound = parselmouth.Sound(path)
cg = call(praat_sound, "To PowerCepstrogram", 75, 0.005, 5000, 50)
praat_f0 = np.full(len(times), np.nan)
for k, t in enumerate(times):
    if 0.03 < t < snd.duration - 0.03:
        sl = call(cg, "To PowerCepstrum (slice)", t)
        praat_f0[k] = 1 / call(sl, "Get quefrency of peak", 75, 400, "parabolic")
ok = np.isfinite(praat_f0) & (harvest > 0)
print(f"Praat {parselmouth.PRAAT_VERSION} PowerCepstrogram (floor 75 Hz, 5 ms, to 5 kHz), search 75-400 Hz")
report("  Harvest-voiced frames compared", ok.sum())
report(
    "  sonore and Praat peak agree within 5%, fraction",
    np.mean(np.abs(ours_at[ok] / praat_f0[ok] - 1) < 0.05),
)
report(
    "  Praat peak and Harvest agree within 5%, fraction",
    np.mean(np.abs(praat_f0[ok] / harvest[ok] - 1) < 0.05),
)
report(
    "  sonore peak and Harvest agree within 5%, fraction",
    np.mean(np.abs(ours_at[ok] / harvest[ok] - 1) < 0.05),
)
sure = ok & (peak_at > 0.1)
report("  frames where sonore's peak also exceeds 0.1 (its voicing rule)", sure.sum())
report(
    "  on those, sonore and Praat agree within 5%, fraction",
    np.mean(np.abs(ours_at[sure] / praat_f0[sure] - 1) < 0.05),
)
