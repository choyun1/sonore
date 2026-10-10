"""How closely sounds made from so.gaussian_spectrogram keep the drawn cells
(docs/design/sources/gaussian-spectrogram.md, G5 and G6). Uses sonore.

  G5  for each carrier, re-analyze the sound with the same filterbank and
      windows and compare its cell levels [dB] with the drawn envelopes'
      cell levels: correlation, slope, and rms difference, over seeds
  G6  lag-1 correlations of the drawn cells across bands and windows,
      measured over many draws, against the intended values
"""

import numpy as np

import sonore as so

fs, duration, hop = 20000, 0.4, 0.010
n_seeds = 20


def cell_levels(bands_power: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Levels [dB] of (band, window) cells: power weighted by each squared
    raised-cosine window, as the paper's rms per window."""
    n_windows = int(np.ceil(t[-1] / hop)) + 1
    offset = (t[:, None] - hop * np.arange(n_windows)[None, :]) / (2 * hop)
    windows = np.where(np.abs(offset) < 0.5, 0.5 * (1 + np.cos(2 * np.pi * offset)), 0.0)
    weights = windows**2 / np.sum(windows**2, axis=0)
    return 10 * np.log10(bands_power.T @ weights + 1e-30)[:, 2:-2]  # drop the half windows at the ends


results = {"fine structure of a noise": [], "'noise' (bands keep their own envelopes)": []}
for seed in range(n_seeds):
    env = so.gaussian_spectrogram(duration, fs, rng=seed)
    bank = env.filterbank
    drawn = cell_levels(env.data[:, 1:-1, 0] ** 2, env.t)
    sounds = {
        "fine structure of a noise": env.to_sound(so.gaussian_noise(duration, fs, rng=1000 + seed)),
        "'noise' (bands keep their own envelopes)": env.to_sound("noise", rng=1000 + seed),
    }
    for name, sound in sounds.items():
        bands = bank.analyze(sound).data[:, 1:-1, 0]
        got = cell_levels(bands**2, env.t)
        # compare shapes, not overall level
        difference = (got - got.mean()) - (drawn - drawn.mean())
        corr = np.corrcoef(drawn.ravel(), got.ravel())[0, 1]
        slope = np.polyfit(drawn.ravel(), got.ravel(), 1)[0]
        results[name].append((corr, slope, np.sqrt(np.mean(difference**2))))
for name, values in results.items():
    corr, slope, rms = np.mean(values, axis=0)
    print(f"G5 {name}: corr {corr:.3f}, slope {slope:.3f}, rms difference {rms:.2f} dB ({n_seeds} seeds)")

# G6: correlation of the drawn cells themselves (the field, before any sound)
all_cells = []
for seed in range(200):
    env = so.gaussian_spectrogram(duration, fs, rng=seed)
    n_windows = int(np.ceil(env.t[-1] / hop)) + 1
    centers = np.minimum(np.round(hop * np.arange(n_windows) * fs).astype(int), len(env.t) - 1)
    all_cells.append(20 * np.log10(env.data[centers, 1:-1, 0].T))  # at a center only that window is nonzero
all_cells = np.array(all_cells)
# remove each band's mean level over all draws (a mean taken per draw would bias the correlations down)
all_cells -= all_cells.mean(axis=(0, 2), keepdims=True)
band_corr = np.corrcoef(all_cells[:, :-1].ravel(), all_cells[:, 1:].ravel())[0, 1]
time_corr = np.corrcoef(all_cells[:, :, :-1].ravel(), all_cells[:, :, 1:].ravel())[0, 1]
spacing = so.cosine_filterbank(39, 20, 4000).spacing
print(
    f"G6 lag-1 across bands {band_corr:.3f} (intended {np.exp(-spacing / 8.78):.3f}), "
    f"across windows {time_corr:.3f} (intended {np.exp(-hop / 0.154):.3f}); 200 draws"
)
