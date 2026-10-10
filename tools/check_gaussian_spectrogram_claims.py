"""Numbers in docs/design/sources/gaussian-spectrogram.md, from NumPy alone.

Shares no code with sonore. Prints each claim's number with its tag.

  G1  grid spacing of McDermott, Wrobleski & Oxenham (2011): 39 half-cosine
      filters equally spaced on the ERB-number scale from 20 to 4000 Hz,
      neighbors overlapping by half, and 20 ms raised-cosine windows
      overlapping by half; their decay constants in units per ERB and per s
  G2  exponential correlation on a regular grid is first-order
      autoregressive, so a separable field can be drawn by running that
      recursion along each axis; it equals the Cholesky draw's covariance
      exactly and costs far less
  G3  the level of the paper's cells in dB: a log10-amplitude standard
      deviation of sqrt(0.5) (the 2017 prototype's variance) in dB
"""

import time

import numpy as np


def erb_number(f):
    """Glasberg & Moore (1990) ERB-number scale."""
    return 21.4 * np.log10(1 + 0.00437 * f)


# G1. With N filters whose neighbors overlap by half and which tile the range
# together with a lowpass and a highpass edge, the N + 2 filters share N + 1
# equal steps between the ends (as in McDermott & Simoncelli's cosine bank).
n_filters = 39
erb_range = erb_number(4000.0) - erb_number(20.0)
erb_per_filter = erb_range / (n_filters + 1)
hop_s = 0.020 / 2
per_filter, per_window = 0.075, 0.065
print(f"G1 ERB-number range 20-4000 Hz: {erb_range:.3f}; spacing {erb_per_filter:.4f} ERB per filter")
print(f"G1 decay {per_filter} per filter = {per_filter / erb_per_filter:.4f} per ERB "
      f"(correlation length {erb_per_filter / per_filter:.2f} ERB)")
print(f"G1 decay {per_window} per window (hop {hop_s * 1000:.0f} ms) = {per_window / hop_s:.2f} per s "
      f"(correlation length {hop_s / per_window * 1000:.0f} ms)")
print(f"G1 lag-1 correlations: across filters {np.exp(-per_filter):.4f}, across windows {np.exp(-per_window):.4f}")


# G2. AR(1) along each axis versus Cholesky of the full Kronecker covariance.
def ar1_matrix(n, rho):
    """Lower-triangular A with A @ white = AR(1) sequence started in its
    stationary distribution: x0 = e0, x_k = rho x_{k-1} + sqrt(1 - rho^2) e_k."""
    a = np.zeros((n, n))
    scale = np.sqrt(1 - rho**2)
    for k in range(n):
        for j in range(k + 1):
            a[k, j] = rho ** (k - j) * (1.0 if j == 0 else scale)
    return a


def exp_corr(n, rho):
    lags = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return rho**lags


n_bands, n_windows = 39, 81  # 400 ms of 20 ms windows at a 10 ms hop
rho_f, rho_t = np.exp(-per_filter), np.exp(-per_window)
a_f, a_t = ar1_matrix(n_bands, rho_f), ar1_matrix(n_windows, rho_t)
err_f = np.max(np.abs(a_f @ a_f.T - exp_corr(n_bands, rho_f)))
err_t = np.max(np.abs(a_t @ a_t.T - exp_corr(n_windows, rho_t)))
print(f"G2 AR(1) covariance minus exponential correlation, max abs: bands {err_f:.1e}, windows {err_t:.1e}")

# the separable draw: field = A_f @ white @ A_t.T has covariance kron(C_f, C_t)
kron = np.kron(exp_corr(n_bands, rho_f), exp_corr(n_windows, rho_t))
a_sep = np.kron(a_f, a_t)
print(f"G2 separable AR(1) covariance minus Kronecker covariance, max abs: "
      f"{np.max(np.abs(a_sep @ a_sep.T - kron)):.1e} (cells: {n_bands * n_windows})")
start = time.perf_counter()
np.linalg.cholesky(kron)
chol_s = time.perf_counter() - start
rng = np.random.default_rng(0)
start = time.perf_counter()
white = rng.standard_normal((n_bands, n_windows))
field = np.empty_like(white)
field[:, 0] = white[:, 0]
for k in range(1, n_windows):  # recursion along time
    field[:, k] = rho_t * field[:, k - 1] + np.sqrt(1 - rho_t**2) * white[:, k]
for k in range(1, n_bands):  # then along frequency
    field[k] = rho_f * field[k - 1] + np.sqrt(1 - rho_f**2) * field[k]
ar_s = time.perf_counter() - start
print(f"G2 time: Cholesky of the {n_bands * n_windows}-cell covariance {chol_s * 1000:.0f} ms, "
      f"AR(1) recursion {ar_s * 1000:.2f} ms (one run, this machine)")

# G3. The 2017 prototype's variance 0.5 on a log10-amplitude grid, in dB.
print(f"G3 sd sqrt(0.5) in log10 amplitude = {20 * np.sqrt(0.5):.2f} dB")
