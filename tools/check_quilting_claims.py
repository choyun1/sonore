"""Numbers in docs/design/views/quilting.md, from NumPy alone.

Shares no code with sonore. Prints each claim's number with its tag.

  Q1  segments in a source and in a quilt, for Overath et al.'s set-up
      (20 s sources at 20 kHz, 30 ms windows, 6 s quilts) and for a 30 s
      LibriSpeech passage, under the paper's rule and the toolbox's
  Q2  cost of choosing the order one segment at a time on full-rate
      envelopes (30 ms borders, 32 bands), for the shortest segments:
      differences taken directly, as the toolbox does, against the same
      distances for all pairs from squared norms and one matrix product
"""

import math
import time

import numpy as np

WINDOW = 0.030  # s, the paper's border length and the unit of segment length
SEGMENT_WINDOWS = [1, 2, 4, 8, 16, 32]  # 30 to 960 ms


def segment_counts(source_dur, fs, quilt_dur):
    n_window_samples = round(WINDOW * fs)
    n_windows = int(source_dur * fs) // n_window_samples
    rows = []
    for windows_per_segment in SEGMENT_WINDOWS:
        n_segments = n_windows // windows_per_segment  # the toolbox drops the remainder
        needed = math.ceil(quilt_dur / (WINDOW * windows_per_segment))
        # the toolbox never picks segments 1, 2 or the last two as a later segment
        toolbox_pool = n_segments - 4
        rows.append((windows_per_segment * 30, n_segments, needed, toolbox_pool))
    return rows


print("Q1 [check] 20 s at 20 kHz, 6 s quilt: segment, segments in source, needed, toolbox pool")
for row in segment_counts(20.0, 20000, 6.0):
    print("    {:4d} ms  {:4d}  {:4d}  {:4d}".format(*row))
print("Q1 [check] 30 s source at 16 kHz, 6 s quilt")
for row in segment_counts(30.0, 16000, 6.0):
    print("    {:4d} ms  {:4d}  {:4d}  {:4d}".format(*row))

# Q2. Each step compares the current segment's last 30 ms with the first
# 30 ms of every segment, over all bands, at the full sampling rate.
fs = 20000
n_bands = 32
n_border = round(WINDOW * fs)
rng = np.random.default_rng(0)
n_segments = 666
envelopes = rng.random((n_segments * n_border, n_bands)) ** 0.3
starts = envelopes.reshape(n_segments, n_border, n_bands)  # each segment's first window
ends = starts  # 30 ms segments: the border is the whole segment
needed = math.ceil(6.0 / WINDOW)
t0 = time.perf_counter()
current = 0
used = [current]
for _ in range(needed - 1):
    distance = np.mean((ends[current][None] - starts) ** 2, axis=(1, 2))
    distance[used] = np.inf
    current = int(np.argmin(distance))
    used.append(current)
elapsed = time.perf_counter() - t0
direct_order = used
print(
    f"Q2 [check] direct: {needed - 1} steps, {n_segments} segments x {n_border} samples x {n_bands} bands:"
    f" {elapsed:.2f} s"
)

# The same distances for every pair at once: |e|^2 + |s|^2 - 2 e.s, with
# all the e.s from one matrix product.
t0 = time.perf_counter()
flat_starts = starts.reshape(n_segments, -1)
flat_ends = ends.reshape(n_segments, -1)
pairs = (
    np.sum(flat_ends**2, axis=1)[:, None]
    + np.sum(flat_starts**2, axis=1)[None, :]
    - 2 * flat_ends @ flat_starts.T
) / flat_starts.shape[1]
current = 0
used = [current]
for _ in range(needed - 1):
    distance = pairs[current].copy()
    distance[used] = np.inf
    current = int(np.argmin(distance))
    used.append(current)
elapsed = time.perf_counter() - t0
worst = max(
    np.max(np.abs(pairs[k] - np.mean((ends[k][None] - starts) ** 2, axis=(1, 2)))) for k in direct_order[:20]
)
print(
    f"Q2 [check] one matrix product for all pairs: {elapsed:.2f} s,"
    f" same order as direct: {used == direct_order},"
    f" largest difference between the two distances (20 rows): {worst:.1e}"
)
