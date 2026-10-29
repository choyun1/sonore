"""A running F0 track from the microphone, printed as a bar every 50 ms.

Needs a sound card and the live extra:

    pip install "sonore[live]"
    python docs/examples/live_f0.py            # Ctrl+C stops

Each line is one time window, 50 ms before it is printed (the tracker
decides a time window once 50 ms more sound has arrived): its time, its F0
or a dash where it is unvoiced, and a bar on a log scale from 60 to 500 Hz.
At the end it prints how long each 10 ms block took to analyze; under 10 ms
keeps up with the microphone.
"""

import time

import numpy as np

import sonore as so

FS = 16000
BLOCK = 160  # 10 ms
F_LO, F_HI = 60.0, 500.0
WIDTH = 50

tracker = so.F0TrackStream(f_lo=F_LO, f_hi=F_HI)
state = tracker.start(FS)
step_times = []
print("Sing or speak. Ctrl+C stops.")
try:
    for block in so.live_blocks(FS, BLOCK):
        started = time.perf_counter()
        state, part = tracker.step(state, block)
        step_times.append(time.perf_counter() - started)
        for t, f0 in zip(part.t, part.f0[0], strict=True):
            if round(t / tracker.hop) % 10:
                continue  # one line every 50 ms
            if f0 > 0:
                filled = int(round(WIDTH * np.log(f0 / F_LO) / np.log(F_HI / F_LO)))
                print(f"{t:7.2f} s  {f0:6.1f} Hz  {'#' * filled}")
            else:
                print(f"{t:7.2f} s       -")
except KeyboardInterrupt:
    pass
if step_times:
    times_ms = 1e3 * np.array(step_times)
    print(
        f"\n{len(times_ms)} blocks of 10 ms; analysis took {np.median(times_ms):.1f} ms per block "
        f"(median), {np.percentile(times_ms, 99):.1f} ms (99th percentile)"
    )
