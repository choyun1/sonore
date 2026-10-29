"""The live page's Python side (docs/design/live-page.md): sonore's streams
stepped on the microphone's blocks, in the page's Pyodide worker.

Each block goes through a cochleagram (32 gammatones without edge filters,
envelopes pooled to 10 ms) and the F0 tracker, deciding 50 ms late. The
page draws what each step returns. With ``keep``, the samples are kept so
that :func:`Live.check` can compare the streamed F0s with ``f0_track`` on
the same samples, which is what the page's test does.
"""

import time

import numpy as np

import sonore as so

N_BANDS = 32
POOL = (25e-3, 10e-3)  # pooled envelope window and hop [s]: one column every 10 ms
FLOOR = 1e-5  # envelope floor before taking dB, 100 dB below a full-scale sine


class Live:
    def __init__(self, fs, keep=False):
        self.fs = float(fs)
        self.keep = keep
        self.bank = so.gammatone_filterbank(n_bands=N_BANDS, edges=False)
        self.pooling = so.EnvelopePooling(*POOL, self.bank)
        self.tracker = so.F0TrackStream()
        self.band_numbers = np.arange(N_BANDS, dtype=float)
        self.restart()
        # One second of quiet noise through the streams first, so that their
        # cached kernels and the first slow steps are done before the microphone starts.
        warm_up = np.random.default_rng(0).standard_normal(int(self.fs)) * 1e-3
        for start in range(0, len(warm_up), int(0.05 * self.fs)):
            self.step(warm_up[start : start + int(0.05 * self.fs)])
        self.restart()

    def restart(self):
        """Back to the start state, as after a gap (the page fell behind)."""
        self.states = (self.bank.start(self.fs), self.pooling.start(self.fs), self.tracker.start(self.fs))
        self.kept = []
        self.parts = []

    def band_centers(self):
        return self.bank.cfs.tolist()

    def step(self, samples):
        """One block in (a Float32Array from the page); what to draw out:
        the new cochleagram columns in dB, the newly decided F0s with their
        time windows' column numbers, and how long the step took."""
        started = time.perf_counter()
        block = so.Sound(np.asarray(samples, dtype=float), self.fs)
        bank_state, pooling_state, tracker_state = self.states
        bank_state, subbands = self.bank.step(bank_state, block)
        pooling_state, columns = self.pooling.step(pooling_state, subbands.envelopes())
        tracker_state, track = self.tracker.step(tracker_state, block)
        self.states = (bank_state, pooling_state, tracker_state)
        if self.keep:
            self.kept.append(block.data[:, 0].copy())
            self.parts.append(track)
        levels = 20 * np.log10(np.maximum(columns.data[:, :, 0], FLOOR))
        f0 = track.f0[0]
        return {
            "first_column": int(columns.first_sample),
            "levels": levels.astype(np.float32).ravel().tolist(),
            "f0_columns": (track.t / POOL[1]).tolist(),
            "f0s": f0.tolist(),
            "f0_bands": np.interp(f0, self.bank.cfs, self.band_numbers, left=np.nan, right=np.nan).tolist(),
            "seconds": time.perf_counter() - started,
        }

    def check(self):
        """The largest difference [Hz] between the streamed F0s and
        ``f0_track(..., delay=0.05)`` on the samples kept, over the time
        windows the stream decided, and how many there were."""
        sound = so.Sound(np.concatenate(self.kept), self.fs)
        streamed = so.F0Track.concatenate(self.parts)
        batch = so.f0_track(sound, hop=self.tracker.hop, delay=self.tracker.delay)
        n = streamed.f0.shape[1]
        same_times = bool(np.array_equal(streamed.t, batch.t[:n]))
        return {
            "windows": n,
            "same_times": same_times,
            "max_difference": float(np.max(np.abs(streamed.f0 - batch.f0[:, :n]))) if n else 0.0,
            "voiced": int(np.sum(streamed.f0 > 0)),
        }
