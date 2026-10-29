"""How long each stream takes per 10 ms of sound, for a live page.

Runs the streams over docs/speech/bdl_arctic_a0131.flac (2.5 s at 16 kHz)
in blocks of 10 ms and 50 ms and prints the best of three runs, in ms per
10 ms of sound: under 10 keeps up with a microphone. The same file runs
under Pyodide in a browser (tools/pyodide_timing/, which says how); the
numbers in docs/design/live-page.md come from both.

    python tools/time_streams.py
"""

import sys
import time

import sonore as so

BLOCKS = (160, 800)  # 10 ms and 50 ms at 16 kHz


def pipelines(fs):
    """Name -> a function of the blocks that runs the stream to its end."""

    def cochleagram(n_bands):
        bank = so.gammatone_filterbank(n_bands=n_bands, edges=False)
        pooling = so.EnvelopePooling(25e-3, 10e-3, bank)
        return lambda blocks: so.stream(pooling, (part.envelopes() for part in so.stream(bank, blocks)))

    def modulation(blocks):
        bank = so.gammatone_filterbank(n_bands=32, edges=False)
        pooling = so.EnvelopePooling(2e-3, 1e-3, bank)  # envelopes at 1 kHz
        envelopes = so.stream(pooling, (part.envelopes() for part in so.stream(bank, blocks)))
        return so.stream(so.ModulationSpectrogramStream(bank), envelopes)

    def cochleagram_and_f0(blocks):
        bank = so.gammatone_filterbank(n_bands=32, edges=False)
        pooling = so.EnvelopePooling(25e-3, 10e-3, bank)
        tracker = so.F0TrackStream()
        bank_state, pooling_state, tracker_state = bank.start(fs), pooling.start(fs), tracker.start(fs)
        for block in blocks:
            bank_state, subbands = bank.step(bank_state, block)
            pooling_state, _ = pooling.step(pooling_state, subbands.envelopes())
            tracker_state, _ = tracker.step(tracker_state, block)
        return ()

    frame = so.GaborFrame(25e-3, 10e-3)
    return {
        "STFT, 25 ms windows every 10 ms": lambda blocks: so.stream(frame, blocks),
        "cochleagram, 32 gammatones pooled to 10 ms": cochleagram(32),
        "cochleagram, 64 gammatones pooled to 10 ms": cochleagram(64),
        "modulation spectrogram, 32 bands": modulation,
        "F0 track, decided 50 ms late": lambda blocks: so.stream(so.F0TrackStream(), blocks),
        "cochleagram (32) and F0 track together": cochleagram_and_f0,
    }


def run(path="docs/speech/bdl_arctic_a0131.flac", repeats=3):
    """``{(name, block size): ms per 10 ms of sound}``."""
    sound = so.load(path)
    tens_of_ms = sound.duration / 0.01
    results = {}
    for name, pipeline in pipelines(sound.fs).items():
        for size in BLOCKS:
            best = float("inf")
            for _ in range(repeats):
                started = time.perf_counter()
                for _part in pipeline(so.blocks(sound, size)):
                    pass
                best = min(best, time.perf_counter() - started)
            results[name, size] = 1e3 * best / tens_of_ms
    return results


def report(results):
    lines = [f"{'':45s}{'10 ms blocks':>14s}{'50 ms blocks':>14s}"]
    for name in dict.fromkeys(name for name, _ in results):
        lines.append(f"{name:45s}" + "".join(f"{results[name, size]:11.1f} ms" for size in BLOCKS))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(run(*sys.argv[1:2])))
