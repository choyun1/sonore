"""Live input: blocks of sound from a microphone or sound card, for streams.

:func:`live_blocks` is :func:`~sonore.core.stream.blocks` for a sound that is
still arriving. It yields each buffer the sound card hands over as a
:class:`~sonore.core.sound.Sound`, so any view that streams reads it::

    tracker = so.F0TrackStream()
    for part in so.stream(tracker, so.live_blocks(16000, 160)):
        print(part.t, part.f0[0])

It needs python-sounddevice and PortAudio (``pip install 'sonore[live]'``),
which this module imports only when it is called. A stream equals the batch
analysis of the samples it was given only if none were lost, so a lost sample
is an error here, not a gap: the blocks before it are yielded, then
:class:`DroppedSamplesError` says how many samples arrived intact.
"""

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Iterator

import numpy as np

from sonore.core.sound import Sound

__all__ = ["live_blocks", "DroppedSamplesError"]


BACKLOG = 2.0  # seconds of blocks held while the reader is busy; more than that is dropped


class DroppedSamplesError(RuntimeError):
    """Samples were lost between the sound card and the reader: the card
    overflowed, or the reader fell more than BACKLOG seconds behind.
    ``n_intact`` samples per channel had been yielded before the loss."""

    def __init__(self, message: str, n_intact: int):
        super().__init__(message)
        self.n_intact = n_intact


def live_blocks(
    fs: float,
    size: int,
    *,
    channels: int = 1,
    device: int | str | None = None,
    timeout: float = 2.0,
) -> Iterator[Sound]:
    """Blocks of ``size`` samples from an input device at ``fs`` Hz, each a
    :class:`Sound` of ``channels`` channels, for as long as they are read.

    Recording starts when the first block is asked for and stops when the
    reader stops (the end of a ``for`` loop, ``break``, or closing the
    generator); ``itertools.islice`` takes a fixed number of blocks.
    ``device`` is a sounddevice device number or name (``None``, the
    default input). A ``size`` of 10 ms or more leaves a slow view time to
    keep up; the blocks are always exactly ``size`` samples, so ``size``
    also sets how often a stream steps.

    Raises :class:`DroppedSamplesError` once the blocks before a loss have
    been read: a stream of what follows would no longer be the analysis of
    one continuous sound. Raises ``TimeoutError`` if no block arrives for
    ``timeout`` seconds.
    """
    fs = float(fs)
    size = int(size)
    if fs <= 0:
        raise ValueError(f"fs must be positive, not {fs:g}")
    if size < 1:
        raise ValueError(f"size must be at least one sample, not {size}")
    if channels < 1:
        raise ValueError(f"channels must be at least 1, not {channels}")
    return _read(fs, size, channels, device, timeout)


def _read(fs, size, channels, device, timeout):
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "live input needs sounddevice and PortAudio: pip install 'sonore[live]'"
        ) from error
    buffers: queue.Queue = queue.Queue(maxsize=max(1, math.ceil(BACKLOG * fs / size)))
    loss = _Loss()

    def on_buffer(indata, frames, time, status):  # runs on PortAudio's thread
        if loss.reason is not None:
            return
        if status.input_overflow:
            loss.mark("the sound card dropped samples (input overflow)")
            try:
                buffers.put_nowait(None)  # wakes the reader
            except queue.Full:
                pass  # the reader finds the loss once it has emptied the queue
            return
        try:
            buffers.put_nowait(np.array(indata, dtype=float))
        except queue.Full:
            loss.mark(f"the reader fell more than {BACKLOG:g} s behind the sound card")

    n_intact = 0
    with sd.InputStream(
        samplerate=fs, blocksize=size, channels=channels, device=device, dtype="float32", callback=on_buffer
    ):
        while True:
            if loss.reason is not None and buffers.empty():
                raise DroppedSamplesError(
                    f"{loss.reason}; {n_intact} samples per channel arrived intact", n_intact
                )
            try:
                data = buffers.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(f"no audio arrived for {timeout:g} s") from None
            if data is None:
                continue  # the loss is reported above
            n_intact += len(data)
            yield Sound(data, fs)


class _Loss:
    """The first reason samples were lost, set from the audio thread."""

    def __init__(self):
        self.reason: str | None = None
        self._lock = threading.Lock()

    def mark(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
