"""Streams: views computed one block of sound at a time.

A view that streams has three pure functions (docs/design/streaming.md):

    state = view.start(fs)                     # nothing seen yet
    state, part = view.step(state, block)      # one block in, the time windows it completed out
    part = view.finish(state)                  # the time windows that need the end

``state`` is a value: ``step`` never changes it, it returns a new one. Blocks
are :class:`~sonore.core.sound.Sound` s, consecutive slices of the sound with
no gaps and no overlap, of any lengths (the sound card or the reader decides).
Blocks are not windowed; the view windows internally and the state carries
any time window a block leaves unfinished. Each part is an ordinary view of
the time windows it holds, and joining the parts in order (the view's
``concatenate``) gives exactly what the batch analysis of the whole sound
gives.

:func:`stream` runs the steps over an iterable of blocks lazily, so stages
chain like a pipe; a callback that is handed blocks (a microphone) calls
``step`` itself. :class:`SlidingWindowStream` writes ``start``, ``step`` and
``finish`` once for every view whose time windows each depend on a fixed
stretch of samples, such as the STFT.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np

from sonore.core.sound import Sound

__all__ = ["Streamable", "SlidingWindowStream", "WindowState", "stream", "blocks"]


class Streamable(Protocol):
    """Anything that streams: :func:`stream` takes any object with these three
    methods. Streamed output joined in order equals the batch output, for
    any way of cutting the sound into blocks."""

    def start(self, fs: float) -> Any:
        """The state before any block has arrived, for sampling rate ``fs``."""
        ...

    def step(self, state: Any, block: Any) -> tuple[Any, Any]:
        """``(new state, the part this block completed)``; ``state`` is not changed."""
        ...

    def finish(self, state: Any) -> Any:
        """The part that needs the end of the sound, after the last block."""
        ...


def stream(view: Streamable, blocks: Iterable[Any]) -> Iterator[Any]:
    """The parts of ``view`` for ``blocks``, one per block and then one from
    :meth:`~Streamable.finish`, computed as they are read. Nothing is yielded
    when there are no blocks. The sampling rate is the first block's.

    ::

        frame = so.GaborFrame(20e-3)
        parts = list(so.stream(frame, so.blocks(snd, 441)))
        so.STFT.concatenate(parts)     # the STFT of snd, exactly
    """
    state = None
    for block in blocks:
        if state is None:
            state = view.start(block.fs)
        state, part = view.step(state, block)
        yield part
    if state is not None:
        yield view.finish(state)


def blocks(sound: Sound, size: int) -> Iterator[Sound]:
    """``sound`` cut into consecutive blocks of ``size`` samples (the last one
    shorter if ``size`` does not divide it), each a :class:`Sound`."""
    size = int(size)
    if size < 1:
        raise ValueError(f"size must be at least one sample, not {size}")
    data = sound.data
    for start in range(0, len(data), size):
        yield Sound(data[start : start + size], sound.fs)


@dataclass(frozen=True)
class WindowState:
    """The state of a :class:`SlidingWindowStream`: the samples from the start
    of the next unfinished time window on, that time window's index, and how
    many samples have arrived. Before the first block ``pending`` holds the
    zeros batch puts before sample 0, with no channels yet. When the hop is
    longer than the window, ``skip`` counts the coming samples that fall
    between time windows."""

    fs: float
    pending: np.ndarray  # (n_samples, n_channels), or (n_samples, n_bands, n_channels) for envelopes
    next_window: int
    n_seen: int
    skip: int = 0


class SlidingWindowStream:
    """``start``, ``step`` and ``finish`` for a view whose time window ``i``
    covers ``window_length`` samples starting at ``first_start + i * hop`` and
    depends on nothing else. A subclass says how its time windows are laid
    out (:meth:`window_layout`), how many batch computes for a sound
    (:meth:`window_count`), and what a run of complete time windows becomes
    (:meth:`time_windows`, the same code its batch analysis runs). Samples
    before the start and after the end are zeros, as in batch.

    Blocks are Sounds unless the subclass names another :attr:`block_type`
    with ``fs``, ``n_channels`` and ``data`` of shape ``(n_samples, ...)``,
    such as a :class:`~sonore.views.temporal_envelope.TemporalEnvelope`."""

    block_type: ClassVar[type[Any]] = Sound

    @abstractmethod
    def window_layout(self, fs: float) -> tuple[int, int, int]:
        """``(window_length, hop, first_start)`` in samples at ``fs``;
        ``first_start`` is where time window 0 starts, negative when it reaches
        before sample 0."""

    @abstractmethod
    def window_count(self, n_samples: int, fs: float) -> int:
        """How many time windows the batch analysis of ``n_samples`` samples has."""

    @abstractmethod
    def time_windows(self, samples: np.ndarray, fs: float, first: int, count: int, n_samples: int) -> Any:
        """The part holding time windows ``first`` to ``first + count - 1``,
        from ``samples`` (``(n, n_channels)``, starting at time window ``first``'s
        first sample), after ``n_samples`` samples of the sound have arrived."""

    def start(self, fs: float) -> WindowState:
        """Nothing seen yet: the zeros before sample 0 that time window 0 covers."""
        _, _, first_start = self.window_layout(fs)
        return WindowState(float(fs), np.zeros((max(-first_start, 0), 0)), 0, 0)

    def step(self, state: WindowState, block: Any) -> tuple[WindowState, Any]:
        """Append ``block`` (a :attr:`block_type`) and return every time window now complete."""
        if not isinstance(block, self.block_type):
            raise TypeError(f"a block is a {self.block_type.__name__}, not {type(block).__name__}")
        if block.fs != state.fs:
            raise ValueError(f"this stream runs at {state.fs:g} Hz; the block is at {block.fs:g} Hz")
        sample_shape = block.data.shape[1:]
        if len(block) == 0 and _no_channels_yet(state.pending):  # an empty block does not fix the channels
            return state, self.time_windows(np.zeros((0, *sample_shape)), state.fs, 0, 0, 0)
        pending = self._with_channels(state.pending, sample_shape)
        window_length, hop, _ = self.window_layout(state.fs)
        pending = np.concatenate([pending, block.data[state.skip :]])
        skip = max(state.skip - len(block), 0)
        count = max(0, (len(pending) - window_length) // hop + 1)
        n_seen = state.n_seen + len(block)
        part = self.time_windows(pending, state.fs, state.next_window, count, n_seen)
        skip += max(count * hop - len(pending), 0)
        new_state = WindowState(state.fs, pending[count * hop :], state.next_window + count, n_seen, skip)
        return new_state, part

    def finish(self, state: WindowState) -> Any:
        """The remaining time windows, with zeros after the last sample."""
        window_length, hop, _ = self.window_layout(state.fs)
        count = max(self.window_count(state.n_seen, state.fs) - state.next_window, 0)
        pending = (
            self._with_channels(state.pending, (1,)) if _no_channels_yet(state.pending) else state.pending
        )
        needed = (count - 1) * hop + window_length if count else 0
        if len(pending) < needed:
            pending = np.concatenate([pending, np.zeros((needed - len(pending), *pending.shape[1:]))])
        return self.time_windows(pending, state.fs, state.next_window, count, state.n_seen)

    @staticmethod
    def _with_channels(pending: np.ndarray, sample_shape: tuple[int, ...]) -> np.ndarray:
        """``pending``, given its channels (and bands) when the first block brings them."""
        if _no_channels_yet(pending):
            return np.zeros((len(pending), *sample_shape))
        if pending.shape[-1] != sample_shape[-1]:
            raise ValueError(
                f"this stream has {pending.shape[-1]} channels; the block has {sample_shape[-1]}"
            )
        if pending.shape[1:] != sample_shape:
            raise ValueError(
                f"this stream's samples have shape {pending.shape[1:]}; the block's {sample_shape}"
            )
        return pending


def _no_channels_yet(pending: np.ndarray) -> bool:
    """Whether ``pending`` is still the zeros of :meth:`SlidingWindowStream.start`, before any block."""
    return pending.shape[1:] == (0,)
