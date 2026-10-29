"""Live input, with a stand-in for sounddevice: the cloud and CI have no
sound card, so these tests hand the reader buffers the way PortAudio does."""

import itertools
import sys
import types

import numpy as np
import pytest

import sonore as so
from sonore.core import live

FS = 16000.0
RNG = np.random.default_rng(3)
SOURCE = RNG.standard_normal((int(0.5 * FS), 2)).astype(np.float32) * 0.1


class FakeInputStream:
    """Hands the callback ``SOURCE`` in blocks as soon as it is entered, then
    nothing; ``overflow_at`` flags that block as an input overflow."""

    opened: list = []

    def __init__(
        self, samplerate, blocksize, channels, device, dtype, callback, source=SOURCE, overflow_at=None
    ):
        self.settings = dict(
            samplerate=samplerate, blocksize=blocksize, channels=channels, device=device, dtype=dtype
        )
        self.callback, self.source, self.overflow_at = callback, source[:, :channels], overflow_at
        self.closed = False
        FakeInputStream.opened.append(self)

    def __enter__(self):
        size = self.settings["blocksize"]
        for index, start in enumerate(range(0, len(self.source) - size + 1, size)):
            status = types.SimpleNamespace(input_overflow=index == self.overflow_at)
            self.callback(self.source[start : start + size].copy(), size, None, status)
        return self

    def __exit__(self, *exc):
        self.closed = True


def fake_sounddevice(monkeypatch, **stream_options):
    FakeInputStream.opened = []
    module = types.SimpleNamespace(InputStream=lambda **kw: FakeInputStream(**kw, **stream_options))
    monkeypatch.setitem(sys.modules, "sounddevice", module)


@pytest.fixture
def roomy(monkeypatch):
    """A backlog long enough to hold all of SOURCE."""
    monkeypatch.setattr(live, "BACKLOG", 10.0)


def test_live_blocks_are_the_buffers_as_sounds(monkeypatch, roomy):
    fake_sounddevice(monkeypatch)
    received = list(itertools.islice(so.live_blocks(FS, 160, channels=2, device=3), 50))
    assert FakeInputStream.opened[0].settings == dict(
        samplerate=FS, blocksize=160, channels=2, device=3, dtype="float32"
    )
    assert all(
        isinstance(block, so.Sound) and block.fs == FS and block.n_samples == 160 for block in received
    )
    joined = np.concatenate([block.data for block in received])
    assert joined.dtype == np.float64 and np.array_equal(joined, SOURCE[: 50 * 160])


def test_a_stream_of_live_blocks_is_the_batch_analysis(monkeypatch, roomy):
    fake_sounddevice(monkeypatch)
    tracker = so.F0TrackStream()
    blocks = itertools.islice(so.live_blocks(FS, 160), 40)
    streamed = so.F0Track.concatenate(list(so.stream(tracker, blocks)))
    batch = so.f0_track(so.Sound(SOURCE[: 40 * 160, :1], FS), delay=tracker.delay)
    assert np.array_equal(streamed.f0, batch.f0) and np.array_equal(streamed.t, batch.t)


def test_recording_stops_when_the_reader_does(monkeypatch, roomy):
    fake_sounddevice(monkeypatch)
    reader = so.live_blocks(FS, 160)
    next(reader)
    assert not FakeInputStream.opened[0].closed
    reader.close()
    assert FakeInputStream.opened[0].closed


def test_an_overflow_ends_the_blocks_after_the_last_intact_one(monkeypatch, roomy):
    fake_sounddevice(monkeypatch, overflow_at=7)
    reader = so.live_blocks(FS, 160)
    assert len([next(reader) for _ in range(7)]) == 7
    with pytest.raises(so.DroppedSamplesError, match=r"input overflow\); 1120 samples") as caught:
        next(reader)
    assert caught.value.n_intact == 7 * 160
    assert FakeInputStream.opened[0].closed


def test_falling_behind_is_a_loss(monkeypatch):
    monkeypatch.setattr(live, "BACKLOG", 0.05)  # 5 blocks of 160 at 16 kHz
    fake_sounddevice(monkeypatch)
    reader = so.live_blocks(FS, 160)
    assert len([next(reader) for _ in range(5)]) == 5
    with pytest.raises(so.DroppedSamplesError, match="fell more than 0.05 s behind"):
        next(reader)


def test_no_audio_is_a_timeout(monkeypatch, roomy):
    fake_sounddevice(monkeypatch, source=SOURCE[:0])
    with pytest.raises(TimeoutError, match="no audio arrived for 0.05 s"):
        next(so.live_blocks(FS, 160, timeout=0.05))


def test_live_input_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match=r"sonore\[live\]"):
        next(so.live_blocks(FS, 160))


@pytest.mark.parametrize("fs, size, channels", [(0, 160, 1), (FS, 0, 1), (FS, 160, 0)])
def test_live_blocks_check_their_arguments(fs, size, channels):
    with pytest.raises(ValueError):
        so.live_blocks(fs, size, channels=channels)
