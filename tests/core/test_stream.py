"""Streams: views computed one block at a time equal the batch analysis
exactly (docs/design/streaming.md). The STFT is the first view that streams."""

import numpy as np
import pytest
from helpers import FS, assert_streams_like_batch

import sonore as so

RNG = np.random.default_rng(0)
NOISE = so.Sound(RNG.standard_normal(int(0.1 * FS)), FS)

FRAMES = {
    "default": so.GaborFrame(20e-3),
    "odd window, zero-padded FFT": so.GaborFrame(23e-3, 7e-3, window="hamming", n_fft=2048),
    "hop equal to the window": so.GaborFrame(10e-3, 10e-3),
    "hop longer than the window": so.GaborFrame(5e-3, 8e-3),
    "two-sample window": so.GaborFrame(2 / FS, 1 / FS),
}


@pytest.mark.parametrize("frame", FRAMES.values(), ids=FRAMES.keys())
def test_stft_streams_like_batch(frame):
    assert_streams_like_batch(frame, NOISE, frame.analyze(NOISE), so.STFT.concatenate)


def test_stft_streams_stereo_and_short_sounds():
    frame = FRAMES["default"]
    stereo = so.Sound(RNG.standard_normal((4000, 2)), FS)
    assert_streams_like_batch(frame, stereo, frame.analyze(stereo), so.STFT.concatenate)
    shortest = so.Sound(RNG.standard_normal(441), FS)  # exactly half the 882-sample window
    assert_streams_like_batch(frame, shortest, frame.analyze(shortest), so.STFT.concatenate)


def test_a_sound_shorter_than_half_a_window_is_refused_as_in_batch():
    frame = FRAMES["default"]
    short = so.Sound(np.ones(440), FS)
    with pytest.raises(ValueError, match="shorter than half the window"):
        frame.analyze(short)
    with pytest.raises(ValueError, match="shorter than half the window"):
        list(so.stream(frame, so.blocks(short, 100)))


def test_parts_are_stfts_at_their_own_times():
    frame = FRAMES["default"]
    batch = frame.analyze(NOISE)
    parts = list(so.stream(frame, so.blocks(NOISE, 441)))
    assert all(isinstance(part, so.STFT) for part in parts)
    start = 0
    for part in parts:
        count = part.data.shape[-1]
        assert part.first_window == start
        assert np.array_equal(part.t, batch.t[start : start + count])
        assert np.array_equal(part.data, batch.data[..., start : start + count])
        start += count
    assert sum(1 for part in parts if part.data.shape[-1]) > 10  # time windows arrive as blocks do


def test_the_state_is_a_value():
    """A step returns a new state and leaves the old one as it was, so a
    stream can be run twice from the same state."""
    frame = FRAMES["default"]
    state = frame.start(FS)
    state, _ = frame.step(state, so.Sound(NOISE.data[:1000], FS))
    pending, next_window = state.pending.copy(), state.next_window
    block = so.Sound(NOISE.data[1000:2000], FS)
    first_state, first_part = frame.step(state, block)
    second_state, second_part = frame.step(state, block)
    assert np.array_equal(state.pending, pending) and state.next_window == next_window
    assert np.array_equal(first_part.data, second_part.data)
    assert np.array_equal(first_state.pending, second_state.pending)


def test_a_part_cannot_be_resynthesized_alone():
    frame = FRAMES["default"]
    parts = list(so.stream(frame, so.blocks(NOISE, 1000)))
    with pytest.raises(ValueError, match="STFT.concatenate first"):
        parts[1].to_sound()
    with pytest.raises(ValueError, match="STFT.concatenate first"):
        parts[1].griffin_lim(n_iter=1)
    whole = so.STFT.concatenate(parts)
    assert whole.is_whole
    assert np.allclose(whole.to_sound().data, NOISE.data)


def test_blocks_and_parts_are_checked():
    frame = FRAMES["default"]
    state = frame.start(FS)
    with pytest.raises(ValueError, match="16000 Hz"):
        frame.step(state, so.Sound(np.ones(100), 16000))
    with pytest.raises(TypeError, match="a block is a Sound"):
        frame.step(state, np.ones(100))
    state, _ = frame.step(state, so.Sound(np.ones(100), FS))
    with pytest.raises(ValueError, match="channels"):
        frame.step(state, so.Sound(np.ones((100, 2)), FS))
    parts = list(so.stream(frame, so.blocks(NOISE, 1000)))
    with pytest.raises(ValueError, match="not consecutive"):
        so.STFT.concatenate([parts[0], parts[2]])
    with pytest.raises(ValueError, match="no parts"):
        so.STFT.concatenate([])


def test_blocks_cuts_a_sound_into_consecutive_sounds():
    pieces = list(so.blocks(NOISE, 1000))
    assert all(isinstance(piece, so.Sound) and piece.fs == FS for piece in pieces)
    assert [len(piece) for piece in pieces[:-1]] == [1000] * (len(pieces) - 1)
    assert np.array_equal(np.concatenate([piece.data for piece in pieces]), NOISE.data)
    with pytest.raises(ValueError):
        list(so.blocks(NOISE, 0))
    assert list(so.stream(FRAMES["default"], [])) == []


def test_a_gabor_frame_is_streamable():
    # the Protocol so.stream takes; mypy checks the same in CI
    frame: so.Streamable = FRAMES["default"]
    assert callable(frame.step)


# The cepstrum and MFCCs are maps over a stream's STFT parts. Their floor is
# set below the whole sound's peak unless reference= fixes it, so a stream
# uses reference= (docs/design/streaming.md).
SILENT_STRETCH = NOISE.data.copy()
SILENT_STRETCH[: len(SILENT_STRETCH) // 3] *= 2.0  # the loudest stretch first
SILENT_STRETCH[len(SILENT_STRETCH) // 2 :] = 0.0  # then digital silence, where the floor binds
WITH_SILENCE = so.Sound(SILENT_STRETCH, FS)


@pytest.mark.parametrize("view", [so.Cepstrum, so.MFCC], ids=["cepstrum", "mfcc"])
def test_cepstrum_and_mfcc_of_a_stream_with_a_fixed_floor_equal_batch(view):
    frame = FRAMES["default"]
    batch = view(frame.analyze(WITH_SILENCE), reference=1.0)
    parts = [view(part, reference=1.0) for part in so.stream(frame, so.blocks(WITH_SILENCE, 441))]
    joined = view.concatenate(parts)
    assert np.array_equal(joined.data, batch.data)
    assert np.array_equal(joined.t, batch.t)


@pytest.mark.parametrize("view", [so.Cepstrum, so.MFCC], ids=["cepstrum", "mfcc"])
def test_without_a_fixed_floor_parts_differ_and_do_not_join(view):
    frame = FRAMES["default"]
    batch = view(frame.analyze(WITH_SILENCE))
    parts = [view(part) for part in so.stream(frame, so.blocks(WITH_SILENCE, 441))]
    data = np.concatenate([part.data for part in parts], axis=-1)
    assert not np.array_equal(data, batch.data)  # each part floors below its own peak
    with pytest.raises(ValueError, match="reference="):
        view.concatenate(parts)


def test_reference_none_is_todays_floor():
    coefs = FRAMES["default"].analyze(WITH_SILENCE)
    assert so.Cepstrum(coefs).reference is None and so.MFCC(coefs).reference is None
    with pytest.raises(ValueError, match="positive"):
        so.Cepstrum(coefs, reference=0.0)
