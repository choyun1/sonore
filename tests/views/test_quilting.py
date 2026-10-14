"""Sound quilts."""

from itertools import pairwise

import numpy as np
import pytest

import sonore as so

FS = 16000


def speechlike(seconds, rng):
    """Noise with a slow, random envelope, so segments differ in level."""
    noise = so.gaussian_noise(seconds, FS, rng=rng)
    envelope = np.repeat(np.random.default_rng(rng).random(int(seconds * 20)) + 0.1, FS // 20)
    return so.Sound(noise.data[:, 0] * envelope[: len(noise)], FS)


class TestQuilt:
    source = speechlike(3.0, 0)

    def test_length_and_determinism(self):
        a = so.quilt(self.source, 0.06, 1.0, rng=1)
        b = so.quilt(self.source, 0.06, 1.0, rng=1)
        assert len(a) == FS and a.fs == FS
        np.testing.assert_array_equal(a.data, b.data)
        assert not np.array_equal(a.data, so.quilt(self.source, 0.06, 1.0, rng=2).data)

    def test_order_follows_the_rule(self):
        # without cross-fades or shifts each quilt segment is a copy of a source segment
        n = int(0.06 * FS)
        n_border = int(0.03 * FS)
        q = so.quilt(self.source, 0.06, 1.2, rng=3, crossfade=0, max_shift=0)
        segments = self.source.data[: len(self.source) // n * n, 0].reshape(-1, n)
        pieces = q.data[:, 0].reshape(-1, n)
        order = [int(np.flatnonzero((segments == piece).all(axis=1))[0]) for piece in pieces]
        last = len(segments) - 1
        assert len(set(order)) == len(order)
        assert all(b != a + 1 for a, b in pairwise(order))
        assert last not in order[:-1]
        # each next segment matches the source's own change best, by a direct
        # (not matrix-product) border distance
        bank = so.cosine_filterbank(30, 20, 8000)
        env = bank.analyze(self.source).envelopes().data[:, :, 0] ** 0.3

        def change(k, m):
            return np.mean((env[(k + 1) * n - n_border : (k + 1) * n] - env[m * n : m * n + n_border]) ** 2)

        for position, (k, chosen) in enumerate(pairwise(order), start=1):
            final = position == len(order) - 1
            used = order[:position]
            free = [m for m in range(last + 1) if m not in used and m != k + 1 and (m != last or final)]
            mismatch = {m: abs(change(k, m) - change(k, k + 1)) for m in free}
            assert mismatch[chosen] <= min(mismatch.values()) + 1e-12

    def test_keeps_the_level(self):
        q = so.quilt(self.source, 0.12, 2.0, rng=0)
        assert 0.7 < q.rms / self.source.rms < 1.4

    def test_too_short_a_sound_is_refused(self):
        with pytest.raises(ValueError, match="needs 34 segments, but the sound"):
            so.quilt(self.source, 0.12, 4.0, rng=0)

    def test_mono_only(self):
        with pytest.raises(ValueError, match="mono"):
            so.quilt(self.source.to_channels(2), 0.12, 1.0)
