"""The dB unit: level arithmetic on Sounds."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestDecibels:
    x = so.pure_tone(0.1, FS, 440)

    def level_change(self, y):
        return float(so.amp_to_db(y.rms / self.x.rms))

    @pytest.mark.parametrize(
        "expr, expected",
        [
            (lambda x, dB: x + 6 * dB, 6),
            (lambda x, dB: x - 3 * dB, -3),
            (lambda x, dB: 6 * dB + x, 6),
            (lambda x, dB: x + np.float64(2.5) * dB, 2.5),
            (lambda x, dB: x + (6 * dB + 3 * dB), 9),
            (lambda x, dB: x + -(4 * dB), -4),
            (lambda x, dB: x + (12 * dB) / 2, 6),
        ],
    )
    def test_level_arithmetic(self, expr, expected):
        assert self.level_change(expr(self.x, so.dB)) == pytest.approx(expected)

    def test_matches_gain_db(self):
        np.testing.assert_allclose((self.x + 6 * so.dB).data, self.x.gain_db(6).data)
        np.testing.assert_allclose(self.x.gain_db(6 * so.dB).data, self.x.gain_db(6).data)

    def test_mixing_at_snr(self):
        target, masker = so.pure_tone(1, FS, 500), so.gaussian_noise(1, FS, rng=0)
        mix = target + (masker - 10 * so.dB)
        noise = mix - target
        assert so.amp_to_db(target.rms / noise.rms) == pytest.approx(10)

    def test_traps_raise(self):
        with pytest.raises(TypeError, match="ambiguous"):
            self.x * (6 * so.dB)
        with pytest.raises(TypeError):
            3 * so.dB - self.x

    def test_number_comes_first(self):
        with pytest.raises(TypeError, match=r"did you mean 6\*dB\?"):
            so.dB * 6
        with pytest.raises(TypeError, match=r"did you mean 2\*\(6\*dB\)\?"):
            (6 * so.dB) * 2
        assert 2 * (6 * so.dB) == 12 * so.dB

    def test_repr_and_value(self):
        assert repr(6 * so.dB) == "6 dB"
        assert (6 * so.dB).gain == pytest.approx(10**0.3)
        assert 3 * so.dB < 6 * so.dB
        assert 3 * so.dB <= 3 * so.dB and not 6 * so.dB <= 3 * so.dB
        assert 6 * so.dB - 3 * so.dB == 3 * so.dB

    def test_not_a_plain_number(self):
        with pytest.raises(TypeError, match=r"\(6\*dB\)\.value"):
            float(6 * so.dB)
        assert (6 * so.dB).value == 6
        with pytest.raises(TypeError):
            True * so.dB
