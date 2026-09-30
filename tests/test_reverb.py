"""Synthetic room impulse responses."""

import numpy as np
import pytest
from helpers import FS

import sonore as so


class TestReverb:
    def test_rt60_and_drr(self):
        ir = so.synth_ir(0.5, FS, drr_db=5, rng=0)
        direct = ir.data[0, 0] ** 2
        tail = np.sum(ir.data[1:, 0] ** 2)
        assert 10 * np.log10(direct / tail) == pytest.approx(5, abs=0.5)
        # Schroeder backward integration on the broadband tail
        e = np.cumsum(ir.data[::-1, 0] ** 2)[::-1]
        edc = 10 * np.log10(e / e[0])
        t = np.arange(len(e)) / FS
        sel = (edc < -5) & (edc > -25)
        slope = np.polyfit(t[sel], edc[sel], 1)[0]
        assert -60 / slope == pytest.approx(0.5, rel=0.25)

    def test_no_wraparound_at_the_end(self):
        # regression: circular filtering wrapped onset energy to the IR's last samples
        ir = so.synth_ir(1.0, FS, drr_db=0, rng=0).data[:, 0]
        onset = np.mean(ir[1 : int(0.05 * FS)] ** 2)
        end = np.mean(ir[-int(0.05 * FS) :] ** 2)
        assert 10 * np.log10(end / onset) < -50

    def test_binaural_tails_are_decorrelated(self):
        ir = so.synth_ir(0.5, FS, n_channels=2, rng=0)
        assert abs(np.corrcoef(ir.data.T)[0, 1]) < 0.1
