"""The time-domain gammatone bank and pooled envelopes stream: joined parts
equal the batch analysis exactly (docs/design/streaming.md)."""

import numpy as np
import pytest
from helpers import FS, assert_streams_like_batch

import sonore as so

RNG = np.random.default_rng(1)
NOISE = so.Sound(RNG.standard_normal((int(0.1 * FS), 2)), FS)
BANK = so.gammatone_filterbank(n_bands=6, edges=False)


def test_time_domain_subbands_stream_like_batch():
    batch = BANK.analyze(NOISE, domain="time")
    assert_streams_like_batch(BANK, NOISE, batch, so.Subbands.concatenate)
    parts = list(so.stream(BANK, so.blocks(NOISE, 1000)))
    joined = so.Subbands.concatenate(parts)
    assert np.array_equal(joined.envelopes().data, batch.envelopes().data)
    assert [part.first_sample for part in parts[:3]] == [0, 1000, 2000]


def test_time_domain_subbands_are_close_to_the_frequency_domain():
    time_domain = BANK.analyze(NOISE, domain="time").data
    frequency_domain = BANK.analyze(NOISE).data
    error = np.sqrt(np.mean((time_domain - frequency_domain) ** 2) / np.mean(frequency_domain**2))
    assert error < 2e-5


def test_envelope_times_fine_structure_is_the_bands():
    subbands = BANK.analyze(NOISE, domain="time")
    rebuilt = subbands.envelopes() * subbands.tfs()
    assert rebuilt.domain == "time"
    assert np.allclose(rebuilt.data, subbands.data)
    assert (subbands * 2.0).domain == "time"


def test_time_domain_subbands_are_not_resynthesized():
    subbands = BANK.analyze(NOISE, domain="time")
    for back in (subbands.to_sound, lambda: BANK.adjoint(subbands), lambda: (subbands * 0.5).to_sound()):
        with pytest.raises(ValueError, match="not resynthesized"):
            back()


def test_only_causal_gammatones_without_edges_run_in_the_time_domain():
    for bank in (
        so.gammatone_filterbank(n_bands=6),
        so.gammatone_filterbank(n_bands=6, edges=False, phase="zero"),
        so.cosine_filterbank(n_bands=6, edges=False),
    ):
        with pytest.raises(ValueError, match="time domain"):
            bank.analyze(NOISE, domain="time")
    with pytest.raises(ValueError, match="domain"):
        BANK.analyze(NOISE, domain="space")


def test_subband_parts_are_checked_when_joined():
    parts = list(so.stream(BANK, so.blocks(NOISE, 1000)))
    with pytest.raises(ValueError, match="where the one before it ended"):
        so.Subbands.concatenate([parts[0], parts[2]])
    with pytest.raises(ValueError, match="time-domain filters"):
        so.Subbands.concatenate([BANK.analyze(NOISE, pad=0)])
    state, _ = BANK.step(BANK.start(FS), so.Sound(np.ones((100, 2)), FS))
    with pytest.raises(ValueError, match="channels"):
        BANK.step(state, so.Sound(np.ones(100), FS))


class Cochleagram:
    """The bank's envelopes pooled, as one stream of two stages."""

    def __init__(self, bank, pooling):
        self.bank, self.pooling = bank, pooling

    def start(self, fs):
        return self.bank.start(fs), self.pooling.start(fs)

    def step(self, state, block):
        bank_state, pooling_state = state
        bank_state, subbands = self.bank.step(bank_state, block)
        pooling_state, part = self.pooling.step(pooling_state, subbands.envelopes())
        return (bank_state, pooling_state), part

    def finish(self, state):
        return self.pooling.finish(state[1])


POOLINGS = {
    "25 ms every 10 ms": (25e-3, 10e-3),
    "odd window": (5.01e-3, 2e-3),
    "hop longer than the window": (2e-3, 3e-3),
}


@pytest.mark.parametrize("durations", POOLINGS.values(), ids=POOLINGS.keys())
def test_pooled_envelopes_stream_like_batch(durations):
    pooling = so.EnvelopePooling(*durations, BANK)
    batch = BANK.analyze(NOISE, domain="time").envelopes().pooled(*durations)
    assert_streams_like_batch(Cochleagram(BANK, pooling), NOISE, batch, so.TemporalEnvelope.concatenate)


def test_pooling_is_hann_weighted_power():
    envelope = so.TemporalEnvelope(RNG.random(1000), 1000.0)
    pooled = envelope.pooled(10e-3, 4e-3)
    weights = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(10) / 10)
    padded = np.concatenate([np.zeros(5), envelope.data[:, 0], np.zeros(10)])
    expected = [
        np.sqrt(np.sum(weights * padded[k * 4 : k * 4 + 10] ** 2) / weights.sum()) for k in range(250)
    ]
    assert pooled.fs == 250.0 and pooled.data.shape == (250, 1)
    assert np.allclose(pooled.data[:, 0], expected)
    assert np.allclose(so.TemporalEnvelope(np.ones(1000), 1000.0).pooled(10e-3, 4e-3).data[2:-3], 1.0)


def test_pooling_checks_its_blocks():
    pooling = so.EnvelopePooling(25e-3, 10e-3, BANK)
    state = pooling.start(FS)
    with pytest.raises(TypeError, match="TemporalEnvelope"):
        pooling.step(state, NOISE)
    with pytest.raises(ValueError, match="different filterbank"):
        pooling.step(state, NOISE.envelope())
    with pytest.raises(ValueError, match="padding"):
        BANK.analyze(NOISE).envelopes().pooled()
    with pytest.raises(ValueError, match="positive"):
        so.EnvelopePooling(0.0)
    later_part = list(so.stream(BANK, so.blocks(NOISE, 1000)))[1].envelopes()
    with pytest.raises(ValueError, match="part of a stream"):
        later_part.resample(50.0)
    with pytest.raises(ValueError, match="part of a stream"):
        later_part.pooled()
