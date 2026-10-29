"""Numerical checks for the claims in docs/design/streaming.md (C1-C11).

The streaming versions are written out here from their definitions, with
NumPy and SciPy only, as the step functions the design proposes: a step takes
the state and a block of samples and returns the new state and whatever time
windows the block completed. sonore is imported only for the batch results
they are compared with. Each line prints the claim number and the number that
supports it.

    python tools/check_streaming_claims.py
"""

import time
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import scipy.fft as sp_fft
import soundfile as sf
from scipy.signal import freqz, hilbert, lfilter

import sonore as so
from sonore.core.utils import erb_bandwidth

FS = 44100.0
DURATION = 3.0
RNG = np.random.default_rng(0)
NOISE = RNG.standard_normal(int(FS * DURATION))
SPEECH = Path(__file__).resolve().parent.parent / "docs" / "speech"


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


def random_blocks(x, rng, largest):
    """``x`` cut into blocks of random length between 1 and ``largest``."""
    cuts = [0]
    while cuts[-1] < len(x):
        cuts.append(min(len(x), cuts[-1] + int(rng.integers(1, largest + 1))))
    return [x[a:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def scan(step, state, blocks):
    """Run ``step`` over ``blocks``: the fold that the design calls a scan.
    Returns the final state and every block's output, in order."""
    outputs = []
    for block in blocks:
        state, out = step(state, block)
        outputs.append(out)
    return state, outputs


# ------------------------------------------------------------------- the STFT
@dataclass(frozen=True)
class STFTState:
    pending: np.ndarray  # samples from the start of the next time window on
    next_window: int  # index q of the next time window, centered at q * hop


def stft_setup(frame, fs):
    sft = frame.sft(fs)
    window = sft.win
    return sft, window, sft.m_num, sft.m_num_mid, sft.hop, sft.mfft


def stft_window(segment, window, m_num_mid, n_fft):
    """One time window, as SciPy's ShortTimeFFT computes it with its default
    phase: the window's middle sample is time 0 of the FFT."""
    padded = np.zeros(n_fft)
    padded[: len(window)] = segment * window
    return sp_fft.rfft(np.roll(padded, -m_num_mid))


def stft_start(frame, fs):
    sft, window, m_num, m_num_mid, hop, n_fft = stft_setup(frame, fs)
    first = sft.p_min  # the first time window reaches back before sample 0
    start = first * hop - m_num_mid
    return STFTState(np.zeros(-start), first)


def make_stft_step(frame, fs):
    sft, window, m_num, m_num_mid, hop, n_fft = stft_setup(frame, fs)

    def step(state, block):
        pending = np.concatenate([state.pending, block])
        columns = []
        q = state.next_window
        offset = 0  # where window q starts in pending
        while offset + m_num <= len(pending):
            columns.append(stft_window(pending[offset : offset + m_num], window, m_num_mid, n_fft))
            q += 1
            offset += hop
        out = np.array(columns).T if columns else np.zeros((n_fft // 2 + 1, 0), complex)
        return STFTState(pending[offset:], q), out

    def flush(state, n_samples):
        """The end of the stream: the time windows that reach past the last
        sample, which SciPy computes with zeros there."""
        last = sft.p_max(n_samples)
        missing = (last - state.next_window) * hop + m_num
        _, out = step(state, np.zeros(max(missing - len(state.pending), 0)))
        return out[:, : last - state.next_window]

    return step, flush


def check_stft():
    frame = so.GaborFrame(20e-3)  # sonore's default: 20 ms Hann, 5 ms hop
    batch = so.STFT(so.Sound(NOISE, FS), frame=frame).data[0]
    peak = np.abs(batch).max()
    step, flush = make_stft_step(frame, FS)
    worst = 0.0
    for largest in (1, 64, 441, 4410):
        blocks = random_blocks(NOISE, np.random.default_rng(largest), largest)
        state, outputs = scan(step, stft_start(frame, FS), blocks)
        streamed = np.concatenate(outputs + [flush(state, len(NOISE))], axis=1)
        assert streamed.shape == batch.shape, (streamed.shape, batch.shape)
        worst = max(worst, np.abs(streamed - batch).max() / peak)
    report("C1", "STFT, streamed in random blocks of 1..4410 samples: worst difference / peak", worst)
    m_num, m_num_mid = frame.sft(FS).m_num, frame.sft(FS).m_num_mid
    report(
        "C1",
        "STFT latency: samples after a window's center before it can be computed [ms]",
        (m_num - m_num_mid) / FS * 1e3,
    )
    return frame, batch


def check_cepstrum_mfcc(frame):
    """Cepstrum and MFCC are computed from each time window alone, so the
    streamed STFT's time windows give them exactly; checked on halves."""
    snd = so.Sound(NOISE, FS)
    coefs = so.STFT(snd, frame=frame)
    whole = so.Cepstrum(coefs).data
    first = coefs._with_data(coefs.data[..., :300])
    second = coefs._with_data(coefs.data[..., 300:])
    halves = np.concatenate([so.Cepstrum(first).data, so.Cepstrum(second).data], axis=-1)
    report(
        "C2",
        "Cepstrum of two halves of the time windows vs of all: worst difference",
        np.abs(halves - whole).max(),
    )
    whole_mfcc = so.MFCC(coefs).data
    halves_mfcc = np.concatenate([so.MFCC(first).data, so.MFCC(second).data], axis=-1)
    report(
        "C2",
        "MFCC of two halves of the time windows vs of all: worst difference",
        np.abs(halves_mfcc - whole_mfcc).max(),
    )
    # Both floor the magnitude a fixed number of dB below each channel's
    # largest value over the whole sound, which is not local: on a sound with
    # silence, or one whose level changes, the halves differ from the whole.
    signal = NOISE.copy()
    signal[: int(1.0 * FS)] *= 2.0  # the loudest stretch in the first part
    signal[int(2.0 * FS) : int(2.5 * FS)] = 0.0  # half a second of digital silence, in the second part
    coefs = so.STFT(so.Sound(signal, FS), frame=frame)
    first = coefs._with_data(coefs.data[..., :300])
    second = coefs._with_data(coefs.data[..., 300:])
    halves = np.concatenate([so.Cepstrum(first).data, so.Cepstrum(second).data], axis=-1)
    halves_mfcc = np.concatenate([so.MFCC(first).data, so.MFCC(second).data], axis=-1)
    report(
        "C2",
        "The same with 0.5 s of digital silence, cepstrum: worst difference",
        np.abs(halves - so.Cepstrum(coefs).data).max(),
    )
    report(
        "C2",
        "                                          MFCC: worst difference",
        np.abs(halves_mfcc - so.MFCC(coefs).data).max(),
    )


# ------------------------------------------------ the sampled gammatone as IIR
def eulerian(n):
    """Eulerian numbers A(n, k), k = 0..n-1: sum_k k**n x**k = x A_n(x) / (1 - x)**(n + 1)."""
    return np.array(
        [sum((-1) ** j * comb(n + 1, j) * (k + 1 - j) ** n for j in range(k + 2)) for k in range(n)]
    )


def gammatone_iir(cf, fs, order=4, bandwidth_factor=1.019):
    """The complex gammatone ``k**(order-1) p**k`` (``p = exp((-2 pi b + 2 pi i cf) / fs)``),
    the analytic impulse response ``t**(order-1) exp(-2 pi b t) exp(2 pi i cf t)``
    sampled at ``t = k / fs``, as an exact rational filter: numerator
    ``p z^-1 A(p z^-1)``, denominator ``(1 - p z^-1)**order``. Scaled so the
    real part (the real gammatone) has unit gain at ``cf``."""
    b = bandwidth_factor * erb_bandwidth(cf)
    p = np.exp((-2 * np.pi * b + 2j * np.pi * cf) / fs)
    num = np.concatenate([[0.0], eulerian(order - 1) * p ** np.arange(1, order)])
    den = np.poly(np.full(order, p))
    w = 2 * np.pi * cf / fs
    _, h_pos = freqz(num, den, worN=[w])
    _, h_neg = freqz(num, den, worN=[-w])
    real_gain = np.abs((h_pos[0] + np.conj(h_neg[0])) / 2)  # response of Re(h) at cf
    return num / real_gain, den


def check_gammatone():
    bank = so.gammatone_filterbank(n_bands=32, f_lo=80, f_hi=8000, edges=False)
    snd = so.Sound(NOISE, FS)
    subbands = bank.analyze(snd)
    batch_real = subbands.data[:, :, 0] if subbands.data.ndim == 3 else subbands.data
    batch_env = np.abs(hilbert(batch_real, axis=0))
    rng = np.random.default_rng(1)
    worst_stream = 0.0
    shape_rel, env_rel = [], []
    for band, cf in enumerate(bank.band_cfs):
        num, den = gammatone_iir(cf, FS)
        whole = lfilter(num, den, NOISE.astype(complex))
        # streamed: the same recursion with its state carried between blocks
        state = np.zeros(max(len(num), len(den)) - 1, complex)
        parts = []
        for block in random_blocks(NOISE, rng, 4410):
            y, state = lfilter(num, den, block.astype(complex), zi=state)
            parts.append(y)
        streamed = np.concatenate(parts)
        worst_stream = max(worst_stream, np.abs(streamed - whole).max() / np.abs(whole).max())
        settled = slice(int(0.1 * FS), None)  # past the onset, where the FFT bank's padding differs
        diff = whole.real[settled] - batch_real[settled, band]
        shape_rel.append(np.sqrt(np.mean(diff**2) / np.mean(batch_real[settled, band] ** 2)))
        env_diff = np.abs(whole[settled]) - batch_env[settled, band]
        env_rel.append(np.sqrt(np.mean(env_diff**2) / np.mean(batch_env[settled, band] ** 2)))
    report(
        "C3",
        "Complex gammatone IIR, streamed in random blocks vs one call: worst difference / peak",
        worst_stream,
    )
    report(
        "C4",
        "Its real part vs today's FFT gammatone band (rms difference / rms), median over 32 bands",
        np.median(shape_rel),
    )
    report("C4", "                                                             worst band", np.max(shape_rel))
    print(
        f"     lowest band {bank.band_cfs[0]:.0f} Hz: {shape_rel[0]:.3g}, "
        f"highest {bank.band_cfs[-1]:.0f} Hz: "
        f"{shape_rel[-1]:.3g}"
    )
    report(
        "C5",
        "Its magnitude vs Hilbert envelope of today's band (rms difference / rms), median",
        np.median(env_rel),
    )
    report("C5", "                                                               worst band", np.max(env_rel))
    # cost: 32 bands, 10 ms blocks, the recursion only
    block = NOISE[:441].astype(complex)
    coefficients = [gammatone_iir(cf, FS) for cf in bank.band_cfs]
    states = [np.zeros(4, complex) for _ in coefficients]
    start = time.perf_counter()
    n_blocks = 300
    for _ in range(n_blocks):
        for i, (num, den) in enumerate(coefficients):
            _, states[i] = lfilter(num, den, block, zi=states[i])
    per_block = (time.perf_counter() - start) / n_blocks
    report("C6", "32-band complex gammatone, one 10 ms block (441 samples): time [ms]", per_block * 1e3)


# ------------------------------------------------------------- copying state
def check_state_copy():
    """The largest state in sight: the modulation spectrogram's 0.5 Hz band
    keeps 6 s of envelope (its kernel length), here 24 bands at 1 kHz."""
    history = RNG.standard_normal((6000, 24))
    block = RNG.standard_normal((10, 24))  # 10 ms of envelope at 1 kHz
    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        history = np.concatenate([history[len(block) :], block])  # a new array each step
    copy = (time.perf_counter() - start) / n
    ring = np.zeros((6000, 24))
    position = 0
    start = time.perf_counter()
    for _ in range(n):
        index = (position + np.arange(len(block))) % len(ring)
        ring[index] = block  # written in place
        position = (position + len(block)) % len(ring)
    in_place = (time.perf_counter() - start) / n
    report(
        "C7",
        f"State of {history.nbytes / 1e6:.2f} MB copied into a new array per 10 ms block: time [ms]",
        copy * 1e3,
    )
    report("C7", "Same state as a ring buffer written in place: time [ms]", in_place * 1e3)


def check_sound_blocks():
    """Blocks as Sounds: what wrapping each 10 ms block costs (Sound copies
    its data and checks its shape)."""
    block = NOISE[:441]
    n = 20000
    start = time.perf_counter()
    for _ in range(n):
        so.Sound(block, FS)
    per_block = (time.perf_counter() - start) / n
    report("C10", "Wrapping one 10 ms block (441 samples) in a Sound: time [ms]", per_block * 1e3)


def check_modulation_direct():
    """The modulation spectrogram's batch path convolves by FFT and keeps every
    hop-th sample; a stream computes each kept time window directly. Both on
    10 s of 24 envelopes at 1 kHz, hop 10 ms, the default 15 kernels."""
    from sonore.views.modspectrogram import HannModulationFilterbank
    from sonore.views.modulation import _correlate

    fs, n_bands = 1000.0, 24
    env = np.abs(RNG.standard_normal((int(10 * fs), n_bands, 1)))
    positions = np.arange(0, len(env), 10)
    kernels = [kernel for kernel, _ in HannModulationFilterbank(0.5, 64.0, 2, 3, None).kernels(fs)]
    start = time.perf_counter()
    by_fft = [_correlate(env, kernel, "causal")[positions] for kernel in kernels]
    fft_time = time.perf_counter() - start
    start = time.perf_counter()
    direct = []
    for kernel in kernels:
        padded = np.concatenate([np.zeros((len(kernel) - 1, n_bands, 1)), env])
        stretches = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=0)[positions]
        direct.append(stretches @ np.conj(kernel))
    direct_time = time.perf_counter() - start
    worst = max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(by_fft, direct, strict=True))
    report("C11", "Modulation spectrogram, direct per time window vs FFT: worst difference / peak", worst)
    report("C11", "Time of the FFT path [ms]", fft_time * 1e3)
    report("C11", "Time computing each kept time window directly [ms]", direct_time * 1e3)


def check_modulation_sliding_sums():
    """The stream as written (sliding sums restarted every 1024 samples)
    against the batch causal spectrogram: 10 min of 4 envelopes at 1 kHz cut
    into random blocks, and the time per 10 ms block for 24 and 64 bands."""
    rng = np.random.default_rng(1)
    bank = so.gammatone_filterbank(n_bands=4, edges=False)
    n = 600_000
    slow_wave = np.sin(np.arange(n) / 50)[:, None, None]
    env = so.TemporalEnvelope(np.abs(rng.standard_normal((n, 4, 1))) + slow_wave + 1.5, 1000.0, bank)
    batch = so.ModulationSpectrogram(env, align="causal")
    edges = np.concatenate([[0], np.cumsum(rng.integers(1, 700, size=3000))])
    edges = [*edges[edges < n].tolist(), n]
    blocks = [
        so.TemporalEnvelope(env.data[a:b], 1000.0, bank) for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    joined = so.ModulationSpectrogram.concatenate(
        list(so.stream(so.ModulationSpectrogramStream(bank), blocks))
    )
    worst = np.abs(joined.power - batch.power).max(axis=(0, 1, 2)) / batch.power.max()
    report("C12", "Sliding sums vs batch FFT, 10 min in random blocks: worst difference / peak", worst.max())
    report("C12", "The same, in the last minute only", worst[-6000:].max())
    for n_bands in (24, 64):
        bank = so.gammatone_filterbank(n_bands=n_bands, edges=False)
        env = so.TemporalEnvelope(np.abs(rng.standard_normal((10_000, n_bands, 1))), 1000.0, bank)
        blocks = [so.TemporalEnvelope(env.data[i : i + 10], 1000.0, bank) for i in range(0, 10_000, 10)]
        start = time.perf_counter()
        list(so.stream(so.ModulationSpectrogramStream(bank), blocks))
        per_block = (time.perf_counter() - start) / len(blocks)
        report(
            "C12",
            f"Streamed modulation spectrogram, {n_bands} bands, one 10 ms block: time [ms]",
            per_block * 1e3,
        )


def check_stft_cost(frame):
    step, _ = make_stft_step(frame, FS)
    blocks = [NOISE[i : i + 441] for i in range(0, len(NOISE) - 441, 441)]
    start = time.perf_counter()
    scan(step, stft_start(frame, FS), blocks)
    per_block = (time.perf_counter() - start) / len(blocks)
    report("C6", "Streamed STFT (20 ms Hann, 5 ms hop), one 10 ms block: time [ms]", per_block * 1e3)


# ---------------------------------------------------------------- the F0 track
def check_f0_lag():
    """A live F0 track must commit to each time window some time after it.
    Fixed-lag decoding: run the Viterbi pass up to ``lag`` past window i,
    trace back from the cheapest end state, and keep only window i's choice."""
    from sonore.views.f0 import _viterbi

    files = ["bdl_arctic_a0131.flac"] + sorted(p.name for p in SPEECH.glob("librispeech_*.flac"))
    for lag_ms in (0, 25, 50, 100, 200):
        changed, total = 0, 0
        for name in files:
            data, fs = sf.read(SPEECH / name)
            snd = so.Sound(data[: int(4 * fs)], fs)
            track = so.f0_track(snd)
            cand, score = track.candidates[0], track.candidate_scores[0]
            full, _ = _viterbi(cand, score, track.threshold, 2.0, 0.5, 0.05)
            lag = int(round(lag_ms / 5))
            n = len(full)
            for i in range(n):
                end = min(n, i + lag + 1)
                prefix, _ = _viterbi(cand[:end], score[:end], track.threshold, 2.0, 0.5, 0.05)
                changed += prefix[i] != full[i]
            total += n
        print(f"     {len(files)} recordings, {total} time windows")
        report(
            "C8",
            f"F0 decided {lag_ms} ms after each time window: share of windows unlike the whole-track pass",
            changed / total,
        )


def check_f0_locality():
    """Everything before the Viterbi pass looks at a bounded stretch around
    each time window: cut the sound in two and compare candidates."""
    data, fs = sf.read(SPEECH / "bdl_arctic_a0131.flac")
    snd = so.Sound(data, fs)  # 2.5 s
    whole = so.f0_track(snd).candidates[0]
    cut = int(1.2 * fs)
    margin = (
        0.1  # s each side of the cut, an estimate of the reach (YIN 25 ms, refinement 3 periods at 54 Hz)
    )
    later = so.f0_track(so.Sound(data[cut - int(margin * fs) :], fs)).candidates[0][
        int(round(2 * margin / 0.005)) :
    ]
    start = int(round((1.2 + margin) / 0.005))
    a, b = whole[start : start + len(later)], later[: len(whole) - start]
    same = np.isclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), rtol=0, atol=1e-9)
    report(
        "C9",
        "F0 candidates computed from the sound cut 0.1 s before: share of windows identical",
        same.mean(),
    )
    # Before D8 (a), a few differed because the code worked on chunks of
    # time windows that shared a window length taken from the chunk's
    # lowest candidate; with one window per chunk they all agreed. Now the
    # row lengths are set by each candidate, so both lines read 1.
    import sonore.views.f0 as f0_module

    default_chunk = f0_module.CHUNK
    f0_module.CHUNK = 1
    try:
        whole_alone = so.f0_track(snd).candidates[0]
        later_alone = so.f0_track(so.Sound(data[cut - int(margin * fs) :], fs)).candidates[0][
            int(round(2 * margin / 0.005)) :
        ]
        a, b = whole_alone[start : start + len(later_alone)], later_alone[: len(whole_alone) - start]
        same = np.isclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), rtol=0, atol=1e-9)
        report("C9", "The same, with every time window computed on its own: share identical", same.mean())
        changed, voiced_changed, total = 0, 0, 0
        files = ["bdl_arctic_a0131.flac"] + sorted(p.name for p in SPEECH.glob("librispeech_*.flac"))
        for name in files:
            samples, rate = sf.read(SPEECH / name)
            sound = so.Sound(samples[: int(4 * rate)], rate)
            f0_module.CHUNK = default_chunk
            today = so.f0_track(sound).f0[0]
            f0_module.CHUNK = 1
            alone = so.f0_track(sound).f0[0]
            both = (today > 0) & (alone > 0)
            changed += np.sum((today > 0) != (alone > 0))
            voiced_changed += np.sum(both & (np.abs(alone / np.where(both, today, 1) - 1) > 0.01))
            total += len(today)
        report(
            "C9",
            "F0 track, every window on its own vs today: share of windows whose voicing changes",
            changed / total,
        )
        report(
            "C9",
            "                                   share voiced in both whose F0 moves by over 1%",
            voiced_changed / total,
        )
    finally:
        f0_module.CHUNK = default_chunk


if __name__ == "__main__":
    frame, _ = check_stft()
    check_cepstrum_mfcc(frame)
    check_gammatone()
    check_state_copy()
    check_stft_cost(frame)
    check_sound_blocks()
    check_modulation_direct()
    check_modulation_sliding_sums()
    check_f0_locality()
    check_f0_lag()
