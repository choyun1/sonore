"""Shared test constants and helpers."""

import numpy as np

import sonore as so

FS = 44100


def peak_freq(s: so.Sound) -> float:
    X = np.abs(np.fft.rfft(s.data[:, 0]))
    return np.fft.rfftfreq(len(s), 1 / s.fs)[np.argmax(X)]


def dominant_freq(s: so.Sound) -> float:
    """Peak frequency with parabolic interpolation on a zero-padded spectrum."""
    x = s.data[:, 0] * np.hanning(len(s))
    n = 8 * len(x)
    X = np.abs(np.fft.rfft(x, n))
    k = np.argmax(X[1:-1]) + 1
    a, b, c = np.log(X[k - 1 : k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * s.fs / n


def cents(f, ref):
    return 1200 * np.log2(f / ref)
