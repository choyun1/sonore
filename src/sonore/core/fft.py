"""How many threads sonore's large FFTs use, and FFT-friendly lengths.

The whole-signal transforms (filterbank analysis and synthesis, Hilbert
envelopes, FFT resampling, modulation filtering) run a batch of 1-D FFTs, one
per band and channel. SciPy can split such a batch across threads. Each
transform is computed exactly as on one thread, so results are bit-identical
whatever the number of threads; only the speed changes.

By default sonore uses every core this process may run on. When you run
several sonore jobs in parallel (one per core, say), set it to 1 so they don't
compete::

    so.set_fft_workers(1)          # from now on
    with so.set_fft_workers(1):    # or only inside the block
        ...
"""

from __future__ import annotations

import os

import scipy.fft

__all__ = ["fft_workers", "set_fft_workers", "fast_padding"]


def _available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects CPU limits set for this process
    except AttributeError:  # not available on macOS or Windows
        return os.cpu_count() or 1


_workers = _available_cores()


def fft_workers() -> int:
    """Number of threads sonore's large FFTs use."""
    return _workers


class set_fft_workers:
    """Set the number of threads for sonore's large FFTs (``None``: every
    available core, the default). Takes effect at once; used in a ``with``
    block, the previous value is restored on exit."""

    def __init__(self, n: int | None = None):
        global _workers
        n = _available_cores() if n is None else int(n)
        if n < 1:
            raise ValueError("the number of FFT workers must be at least 1")
        self._previous = _workers
        _workers = n

    def __enter__(self) -> set_fft_workers:
        return self

    def __exit__(self, *exc) -> None:
        global _workers
        _workers = self._previous


def threads():
    """Context in which SciPy FFTs (also inside ``scipy.signal``) use
    :func:`fft_workers` threads."""
    return scipy.fft.set_workers(_workers)


def _is_fast(n: int) -> bool:
    for prime in (2, 3, 5, 7, 11):
        while n % prime == 0:
            n //= prime
    return n == 1


def fast_padding(n: int, pad: int) -> int:
    """The smallest padding ``>= pad`` for which ``n + 2 * padding`` has no
    prime factor above 11. FFTs of such lengths are several times faster than
    of lengths with a large prime factor. Padding on both sides keeps the
    parity of ``n``, so an odd ``n`` cannot use the factor 2. For ``n`` above
    10000 the padded length grows by at most about 5% (median 0.3%) for odd
    ``n``, and by under 2% for even ``n``."""
    while not _is_fast(n + 2 * pad):
        pad += 1
    return pad
