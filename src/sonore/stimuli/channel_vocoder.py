"""The channel vocoder (Shannon et al., 1995): the band envelopes of one sound
imposed on the fine structure of another, as in simulations of
cochlear-implant hearing."""

from __future__ import annotations

import numpy as np

from sonore.core.sound import Sound
from sonore.core.utils import _below_nyquist, as_rng
from sonore.frames.filterbank import ERBFilterbank, Subbands

__all__ = ["noise_vocode"]


def noise_vocode(
    sound: Sound,
    n_bands: int = 16,
    f_lo: float = 80.0,
    f_hi: float = 8000.0,
    carrier: Sound | str = "noise",
    env_lowpass: float | None = 50.0,
    rng=None,
) -> Sound:
    """Channel vocoder (Shannon et al., 1995).

    Band envelopes of ``sound`` (lowpassed at ``env_lowpass`` Hz) modulate the
    fine structure of the ``carrier``: ``"noise"``, ``"tone"`` (sinusoids at the
    band centers), or any Sound at least as long. The edge bands (outside
    ``f_lo..f_hi``) are silenced. The output matches the input's RMS.
    """
    filterbank = ERBFilterbank(n_bands, f_lo, min(f_hi, _below_nyquist(sound.fs)))
    envelopes = filterbank.analyze(sound).envelopes(lowpass=env_lowpass).without_edges()
    if isinstance(carrier, Sound):
        if len(carrier) < len(sound):
            raise ValueError("carrier is shorter than the sound")
        fine = filterbank.analyze(Sound(carrier.data[: len(sound)], carrier.fs)).tfs()
    elif carrier == "noise":
        from sonore.signals.generators import gaussian_noise

        noise = gaussian_noise(sound.duration, sound.fs, n_channels=sound.n_channels, rng=as_rng(rng))
        fine = filterbank.analyze(noise, pad=0).tfs()  # generated noise is periodic: circular is exact
    elif carrier == "tone":
        tones = np.cos(2 * np.pi * filterbank.cfs[None, :, None] * sound.t[:, None, None])
        fine = Subbands(
            np.broadcast_to(tones, (len(sound), len(filterbank.cfs), sound.n_channels)).copy(),
            sound.fs,
            filterbank,
        )
    else:
        raise ValueError("carrier must be 'noise', 'tone', or a Sound")
    return (envelopes * fine).synthesize().normalize(sound.rms)
