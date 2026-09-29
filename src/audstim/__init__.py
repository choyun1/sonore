"""audstim: signals and stimuli for auditory research.

Typical use in a notebook::

    import audstim as au
    from audstim import dB

    x = au.pure_tone(1.0, 44100, 440).ramp(0.01)
    y = x + (au.gaussian_noise(1.0, 44100, rng=0) - 10 * dB)   # tone at +10 dB SNR
    y                      # displays an audio player
    au.overview(y)         # waveform, spectrum, spectrogram, modulation spectrum
"""

from audstim.binaural import (
    InterauralCues,
    apply_itd_ild,
    interaural_cues,
    oscor,
    phasewarp,
    simple_bir,
)
from audstim.filterbank import ERBFilterbank, Subbands, noise_vocode, subbands
from audstim.generators import (
    correlated_noise,
    exponential_chirp,
    gaussian_noise,
    harmonic_complex,
    iterated_ripple_noise,
    linear_chirp,
    pulse_train,
    pure_tone,
    sawtooth_wave,
    schroeder_complex,
    silence,
    square_wave,
)
from audstim.plotting import overview
from audstim.processing import (
    amplitude_modulate,
    bandpass,
    butter_filter,
    concat,
    match_channels,
    match_fs,
    mix,
    normalize,
    pad,
    relative_db,
    truncate,
)
from audstim.representations import (
    STFT,
    Mask,
    ModulationSpectrum,
    Spectrum,
    ideal_binary_mask,
    ideal_ratio_mask,
    long_term_spectrum,
)
from audstim.reverb import synth_ir
from audstim.sound import Sound, load
from audstim.spatialization import (
    HRIRSet,
    circular_trajectory,
    distance_gain_db,
    hcc_to_rect,
    linear_trajectory,
    move_sound,
    rect_to_hcc,
    spatialize,
)
from audstim.units import Decibels, dB
from audstim.utils import amp_to_db, db_to_amp, erb_to_freq, freq_to_erb, rms

__version__ = "0.2.0"

__all__ = [
    "amp_to_db",
    "amplitude_modulate",
    "apply_itd_ild",
    "bandpass",
    "butter_filter",
    "circular_trajectory",
    "concat",
    "correlated_noise",
    "dB",
    "db_to_amp",
    "Decibels",
    "distance_gain_db",
    "erb_to_freq",
    "ERBFilterbank",
    "exponential_chirp",
    "freq_to_erb",
    "gaussian_noise",
    "harmonic_complex",
    "hcc_to_rect",
    "HRIRSet",
    "ideal_binary_mask",
    "ideal_ratio_mask",
    "interaural_cues",
    "InterauralCues",
    "iterated_ripple_noise",
    "linear_chirp",
    "linear_trajectory",
    "load",
    "long_term_spectrum",
    "Mask",
    "match_channels",
    "match_fs",
    "mix",
    "ModulationSpectrum",
    "move_sound",
    "noise_vocode",
    "normalize",
    "oscor",
    "overview",
    "pad",
    "phasewarp",
    "pulse_train",
    "pure_tone",
    "rect_to_hcc",
    "relative_db",
    "rms",
    "sawtooth_wave",
    "schroeder_complex",
    "silence",
    "simple_bir",
    "Sound",
    "spatialize",
    "Spectrum",
    "square_wave",
    "STFT",
    "Subbands",
    "subbands",
    "synth_ir",
    "truncate",
]
