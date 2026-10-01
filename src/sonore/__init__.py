"""sonore: signals and stimuli for auditory research.

Typical use in a notebook::

    import sonore as so
    from sonore import dB

    x = so.pure_tone(1.0, 44100, 440).ramp(0.01)
    y = x + (so.gaussian_noise(1.0, 44100, rng=0) - 10 * dB)   # tone at +10 dB SNR
    y                      # displays an audio player
    so.overview(y)         # waveform, spectrum, spectrogram, modulation spectrum
"""

from sonore import texture
from sonore.binaural import (
    InterauralCues,
    apply_itd_ild,
    interaural_cues,
    oscor,
    phasewarp,
    simple_bir,
)
from sonore.envelopes import Envelope, Envelopes
from sonore.filterbank import (
    CosineFilterbank,
    ERBFilterbank,
    GammatoneFilterbank,
    MorletFilterbank,
    OctaveFilterbank,
    Subbands,
    noise_vocode,
    subbands,
)
from sonore.frames import Filterbank, Frame, GaborFrame, TVGaborFrame
from sonore.generators import (
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
from sonore.modulation import ConstantQModulationFilterbank, ModulationFilterbank, OctaveModulationFilterbank
from sonore.phasevocoder import PVAnalysis, pitch_shift, pv_analyze, time_stretch
from sonore.plotting import overview
from sonore.processing import (
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
from sonore.representations import (
    STFT,
    TVSTFT,
    Mask,
    ModulationSpectrum,
    Spectrum,
    TFPower,
    ideal_binary_mask,
    ideal_ratio_mask,
    long_term_spectrum,
    tandem_power,
)
from sonore.reverb import band_rt60s, measure_rt60, synth_ir
from sonore.ripples import DynamicRipple, Ripple, RippleSum, ripple_sound
from sonore.sound import Sound, load
from sonore.spatialization import (
    HRIRSet,
    circular_trajectory,
    distance_gain_db,
    hcc_to_rect,
    linear_trajectory,
    move_sound,
    rect_to_hcc,
    spatialize,
)
from sonore.units import Decibels, dB
from sonore.utils import amp_to_db, db_to_amp, erb_to_freq, freq_to_erb, rms

__version__ = "0.2.0"

__all__ = [
    "texture",
    "ConstantQModulationFilterbank",
    "ModulationFilterbank",
    "OctaveModulationFilterbank",
    "measure_rt60",
    "band_rt60s",
    "CosineFilterbank",
    "Filterbank",
    "Frame",
    "GaborFrame",
    "GammatoneFilterbank",
    "MorletFilterbank",
    "TVGaborFrame",
    "TVSTFT",
    "TFPower",
    "tandem_power",
    "Envelopes",
    "Envelope",
    "ripple_sound",
    "RippleSum",
    "Ripple",
    "DynamicRipple",
    "OctaveFilterbank",
    "time_stretch",
    "pv_analyze",
    "pitch_shift",
    "PVAnalysis",
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
