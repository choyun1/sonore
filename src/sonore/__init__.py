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
from sonore.analysis.cepstrum import Cepstrum
from sonore.analysis.envelopes import Envelope, Envelopes
from sonore.analysis.filterbank import (
    CosineFilterbank,
    ERBFilterbank,
    GammatoneFilterbank,
    MorletFilterbank,
    OctaveFilterbank,
    Subbands,
    noise_vocode,
    subbands,
)
from sonore.analysis.frames import Filterbank, Frame, GaborFrame, TVGaborFrame
from sonore.analysis.modspectrogram import ModulationSpectrogram
from sonore.analysis.modulation import (
    ConstantQModulationFilterbank,
    HannModulationFilterbank,
    ModulationFilterbank,
    OctaveModulationFilterbank,
)
from sonore.analysis.representations import (
    STFT,
    TVSTFT,
    Mask,
    ModulationSpectrum,
    ReassignedSpectrogram,
    Spectrum,
    TFPower,
    ideal_binary_mask,
    ideal_ratio_mask,
    long_term_spectrum,
    reassigned_spectrogram,
    tandem_power,
)
from sonore.core.sound import Sound, load
from sonore.core.units import Decibels, dB
from sonore.core.utils import amp_to_db, db_to_amp, erb_to_freq, freq_to_erb, rms
from sonore.plotting import overview
from sonore.signals.generators import (
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
from sonore.signals.processing import (
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
from sonore.stimuli.binaural import (
    InterauralCues,
    apply_itd_ild,
    interaural_cues,
    oscor,
    phasewarp,
    simple_bir,
)
from sonore.stimuli.hrir_data import load_hrirs
from sonore.stimuli.phasevocoder import PVAnalysis, pitch_shift, pv_analyze, time_stretch
from sonore.stimuli.reverb import band_rt60s, measure_rt60, synth_ir
from sonore.stimuli.ripples import DynamicRipple, Ripple, RippleSum, ripple_sound
from sonore.stimuli.spatialization import (
    HRIRSet,
    circular_trajectory,
    distance_gain_db,
    hcc_to_rect,
    linear_trajectory,
    move_sound,
    rect_to_hcc,
    spatialize,
)

__version__ = "0.3.1"

__all__ = [
    "texture",
    "ConstantQModulationFilterbank",
    "HannModulationFilterbank",
    "ModulationFilterbank",
    "ModulationSpectrogram",
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
    "Cepstrum",
    "TFPower",
    "tandem_power",
    "ReassignedSpectrogram",
    "reassigned_spectrogram",
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
    "load_hrirs",
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
