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
from sonore.core.fft import fft_workers, set_fft_workers
from sonore.core.sound import Sound, load
from sonore.core.units import Decibels, dB
from sonore.core.utils import (
    amp_to_db,
    db_to_amp,
    db_to_power,
    erb_to_freq,
    freq_to_erb,
    freq_to_mel,
    mel_to_freq,
    power_to_db,
    rms,
)
from sonore.frames.filterbank import (
    CosineFilterbank,
    ERBFilterbank,
    Filterbank,
    GammatoneFilterbank,
    MorletFilterbank,
    OctaveFilterbank,
    Subbands,
    subbands,
)
from sonore.frames.frame import Frame
from sonore.frames.gabor import STFT, TVSTFT, GaborFrame, TVGaborFrame
from sonore.frames.mask import Mask, ideal_binary_mask, ideal_ratio_mask
from sonore.plotting import overview
from sonore.signals.generators import (
    correlated_noise,
    exponential_chirp,
    gaussian_noise,
    glottal_source,
    harmonic_complex,
    iterated_ripple_noise,
    lf_harmonics,
    lf_pulse,
    linear_chirp,
    pulse_train,
    pure_tone,
    sawtooth_wave,
    schroeder_complex,
    silence,
    square_wave,
)
from sonore.signals.klatt import KLATT_DEFAULTS, klatt_continuum, klatt_synthesize
from sonore.signals.processing import (
    amplitude_modulate,
    antiresonator,
    bandpass,
    butter_filter,
    concat,
    match_channels,
    match_fs,
    mix,
    normalize,
    pad,
    relative_db,
    resonator,
    truncate,
)
from sonore.signals.world import DIFFERENCES_FROM_WORLD, world_synthesize
from sonore.spatial.binaural import (
    InterauralCues,
    apply_itd_ild,
    interaural_cues,
    oscor,
    phasewarp,
    simple_bir,
)
from sonore.spatial.hrir_data import load_hrirs
from sonore.spatial.reverb import band_rt60s, measure_rt60, synth_ir
from sonore.spatial.spatialization import (
    SPEED_OF_SOUND,
    HRIRSet,
    circular_trajectory,
    distance_gain_db,
    hcc_to_rect,
    hcc_trajectory,
    linear_trajectory,
    move_sound,
    rect_to_hcc,
    spatialize,
)
from sonore.stimuli.channel_vocoder import noise_vocode
from sonore.stimuli.phasevocoder import PVAnalysis, pitch_shift, pv_analyze, time_stretch
from sonore.stimuli.ripples import DynamicRipple, Ripple, RippleSum, ripple_sound
from sonore.views.aperiodicity import Aperiodicity, d4c, harmonic_aperiodicity
from sonore.views.cepstrum import Cepstrum
from sonore.views.envelopes import Envelope, Envelopes
from sonore.views.f0 import F0Track, f0_track, scale_f0
from sonore.views.mfcc import MFCC
from sonore.views.modspectrogram import ModulationSpectrogram
from sonore.views.modulation import (
    ConstantQModulationFilterbank,
    HannModulationFilterbank,
    ModulationFilterbank,
    ModulationSpectrum,
    OctaveModulationFilterbank,
)
from sonore.views.reassigned import ReassignedSpectrogram, reassigned_spectrogram
from sonore.views.spectral_envelope import GridEnvelope, SpectralEnvelope, cheaptrick, warp_frequency
from sonore.views.spectrum import Spectrum, TFPower, long_term_spectrum, tandem_power
from sonore.views.view import NotInvertibleError, View

__version__ = "0.4.0"

__all__ = [
    "SPEED_OF_SOUND",
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
    "View",
    "NotInvertibleError",
    "GaborFrame",
    "GammatoneFilterbank",
    "MorletFilterbank",
    "TVGaborFrame",
    "TVSTFT",
    "Cepstrum",
    "MFCC",
    "F0Track",
    "f0_track",
    "Aperiodicity",
    "DIFFERENCES_FROM_WORLD",
    "SpectralEnvelope",
    "cheaptrick",
    "d4c",
    "harmonic_aperiodicity",
    "world_synthesize",
    "GridEnvelope",
    "scale_f0",
    "warp_frequency",
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
    "KLATT_DEFAULTS",
    "klatt_continuum",
    "klatt_synthesize",
    "PVAnalysis",
    "amp_to_db",
    "power_to_db",
    "db_to_power",
    "amplitude_modulate",
    "antiresonator",
    "resonator",
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
    "fft_workers",
    "freq_to_erb",
    "freq_to_mel",
    "gaussian_noise",
    "glottal_source",
    "harmonic_complex",
    "hcc_to_rect",
    "HRIRSet",
    "ideal_binary_mask",
    "ideal_ratio_mask",
    "interaural_cues",
    "InterauralCues",
    "iterated_ripple_noise",
    "lf_harmonics",
    "lf_pulse",
    "linear_chirp",
    "hcc_trajectory",
    "linear_trajectory",
    "load",
    "load_hrirs",
    "long_term_spectrum",
    "Mask",
    "match_channels",
    "match_fs",
    "mel_to_freq",
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
    "set_fft_workers",
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
