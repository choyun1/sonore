"""sigtools: signals and stimuli for auditory research.

Typical use in a notebook::

    import sigtools as st
    x = st.pure_tone(1.0, 44100, 440).ramp(0.01)
    x                      # displays an audio player
    st.overview(x)         # waveform, spectrum, spectrogram, modulation spectrum
"""

from sigtools.binaural import (
    InterauralCues,
    apply_itd_ild,
    interaural_cues,
    oscor,
    phasewarp,
    simple_bir,
)
from sigtools.filterbank import ERBFilterbank, Subbands, noise_vocode, subbands
from sigtools.generators import (
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
from sigtools.plotting import overview
from sigtools.processing import (
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
from sigtools.representations import (
    STFT,
    Mask,
    ModulationSpectrum,
    Spectrum,
    ideal_binary_mask,
    ideal_ratio_mask,
    long_term_spectrum,
)
from sigtools.reverb import synth_ir
from sigtools.sound import Sound, load
from sigtools.spatialization import (
    HRIRSet,
    circular_trajectory,
    distance_gain_db,
    hcc_to_rect,
    linear_trajectory,
    move_sound,
    rect_to_hcc,
    spatialize,
)
from sigtools.utils import amp_to_db, db_to_amp, erb_to_freq, freq_to_erb, rms

__version__ = "0.2.0"
