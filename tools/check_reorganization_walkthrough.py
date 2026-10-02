"""Runs the "What a user sees" walkthrough of docs/design/reorganization.md (D13) against
today's sonore, and prints what each step gives.

    python tools/check_reorganization_walkthrough.py

It prints the reason each view gives when asked to synthesize (NotInvertibleError), and
the errors and durations of the steps that make a sound.
"""

import numpy as np

import sonore as so

sentence = so.load("docs/speech/bdl_arctic_a0131.flac")
print(f"sentence: {sentence.duration:.3f} s at {sentence.fs:.0f} Hz")

frame = so.GaborFrame(win_dur=0.025, hop_dur=0.005)
coefs = frame.analyze(sentence)
error = np.max(np.abs(frame.synthesize(coefs).data - sentence.data))
print(f"GaborFrame: analyze gives {type(coefs).__name__}, synthesize max error {error:.1e}")

bands = so.subbands(sentence, n_bands=16)
error = np.max(np.abs(bands.synthesize().data - sentence.data))
print(f"subbands: synthesize max error {error:.1e}")

for name, view in [("MFCC", so.MFCC(sentence)), ("Envelopes", bands.envelopes())]:
    try:
        view.synthesize()
    except so.NotInvertibleError as err:
        print(f"{name}.synthesize(): NotInvertibleError: {err}")

cepstrum = so.Cepstrum(coefs)
error = np.max(np.abs(cepstrum.to_sound().data - sentence.data))
print(f"Cepstrum.to_sound() (original phase, unliftered): max error {error:.1e}")

times, f0 = np.loadtxt("docs/speech/bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2).T
aperiodicity = so.d4c(sentence, (times, f0))
voice = so.world_synthesize((times, f0), so.MFCC(sentence).envelope_view(), aperiodicity)
print(f"world_synthesize from the MFCC envelope: {voice.duration:.3f} s of sound")
