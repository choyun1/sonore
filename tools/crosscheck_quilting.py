"""Dev-time cross-check of so.quilt and so.splice against McDermott's Sound
Quilting Toolbox (MATLAB), run in GNU Octave. Not part of the test suite.

The toolbox has no license, so it is not in this repository: give the folder
of a local copy and the Octave binary (Octave 10.3 from conda-forge, with the
signal package, works; set OCTAVE_HOME to its prefix).

    python tools/crosscheck_quilting.py /path/to/Sound_Quilting_Toolbox octave-cli

Checks, on the toolbox's example speech at 20 kHz, 30 ms windows, 30 bands
from 20 Hz to 10 kHz (the paper's settings):

  X1  the band filters: sonore's cosine bank and the toolbox's, on the same
      2N-point frequency grid
  X2  the border changes the toolbox records (between neighbors in the
      source, and in its quilt), against sonore's border-change matrix,
      for envelopes made as the toolbox makes them and as sonore does
  X3  the order: the toolbox's exclusion rule run on sonore's changes,
      from the toolbox's first segment, gives the toolbox's order
  X4  the joining: so.splice against the toolbox's reorder_segs on the
      toolbox's order
  X5  the three bugs found by reading the toolbox: stereo input, the 4x
      overcomplete filterbank option, and running out of segments
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert

import sonore as so
from sonore.views.quilting import _border_changes

TOOLBOX, OCTAVE = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "octave-cli"
FS, WIN_MS, N_BANDS, F_LO, F_HI = 20000, 30, 30, 20, 10000

work = Path(tempfile.mkdtemp())


def octave(script):
    """Run an Octave script with the toolbox on the path; return what it printed."""
    path = work / "run.m"
    path.write_text(f"pkg load signal; addpath('{TOOLBOX}'); cd('{work}');\n" + script)
    done = subprocess.run([OCTAVE, "--no-gui", "--quiet", str(path)], capture_output=True, text=True)
    return done.stdout + done.stderr


# The toolbox's run: its example speech at 20 kHz, 60 ms segments, a 6 s quilt,
# with Octave's generator seeded; everything saved for comparison.
out = octave(f"""
[s, sr] = audioread('{TOOLBOX}/example_speech_source_file.wav');
s = resample(s, {FS}, sr); s = s(:)'; s = s / rms(s) * .03;
P.filt_density = 1; P.N_audio_channel = {N_BANDS}; P.audio_low_lim_Hz = {F_LO};
P.audio_high_lim_Hz = {F_HI}; P.audio_sr = {FS};
rand('seed', 1);
[order, source_changes, quilt_changes] = make_seg_order(s, {WIN_MS}, 2, 6, P);
win = round({WIN_MS} / 1000 * {FS});
[quilted, shifts] = reorder_segs(s, win * 2, order, 3, win, win / 2);
filts = make_erb_cos_filters(length(s) * 2, {FS}, {N_BANDS}, {F_LO}, {F_HI});
envs = abs(hilbert(generate_subbands_zp(s', filts))) ;
save('-v7', 'toolbox.mat', 's', 'order', 'source_changes', 'quilt_changes', ...
     'quilted', 'shifts', 'filts', 'envs');
""")
if not (work / "toolbox.mat").exists():
    sys.exit("Octave run failed:\n" + out)
tb = loadmat(work / "toolbox.mat")
source = so.Sound(tb["s"].ravel(), FS)
order = tb["order"].ravel().astype(int) - 1  # to 0-based
n_window = round(WIN_MS / 1000 * FS)
n_segment = 2 * n_window
n = len(source)

# X1. The toolbox's filters sample the 2N-point grid from 0 to Nyquist; sonore's
# bank padded by N/2 at each end is evaluated on the same grid.
bank = so.cosine_filterbank(N_BANDS, F_LO, F_HI)
freqs = np.arange(n + 1) * FS / (2 * n)
ours = bank.response(freqs)
filts = tb["filts"]
print(
    f"X1 filters: shapes {ours.shape} and {filts.shape},"
    f" largest difference {np.max(np.abs(ours - filts)):.1e}"
)

# X2. Envelopes as the toolbox makes them (Hilbert after removing the padding)
# from sonore's subbands, and sonore's own (Hilbert on the padded subbands).
subbands = bank.analyze(source, pad=n / 2 / FS)
like_toolbox = np.abs(hilbert(subbands.data[:, :, 0], axis=0))
own = subbands.envelopes().data[:, :, 0]
print(
    f"X2 envelopes, Hilbert as the toolbox:"
    f" largest difference {np.max(np.abs(like_toolbox - tb['envs'])):.1e}"
)
print(f"X2 envelopes, sonore's own: largest difference {np.max(np.abs(own - tb['envs'])):.1e}")
for label, env in (("as the toolbox", like_toolbox), ("sonore's own", own)):
    change = _border_changes(env, n_segment, n_window)
    source_changes = np.diag(change, 1)
    quilt_changes = change[order[:-1], order[1:]]
    tb_source = tb["source_changes"].ravel()
    tb_quilt = tb["quilt_changes"].ravel()
    print(
        f"X2 border changes, envelopes {label}:"
        f" source {np.max(np.abs(source_changes - tb_source)) / tb_source.max():.1e},"
        f" quilt {np.max(np.abs(quilt_changes - tb_quilt)) / tb_quilt.max():.1e} (relative to the largest)"
    )

    # X3. The toolbox's rule: never segments 1 and 2, the current one and its
    # neighbors, used ones, or the last two (candidates run to n_seg - 2, 1-based).
    n_segments = len(change)
    rule = [int(order[0])]
    for _ in range(1, len(order)):
        k = rule[-1]
        avoid = set(rule) | {k - 1, k, k + 1, 0, 1}
        candidates = [j for j in range(n_segments - 2) if j not in avoid]
        mismatch = {j: abs(change[k, j] - change[k, k + 1]) for j in candidates}
        rule.append(min(mismatch, key=mismatch.get))
    agree = int(np.sum(np.array(rule) == order))
    print(f"X3 order, envelopes {label}: {agree} of {len(order)} segments as the toolbox's")

# X4. The joining. (a) Without shifts, both are plain overlap-add with the same
# ramps. The toolbox fades the outer ends (sonore starts and ends at full level),
# so the outer half cross-fades are left out. (b) The shifts: the toolbox's output
# is rebuilt join by join from its own shifts, and at each join sonore's search
# runs on what the toolbox had written there.
out = octave(f"""
load('toolbox.mat');
win = round({WIN_MS} / 1000 * {FS});
[plain, ~] = reorder_segs(s, win * 2, order, 3, win, 0);
save('-v7', 'plain.mat', 'plain');
""")
plain = loadmat(work / "plain.mat")["plain"].ravel()
joined = so.splice(source, order * n_segment / FS, n_segment / FS, max_shift=0).data[:, 0]
inner = slice(n_window // 2, len(plain) - n_window // 2)
print(
    f"X4 overlap-add without shifts:"
    f" largest difference inside {np.max(np.abs(joined[inner] - plain[inner])):.1e}"
)

signal = source.data[:, 0]
half = n_window // 2
rise = 0.5 * (1 - np.cos(np.pi * np.arange(n_window) / n_window))
window = np.concatenate([rise, np.ones(n_segment - n_window), 1 - rise])
toolbox_shifts = tb["shifts"].ravel().astype(int)
written = np.zeros(n_segment * len(order) + half)
start = order[0] * n_segment
written[: n_segment + half] = signal[start : start + n_segment + half] * window[half:]
ours, one_later = [], []
for i in range(1, len(order)):
    start, join = order[i] * n_segment, i * n_segment
    target = written[join - half : join + half] * rise
    block = signal[start - half - half : start + half + half]  # max_shift is half a window
    ours.append(int(np.argmax(np.correlate(block, target, mode="valid"))) - half)
    # the same search, compared one sample later than the read, as reorder_segs indexes it
    block = signal[start - half - half + 1 : start + half + half + 1]
    one_later.append(int(np.argmax(np.correlate(block, target, mode="valid"))) - half)
    shift = toolbox_shifts[i]
    written[join - half : join + n_segment + half] += (
        window * signal[start + shift - half : start + shift + n_segment + half]
    )
rebuilt = written[: n_segment * len(order)]
values, counts = np.unique(np.array(ours) - toolbox_shifts[1:], return_counts=True)
offsets = dict(zip(values.tolist(), counts.tolist(), strict=True))
print(f"X4 rebuilt toolbox output: largest difference {np.max(np.abs(rebuilt - tb['quilted'].ravel())):.1e}")
print(f"X4 sonore's shift minus the toolbox's, count at {len(ours)} joins: {offsets}")
same = int(np.sum(np.array(one_later) == toolbox_shifts[1:]))
print(f"X4 compared one sample later: {same} of {len(ours)} shifts as the toolbox's")

# X5. The three bugs.
print(
    "X5 stereo:",
    octave(f"""
s = randn({FS} * 20, 2);
P.filt_density = 1; P.N_audio_channel = {N_BANDS}; P.audio_low_lim_Hz = {F_LO};
P.audio_high_lim_Hz = {F_HI}; P.audio_sr = {FS};
try, q = generate_quilt(s, {WIN_MS}, 2, 1, P); printf('quilt of %d samples\\n', numel(q));
catch e, disp(e.message); end
""")
    .strip()
    .splitlines()[-1],
)
print(
    "X5 filt_density 2:",
    octave(f"""
s = randn(1, {FS} * 5);
P.filt_density = 2; P.N_audio_channel = {N_BANDS}; P.audio_low_lim_Hz = {F_LO};
P.audio_high_lim_Hz = {F_HI}; P.audio_sr = {FS};
try, q = generate_quilt(s, {WIN_MS}, 2, 1, P); disp('ran');
catch e, disp(e.message); end
""")
    .strip()
    .splitlines()[-1],
)
print(
    "X5 running out:",
    octave(f"""
s = randn(1, {FS} * 2);
P.filt_density = 1; P.N_audio_channel = {N_BANDS}; P.audio_low_lim_Hz = {F_LO};
P.audio_high_lim_Hz = {F_HI}; P.audio_sr = {FS};
try, q = generate_quilt(s, {WIN_MS}, 8, 2, P); disp('ran');
catch e, disp(e.message); end
""")
    .strip()
    .splitlines()[-1],
)
