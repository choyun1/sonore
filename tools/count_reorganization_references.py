"""Counts what the reorganization proposed in docs/design/layout/reorganization.md would touch.

For every module the proposal moves, it counts the files in the repository that name the
module's current path, grouped by where they live, and the lines of the module itself. It
also lists the gallery pages by group as build.py's TOPICS has them, and which modules a
released version (the git tag given with --release) already shipped, since only those deep
import paths can be in anyone's code. Nothing is changed.

    python tools/count_reorganization_references.py [--release v0.3.0]

It measures the tree before the move, so it runs on a checkout of main at b08063c,
where the doc's counts were taken (git worktree add /tmp/before b08063c).

A reference is any text match of the dotted path (``analysis.vocoder``, which also matches
``sonore.analysis.vocoder``) or the file path (``analysis/vocoder.py``). Moving a module
means editing each file counted; the count is files, not lines.
"""

import argparse
import ast
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "sonore"
if not (SRC / "analysis").is_dir():
    raise SystemExit("the package is already reorganized; run this on a checkout of b08063c")

# Recommended option (A): current module -> proposed module. A split module has
# several targets, separated by " + ". The trunk's analysis layer becomes two,
# frames (invertible analyses and their coefficients) and views (one-way analyses);
# there is no voice branch: synthesizers go to signals, voice analyses to views.
MOVES_A = {
    "analysis/frames": "frames/frame + frames/filterbank + frames/gabor",
    "analysis/filterbank": "frames/filterbank + stimuli/channel_vocoder",
    "analysis/representations": (
        "views/spectrum + frames/gabor + frames/mask + views/reassigned + views/modulation"
    ),
    "analysis/cepstrum": "views/cepstrum",
    "analysis/mfcc": "views/mfcc",
    "analysis/envelopes": "views/envelopes",
    "analysis/modulation": "views/modulation",
    "analysis/modspectrogram": "views/modspectrogram",
    "analysis/f0": "views/f0",
    "analysis/vocoder": "views/aperiodicity + views/spectral_envelope + signals/world",
    "stimuli/vocoder": "signals/world",
    "analysis/voice": "views/spectral_envelope + views/f0 + signals/world",
    "signals/glottal": "signals/generators",
    "stimuli/klatt": "signals/klatt",
    "stimuli/binaural": "spatial/binaural",
    "stimuli/spatialization": "spatial/spatialization",
    "stimuli/hrir_data": "spatial/hrir_data",
    "stimuli/reverb": "spatial/reverb",
}
_WORLD_SHARED = [
    "world_randn",
    "_Stream",
    "_step_words",
    "_to_bits",
    "_lane_jump",
    "_extend_stream",
    "_matlab_round",
    "world_fft_size",
    "_interp1q",
    "_windowed_waveform",
    "_time_windows",
    "_integer_fs",
    "_DEFAULT_F0",
    "_SAFEGUARD",
    "DIFFERENCES_FROM_WORLD",
]
_ENVELOPE = [
    "cheaptrick",
    "SpectralEnvelope",
    "_FrequencyView",
    "_positions",
    "_pointwise",
    "_linear_smoothing",
    "_dc_correction",
]
# Where each top-level name of a split module goes; names not listed go to the first target.
SPLIT_NAMES = {
    "analysis/frames": {
        "Filterbank": "frames/filterbank",
        "_ringing_samples": "frames/filterbank",
        **dict.fromkeys(
            [
                "GaborFrame",
                "_gabor_sft",
                "_gabor_adjoint_sft",
                "_window",
                "TVGaborFrame",
                "_bridged_f0",
                "_TVLayout",
                "_tv_layout",
            ],
            "frames/gabor",
        ),
    },
    "analysis/filterbank": {"noise_vocode": "stimuli/channel_vocoder"},
    "analysis/representations": {
        "STFT": "frames/gabor",
        "TVSTFT": "frames/gabor",
        "Mask": "frames/mask",
        "ideal_binary_mask": "frames/mask",
        "ideal_ratio_mask": "frames/mask",
        "ReassignedSpectrogram": "views/reassigned",
        "reassigned_spectrogram": "views/reassigned",
        "_window_from_formula": "views/reassigned",
        "_window_tau_and_derivative": "views/reassigned",
        "ModulationSpectrum": "views/modulation",
    },
    "analysis/vocoder": {
        **dict.fromkeys(_WORLD_SHARED, "signals/world"),
        **dict.fromkeys(_ENVELOPE, "views/spectral_envelope"),
    },
    "analysis/voice": {
        **dict.fromkeys(["_scaled", "scale_f0", "_scaled_candidates"], "views/f0"),
        "_contour_on_grid": "signals/world",
    },
}
# Module-level imports of a split module that belong to a part other than its first target.
MODULE_LEVEL_OWNER = {
    ("analysis/voice", "F0Track"): "views/f0",  # used by scale_f0
    ("analysis/vocoder", "F0Track"): "signals/world",  # used by _time_windows
}
# Imports the source PR removes by accepting inputs by what they provide (.t and .f0,
# env(t, f), a grid's .t, .f and .data) instead of checking their type, as
# harmonic_complex and klatt_synthesize already do.
DUCK_TYPED = {
    ("signals/world", "F0Track"),
    ("signals/world", "SpectralEnvelope"),
    ("signals/world", "Aperiodicity"),
}
# The trunk is ranked; the branches share the rank above it and must not import each other.
RANK = {"core": 0, "signals": 1, "frames": 2, "views": 3, "spatial": 4, "stimuli": 4, "texture": 4}
# Option B, the smallest change: only the names that collide.
MOVES_B = {
    "analysis/vocoder": "analysis/world",
    "stimuli/vocoder": "analysis/world",
}

# Recommended gallery groups: page script (folder/name) -> its new folder.
GALLERY_MOVES = {
    "stimuli/binaural": "spatial",
    "listeners/vocoder": "seeing",
    "listeners/reverb": "spatial",
    "listeners/moving": "spatial",
    "seeing/cepstrum": "voice",
    "seeing/harmonics": "voice",
    "seeing/formants": "voice",
    "seeing/aperiodicity": "voice",
    "seeing/voice": "voice",
}

SKIP_DIRS = {".git", "audio", "img", "_build", "__pycache__", ".pytest_cache", "build", "dist"}
SKIP_SUFFIXES = {".wav", ".png", ".npy", ".npz", ".mp3", ".svg", ".pdf", ".zip", ".pyc"}


def area(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    if parts[0] == "README.md":
        return "README"
    if parts[:2] == ("docs", "gallery"):
        return "gallery html" if path.suffix == ".html" else "gallery scripts"
    if parts[:2] == ("docs", "design"):
        return "design docs"
    if parts[:2] == ("docs", "api"):
        return "API docs"
    return parts[0]


def text_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).parts
        if not path.is_file() or SKIP_DIRS.intersection(rel) or path.suffix in SKIP_SUFFIXES:
            continue
        if path.name in (Path(__file__).name, "reorganization.md"):
            continue  # this tool and the proposal it measures
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def references(module: str, files) -> Counter:
    dotted = re.compile(r"(?<![\w.])(?:sonore\.)?" + re.escape(module.replace("/", ".")) + r"\b")
    filename = re.compile(re.escape(module) + r"\.py\b")
    counts = Counter()
    for path, text in files:
        if path.with_suffix("") == SRC / module:
            continue  # the module itself
        if dotted.search(text) or filename.search(text):
            counts[area(path)] += 1
    return counts


def noise_vocode_references(files) -> Counter:
    """noise_vocode leaves analysis/filterbank.py; only callers that name the module break."""
    pattern = re.compile(
        r"filterbank\.noise_vocode|filterbank import[^\n]*noise_vocode|filterbank import \([^)]*noise_vocode"
    )
    counts = Counter()
    for path, text in files:
        if pattern.search(text):
            counts[area(path)] += 1
    return counts


def module_imports():
    """Every import of a sonore name, as (importer module, the importer's top-level
    definition or "" at module level, imported module, imported name, inside a function)."""
    rows = []
    for path in SRC.rglob("*.py"):
        module = "/".join(path.relative_to(SRC).with_suffix("").parts)
        tree = ast.parse(path.read_text())
        for top in tree.body:
            top_name = getattr(top, "name", "")
            for node in ast.walk(top):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sonore."):
                    imported = node.module.removeprefix("sonore.").replace(".", "/")
                    in_function = node is not top and top_name != ""
                    if isinstance(top, ast.If):  # TYPE_CHECKING blocks
                        in_function = True
                    for alias in node.names:
                        rows.append((module, top_name, imported, alias.name, in_function))
    return rows


def destination(module: str, name: str) -> str:
    """Where ``name`` of ``module`` goes under option A."""
    if module in SPLIT_NAMES and name in SPLIT_NAMES[module]:
        return SPLIT_NAMES[module][name]
    if module in MOVES_A:
        return MOVES_A[module].split(" + ")[0]
    return module


def report(title, moves, files):
    print(f"== {title}")
    total = Counter()
    for module, target in moves.items():
        lines = len((SRC / f"{module}.py").read_text().splitlines())
        counts = references(module, files)
        total.update(counts)
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        print(f"{module} ({lines} lines) -> {target}: {sum(counts.values())} files [{detail}]")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="v0.3.0")
    args = parser.parse_args()
    files = list(text_files())

    report("Option A: topic subpackages", MOVES_A, files)
    nv = noise_vocode_references(files)
    detail = ", ".join(f"{k} {v}" for k, v in sorted(nv.items()))
    print(f"analysis/filterbank.noise_vocode -> stimuli/channel_vocoder: {sum(nv.values())} files [{detail}]")
    distinct_a = {
        str(path)
        for path, text in files
        for module in list(MOVES_A) + ["analysis/filterbank.noise_vocode"]
        if module.replace("/", ".") in text or f"{module}.py" in text
    }
    print(f"distinct files touched by option A: {len(distinct_a)}")
    report("Option B: rename the WORLD modules only", MOVES_B, files)
    distinct_b = {
        str(path)
        for path, text in files
        for module in MOVES_B
        if module.replace("/", ".") in text or f"{module}.py" in text
    }
    print(f"distinct files touched by option B: {len(distinct_b)}")

    print(f"== Shipped in {args.release}")
    shipped = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", args.release, "src/sonore"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    shipped = {p.removeprefix("src/sonore/").removesuffix(".py") for p in shipped if p.endswith(".py")}
    moved = sorted(m for m in MOVES_A if m in shipped)
    print(f"moved by option A and already released: {', '.join(moved)}")
    print(f"moved by option A and unreleased: {', '.join(sorted(m for m in MOVES_A if m not in shipped))}")

    print("== Imports between proposed subpackages (option A), resolved name by name")
    edges = Counter()
    wrong_way = []
    duck = []
    for importer, top_name, imported, name, in_function in module_imports():
        a = MODULE_LEVEL_OWNER.get((importer, name)) if not top_name else None
        a, b = a or destination(importer, top_name), destination(imported, name)
        pa, pb = a.split("/")[0], b.split("/")[0]
        if pa == pb or "plotting" in (pa, pb) or pa == "__init__":
            continue
        if (a, name) in DUCK_TYPED:
            duck.append(f"{a} imports {name} from {b} ({importer}.{top_name or 'module level'})")
            continue
        edges[(pa, pb)] += 1
        if RANK[pb] >= RANK[pa]:
            where = "inside a function" if in_function else "at module level"
            wrong_way.append(f"{a} imports {name} from {b}, {where} ({importer}.{top_name})")
    for (a, b), n in sorted(edges.items()):
        print(f"{a} -> {b}: {n}")
    for line in sorted(set(wrong_way)):
        print(f"  not downward: {line}")
    for line in sorted(set(duck)):
        print(f"  removed by duck typing: {line}")

    print("== Gallery pages by group (build.py TOPICS)")
    build = (ROOT / "docs" / "gallery" / "build.py").read_text()
    tree = ast.parse(build)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TOPICS":
            for group, folder, pages in ast.literal_eval(node.value):
                print(f"{group} ({folder}): {len(pages)} pages: {', '.join(href for href, _ in pages)}")

    print("== Gallery scripts that change folder, and the files that name their path")
    for script, folder in GALLERY_MOVES.items():
        old = f"docs/gallery/{script}.py"
        named = sorted(
            str(path.relative_to(ROOT))
            for path, text in files
            if old in text or f"gallery/{script}.py" in text
        )
        print(f"{old} -> docs/gallery/{folder}/{script.split('/')[1]}.py: {len(named)} files {named}")
    pages = list((ROOT / "docs" / "gallery").glob("*.html"))
    print(f"{len(pages)} HTML pages, all at docs/gallery/<name>.html whatever their folder")


if __name__ == "__main__":
    main()
