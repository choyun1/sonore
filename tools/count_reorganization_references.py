"""Counts what the reorganization proposed in docs/design/reorganization.md would touch.

For every module the proposal moves, it counts the files in the repository that name the
module's current path, grouped by where they live, and the lines of the module itself. It
also lists the gallery pages by group as build.py's TOPICS has them, and which modules a
released version (the git tag given with --release) already shipped, since only those deep
import paths can be in anyone's code. Nothing is changed.

    python tools/count_reorganization_references.py [--release v0.3.0]

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

# Recommended option (A): current module -> proposed module. A split module has
# several targets, separated by " + ".
MOVES_A = {
    "analysis/f0": "voice/f0",
    "analysis/vocoder": "voice/world + voice/aperiodicity",
    "stimuli/vocoder": "voice/world",
    "analysis/voice": "analysis/spectral_envelope + voice/change",
    "signals/glottal": "voice/glottal",
    "stimuli/klatt": "voice/klatt",
    "stimuli/binaural": "spatial/binaural",
    "stimuli/spatialization": "spatial/spatialization",
    "stimuli/hrir_data": "spatial/hrir_data",
    "stimuli/reverb": "spatial/reverb",
    "stimuli/phasevocoder": "analysis/phasevocoder",
}
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
        if path.name == Path(__file__).name:
            continue
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
    """Module-level and in-function imports between sonore modules, as (importer, imported)."""
    edges = set()
    for path in SRC.rglob("*.py"):
        name = ".".join(path.relative_to(SRC.parent).with_suffix("").parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sonore."):
                edges.add((name, node.module))
    return edges


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

    print("== Imports between proposed subpackages (option A)")
    rename = {m.replace("/", "."): t.split(" + ")[0].replace("/", ".") for m, t in MOVES_A.items()}
    rename["analysis.voice"] = "analysis.spectral_envelope"  # what cepstrum and mfcc import (GridEnvelope)

    def package(module):
        module = module.removeprefix("sonore.")
        module = rename.get(module, module)
        return module.split(".")[0]

    crossing = Counter()
    upward = []
    for importer, imported in sorted(module_imports()):
        a, b = package(importer), package(imported)
        if a != b and "plotting" not in (a, b):
            crossing[(a, b)] += 1
            if (a, b) in {("analysis", "voice"), ("core", "analysis")}:
                upward.append(f"{importer} imports {imported}")
    for (a, b), n in sorted(crossing.items()):
        print(f"{a} -> {b}: {n}")
    # analysis/voice.py is split, and this count treats the whole file as its first part;
    # these are the imports to look at by hand.
    for line in upward:
        print(f"  upward: {line}")

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
