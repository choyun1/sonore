# Contributing to sonore

Thank you for looking. sonore has one maintainer, Adrian Cho, and is developed
with AI assistance (see "How sonore was developed" in the README). Issues and
pull requests are welcome.

## Reporting a problem

Open an [issue](https://github.com/choyun1/sonore/issues). The most useful
reports include:

- the sonore, NumPy and SciPy versions (`so.__version__`, `np.__version__`,
  `scipy.__version__`) and your operating system;
- a short script that shows the problem, ideally one that makes its sound
  with a fixed `rng=` so the output can be compared;
- what you expected and what you got. For a numerical claim, say where the
  expected value comes from (a paper, another implementation, a hand
  calculation).

Wrong numbers, wrong citations and wrong wording in the gallery are all
bugs.

## Asking a question or proposing a feature

Open an issue as well. sonore's scope is the sound side of hearing
research: making stimuli and representing them in ways that turn back into
sound. Listener models and psychophysics (thresholds, loudness, periphery
models) are out of scope for now; the README's "Related projects" lists
packages that cover them.

## Making a change

```bash
pip install -e ".[dev]"
python -m pytest                   # one test file per module
ruff check . && ruff format .
```

Work on a branch and open a pull request against `main`. The working rules
are in [docs/design/philosophy.md](docs/design/philosophy.md). The ones that
matter most for a pull request:

- **Design document first** for anything larger than a fix. It goes in
  `docs/design/<folder>/`, lists its decisions with the options, and is
  agreed before the code is written.
- **Claims are checked.** A mathematical claim in a docstring or design
  document gets a test, and where it helps, a standalone checker in `tools/`
  that shares no code with the library.
- **No silent changes.** A change that alters existing output says so in
  `CHANGELOG.md`, with how large the change is. Texture synthesis output stays
  bit for bit identical.
- **Readable names** and American English spelling, in code, comments and
  docs.
- Gallery pages are built by `python docs/gallery/build.py`. Commit
  re-rendered audio and figures only for pages whose code changed.

If you used an AI assistant, say so in the pull request.

## What to expect

sonore is maintained in spare time, on a best-effort basis. There is no
guaranteed response time, though issues are usually read within a week or
two. Bug reports with a reproducing script get priority. Until 1.0, a minor
release (0.x.0) may change import paths or behavior, as described in
`CHANGELOG.md`.

## License

By contributing you agree that your contribution is released under the
project's [MIT license](LICENSE).
