# Releasing sonore

Releases are built and uploaded by `.github/workflows/release.yml`. No password
or API token is stored anywhere: PyPI's *trusted publishing* lets PyPI accept an
upload because it comes from this repository's release workflow, running in a
named GitHub environment.

## One-time setup

1. On [PyPI](https://pypi.org/manage/account/publishing/) and on
   [TestPyPI](https://test.pypi.org/manage/account/publishing/) (separate accounts),
   add a *pending publisher*: project `sonore`, owner `choyun1`, repository `sonore`,
   workflow `release.yml`, environment `pypi` (on PyPI) or `testpypi` (on TestPyPI).
   The project is created by the first upload.
2. In the repository settings, under Environments, create `pypi` and `testpypi`.
   Adding yourself as a required reviewer on `pypi` means every real release
   waits for one click.

## Each release

1. Set the version in `src/sonore/__init__.py` and move the `Unreleased` notes in
   `CHANGELOG.md` under a heading for that version. Merge to `main`.
2. Trial run on TestPyPI: Actions, *release*, *Run workflow* on `main`. It uploads
   whatever version `main` holds. TestPyPI and PyPI are separate, so a version
   uploaded to TestPyPI can still go to PyPI, but neither accepts the same version
   twice. If a trial needs a second attempt, run the workflow from a branch whose
   version is a release candidate such as `0.3.0rc1`. Check it installs in a fresh
   environment (the extra index lets pip fetch numpy and the other dependencies
   from PyPI):

   ```
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "sonore==0.3.0"
   ```

3. Real release: on GitHub, *Releases*, *Draft a new release*, tag `v0.3.0` (it must
   match the version exactly, or the workflow stops), paste the changelog
   section, *Publish release*. The workflow uploads to PyPI.
