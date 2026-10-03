"""Every view refuses to synthesize, says what it drops, and has to_sound only where a route back exists."""

import pytest

import sonore as so
from sonore.views.view import NotInvertibleError, View

VIEW_NAMES = [
    "Spectrum",
    "TFPower",
    "ReassignedSpectrogram",
    "Envelope",
    "Envelopes",
    "ModulationSpectrum",
    "ModulationSpectrogram",
    "Cepstrum",
    "MFCC",
    "GridEnvelope",
    "SpectralEnvelope",
    "Aperiodicity",
    "Mask",
    "F0Track",
    "InterauralCues",
    "TextureStats",
    "PVAnalysis",
]


def _public_subclasses(cls):
    for subclass in cls.__subclasses__():
        if not subclass.__name__.startswith("_"):
            yield subclass
        yield from _public_subclasses(subclass)


# Every other public class, by kind. A new class has to be put in one of
# these on purpose, so a one-way analysis cannot arrive without being a View.
FRAME_SIDE = ["Frame", "Filterbank", "STFT", "TVSTFT", "Subbands"]  # with their subclasses
NOT_ANALYSES = [
    "Sound",
    "Decibels",
    "set_fft_workers",
    "Ripple",
    "DynamicRipple",
    "RippleSum",
    "HRIRSet",
    "TextureModel",
    "ModulationFilterbank",
]  # ModulationFilterbank: a tool that makes views


def _public_classes():
    namespaces = [(so, so.__all__), (so.texture, so.texture.__all__)]
    return {
        name: getattr(module, name)
        for module, names in namespaces
        for name in names
        if isinstance(getattr(module, name), type) and not issubclass(getattr(module, name), BaseException)
    }


def test_every_analysis_that_is_not_a_frame_is_a_view():
    assert {view.__name__ for view in _public_subclasses(View)} == set(VIEW_NAMES)
    classes = _public_classes()
    bases = tuple(classes[name] for name in FRAME_SIDE + NOT_ANALYSES)
    unsorted = [name for name, cls in classes.items() if not issubclass(cls, (View, *bases))]
    assert unsorted == [], f"make these frames or views: {unsorted}"


@pytest.mark.parametrize("view", list(_public_subclasses(View)), ids=lambda view: view.__name__)
def test_view_says_what_it_drops_and_refuses(view):
    assert view.discards.strip(), f"{view.__name__} has no discards sentence"
    assert view.discards.startswith(view.__name__)
    assert view.discards.endswith(".")
    assert view.back_to_sound == "" or view.back_to_sound.endswith(".")
    # synthesize needs no state to refuse, so an instance made without __init__ will do
    instance = object.__new__(view)
    with pytest.raises(NotInvertibleError) as refusal:
        instance.synthesize()
    assert view.discards in str(refusal.value)
    assert (view.back_to_sound or "no canonical route") in str(refusal.value)


def test_refusal_is_a_not_implemented_error():
    assert issubclass(NotInvertibleError, NotImplementedError)
    assert so.NotInvertibleError is NotInvertibleError and so.View is View


def test_frames_and_tools_are_not_views():
    for name in ["Frame", "GaborFrame", "Filterbank", "STFT", "Subbands", "ModulationFilterbank"]:
        assert not issubclass(getattr(so, name), View), name


# Views with a canonical route back to a sound; every other view refuses to_sound.
ROUTES_BACK = {"Spectrum", "Cepstrum", "PVAnalysis"}


@pytest.mark.parametrize("view", list(_public_subclasses(View)), ids=lambda view: view.__name__)
def test_to_sound_exists_only_where_a_route_does(view):
    if view.__name__ in ROUTES_BACK:
        assert view.to_sound is not View.to_sound
        assert f"{view.__name__}.to_sound" in view.back_to_sound
        return
    assert view.to_sound is View.to_sound, f"{view.__name__}.to_sound: add it to ROUTES_BACK on purpose"
    instance = object.__new__(view)
    with pytest.raises(NotInvertibleError) as refusal:
        instance.to_sound()
    assert view.discards in str(refusal.value)
    assert (view.back_to_sound or "no canonical route") in str(refusal.value)
