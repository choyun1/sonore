"""Every public function, class, method and property has a docstring, since the
API reference is generated from them. Names still waiting for one are listed in
undocumented.txt; the audit (issue #48) removes them a module at a time."""

import importlib
import inspect
import pkgutil
from pathlib import Path

import sonore

WAITING = Path(__file__).parent / "undocumented.txt"


def _inherited_doc(cls, name):
    """The member's docstring, or the one it overrides in a base class (Sphinx shows that one)."""
    for base in cls.__mro__:
        member = vars(base).get(name)
        if member is None:
            continue
        if isinstance(member, property):
            member = member.fget
        elif isinstance(member, classmethod | staticmethod):
            member = member.__func__
        if getattr(member, "__doc__", None):
            return member.__doc__
    return None


def undocumented_names():
    missing = []
    for info in pkgutil.walk_packages(sonore.__path__, "sonore."):
        module = importlib.import_module(info.name)
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            if not (inspect.isclass(obj) or inspect.isfunction(obj)) or obj.__module__ != module.__name__:
                continue
            qualified = f"{module.__name__}.{name}"
            if not obj.__doc__:
                missing.append(qualified)
            if inspect.isclass(obj):
                for member_name, member in vars(obj).items():
                    if member_name.startswith("_"):
                        continue
                    is_api = isinstance(member, property | classmethod | staticmethod) or callable(member)
                    if is_api and not _inherited_doc(obj, member_name):
                        missing.append(f"{qualified}.{member_name}")
    return sorted(missing)


def test_public_names_have_docstrings():
    waiting = {line.strip() for line in WAITING.read_text().splitlines() if line.strip() and line[0] != "#"}
    missing = set(undocumented_names())
    assert not missing - waiting, f"public names without a docstring: {sorted(missing - waiting)}"
    assert not waiting - missing, f"documented now, remove from undocumented.txt: {sorted(waiting - missing)}"
