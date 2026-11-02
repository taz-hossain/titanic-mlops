import keyword

from types import ModuleType
from typing import Any


def resolve_name(thing: Any, mdict: dict[str, Any]) -> str:
    """Infer the name that ``thing`` should be known by in ``mdict``.

    ``mdict`` is always the globals of the module where the call appears, never the
    globals of the module where ``thing`` was defined.  The name has to be the one bound
    in ``mdict``, since that's the name ``from <module> import *`` will look for.

    :param thing: A string, a module, or an object with a ``__name__``.
    :param mdict: The globals to resolve the name against.
    :return: The name to add to or remove from ``__all__``.
    :raises TypeError: When no name can be inferred from ``thing``.
    :raises ValueError: When ``thing`` is a string that can't be used as a name.
    """
    if isinstance(thing, str):
        if not thing.isidentifier():
            msg = f'Not a valid Python identifier: {thing!r}'
            raise ValueError(msg)
        # Reserved words pass isidentifier(), but nothing can ever be bound to one, so
        # such a name in __all__ is guaranteed to be dangling.  Soft keywords are
        # ordinary names, so keyword.issoftkeyword() must not be consulted here.
        if keyword.iskeyword(thing):
            msg = f'Cannot use a Python keyword as a name: {thing!r}'
            raise ValueError(msg)
        return thing
    if isinstance(thing, ModuleType):
        # A module's __name__ is its dotted path, and it can be bound under any name at
        # all, so only an identity search can find the local binding.
        if (name := _search(thing, mdict)) is None:
            msg = f'Module is not bound in the calling namespace: {thing.__name__}'
            raise TypeError(msg)
        return name
    if (name := getattr(thing, '__name__', None)) is None:
        msg = f'Cannot infer a name from: {type(thing)}; use the keyword argument form'
        raise TypeError(msg)
    # The object is already bound under its own name.  This covers a local definition as
    # well as the common `from bar import Foo` re-export.
    if mdict.get(name) is thing:
        return name
    # The decorator case, where the object is being defined right here and so isn't bound
    # yet.  This is what keeps @public from paying for the search below.
    if getattr(thing, '__module__', None) == mdict.get('__name__'):
        return name
    # Imported under some other name, so go find the local binding.  Falling back to
    # __name__ keeps the pathological `Bar = Foo; del Foo; public(Bar)` from blowing up.
    found = _search(thing, mdict)
    return name if found is None else found


def _search(thing: Any, mdict: dict[str, Any]) -> str | None:
    for key, value in mdict.items():
        if value is thing:
            return key
    return None
