# https://docs.astral.sh/ruff/rules/future-rewritable-type-annotation/
#
# 2024-05-02(bwarsaw): We must ignore I001 on this line or the ruff formatter
# and linter will be in conflict between I001 and F404 (which wants to move
# this import to below `import sys`.  Ruff's unified linter and formatter
# will hopefully resolve this: https://github.com/astral-sh/ruff/issues/8232
from __future__ import annotations  # noqa: I001

import sys

from typing import Any, TYPE_CHECKING, overload

from ._resolve import resolve_name

if TYPE_CHECKING:
    from ._types import ModuleAware


# fmt: off
@overload
def public(thing: ModuleAware) -> ModuleAware:
    ...


@overload
def public(**kws: Any) -> Any | tuple[Any]:
    ...


# fmt: on
def public(thing: Any | None = None, **kws: Any) -> ModuleAware | Any | tuple[Any]:
    """Add a name or names to __all__.

    The name is always added to the ``__all__`` of the module where this call
    appears, creating that list if necessary.  The ``__all__`` of the module
    where ``thing`` happens to have been defined is never touched.

    Most commonly this is used as a decorator on a class or function at module
    scope, in which case ``thing`` is the object being defined and its
    ``__name__`` is the name that gets added.

    It can also be called directly with a single positional argument, which
    may be an object imported from another module, or a module or submodule.
    The name added is the one that object is bound to in the calling module's
    globals, since that's the name ``from <module> import *`` has to find.  So
    given ``from bar import Foo as Baz``, ``public(Baz)`` adds ``'Baz'``.

    That single argument may instead be a string, which is added to ``__all__``
    as-is.  It must be a valid Python identifier and must not be a reserved
    word, since nothing can ever be bound to one of those; soft keywords such
    as ``match`` and ``type`` are ordinary names and are accepted.  Beyond
    that, nothing checks a string against the module's contents, so the name
    need not be bound, or even exist.  This is the escape hatch for names that
    don't appear in the source, such as bindings made dynamically or
    re-exports guarded by ``try``/``except ImportError``.  Reach for it last;
    a string is precisely the kind of thing that goes stale, which is what
    this library exists to prevent.

    The other call form takes keyword arguments.  Each key is added to
    ``__all__`` and bound to its value in the calling module's globals.  This
    form returns the keyword argument values in order; if only a single
    keyword argument is given its value is returned, otherwise a tuple of the
    values is returned.  Use it for constants and instances, which have no
    ``__name__`` to infer a name from.

    Only one or the other format may be used.

    :param thing: None, a string, a module, or an object with a ``__name__``.
    :param kws: Keyword arguments.
    :return: In the decorator and single argument forms, the original
        ``thing`` object is returned.  In the keyword argument form, the
        keyword argument value is returned if only a single keyword argument
        is given, otherwise a tuple of the keyword argument values is
        returned.
    :raises TypeError: When no name can be inferred from ``thing``, or this
        function finds a non-list ``__all__`` attribute.
    :raises ValueError: When ``thing`` is a string that can't be used as a
        name, i.e. it isn't an identifier, or it's a reserved word.
    """
    # 2020-07-14(warsaw): I considered using inspect.getmodule() here but looking at its
    # implementation, I feel like it does a ton of unnecessary work in the oddball cases
    # (i.e. where the object does not have an __module__ attribute).  Because @public runs
    # at module import time, and because I'm not really sure we even want to support those
    # oddball cases, I'm taking the more straightforward approach.
    #
    # 2026-08-16(warsaw): That straightforward approach used to be sys.modules[thing.__module__],
    # i.e. the module where thing was *defined*.  That's the wrong module for anything imported from
    # elsewhere, so every form now reads the caller's frame globals instead.  The note above still
    # applies to resolve_name(): this runs at import time, so name resolution has to stay cheap.
    #
    # 2026-08-31(warsaw): I considered switching to inspect.currentframe().f_back instead of
    # sys._getframe(1) -- note that currentframe() alone is sys._getframe(0), i.e. our own frame,
    # so the naive swap would silently return this module's globals rather than the caller's.  It
    # doesn't buy us much, and it does incur the cost of importing the entire inspect module for
    # just one function.  The other consideration is that inspect.currentframe() returns None in
    # interpreters without sys._getframe(), so we'd either have to test for that value (and then
    # what? raise a RuntimeError instead of the AttributeError that'll result for None.f_globals?).
    # We'll get that anywhere here in non-conforming interpreters, and the direct call's
    # "module 'sys' has no attribute '_getframe'" is the more informative message anyway.
    mdict = sys._getframe(1).f_globals
    dunder_all = mdict.setdefault('__all__', [])
    if not isinstance(dunder_all, list):
        # https://docs.astral.sh/ruff/rules/f-string-in-exception/
        msg = f'__all__ must be a list not: {type(dunder_all)}'
        raise TypeError(msg)
    if thing is None:
        # The keyword argument form.
        retval = []
        for key, value in kws.items():
            # This overwrites any previous similarly named __all__ entry.
            if key not in dunder_all:
                dunder_all.append(key)
            # We currently do not check for duplications in the globals.
            mdict[key] = value
            retval.append(value)
        if len(retval) == 1:
            return retval[0]
        return tuple(retval)
    # I think it's impossible to use the @public decorator and pass in keyword
    # arguments.  Not quite syntactically impossible, but you'll get a
    # TypeError if you try it, before you even get to this code.
    assert len(kws) == 0, 'Keyword arguments are incompatible with use as decorator'
    name = resolve_name(thing, mdict)
    if name not in dunder_all:
        dunder_all.append(name)
    return thing
