import sys

from ._resolve import resolve_name
from ._types import ModuleAware


def private(thing: ModuleAware) -> ModuleAware:
    """Remove names from __all__.

    This decorator documents private names and ensures that the names do not
    appear in the module's __all__.

    The name is always removed from the ``__all__`` of the module where this
    call appears.  The ``__all__`` of the module where ``thing`` happens to
    have been defined is never touched.

    Unlike ``public()``, this never creates an ``__all__``.  If the module
    doesn't have one there is nothing to remove the name from, and an empty
    ``__all__`` is not the same thing as no ``__all__`` at all: the first
    exports nothing, while the second exports every name that doesn't start
    with an underscore.  The argument is still checked, so passing something
    no name can be inferred from raises whether or not the module has an
    ``__all__``.

    Like ``public()``, this can be called directly with a single positional
    argument instead of used as a decorator, and the name is resolved the same
    way: an object imported from another module, or a module or submodule,
    resolves to the name it is bound to in the calling module's globals, while
    a string must be a valid Python identifier that isn't a reserved word, and
    is otherwise taken as given.  There is no keyword argument form.

    :param thing: A string, a module, or an object with a ``__name__``.
    :return: The original `thing` object.
    :raises TypeError: When no name can be inferred from ``thing``, or this
        function finds a non-list ``__all__`` attribute.
    :raises ValueError: When ``thing`` is a string that can't be used as a
        name, i.e. it isn't an identifier, or it's a reserved word.
    """
    # 2026-08-31(warsaw): See the comment in public() for why we use sys._getframe() explicitly here.
    mdict = sys._getframe(1).f_globals
    if '__all__' not in mdict:
        # Unlike public(), private() never creates an __all__.  There would be nothing to
        # remove the name from, and an empty __all__ is not the same thing as no __all__
        # at all: the former exports nothing, while the latter exports every name that
        # does not begin with an underscore.
        #
        # The name is still resolved here, and the result deliberately discarded, so that
        # an argument this function cannot make sense of is rejected whether or not the
        # module happens to have an __all__ yet.  Otherwise a bad argument would sit
        # unnoticed until something else in the module created an __all__.
        resolve_name(thing, mdict)
        return thing
    dunder_all = mdict['__all__']
    if not isinstance(dunder_all, list):
        # https://docs.astral.sh/ruff/rules/f-string-in-exception/
        msg = f'__all__ must be a list not: {type(dunder_all)}'
        raise TypeError(msg)
    name = resolve_name(thing, mdict)
    if name in dunder_all:
        dunder_all.remove(name)
    return thing
