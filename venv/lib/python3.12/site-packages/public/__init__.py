"""Decorators that keep a module's __all__ in sync with its public names.

Two decorators document the visibility of a name where the name is defined, rather than in a list
kept somewhere else that has to be maintained by hand:

* @public adds the decorated name to the module's __all__.
* @private keeps it out, and removes it if something else already added it.

populate_all() is the bulk alternative: called at the bottom of a module, it infers the public names
and fills in __all__ for you.

install() puts public and private into builtins so they don't need explicit imports in any module
that uses them.

This module is a hub -- everything here is re-exported from a private implementation module, and
__all__ below states that intent for type checkers.  You should always import these names from here.

Full documentation: https://public.readthedocs.io
"""

from ._modules import populate_all
from ._private import private
from ._public import public
from ._startup import install


__version__ = '9.0.0'


# 2026-08-18(warsaw): Irony alert!
#
# These names are re-exports, and a type checker has no way to tell a re-export from an
# implementation detail that merely happens to be imported.  Under a strict configuration mypy
# rejects "from public import public" unless the intent is stated, and this list states it.  It
# matters for people type checking their own code against this package, not for us; pyrefly, which
# is what this project runs, is happy either way.
#
# Using our own decorator here instead would be nicer, but neither call form manages it.  Unpacking
# the keyword form's return value loses every type, since that overload can only return Any.  The
# single argument form does keep the types, but mypy does not accept an assignment as an explicit
# export, so this list would still be needed.
__all__ = [
    'install',
    'populate_all',
    'private',
    'public',
]
