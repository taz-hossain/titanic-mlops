"""NFS-safe file locking with timeouts for POSIX and Windows.

The Lock class is the primary API.  Create a Lock instance naming a file system lock file, then
acquire and release it, usually as a context manager.  Its own exceptions all derive from LockError,
and LockState describes what Lock.state infers about a lock file.  Everything public is re-exported
here, so flufl.lock is the only import path any consumer needs.
"""

from flufl.lock._lockfile import (
    AlreadyLockedError,
    Lock,
    LockError,
    LockState,
    NotLockedError,
    SEP,
    TimeOutError,
)


__version__ = '9.2.0'


# These names are re-exports, and a type checker has no way to tell a re-export from an
# implementation detail that merely happens to be imported.  mypy under --strict, and pyright in
# its default mode, reject `from flufl.lock import Lock` for an installed package unless the intent
# is stated, and this list states it.  (pyrefly and ty accept it either way.)
#
# A literal list is what makes that work, which is why this module doesn't use @public: the
# decorator builds __all__ at runtime, where no type checker can see it.  The list is kept honest in
# both directions: ruff's F401 flags an import missing from it, and tests/test_api.py a name in it
# that nothing defines.  (ruff's F822 would catch the latter too, but outside preview it skips
# __init__.py, where an undefined name might be a submodule.)
__all__ = [
    'SEP',
    'AlreadyLockedError',
    'Lock',
    'LockError',
    'LockState',
    'NotLockedError',
    'TimeOutError',
]
