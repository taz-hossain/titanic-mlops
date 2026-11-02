import builtins

from ._private import private
from ._public import public


def install() -> None:
    """Add the ``@public`` and ``@private`` decorators to :mod:`builtins`.

    Once installed, both decorators are available in every module without importing anything.  The
    companion ``atpublic-install`` package calls this at interpreter startup, but you can call it
    yourself only if you are not using that package.
    """
    # pyrefly: ignore[missing-attribute]
    builtins.public = public
    # pyrefly: ignore[missing-attribute]
    builtins.private = private
