from typing import TypeVar


# 2026-08-16(warsaw): This used to be bound to Callable[..., Any], which was fine when the only
# things you could hand to public() and private() were the functions and classes you can decorate.
# The single argument call form also takes modules and strings, and neither of those is callable, so
# the bound had to go.
ModuleAware = TypeVar('ModuleAware')
