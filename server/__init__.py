"""HTTP / console application shell."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import create_app, main

__all__ = ["create_app", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .app import create_app, main

        return {"create_app": create_app, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
