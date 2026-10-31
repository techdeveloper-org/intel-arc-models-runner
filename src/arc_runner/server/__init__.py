"""FastAPI server subsystem providing Ollama and OpenAI wire protocol parity."""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from arc_runner.server.app import create_app
    from arc_runner.server.model_manager import ModelManager


def __getattr__(name: str) -> object:
    if name == "create_app":
        from arc_runner.server.app import create_app
        return create_app
    if name == "ModelManager":
        from arc_runner.server.model_manager import ModelManager
        return ModelManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: List[str] = [
    "create_app",
    "ModelManager",
]
