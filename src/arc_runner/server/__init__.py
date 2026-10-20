"""FastAPI server subsystem providing Ollama and OpenAI wire protocol parity."""

from typing import List
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager

__all__: List[str] = [
    "create_app",
    "ModelManager",
]
