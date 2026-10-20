"""Intel Arc Models Runner.

Native local model execution engine and OpenAI/Ollama compatible API server
optimized specifically for Intel Arc GPUs (discrete and integrated) and
Intel AI Boost NPUs.
"""

from typing import List
from arc_runner.utils.dll_loader import initialize_windows_dll_environment

initialize_windows_dll_environment()

__version__: str = "0.1.0"
__all__: List[str] = ["__version__", "initialize_windows_dll_environment"]
