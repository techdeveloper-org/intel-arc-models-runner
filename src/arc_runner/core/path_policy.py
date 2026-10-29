"""Path and name policy for anything that reaches the filesystem from a request or a model name.

Model names are flat directory names. Filesystem paths taken from requests must lie below an allowed root:
the models directories plus any directories listed in ARC_ALLOWED_PATHS (os.pathsep separated).
"""

import os
from pathlib import Path
import re
from typing import Iterable, List, Union

from arc_runner.core.exceptions import ArcRunnerError

ALLOWED_PATHS_ENV = "ARC_ALLOWED_PATHS"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def validate_model_name(name: str) -> str:
    """Accept only an ordinary flat directory name as a model name.

    Args:
        name: Candidate model name (an optional ":tag" suffix must be removed by the caller).

    Returns:
        str: The stripped name.

    Raises:
        ArcRunnerError: If the name is empty, too long, absolute, contains a separator, NUL, "..", or does not
            start with a letter or digit.
    """
    if not isinstance(name, str):
        raise ArcRunnerError("Model name must be a string", "INVALID_MODEL_NAME", 400)
    cleaned = name.strip()
    if not _MODEL_NAME.match(cleaned) or ".." in cleaned:
        raise ArcRunnerError(f"Invalid model name {cleaned[:40]!r}", "INVALID_MODEL_NAME", 400)
    return cleaned


def resolve_inside(base: Path, name: str) -> Path:
    """Resolve a validated model name to a path strictly below a base directory.

    Symbolic links are resolved, so a link inside the base that points outside is rejected.

    Args:
        base: Base directory.
        name: Model name or relative child name.

    Returns:
        Path: Resolved path below base.

    Raises:
        ArcRunnerError: If the name is invalid or the resolved path is not strictly below base.
    """
    base_resolved = Path(base).resolve()
    candidate = (base_resolved / validate_model_name(name)).resolve()
    if candidate == base_resolved or base_resolved not in candidate.parents:
        raise ArcRunnerError("Path escapes the models directory", "PATH_NOT_ALLOWED", 400)
    return candidate


def allowed_roots(extra: Iterable[Union[str, Path]] = ()) -> List[Path]:
    """Return the directories request-supplied paths may point into.

    Args:
        extra: Directories always allowed (for example the models directories).

    Returns:
        List[Path]: Resolved roots: extra plus ARC_ALLOWED_PATHS entries.
    """
    roots = [Path(item).resolve() for item in extra]
    for entry in os.environ.get(ALLOWED_PATHS_ENV, "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry.strip()).resolve())
    return roots


def ensure_path_allowed(path: Union[str, Path], roots: Iterable[Path]) -> Path:
    """Resolve a path and require it to lie below one of the roots.

    Args:
        path: Path from a request.
        roots: Allowed roots.

    Returns:
        Path: The resolved path.

    Raises:
        ArcRunnerError: If the path is outside every root.
    """
    resolved = Path(path).resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    raise ArcRunnerError("Path is outside the allowed directories (set ARC_ALLOWED_PATHS to add one)", "PATH_NOT_ALLOWED", 403)
