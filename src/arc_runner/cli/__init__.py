"""Command-line interface module for Intel Arc Models Runner."""

from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Delegate CLI execution to arc_runner.cli.main.

    Args:
        argv: Optional command-line argument list.

    Returns:
        int: Process exit code.
    """
    from arc_runner.cli.main import main as cli_entrypoint
    return cli_entrypoint(argv)


__all__: List[str] = ["main"]
