"""Direct executable entry point for python -m arc_runner.

Enables invocation via python -m arc_runner without requiring binary PATH resolution.
"""

from arc_runner.cli.main import main


if __name__ == "__main__":
    main()
