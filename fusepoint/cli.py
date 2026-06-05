"""
Command-line entry points for fusepoint.

Installed scripts (declared in pyproject.toml [project.scripts]):
- fuse-ui      launches the Streamlit web app
"""
from __future__ import annotations
import os
import sys
import subprocess


def run_streamlit() -> int:
    """Launch the bundled Streamlit web UI via `fuse-ui`."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "The web UI requires Streamlit. Install with:\n"
            "    pip install 'fusepoint[ui]'",
            file=sys.stderr,
        )
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    web_path = os.path.join(here, "web.py")
    args = [sys.executable, "-m", "streamlit", "run", web_path, *sys.argv[1:]]
    return subprocess.call(args)


def main() -> None:
    sys.exit(run_streamlit())


if __name__ == "__main__":
    main()
