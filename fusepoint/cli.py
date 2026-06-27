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
    """Launch the bundled Streamlit web UI via `fuse-ui`.

    Forces the dark theme via env vars: the in-page colours
    (#e0e0e0 / #9ca3af on muted-blue boxes) are tuned for a dark
    background; on a light theme they collapse onto white and become
    unreadable. setdefault leaves any user override intact.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "The web UI requires Streamlit. Install with:\n"
            "    pip install 'fusepoint[ui]'",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    env.setdefault("STREAMLIT_THEME_BASE", "dark")
    env.setdefault("STREAMLIT_THEME_BACKGROUND_COLOR", "#0e1117")
    env.setdefault("STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR", "#1a1f2c")
    env.setdefault("STREAMLIT_THEME_TEXT_COLOR", "#e5e7eb")
    env.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#3b82f6")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    here = os.path.dirname(os.path.abspath(__file__))
    web_path = os.path.join(here, "web.py")
    args = [sys.executable, "-m", "streamlit", "run", web_path, *sys.argv[1:]]
    return subprocess.call(args, env=env)


def main() -> None:
    sys.exit(run_streamlit())


if __name__ == "__main__":
    main()
