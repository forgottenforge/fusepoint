# FUSE Streamlit entrypoint for Hugging Face Spaces.
#
# The Streamlit app lives inside the installed package at `fusepoint/web.py`
# as module-level imperative code (it issues `st.*` calls at import time),
# so simply importing the module runs the app. Streamlit picks up this file
# via the `app_file: app.py` field in README.md's YAML frontmatter.
#
# Note: on the first launch, Hugging Face Spaces builds the Python environment
# from requirements.txt — this can take a couple of minutes before the UI
# becomes available.

import fusepoint.web  # noqa: F401  -- runs the Streamlit app on import
