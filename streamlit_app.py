"""Streamlit Community Cloud root entrypoint for AIravat."""
from pathlib import Path
import runpy
import sys

# Ensure src/dashboard is on sys.path
dash_dir = Path(__file__).resolve().parent / "src" / "dashboard"
if str(dash_dir) not in sys.path:
    sys.path.insert(0, str(dash_dir))

# Run the main dashboard application
runpy.run_path(str(dash_dir / "app.py"), run_name="__main__")
