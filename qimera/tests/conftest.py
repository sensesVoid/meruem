import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(_REPO_ROOT))

# Make Julia discoverable for juliacall even in terminals opened before install.
_juliaup = Path.home() / ".juliaup" / "bin"
if not shutil.which("julia") and _juliaup.exists():
    os.environ["PATH"] = str(_juliaup) + os.pathsep + os.environ.get("PATH", "")