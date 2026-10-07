"""ReproFix: an autonomous reproduce -> diagnose -> repair -> verify agent for broken ML repos."""

__version__ = "0.1.0"

import os
import sys

# Ensure stdout and stderr handle UTF-8 cleanly on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if os.name == "nt":
    for _extra in [r"C:\Program Files\Git\usr\bin", r"C:\Program Files (x86)\Git\usr\bin"]:
        if os.path.exists(_extra) and _extra not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")
