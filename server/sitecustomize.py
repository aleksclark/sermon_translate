"""Auto-loaded by Python site startup when this directory is on sys.path."""

from __future__ import annotations

try:
    from src.runtime.nvidia_libs import ensure_nvidia_library_path

    ensure_nvidia_library_path()
except Exception:
    pass
