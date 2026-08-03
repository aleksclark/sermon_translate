from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PRELOAD_NAMES = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcusparse.so.12",
    "libcusparseLt.so.0",
    "libcudnn.so.9",
    "libnccl.so.2",
)


def nvidia_lib_dirs() -> list[Path]:
    roots: list[Path] = []
    for entry in sys.path:
        candidate = Path(entry) / "nvidia"
        if candidate.is_dir():
            roots.append(candidate)
    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    if site.is_dir():
        roots.append(site)

    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for lib_dir in root.glob("*/lib"):
            resolved = lib_dir.resolve()
            if resolved.is_dir() and resolved not in seen:
                seen.add(resolved)
                dirs.append(resolved)
    return dirs


def ensure_nvidia_library_path() -> str:
    """Expose wheel-shipped NVIDIA .so dirs to this process and children."""
    dirs = [str(path) for path in nvidia_lib_dirs()]
    if not dirs:
        return os.environ.get("LD_LIBRARY_PATH", "")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = dirs + ([p for p in existing.split(":") if p] if existing else [])
    # de-dupe preserving order
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    joined = ":".join(ordered)
    os.environ["LD_LIBRARY_PATH"] = joined

    # Setting LD_LIBRARY_PATH after process start does not affect this process's
    # dynamic linker. Preload critical libs so torch can resolve them.
    by_name: dict[str, Path] = {}
    for lib_dir in nvidia_lib_dirs():
        for so in lib_dir.glob("lib*.so*"):
            by_name.setdefault(so.name, so)
            # also index bare soname without extra version suffixes after .so.N
            name = so.name
            if ".so." in name:
                base = name.split(".so.")[0] + ".so." + name.split(".so.")[1].split(".")[0]
                by_name.setdefault(base, so)

    for name in _PRELOAD_NAMES:
        path = by_name.get(name)
        if path is None:
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.debug("nvidia preload skipped %s: %s", path, exc)

    return joined
