from __future__ import annotations

from pathlib import Path


class ModelCache:
    """Shared root for downloaded model weights (local disk or MooseFS mount)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def path_for(self, *parts: str, create_parents: bool = True) -> Path:
        if not parts:
            raise ValueError("path_for requires at least one path part")
        cleaned: list[str] = []
        for part in parts:
            if part is None or part == "":
                raise ValueError("path parts must be non-empty")
            segment = str(part)
            if segment in {".", ".."} or "/" in segment or "\\" in segment:
                raise ValueError(f"invalid path part: {part!r}")
            cleaned.append(segment)

        target = self.root.joinpath(*cleaned).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes model cache root: {target}") from exc

        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def environ(self) -> dict[str, str]:
        hf_home = str(self.path_for("huggingface"))
        torch_home = str(self.path_for("torch"))
        return {
            "MODEL_CACHE_DIR": str(self.root),
            "HF_HOME": hf_home,
            "TRANSFORMERS_CACHE": str(Path(hf_home) / "transformers"),
            "HUGGINGFACE_HUB_CACHE": str(Path(hf_home) / "hub"),
            "TORCH_HOME": torch_home,
        }
