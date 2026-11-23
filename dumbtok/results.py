"""Write results/*.json, each with a header saying how it was made.

Every number in the episode comes from one of these files. The "meta" header
records when the file was written, on what kind of machine, with which
library versions and settings, and how long it took, so anyone can tell
whether their numbers should match ours.
"""

import importlib.metadata
import json
import os
import platform
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import config

LIBRARIES = ["tiktoken", "tokenizers", "huggingface-hub", "marimo", "torch", "transformers"]


def library_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None  # e.g. torch, if you skipped the models group
    return versions


def machine() -> dict:
    """What kind of computer this ran on (no names or addresses)."""
    info = {"os": platform.platform(), "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    return info


def project_version() -> str:
    with open(config.ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def metadata(settings: dict, seconds: float) -> dict:
    return {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_version": project_version(),
        "libraries": library_versions(),
        "machine": machine(),
        "settings": settings,
        "wall_seconds": round(seconds, 2),
    }


def write(path: Path, data: dict, settings: dict, seconds: float) -> Path:
    """Write data to path as JSON, with a "meta" header first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": metadata(settings, seconds), **data}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n")
    return path


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())
