"""Project root resolution and local ``.env`` loading."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_MARKERS = ("pyproject.toml", "load_env.py")


def resolve_project_dir() -> Path:
    """Find repo root from cwd, notebook path, or script location."""
    starts: list[Path] = [Path.cwd().resolve()]
    for name in ("__vsc_ipynb_file__", "__file__"):
        if name in globals():
            starts.insert(0, Path(globals()[name]).resolve().parent)
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ns = getattr(ip, "user_ns", {})
            for key in ("__vsc_ipynb_file__", "__file__"):
                if key in ns:
                    starts.insert(0, Path(ns[key]).resolve().parent)
    except ImportError:
        pass

    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if all((candidate / name).is_file() for name in _PROJECT_MARKERS):
                return candidate
            if (candidate / "data_access_env_audit.ipynb").is_file():
                return candidate
    return starts[0]


def ensure_project_on_path(project_dir: Path) -> None:
    root = str(project_dir.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def import_project_module(project_dir: Path, module_name: str):
    """Import ``module_name.py`` from the project root (works in notebooks)."""
    module_path = project_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Expected project module at {module_path}")

    ensure_project_on_path(project_dir)
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_project_env(project_dir: Path) -> None:
    """Load ``project_dir/.env`` into ``os.environ`` (existing vars are kept)."""
    load_dotenv(project_dir / ".env", override=False)


def huggingface_token_configured() -> bool:
    return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))


def bootstrap_project() -> Path:
    """Resolve root, load ``.env``, and return ``PROJECT_DIR``."""
    project_dir = resolve_project_dir()
    load_project_env(project_dir)
    ensure_project_on_path(project_dir)
    return project_dir
