# Migrate `tracking-markers` from Poetry to uv

**Date:** 2026-05-27
**Status:** Approved

## Goal

Replace Poetry with [uv](https://docs.astral.sh/uv/) as the project's package
manager and build system, with no change to the package's runtime behavior,
public API, dependency semantics, or PyPI distribution.

## Context

`tracking-markers` is a published PyPI package (v0.9.0) using Poetry with the
`poetry-core` build backend.

- **Runtime deps:** numpy, opencv-python, tqdm
- **Dev dep:** pytest
- **Python:** `>=3.9,<3.13`
- **CLI entry point:** `tracking-markers` → `tracking_markers.tracking_points:main`
- **Layout:** flat — the `tracking_markers/` package sits at the repo root (not under `src/`)
- **Version** is currently duplicated in `pyproject.toml` and `tracking_markers/__init__.py`
- No CI / `.github` workflows exist; publishing to PyPI is manual
- `uv 0.11.7` is installed locally

## Decisions

1. **Build backend:** `uv_build` (uv's native backend — ships with uv, fast,
   minimal, no extra build dependency).
2. **Versioning:** single source of truth. `version` lives only in
   `pyproject.toml`; `__init__.py` reads it at runtime via
   `importlib.metadata`. (`uv_build` does not support reading a dynamic version
   from `__init__.py`, so the static-version-in-pyproject + importlib.metadata
   pattern is the clean option.)
3. **Scope:** migration only. No CI workflow is added; publishing stays manual.
4. **`uv.lock`** is committed to git for reproducible development.
5. README gets a short "Development" section documenting the new local commands.

## Changes

### 1. Replace `pyproject.toml`

```toml
[project]
name = "tracking-markers"
version = "0.9.0"
description = "A humble image tracking code"
authors = [{ name = "Giovanni Bordiga", email = "gbordiga@seas.harvard.edu" }]
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
keywords = ["image-tracking", "opencv"]
requires-python = ">=3.9,<3.13"
dependencies = [
    "numpy>=1.26.1,<2",
    "opencv-python>=4.8.1.78,<5",
    "tqdm>=4.67.1,<5",
]

[project.urls]
Homepage = "https://github.com/bertoldi-collab/tracking-markers"
Repository = "https://github.com/bertoldi-collab/tracking-markers"

[project.scripts]
tracking-markers = "tracking_markers.tracking_points:main"

[dependency-groups]
dev = ["pytest>=7.4.3,<8"]

[tool.uv.build-backend]
module-root = ""

[build-system]
requires = ["uv_build>=0.11.7,<0.12.0"]
build-backend = "uv_build"
```

Conversion notes:

- Poetry caret ranges → explicit PEP 440 ranges with identical semantics:
  `^1.26.1` → `>=1.26.1,<2`, `^4.8.1.78` → `>=4.8.1.78,<5`, `^4.67.1` →
  `>=4.67.1,<5`, dev `^7.4.3` → `>=7.4.3,<8`.
- `[tool.poetry.group.dev.dependencies]` → PEP 735 `[dependency-groups]` `dev`
  (installed by default by `uv sync`).
- `packages = [{ include = "tracking_markers" }]` → `module-root = ""` (flat
  layout). Module name auto-normalizes from `tracking-markers` to
  `tracking_markers`, so `module-name` need not be set.
- `homepage` / `repository` → `[project.urls]`.

### 2. `tracking_markers/__init__.py`

```python
from importlib.metadata import version

__version__ = version("tracking-markers")
```

### 3. File operations

- Delete `poetry.lock`.
- Run `uv lock` to generate `uv.lock`; commit it.
- Run `uv sync` to create `.venv`.
- Verify `.gitignore` ignores `.venv/`; add the entry if missing.

### 4. README "Development" section

Add a concise section documenting the new local workflow (existing install
instructions stay valid — `pip install -e .` still works because pip builds via
`uv_build`):

- `uv sync` — set up the environment
- `uv run pytest` — run tests
- `uv build` — build sdist + wheel
- `uv publish` — publish to PyPI

## Verification

Before declaring the migration complete:

- `uv sync` succeeds.
- `uv run pytest` passes (same result as before migration).
- `uv build` produces a wheel + sdist in `dist/`.
- `uv run python -c "import tracking_markers; print(tracking_markers.__version__)"`
  prints `0.9.0`.
- `uv run tracking-markers --help` works (CLI entry point intact).

## Out of scope

- CI / GitHub Actions publish workflow.
- Changing the supported Python range or any dependency versions.
- Any change to package source code beyond `__init__.py`'s version lookup.
