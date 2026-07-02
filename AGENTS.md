# Repository Guidelines

## Project Structure & Module Organization

MemEvo is an early-stage Python 3.12 project for evaluating conversational memory algorithms.

- `main.py` is the current executable entry point.
- `src/memevo/algorithms/` contains memory implementations. Shared interfaces belong in `base/`; each algorithm should have its own subpackage.
- `src/memevo/datasets/` contains dataset loaders and evaluation logic, including LoCoMo support.
- `src/memevo/utils/` provides LLM clients, run helpers, and serialization utilities.
- `src/configs/` stores example TOML run configurations. Keep generated results under `runs/` and datasets under `data/`; do not commit large artifacts.

Tests live under `tests/`; mirror source areas when adding coverage (for example, `tests/test_locomo.py`).

## Build, Test, and Development Commands

Create an isolated Python 3.12 environment before development:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
memevo --config src/configs/full_context.example.toml
python -m compileall main.py src
```

`pip install -e .` installs the project for iterative development; `memevo` runs a configured benchmark; `compileall` performs a quick syntax check. Prefer `uv sync` when uv is available.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, explicit type hints, and `pathlib.Path` for filesystem work. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and leading underscores for private helpers. Keep async boundaries explicit for network operations. Prefer small dataclasses for structured data. Run `uv run ruff check .` and `uv run ruff format --check .` before submitting.

## Testing Guidelines

Use `pytest`; run it with `uv run pytest`. Name files `test_*.py` and tests `test_<behavior>`. Cover parsing edge cases, missing files, async client behavior, and algorithm ingest/retrieve/reset contracts. Mock external LLM calls; unit tests must not require API keys or network access.

## Commit & Pull Request Guidelines

This repository has no commit history yet. Use short, imperative subjects such as `Add LoCoMo timestamp tests`, optionally with Conventional Commit prefixes (`feat:`, `fix:`, `test:`). Pull requests should explain the change, list validation commands, link relevant issues, and include sample output for benchmark or configuration changes. Keep unrelated refactors separate.

## Security & Configuration

Copy `.env.example` to `.env` and populate keys locally. Never commit credentials, private service URLs, datasets, or generated memory/output files. Reference secrets through environment-variable names in example TOML files.
