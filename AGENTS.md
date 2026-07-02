# Repository Guidelines

## Project Structure & Module Organization

MemEvo is an early-stage Python 3.12 project for evaluating conversational memory algorithms.

- `main.py` is the current executable entry point.
- `src/algorithms/` contains memory implementations. Shared interfaces belong in `base/`; each algorithm should have its own subpackage.
- `src/datasets/` contains dataset loaders and evaluation logic, including LoCoMo support.
- `src/utils/` provides LLM clients, run helpers, and serialization utilities.
- `src/configs/` stores example TOML run configurations. Keep generated results under `runs/` and datasets under `data/`; do not commit large artifacts.

There is no test directory yet. Add tests under `tests/`, mirroring source paths (for example, `tests/datasets/test_locomo.py`).

## Build, Test, and Development Commands

Create an isolated Python 3.12 environment before development:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py
python -m compileall main.py src
```

`pip install -e .` installs the project for iterative development; `python main.py` runs the placeholder entry point; `compileall` performs a quick syntax check. Runtime imports currently include `openai` and `pydantic`, but dependencies and a formal test command are not yet fully configured in `pyproject.toml`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, explicit type hints, and `pathlib.Path` for filesystem work. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and leading underscores for private helpers. Keep async boundaries explicit for network operations. Prefer small dataclasses or Pydantic models for structured data. No formatter is configured; format consistently and remove trailing whitespace before submitting.

## Testing Guidelines

Use `pytest` when introducing the test infrastructure. Name files `test_*.py` and tests `test_<behavior>`. Cover parsing edge cases, missing files, async client behavior, and algorithm ingest/retrieve/reset contracts. Mock external LLM calls; unit tests must not require API keys or network access.

## Commit & Pull Request Guidelines

This repository has no commit history yet. Use short, imperative subjects such as `Add LoCoMo timestamp tests`, optionally with Conventional Commit prefixes (`feat:`, `fix:`, `test:`). Pull requests should explain the change, list validation commands, link relevant issues, and include sample output for benchmark or configuration changes. Keep unrelated refactors separate.

## Security & Configuration

Copy `.env.example` to `.env` and populate keys locally. Never commit credentials, private service URLs, datasets, or generated memory/output files. Reference secrets through environment-variable names in example TOML files.
