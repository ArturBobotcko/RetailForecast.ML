# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python 3.14 ML forecasting service. `main.py` is the FastAPI entry point and owns the training workflow, dataset loading, model selection, metrics, and callbacks. `predict.py` is a standalone forecasting script for local experiments. `DTOs/` contains Pydantic request and response schemas. `models/` contains shared enums and domain definitions. There is currently no committed `tests/` directory; add one with new automated coverage.

Generated files and local environments are ignored through `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, and `.ruff_cache/`.

## Build, Test, and Development Commands

- `uv sync`: create or update the local virtual environment from `pyproject.toml` and `uv.lock`.
- `uv run fastapi dev main.py`: run the API locally with reload for development.
- `uv run uvicorn main:app --reload`: alternative local API runner if using Uvicorn directly.
- `uv run python predict.py --file data.xlsx --target "Revenue" --period 2030`: run the standalone prediction script.
- `uv run python -m py_compile main.py predict.py DTOs/TrainingRunRequest.py models/Model.py`: quick syntax check for touched modules.
- `uvx ruff check .` and `uvx ruff format .`: lint and format the codebase when Ruff is not installed in the environment.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for functions and variables, `PascalCase` for Pydantic models and enums, and uppercase constants such as `GLOBAL_RANDOM_SEED`. Keep request/response contracts in `DTOs/`, shared enums in `models/`, and API orchestration in `main.py` unless a new module clearly reduces complexity. Prefer type hints on new functions.

## Testing Guidelines

No formal test framework is configured yet. For new behavior, add `pytest` tests under `tests/` with names like `test_training_run_validation.py`. Focus coverage on validation, forecast frequency normalization, feature selection, metrics, and callback payload shape. Until a suite exists, run `py_compile` and exercise the relevant API or CLI path manually.

## Commit & Pull Request Guidelines

Recent history uses short imperative commits, often Conventional Commit style, for example `feat: add model selection for training`. Prefer `feat:`, `fix:`, `refactor:`, or a concise plain sentence when appropriate.

Pull requests should include a behavior summary, validation steps or command output, linked issue when available, and notes about API contract changes. Include sample payloads when changing DTOs or callbacks.

## Security & Configuration Tips

Do not commit datasets, trained artifacts, secrets, or local virtual environments. Treat `downloadUrl` and `callbackUrl` as external inputs; validate assumptions and avoid logging sensitive URLs.
