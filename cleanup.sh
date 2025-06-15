#!/usr/bin/env bash

# Check if uv is available, install if not
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing with pip..."
    python -m pip install uv
fi

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Not in a virtual environment, creating and activating one..."
    python -m uv sync --all-extras
    echo "Re-running script in virtual environment..."
    exec "$0" "$@"
fi

python -m uv run hatch clean
python -m uv run hatch build
echo "python -m uzpy run -e src"
python -m uzpy run -e src
echo "find . -name '*.py' -exec python -m uv run autoflake -i {} +"
for p in src tests; do find "$p" -name "*.py" -exec python -m uv run autoflake -i {} +; done
echo "find . -name '*.py' -type f -exec python -m uv run pyupgrade --py311-plus {} +"
for p in src tests; do find "$p" -type f -exec python -m uv run pyupgrade --py311-plus {} +; done
echo "find . -name '*.py' -exec python -m uv run ruff check --output-format=github --fix --unsafe-fixes {} +"
for p in src tests; do find "$p" -name "*.py" -exec python -m uv run ruff check --output-format=github --fix --unsafe-fixes {} +; done
echo "find . -name '*.py' -exec python -m uv run ruff format --respect-gitignore --target-version py311 {} +"
for p in src tests; do find "$p" -name "*.py" -exec python -m uv run ruff format --respect-gitignore --target-version py311 {} +; done
echo "python -m uv run ty check"
python -m uv run ty check
echo "PYTHONPATH=src python -m mypy -p imgcolorshine"
PYTHONPATH=src python -m mypy -p imgcolorshine
if command -v npx >/dev/null 2>&1; then
    echo "npx repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs -o llms.txt ."
    npx repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs -o llms.txt .
else
    echo "npx not found, skipping repomix"
fi
echo "python -m uv run hatch test"
python -m uv run hatch test
