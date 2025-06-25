#!/usr/bin/env bash

# Notice before redirecting output
echo "Starting cleanup process... All output will be logged to cleanup.log"

# Redirect all subsequent output to cleanup.log
exec >cleanup.log 2>&1

echo "=== Cleanup started at $(date) ==="

# Check if uv is available, install if not
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing with pip..."
    python -m pip install uv
fi

echo "python -m uv pip install .[all]"
python -m uv pip install .[all] --quiet # Add --quiet to reduce log noise from this step
echo "python -m uv run --with hatch hatch clean"
python -m uv run --with hatch hatch clean
echo "python -m uv run --with hatch hatch build"
python -m uv run --with hatch hatch build
#echo "python -m uzpy run -e src"
#python -m uzpy run -e src

echo "python -m uv run --with autoflake find . -name '*.py' -exec autoflake -i {} +"
# Corrected loop for find + exec with uv run
find src tests -name "*.py" -exec python -m uv run --with autoflake autoflake -i {} \;
echo "python -m uv run --with pyupgrade find . -name '*.py' -exec pyupgrade --py311-plus {} +"
find src tests -name "*.py" -exec python -m uv run --with pyupgrade pyupgrade --py311-plus {} \;
echo "python -m uv run --with ruff find . -name '*.py' -exec ruff check --output-format=github --fix --unsafe-fixes {} +"
find src tests -name "*.py" -exec python -m uv run --with ruff ruff check --output-format=github --fix --unsafe-fixes {} \;
echo "python -m uv run --with ruff find . -name '*.py' -exec ruff format --respect-gitignore --target-version py311 {} +"
find src tests -name "*.py" -exec python -m uv run --with ruff ruff format --respect-gitignore --target-version py311 {} \;
echo "python -m uv run --with ty ty check"
python -m uv run --with ty ty check
echo "python -m uv run --with mypy mypy --config-file pyproject.toml src/imgcolorshine tests"
python -m uv run --with mypy mypy --config-file pyproject.toml src/imgcolorshine tests

# Ensure npx is available or skip
if command -v npx >/dev/null 2>&1; then
    echo "npx repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs,.log -o llms.txt ."
    npx repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs,.log -o llms.txt .
else
    echo "npx not found, skipping repomix"
fi
echo "python -m uv run --with hatch hatch test"
python -m uv run --with hatch hatch test

echo "=== Cleanup completed at $(date) ==="
