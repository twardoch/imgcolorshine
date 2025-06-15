#!/usr/bin/env bash

rm -rf dist/imgcolorshine*.*
uv build

python -m uzpy run -e src
find . -name "*.py" -exec uvx autoflake -i {} \;
find . -name "*.py" -exec uvx pyupgrade --py311-plus {} \;
find . -name "*.py" -exec uvx ruff check --output-format=github --fix --unsafe-fixes {} \;
find . -name "*.py" -exec uvx ruff format --respect-gitignore --target-version py311 {} \;
uvx ty check
PYTHONPATH=src mypy -p imgcolorshine

repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs -o llms.txt .
python -m pytest
