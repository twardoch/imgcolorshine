#!/usr/bin/env bash

rm -rf dist/imgcolorshine*.*
uv build

python -m uzpy run -e src
fd -e py -x uvx autoflake -i {}
fd -e py -x uvx pyupgrade --py311-plus {}
fd -e py -x uvx ruff check --output-format=github --fix --unsafe-fixes {}
fd -e py -x uvx ruff format --respect-gitignore --target-version py311 {}
uvx ty check
PYTHONPATH=src mypy -p imgcolorshine

repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs -o llms.txt .
python -m pytest
