#!/usr/bin/env bash
echo "rm -rf dist/imgcolorshine*.*"
rm -rf dist/imgcolorshine*.*
echo "uv build"
uv build
echo "python -m uzpy run -e src"
python -m uzpy run -e src
echo "find . -name '*.py' -exec uvx autoflake -i {} +"
for p in src tests; do find "$p" -name "*.py" -exec uvx autoflake -i {} +; done
echo "find . -name '*.py' -exec uvx pyupgrade --py311-plus {} +"
for p in src tests; do find "$p" -exec uvx pyupgrade --py311-plus {} +; done
echo "find . -name '*.py' -exec uvx ruff check --output-format=github --fix --unsafe-fixes {} +"
for p in src tests; do find "$p" -name "*.py" -exec uvx ruff check --output-format=github --fix --unsafe-fixes {} +; done
echo "find . -name '*.py' -exec uvx ruff format --respect-gitignore --target-version py311 {} +"
for p in src tests; do find "$p" -name "*.py" -exec uvx ruff format --respect-gitignore --target-version py311 {} +; done
echo "uvx ty check"
uvx ty check
echo "PYTHONPATH=src mypy -p imgcolorshine"
PYTHONPATH=src mypy -p imgcolorshine
echo "repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs -o llms.txt ."
repomix -i varia,.specstory,AGENT.md,CLAUDE.md,PLAN.md,SPEC.md,llms.txt,.cursorrules,docs -o llms.txt .
echo "python -m pytest"
python -m pytest
