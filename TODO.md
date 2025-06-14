# TODO

## Completed

- [x] Create TODO.md and PLAN.md files for project organization
- [x] Review existing old implementation files to understand complete feature set
- [x] Port color_engine.py from old implementation to new structure
- [x] Port image_io.py, transforms.py, and other core modules
- [x] Implement CLI interface using Click
- [x] Update README.md with proper documentation
- [x] Create CHANGELOG.md to track changes

## Pending

- [ ] Add comprehensive tests for all modules
- [ ] Run Python formatting and linting tools
- [ ] Test the CLI with sample images
- [ ] Add GPU acceleration support (optional)

## Implementation Notes

- The old implementation in `old/imgcolorshine/` contains a complete working version
- Core modules to port: color_engine.py, image_io.py, transforms.py, gamut.py, falloff.py, utils.py
- Use modern Python packaging structure with src/ layout
- Maintain compatibility with the existing pyproject.toml configuration