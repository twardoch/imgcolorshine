# TODO

This file tracks immediate actionable tasks for imgcolorshine development. For detailed implementation plans, see `PLAN.md`.

## 🚨 Critical - Fix Build System (Week 1)

- [ ] **Debug Mypyc compilation errors**
  - [ ] Run `uvx hatch clean && uvx hatch build` and capture full error log
  - [ ] Create module dependency graph to identify circular imports
  - [ ] Fix numpy type annotations (use `numpy.typing.NDArray`)
  - [ ] Add type stubs for coloraide dependency
  - [ ] Test build with `IMGCS_DISABLE_MYPYC=1` environment variable

- [ ] **Implement graceful fallback mechanism**
  - [ ] Create pure Python copies of mypyc modules in `src/imgcolorshine/pure/`
  - [ ] Update imports to use try/except pattern
  - [ ] Add logging when falling back to pure Python
  - [ ] Test performance difference between mypyc and pure Python

## 🧪 Testing Infrastructure (Week 2)

- [ ] **Fix existing test failures**
  - [ ] Run `python -m pytest -xvs` to identify all failures
  - [ ] Fix import errors in test files
  - [ ] Update tests for new module structure
  - [ ] Ensure all tests pass locally

- [ ] **Set up GitHub Actions CI/CD**
  - [ ] Create `.github/workflows/test.yml`
  - [ ] Add test matrix: Python 3.9-3.12, Windows/macOS/Linux
  - [ ] Add coverage reporting with codecov
  - [ ] Set up automatic wheel building
  - [ ] Add badge to README.md

- [ ] **Expand test coverage to 80%**
  - [ ] Add tests for CLI module
  - [ ] Add tests for GPU acceleration path
  - [ ] Add tests for LUT generation
  - [ ] Create integration tests for full pipeline
  - [ ] Add property-based tests with Hypothesis

## 🎯 Quick Wins (Week 3)

- [ ] **Enhance CLI user experience**
  - [ ] Add progress bar using Rich for image processing
  - [ ] Add `--verbose` flag with detailed logging
  - [ ] Improve error messages with helpful suggestions
  - [ ] Add `--dry-run` mode to preview operations

- [ ] **Create example scripts**
  - [ ] `examples/batch_process.py` - Process directory of images
  - [ ] `examples/preset_transforms.py` - Common color transformations
  - [ ] `examples/api_usage.py` - Demonstrate Python API
  - [ ] `examples/performance_comparison.py` - Compare CPU/GPU/LUT

- [ ] **Documentation improvements**
  - [ ] Add docstrings to all public functions
  - [ ] Create `docs/api.md` with API reference
  - [ ] Add performance tuning guide
  - [ ] Document all CLI options with examples

## 📦 Packaging & Distribution (Week 4)

- [ ] **Create Docker image**
  - [ ] Write `Dockerfile` with multi-stage build
  - [ ] Include all dependencies and optimizations
  - [ ] Add GPU support variant
  - [ ] Publish to Docker Hub

- [ ] **Improve installation experience**
  - [ ] Create install script for common platforms
  - [ ] Add platform-specific installation instructions
  - [ ] Test installation on fresh systems
  - [ ] Create troubleshooting guide

- [ ] **Prepare for release**
  - [ ] Update version to 3.4.0
  - [ ] Finalize CHANGELOG.md
  - [ ] Tag release in git
  - [ ] Build and test wheels locally
  - [ ] Publish to TestPyPI first

## 🔧 Code Quality (Ongoing)

- [ ] **Linting and formatting**
  - [ ] Run `fd -e py -x ruff check --fix {}`
  - [ ] Run `fd -e py -x ruff format {}`
  - [ ] Fix any mypy errors
  - [ ] Add pre-commit hooks

- [ ] **Performance profiling**
  - [ ] Profile memory usage with large images
  - [ ] Identify bottlenecks with cProfile
  - [ ] Create benchmark suite
  - [ ] Document performance characteristics

- [ ] **Code cleanup**
  - [ ] Remove commented-out code
  - [ ] Consolidate duplicate functions
  - [ ] Update imports to use new structure
  - [ ] Add `__all__` to all modules

## 🚀 Future Features (Backlog)

- [ ] **Memory optimization**
  - [ ] Implement streaming tile processor
  - [ ] Add LRU cache for color conversions
  - [ ] Profile and reduce memory allocations

- [ ] **Advanced CLI features**
  - [ ] Interactive mode for parameter tuning
  - [ ] Configuration file support
  - [ ] Preset management system

- [ ] **API enhancements**
  - [ ] High-level `ColorShine` class
  - [ ] Async/await support
  - [ ] Plugin system for custom transforms

- [ ] **Platform integration**
  - [ ] GIMP plugin
  - [ ] Photoshop extension
  - [ ] ImageMagick delegate

## Notes

- Always run tests after making changes: `python -m pytest`
- Use `uv` for all Python operations
- Update this TODO list as tasks are completed
- Check PLAN.md for detailed implementation guidance
- Create issues on GitHub for bugs and feature requests

---

Last updated: 2025-06-29