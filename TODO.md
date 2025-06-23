# TODO

This file tracks the high-level tasks for the current development effort. Detailed steps are in `PLAN.md`.

## [ ] Overall Modernization Project

-   **[ ] Initial Setup & Cleanup:** (Corresponds to Plan Step 1)
    -   [x] Create `PLAN.md` with the detailed plan.
    -   [ ] Create/Update `TODO.md` (this file).
    -   [ ] Remove `package.toml`.
    -   [ ] Ensure `uv` usage is consistent.
-   **[ ] Packaging & Dependency Management (Hatch & `uv`):** (Corresponds to Plan Step 2)
    -   [ ] Refine `pyproject.toml` for Hatch.
    -   [ ] Update GitHub Actions for Hatch + `uv`.
-   **[ ] Codebase Modernization & Refactoring:** (Corresponds to Plan Step 3)
    -   [ ] **Refactor for Performance Modules:**
        -   [ ] Identify Mypyc-optimizable code.
        -   [ ] Move Mypyc candidates to `src/imgcolorshine/fast_mypyc/`.
        -   [ ] Identify Numba-accelerated code.
        -   [ ] Move Numba candidates to `src/imgcolorshine/fast_numba/`.
        -   [ ] Ensure core `src/imgcolorshine/` modules are primarily pure Python.
        -   [ ] Update `pyproject.toml` for Mypyc compilation of `fast_mypyc`.
    -   [ ] Apply general code modernization.
    -   [ ] Add `this_file` comments to Python source files.
-   **[ ] Quality Assurance (QA) Enhancements:** (Corresponds to Plan Step 4)
    -   [ ] Review and update `.pre-commit-config.yaml`.
    -   [ ] Enhance GitHub Actions checks.
-   **[ ] Testing:** (Corresponds to Plan Step 5)
    -   [ ] Review existing tests.
    *   [ ] Add tests for new/refactored components.
    *   [ ] Ensure all tests pass.
-   **[ ] Documentation Update:** (Corresponds to Plan Step 6)
    *   [ ] Update `README.md` (installation, usage, development, architecture).
    *   [ ] Update `CHANGELOG.md`.
-   **[ ] Final Review & Build:** (Corresponds to Plan Step 7)
    *   [ ] Ensure project builds with `hatch build`.
    *   [ ] Run all QA checks.
    *   [ ] Execute post-change script from `AGENT.md`.
-   **[ ] Submit:** (Corresponds to Plan Step 8)
    *   [ ] Commit and submit changes.

*(Self-correction: Marked the `PLAN.md` creation as done in this new TODO, as it was done previously. Updating this TODO file itself is the current action.)*
