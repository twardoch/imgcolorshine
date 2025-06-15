# Development Plan for imgcolorshine - Remaining Tasks

## Task: Fix MyPyc Build Errors

The build process is failing during the MyPyc compilation phase with missing module stubs. The error output shows:

```
Exception: Error while invoking Mypyc:
src/imgcolorshine/utils.py:18: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/utils.py:19: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
[... many more similar errors ...]
```

### Root Cause

MyPyc requires type stubs for all imported modules during compilation. The build environment is missing type stubs for the project's dependencies.

### Solution Plan

1. **Install Type Stubs in Build Environment**
   - Add type stub packages to the build requirements
   - Ensure MyPyc can find all necessary type information

2. **Fix Module-Specific Issues**
   - `gpu.py:24`: "Name 'cp' already defined on line 20" - Fix the duplicate definition
   - `colorshine.py:169`: "Name 'transformed' already defined on line 166" - Fix variable redefinition
   - Handle the scipy-stubs requirement for `lut.py`

3. **Update pyproject.toml**
   ```toml
   [build-system]
   requires = [
       'hatchling>=1.27.0',
       'hatch-vcs>=0.4.0',
       'hatch-mypyc>=0.16.0',
       # Add type stubs for build
       'numpy',
       'types-Pillow',
       'scipy-stubs',
   ]
   ```

4. **Alternative: Selective MyPyc Compilation**
   - If type stubs are problematic, exclude modules with heavy external dependencies
   - Focus MyPyc on pure Python modules that benefit most from compilation

5. **Fallback Option**
   - Temporarily disable MyPyc compilation for release
   - Ship with Numba JIT compilation only (still provides excellent performance)

### Implementation Steps

#### Step 1: Fix Code Issues First

1. Fix duplicate `cp` definition in `gpu.py`:
   ```python
   # Remove line 24 or rename the import
   ```

2. Fix duplicate `transformed` variable in `colorshine.py:166-169`:
   ```python
   # Use different variable name or restructure the logic
   ```

#### Step 2: Configure MyPyc Build Dependencies

Option A - Add stub dependencies to build requirements:
```toml
[tool.hatch.build.hooks.mypyc]
dependencies = [
    "hatch-mypyc",
    "numpy",
    "numba",
    "scipy",
    "scipy-stubs",
    "types-Pillow",
    "loguru",
    "coloraide",
    "fire",
]
```

Option B - Use MyPyc's `--ignore-missing-imports` flag:
```toml
[tool.hatch.build.hooks.mypyc]
mypy-args = ["--ignore-missing-imports"]
```

#### Step 3: Selective Compilation Strategy

If full compilation fails, compile only the modules that:
1. Don't have heavy external dependencies
2. Benefit most from compilation
3. Have clean type annotations

```toml
[tool.hatch.build.hooks.mypyc]
files = [
    "src/imgcolorshine/colorshine.py",
    "src/imgcolorshine/utils.py",
    # Exclude modules with Numba JIT (already optimized)
    # Exclude modules with complex dependencies
]
```

#### Step 4: Testing the Build

```bash
# Clean previous build artifacts
uvx hatch clean

# Test build locally
uvx hatch build

# If successful, publish
uvx hatch publish
```

## Task: Fix cleanup.log Issues

The cleanup.log shows that the build process is failing consistently due to MyPyc compilation errors. The cleanup script runs several commands:

1. `python -m uv sync --all-extras` - Fails due to MyPyc
2. `python -m uv run hatch clean` - Fails due to MyPyc
3. `python -m uv run hatch test` - Fails due to MyPyc

### Root Cause Analysis

The MyPyc build hook is being triggered during editable installs, which is causing all development commands to fail. This is blocking:
- Dependency synchronization
- Test execution
- Package cleaning
- Development workflow

### Immediate Solution

**Temporarily disable MyPyc for development workflow:**

1. **Option A: Comment out MyPyc in pyproject.toml**
   ```toml
   # [tool.hatch.build.hooks.mypyc]
   # dependencies = ["hatch-mypyc"]
   # files = [...]
   ```

2. **Option B: Use environment variable to disable**
   ```bash
   export HATCH_BUILD_HOOK_ENABLE_MYPYC=false
   ```

3. **Option C: Create separate build configurations**
   - Development build without MyPyc
   - Production build with MyPyc

### Long-term Solution

1. **Fix the code issues first:**
   - `gpu.py:24` - Duplicate `cp` definition
   - `colorshine.py:169` - Duplicate `transformed` variable
   - `cli.py:13` - Add proper type ignore for fire import

2. **Install type stubs in build environment:**
   ```bash
   pip install numpy-stubs types-Pillow scipy-stubs
   ```

3. **Configure MyPyc properly:**
   - Use `--ignore-missing-imports` flag
   - Exclude problematic modules
   - Focus on modules that benefit most from compilation

### Action Items

1. **Immediate fix to unblock development:**
   ```bash
   # Temporarily disable MyPyc
   export HATCH_BUILD_HOOK_ENABLE_MYPYC=false
   
   # Run cleanup
   python -m uv sync --all-extras
   python -m uv run hatch clean
   python -m uv run hatch test
   ```

2. **Fix code issues:**
   - Fix duplicate definitions in gpu.py and colorshine.py
   - Add type ignores where appropriate

3. **Update build configuration:**
   - Create development-specific build settings
   - Document the build process
   - Add CI/CD configuration for production builds with MyPyc