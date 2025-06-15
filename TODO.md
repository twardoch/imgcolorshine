# TODO

TASK: Work through the steps below. Once you've completed a step, mark it with `[x]`. Always use `uv` or `hatch` or `python`, not `python3`.

## This tool works like so:

Tolerance of 0% influences no pixels, tolerance of 100% influences all pixels, and tolerance of 50% influences that half of the pixels that are more similar to the attractor than the other half. 

The actual influence of the attractor onto a given pixel should always stronger if the pixel is more similar to the attractor, and less strong if it's less similar. 

The strength of 100% means that the influence of the attractor onto the pixels that are most similar to the attractor is full, that is, these pixels take on the hue and/or saturation and/or luminance of the attractor. But for pixels that are less similar, there's a falloff. 

Aa strength of 50% means that the influence is 50% but only on the most similar pixels, that is, the new value of H or S or L becomes 50% of the old one and 50% of the new one. But the strength of the influence always falls off, the less similar the pixel is to the attractor. 

The strength of 200% means there is no falloff: the influence is always full within the tolerance. 

## Task 1 [x]

Update CHANGELOG.md and README.md to reflect the changes that are marked as done in PLAN.md. 

## Task 2 [x]

Analyze `llms.txt` and completely remove all completed items from PLAN.md. Keep only things that are TBD. 

## Task 3 [x]

```
uvx hatch clean; uvx hatch build; uvx hatch publish

─────────────────────────────────────────────────────────── sdist ───────────────────────────────────────────────────────────
dist/imgcolorshine-3.3.0.tar.gz
─────────────────────────────────────────────────────────── wheel ───────────────────────────────────────────────────────────
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/hatchling/__main__.py", line 6, in <module>
    sys.exit(hatchling())
             ^^^^^^^^^^^
  File "/Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/hatchling/cli/__init__.py", line 26, in hatchling
    command(**kwargs)
  File "/Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/hatchling/cli/build/__init__.py", line 82, in build_impl
    for artifact in builder.build(
                    ^^^^^^^^^^^^^^
  File "/Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/hatchling/builders/plugin/interface.py", line 147, in build
    build_hook.initialize(version, build_data)
  File "/Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/hatch_mypyc/plugin.py", line 338, in initialize
    raise Exception(f'Error while invoking Mypyc:\n{process.stdout.decode("utf-8")}')
Exception: Error while invoking Mypyc:
src/imgcolorshine/utils.py:18: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/utils.py:19: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/trans_numba.py:20: error: Cannot find implementation or library stub for module named "numba"  [import-not-found]
src/imgcolorshine/trans_numba.py:21: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/numba_utils.py:14: error: Cannot find implementation or library stub for module named "numba"  [import-not-found]
src/imgcolorshine/numba_utils.py:15: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/lut.py:21: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/lut.py:22: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/lut.py:23: error: Library stubs not installed for "scipy.interpolate"  [import-untyped]
src/imgcolorshine/lut.py:23: note: Hint: "python3 -m pip install scipy-stubs"
src/imgcolorshine/lut.py:23: note: (or run "mypy --install-types" to install all missing stub packages)
src/imgcolorshine/io.py:19: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/io.py:20: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/io.py:24: error: Cannot find implementation or library stub for module named "cv2"  [import-not-found]
src/imgcolorshine/io.py:32: error: Cannot find implementation or library stub for module named "PIL"  [import-not-found]
src/imgcolorshine/gpu.py:14: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/gpu.py:15: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/gpu.py:24: error: Cannot find implementation or library stub for module named "cupy"  [import-not-found]
src/imgcolorshine/gpu.py:24: error: Name "cp" already defined on line 20  [no-redef]
src/imgcolorshine/gpu.py:26: error: "None" has no attribute "cuda"  [attr-defined]
src/imgcolorshine/gpu.py:29: error: "None" has no attribute "cuda"  [attr-defined]
src/imgcolorshine/falloff.py:18: error: Cannot find implementation or library stub for module named "numba"  [import-not-found]
src/imgcolorshine/falloff.py:19: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/gamut.py:18: error: Cannot find implementation or library stub for module named "numba"  [import-not-found]
src/imgcolorshine/gamut.py:19: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/gamut.py:20: error: Cannot find implementation or library stub for module named "coloraide"  [import-not-found]
src/imgcolorshine/gamut.py:21: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/engine.py:19: error: Cannot find implementation or library stub for module named "numba"  [import-not-found]
src/imgcolorshine/engine.py:20: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/engine.py:21: error: Cannot find implementation or library stub for module named "coloraide"  [import-not-found]
src/imgcolorshine/engine.py:22: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/colorshine.py:17: error: Cannot find implementation or library stub for module named "numpy"  [import-not-found]
src/imgcolorshine/colorshine.py:18: error: Cannot find implementation or library stub for module named "loguru"  [import-not-found]
src/imgcolorshine/colorshine.py:169: error: Name "transformed" already defined on line 166  [no-redef]
src/imgcolorshine/cli.py:13: error: Cannot find implementation or library stub for module named "fire"  [import-not-found]
src/imgcolorshine/cli.py:13: note: Error code "import-not-found" not covered by "type: ignore" comment
src/imgcolorshine/cli.py:13: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports

dist/imgcolorshine-3.3.0.tar.gz ... success

[imgcolorshine]
https://pypi.org/project/imgcolorshine/3.3.0/
```

Into `PLAN.md` write a plan for how to fix the build errors. 

## Task 4 [x]

Read `cleanup.log` (which is the result of running `cleanup.sh`), and into `PLAN.md` write a plan for how to fix the issues, problems, errors recorded there. 