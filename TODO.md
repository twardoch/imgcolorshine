# TODO

This file tracks the high-level tasks for the current development effort. Detailed steps are in `PLAN.md`.

TASK: Work through the steps below. Once you've completed a step, mark it with `[x]`. Always run Pytho tools with `uv` or `uv run hatch` or `python`. 

## [x] Task - Fixed mypyc build errors

```bash
uvx hatch clean; uvx hatch build

─────────────────────────────────────────────────────────── sdist ───────────────────────────────────────────────────────────
dist/imgcolorshine-3.3.4.dev4+g3ba569e.d20250616.tar.gz
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

LOG:  Mypy Version:           1.16.0
LOG:  Config File:            Default
LOG:  Configured Executable:  /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/bin/python
LOG:  Current Executable:     /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/bin/python
LOG:  Cache Dir:              .mypy_cache
LOG:  Compiled:               True
LOG:  Exclude:                []
LOG:  Found source:           BuildSource(path='src/imgcolorshine/cli.py', module='imgcolorshine.cli', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/colorshine.py', module='imgcolorshine.colorshine', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/engine.py', module='imgcolorshine.engine', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/gamut.py', module='imgcolorshine.gamut', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/io.py', module='imgcolorshine.io', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/utils.py', module='imgcolorshine.utils', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/fast_mypyc/__init__.py', module='imgcolorshine.fast_mypyc', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/fast_mypyc/colorshine_helpers.py', module='imgcolorshine.fast_mypyc.colorshine_helpers', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/fast_mypyc/engine_helpers.py', module='imgcolorshine.fast_mypyc.engine_helpers', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/fast_mypyc/gamut_helpers.py', module='imgcolorshine.fast_mypyc.gamut_helpers', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Found source:           BuildSource(path='src/imgcolorshine/fast_mypyc/utils.py', module='imgcolorshine.fast_mypyc.utils', has_text=False, base_dir='/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src', followed=False)
LOG:  Metadata not found for imgcolorshine.cli
LOG:  Parsing src/imgcolorshine/cli.py (imgcolorshine.cli)
LOG:  Metadata not found for imgcolorshine.colorshine
LOG:  Parsing src/imgcolorshine/colorshine.py (imgcolorshine.colorshine)
LOG:  Metadata not found for imgcolorshine.engine
LOG:  Parsing src/imgcolorshine/engine.py (imgcolorshine.engine)
LOG:  Metadata not found for imgcolorshine.gamut
LOG:  Parsing src/imgcolorshine/gamut.py (imgcolorshine.gamut)
LOG:  Metadata not found for imgcolorshine.io
LOG:  Parsing src/imgcolorshine/io.py (imgcolorshine.io)
LOG:  Metadata not found for imgcolorshine.utils
LOG:  Parsing src/imgcolorshine/utils.py (imgcolorshine.utils)
LOG:  Metadata not found for imgcolorshine.fast_mypyc
LOG:  Parsing src/imgcolorshine/fast_mypyc/__init__.py (imgcolorshine.fast_mypyc)
LOG:  Metadata not found for imgcolorshine.fast_mypyc.colorshine_helpers
LOG:  Parsing src/imgcolorshine/fast_mypyc/colorshine_helpers.py (imgcolorshine.fast_mypyc.colorshine_helpers)
LOG:  Metadata not found for imgcolorshine.fast_mypyc.engine_helpers
LOG:  Parsing src/imgcolorshine/fast_mypyc/engine_helpers.py (imgcolorshine.fast_mypyc.engine_helpers)
LOG:  Metadata not found for imgcolorshine.fast_mypyc.gamut_helpers
LOG:  Parsing src/imgcolorshine/fast_mypyc/gamut_helpers.py (imgcolorshine.fast_mypyc.gamut_helpers)
LOG:  Metadata not found for imgcolorshine.fast_mypyc.utils
LOG:  Parsing src/imgcolorshine/fast_mypyc/utils.py (imgcolorshine.fast_mypyc.utils)
LOG:  Metadata not found for imgcolorshine
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/__init__.py (imgcolorshine)
LOG:  Metadata not found for builtins
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/builtins.pyi (builtins)
LOG:  Metadata not found for imgcolorshine.lut
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/lut.py (imgcolorshine.lut)
LOG:  Metadata not found for sys
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sys/__init__.pyi (sys)
LOG:  Metadata not found for pathlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/pathlib.pyi (pathlib)
LOG:  Metadata not found for typing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/typing.pyi (typing)
LOG:  Metadata not found for numpy
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/__init__.pyi (numpy)
LOG:  Metadata not found for loguru
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/loguru/__init__.pyi (loguru)
LOG:  Metadata not found for imgcolorshine.fast_numba.trans_numba
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/trans_numba.py (imgcolorshine.fast_numba.trans_numba)
LOG:  Metadata not found for imgcolorshine.fast_numba.engine_kernels
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/engine_kernels.py (imgcolorshine.fast_numba.engine_kernels)
LOG:  Metadata not found for collections.abc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/collections/abc.pyi (collections.abc)
LOG:  Metadata not found for imgcolorshine.fast_numba
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/__init__.py (imgcolorshine.fast_numba)
LOG:  Metadata not found for imgcolorshine.gpu
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/gpu.py (imgcolorshine.gpu)
LOG:  Metadata not found for dataclasses
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/dataclasses.pyi (dataclasses)
LOG:  Metadata not found for coloraide
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/__init__.py (coloraide)
LOG:  Metadata not found for imgcolorshine.fast_numba.gamut_numba
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/gamut_numba.py (imgcolorshine.fast_numba.gamut_numba)
LOG:  Metadata not found for PIL.Image
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/Image.pyi (PIL.Image)
LOG:  Metadata not found for PIL
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/__init__.pyi (PIL)
LOG:  Metadata not found for __future__
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/__future__.pyi (__future__)
LOG:  Metadata not found for importlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/__init__.pyi (importlib)
LOG:  Metadata not found for _ast
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_ast.pyi (_ast)
LOG:  Metadata not found for _sitebuiltins
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_sitebuiltins.pyi (_sitebuiltins)
LOG:  Metadata not found for _typeshed
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_typeshed/__init__.pyi (_typeshed)
LOG:  Metadata not found for types
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/types.pyi (types)
LOG:  Metadata not found for _collections_abc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_collections_abc.pyi (_collections_abc)
LOG:  Metadata not found for io
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/io.pyi (io)
LOG:  Metadata not found for typing_extensions
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/typing_extensions.pyi (typing_extensions)
LOG:  Metadata not found for hashlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/hashlib.pyi (hashlib)
LOG:  Metadata not found for pickle
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/pickle.pyi (pickle)
LOG:  Metadata not found for _typeshed.importlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_typeshed/importlib.pyi (_typeshed.importlib)
LOG:  Metadata not found for sys._monitoring
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sys/_monitoring.pyi (sys._monitoring)
LOG:  Metadata not found for os
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/os/__init__.pyi (os)
LOG:  Metadata not found for collections
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/collections/__init__.pyi (collections)
LOG:  Metadata not found for abc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/abc.pyi (abc)
LOG:  Metadata not found for re
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/re.pyi (re)
LOG:  Metadata not found for contextlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/contextlib.pyi (contextlib)
LOG:  Metadata not found for numpy._core._internal
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_internal.pyi (numpy._core._internal)
LOG:  Metadata not found for numpy._typing._callable
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_callable.pyi (numpy._typing._callable)
LOG:  Metadata not found for numpy._typing._extended_precision
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_extended_precision.py (numpy._typing._extended_precision)
LOG:  Metadata not found for numpy._core.records
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/records.pyi (numpy._core.records)
LOG:  Metadata not found for numpy._core.function_base
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/function_base.pyi (numpy._core.function_base)
LOG:  Metadata not found for numpy._core.fromnumeric
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/fromnumeric.pyi (numpy._core.fromnumeric)
LOG:  Metadata not found for numpy._core._asarray
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_asarray.pyi (numpy._core._asarray)
LOG:  Metadata not found for numpy._core._type_aliases
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_type_aliases.pyi (numpy._core._type_aliases)
LOG:  Metadata not found for numpy._core._ufunc_config
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_ufunc_config.pyi (numpy._core._ufunc_config)
LOG:  Metadata not found for numpy._core.arrayprint
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/arrayprint.pyi (numpy._core.arrayprint)
LOG:  Metadata not found for numpy._core.einsumfunc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/einsumfunc.pyi (numpy._core.einsumfunc)
LOG:  Metadata not found for numpy._core.multiarray
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/multiarray.pyi (numpy._core.multiarray)
LOG:  Metadata not found for numpy._core.numeric
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/numeric.pyi (numpy._core.numeric)
LOG:  Metadata not found for numpy._core.numerictypes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/numerictypes.pyi (numpy._core.numerictypes)
LOG:  Metadata not found for numpy._core.shape_base
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/shape_base.pyi (numpy._core.shape_base)
LOG:  Metadata not found for numpy.lib.scimath
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/scimath.pyi (numpy.lib.scimath)
LOG:  Metadata not found for numpy.lib._arraypad_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_arraypad_impl.pyi (numpy.lib._arraypad_impl)
LOG:  Metadata not found for numpy.lib._arraysetops_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_arraysetops_impl.pyi (numpy.lib._arraysetops_impl)
LOG:  Metadata not found for numpy.lib._function_base_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_function_base_impl.pyi (numpy.lib._function_base_impl)
LOG:  Metadata not found for numpy.lib._histograms_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_histograms_impl.pyi (numpy.lib._histograms_impl)
LOG:  Metadata not found for numpy.lib._index_tricks_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_index_tricks_impl.pyi (numpy.lib._index_tricks_impl)
LOG:  Metadata not found for numpy.lib._nanfunctions_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.pyi (numpy.lib._nanfunctions_impl)
LOG:  Metadata not found for numpy.lib._npyio_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_npyio_impl.pyi (numpy.lib._npyio_impl)
LOG:  Metadata not found for numpy.lib._polynomial_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.pyi (numpy.lib._polynomial_impl)
LOG:  Metadata not found for numpy.lib._shape_base_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_shape_base_impl.pyi (numpy.lib._shape_base_impl)
LOG:  Metadata not found for numpy.lib._stride_tricks_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.pyi (numpy.lib._stride_tricks_impl)
LOG:  Metadata not found for numpy.lib._twodim_base_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_twodim_base_impl.pyi (numpy.lib._twodim_base_impl)
LOG:  Metadata not found for numpy.lib._type_check_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_type_check_impl.pyi (numpy.lib._type_check_impl)
LOG:  Metadata not found for numpy.lib._ufunclike_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_ufunclike_impl.pyi (numpy.lib._ufunclike_impl)
LOG:  Metadata not found for numpy.lib._utils_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_utils_impl.pyi (numpy.lib._utils_impl)
LOG:  Metadata not found for numpy.__config__
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/__config__.pyi (numpy.__config__)
LOG:  Metadata not found for numpy._pytesttester
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_pytesttester.pyi (numpy._pytesttester)
LOG:  Metadata not found for numpy._typing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/__init__.py (numpy._typing)
LOG:  Metadata not found for numpy._array_api_info
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_array_api_info.pyi (numpy._array_api_info)
LOG:  Metadata not found for numpy.char
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/char/__init__.pyi (numpy.char)
LOG:  Metadata not found for numpy.core
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/core/__init__.pyi (numpy.core)
LOG:  Metadata not found for numpy.ctypeslib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ctypeslib/__init__.pyi (numpy.ctypeslib)
LOG:  Metadata not found for numpy.dtypes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/dtypes.pyi (numpy.dtypes)
LOG:  Metadata not found for numpy.exceptions
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/exceptions.pyi (numpy.exceptions)
LOG:  Metadata not found for numpy.f2py
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/__init__.pyi (numpy.f2py)
LOG:  Metadata not found for numpy.fft
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/fft/__init__.pyi (numpy.fft)
LOG:  Metadata not found for numpy.lib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/__init__.pyi (numpy.lib)
LOG:  Metadata not found for numpy.linalg
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/__init__.pyi (numpy.linalg)
LOG:  Metadata not found for numpy.ma
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/__init__.pyi (numpy.ma)
LOG:  Metadata not found for numpy.polynomial
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/__init__.pyi (numpy.polynomial)
LOG:  Metadata not found for numpy.random
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/__init__.pyi (numpy.random)
LOG:  Metadata not found for numpy.rec
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/rec/__init__.pyi (numpy.rec)
LOG:  Metadata not found for numpy.strings
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/strings/__init__.pyi (numpy.strings)
LOG:  Metadata not found for numpy.testing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/__init__.pyi (numpy.testing)
LOG:  Metadata not found for numpy.typing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/typing/__init__.py (numpy.typing)
LOG:  Metadata not found for numpy.matlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/matlib.pyi (numpy.matlib)
LOG:  Metadata not found for numpy.matrixlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/matrixlib/__init__.pyi (numpy.matrixlib)
LOG:  Metadata not found for numpy.version
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/version.pyi (numpy.version)
LOG:  Metadata not found for numpy._expired_attrs_2_0
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_expired_attrs_2_0.pyi (numpy._expired_attrs_2_0)
LOG:  Metadata not found for numpy._globals
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_globals.pyi (numpy._globals)
LOG:  Metadata not found for mmap
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/mmap.pyi (mmap)
LOG:  Metadata not found for ctypes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ctypes/__init__.pyi (ctypes)
LOG:  Metadata not found for array
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/array.pyi (array)
LOG:  Metadata not found for datetime
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/datetime.pyi (datetime)
LOG:  Metadata not found for decimal
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/decimal.pyi (decimal)
LOG:  Metadata not found for fractions
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/fractions.pyi (fractions)
LOG:  Metadata not found for uuid
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/uuid.pyi (uuid)
LOG:  Metadata not found for multiprocessing.context
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/context.pyi (multiprocessing.context)
LOG:  Metadata not found for asyncio
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/__init__.pyi (asyncio)
LOG:  Metadata not found for logging
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/logging/__init__.pyi (logging)
LOG:  Metadata not found for imgcolorshine.fast_numba.falloff
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/falloff.py (imgcolorshine.fast_numba.falloff)
LOG:  Metadata not found for imgcolorshine.fast_numba.utils
LOG:  Parsing /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/utils.py (imgcolorshine.fast_numba.utils)
LOG:  Metadata not found for enum
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/enum.pyi (enum)
LOG:  Metadata not found for coloraide.__meta__
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/__meta__.py (coloraide.__meta__)
LOG:  Metadata not found for coloraide.color
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/color.py (coloraide.color)
LOG:  Metadata not found for coloraide.interpolate
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/__init__.py (coloraide.interpolate)
LOG:  Metadata not found for coloraide.algebra
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/algebra.py (coloraide.algebra)
LOG:  Metadata not found for coloraide.easing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/easing.py (coloraide.easing)
LOG:  Metadata not found for PIL.PyAccess
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/PyAccess.pyi (PIL.PyAccess)
LOG:  Metadata not found for PIL._imaging
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/_imaging.pyi (PIL._imaging)
LOG:  Metadata not found for PIL.ImageFilter
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/ImageFilter.pyi (PIL.ImageFilter)
LOG:  Metadata not found for PIL.ImagePalette
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/ImagePalette.pyi (PIL.ImagePalette)
LOG:  Metadata not found for importlib._bootstrap
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/_bootstrap.pyi (importlib._bootstrap)
LOG:  Metadata not found for importlib.abc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/abc.pyi (importlib.abc)
LOG:  Metadata not found for ast
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ast.pyi (ast)
LOG:  Metadata not found for importlib.machinery
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/machinery.pyi (importlib.machinery)
LOG:  Metadata not found for _io
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_io.pyi (_io)
LOG:  Metadata not found for _blake2
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_blake2.pyi (_blake2)
LOG:  Metadata not found for _hashlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_hashlib.pyi (_hashlib)
LOG:  Metadata not found for _pickle
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_pickle.pyi (_pickle)
LOG:  Metadata not found for os.path
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/os/path.pyi (os.path)
LOG:  Metadata not found for subprocess
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/subprocess.pyi (subprocess)
LOG:  Metadata not found for resource
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/resource.pyi (resource)
LOG:  Metadata not found for sre_compile
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sre_compile.pyi (sre_compile)
LOG:  Metadata not found for sre_constants
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sre_constants.pyi (sre_constants)
LOG:  Metadata not found for numpy._core
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/__init__.pyi (numpy._core)
LOG:  Metadata not found for numpy._typing._array_like
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_array_like.py (numpy._typing._array_like)
LOG:  Metadata not found for numpy._typing._nbit
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_nbit.py (numpy._typing._nbit)
LOG:  Metadata not found for numpy._typing._nested_sequence
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_nested_sequence.py (numpy._typing._nested_sequence)
LOG:  Metadata not found for numpy._typing._scalars
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_scalars.py (numpy._typing._scalars)
LOG:  Metadata not found for numpy._typing._ufunc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_ufunc.pyi (numpy._typing._ufunc)
LOG:  Metadata not found for numpy.lib._array_utils_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_array_utils_impl.pyi (numpy.lib._array_utils_impl)
LOG:  Metadata not found for numpy.lib._scimath_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_scimath_impl.pyi (numpy.lib._scimath_impl)
LOG:  Metadata not found for numpy.ma.mrecords
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/mrecords.pyi (numpy.ma.mrecords)
LOG:  Metadata not found for numpy.lib._datasource
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_datasource.pyi (numpy.lib._datasource)
LOG:  Metadata not found for zipfile
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/zipfile/__init__.pyi (zipfile)
LOG:  Metadata not found for numpy._typing._char_codes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_char_codes.py (numpy._typing._char_codes)
LOG:  Metadata not found for numpy._typing._dtype_like
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_dtype_like.py (numpy._typing._dtype_like)
LOG:  Metadata not found for numpy._typing._nbit_base
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_nbit_base.pyi (numpy._typing._nbit_base)
LOG:  Metadata not found for numpy._typing._shape
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_shape.py (numpy._typing._shape)
LOG:  Metadata not found for numpy._core.defchararray
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/defchararray.pyi (numpy._core.defchararray)
LOG:  Metadata not found for numpy.ctypeslib._ctypeslib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ctypeslib/_ctypeslib.pyi (numpy.ctypeslib._ctypeslib)
LOG:  Metadata not found for numpy.f2py.f2py2e
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/f2py2e.pyi (numpy.f2py.f2py2e)
LOG:  Metadata not found for numpy.fft._helper
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/fft/_helper.pyi (numpy.fft._helper)
LOG:  Metadata not found for numpy.fft._pocketfft
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/fft/_pocketfft.pyi (numpy.fft._pocketfft)
LOG:  Metadata not found for numpy.lib.array_utils
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/array_utils.pyi (numpy.lib.array_utils)
LOG:  Metadata not found for numpy.lib.format
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/format.pyi (numpy.lib.format)
LOG:  Metadata not found for numpy.lib.introspect
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/introspect.pyi (numpy.lib.introspect)
LOG:  Metadata not found for numpy.lib.mixins
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/mixins.pyi (numpy.lib.mixins)
LOG:  Metadata not found for numpy.lib.npyio
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/npyio.pyi (numpy.lib.npyio)
LOG:  Metadata not found for numpy.lib.stride_tricks
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/stride_tricks.pyi (numpy.lib.stride_tricks)
LOG:  Metadata not found for numpy.lib._arrayterator_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_arrayterator_impl.pyi (numpy.lib._arrayterator_impl)
LOG:  Metadata not found for numpy.lib._version
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_version.pyi (numpy.lib._version)
LOG:  Metadata not found for numpy.linalg._linalg
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/_linalg.pyi (numpy.linalg._linalg)
LOG:  Metadata not found for numpy.linalg._umath_linalg
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.pyi (numpy.linalg._umath_linalg)
LOG:  Metadata not found for numpy.linalg.linalg
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/linalg.pyi (numpy.linalg.linalg)
LOG:  Metadata not found for numpy.ma.core
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/core.pyi (numpy.ma.core)
LOG:  Metadata not found for numpy.ma.extras
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/extras.pyi (numpy.ma.extras)
LOG:  Metadata not found for numpy.polynomial.chebyshev
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/chebyshev.pyi (numpy.polynomial.chebyshev)
LOG:  Metadata not found for numpy.polynomial.hermite
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/hermite.pyi (numpy.polynomial.hermite)
LOG:  Metadata not found for numpy.polynomial.hermite_e
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/hermite_e.pyi (numpy.polynomial.hermite_e)
LOG:  Metadata not found for numpy.polynomial.laguerre
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/laguerre.pyi (numpy.polynomial.laguerre)
LOG:  Metadata not found for numpy.polynomial.legendre
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/legendre.pyi (numpy.polynomial.legendre)
LOG:  Metadata not found for numpy.polynomial.polynomial
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/polynomial.pyi (numpy.polynomial.polynomial)
LOG:  Metadata not found for numpy.random._generator
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_generator.pyi (numpy.random._generator)
LOG:  Metadata not found for numpy.random._mt19937
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_mt19937.pyi (numpy.random._mt19937)
LOG:  Metadata not found for numpy.random._pcg64
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_pcg64.pyi (numpy.random._pcg64)
LOG:  Metadata not found for numpy.random._philox
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_philox.pyi (numpy.random._philox)
LOG:  Metadata not found for numpy.random._sfc64
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_sfc64.pyi (numpy.random._sfc64)
LOG:  Metadata not found for numpy.random.bit_generator
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/bit_generator.pyi (numpy.random.bit_generator)
LOG:  Metadata not found for numpy.random.mtrand
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/mtrand.pyi (numpy.random.mtrand)
LOG:  Metadata not found for numpy._core.strings
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/strings.pyi (numpy._core.strings)
LOG:  Metadata not found for numpy.testing._private.utils
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/_private/utils.pyi (numpy.testing._private.utils)
LOG:  Metadata not found for numpy.testing.overrides
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/overrides.pyi (numpy.testing.overrides)
LOG:  Metadata not found for unittest
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/__init__.pyi (unittest)
LOG:  Metadata not found for numpy._typing._add_docstring
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_add_docstring.py (numpy._typing._add_docstring)
LOG:  Metadata not found for warnings
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/warnings.pyi (warnings)
LOG:  Metadata not found for numpy.matrixlib.defmatrix
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/matrixlib/defmatrix.pyi (numpy.matrixlib.defmatrix)
LOG:  Metadata not found for ctypes._endian
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ctypes/_endian.pyi (ctypes._endian)
LOG:  Metadata not found for _ctypes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_ctypes.pyi (_ctypes)
LOG:  Metadata not found for time
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/time.pyi (time)
LOG:  Metadata not found for numbers
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/numbers.pyi (numbers)
LOG:  Metadata not found for _decimal
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_decimal.pyi (_decimal)
LOG:  Metadata not found for multiprocessing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/__init__.pyi (multiprocessing)
LOG:  Metadata not found for multiprocessing.popen_fork
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_fork.pyi (multiprocessing.popen_fork)
LOG:  Metadata not found for multiprocessing.popen_forkserver
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_forkserver.pyi (multiprocessing.popen_forkserver)
LOG:  Metadata not found for multiprocessing.popen_spawn_posix
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_spawn_posix.pyi (multiprocessing.popen_spawn_posix)
LOG:  Metadata not found for multiprocessing.popen_spawn_win32
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_spawn_win32.pyi (multiprocessing.popen_spawn_win32)
LOG:  Metadata not found for multiprocessing.queues
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/queues.pyi (multiprocessing.queues)
LOG:  Metadata not found for multiprocessing.synchronize
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/synchronize.pyi (multiprocessing.synchronize)
LOG:  Metadata not found for multiprocessing.managers
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/managers.pyi (multiprocessing.managers)
LOG:  Metadata not found for multiprocessing.pool
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/pool.pyi (multiprocessing.pool)
LOG:  Metadata not found for multiprocessing.process
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/process.pyi (multiprocessing.process)
LOG:  Metadata not found for multiprocessing.sharedctypes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/sharedctypes.pyi (multiprocessing.sharedctypes)
LOG:  Metadata not found for multiprocessing.connection
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/connection.pyi (multiprocessing.connection)
LOG:  Metadata not found for asyncio.base_events
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/base_events.pyi (asyncio.base_events)
LOG:  Metadata not found for asyncio.coroutines
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/coroutines.pyi (asyncio.coroutines)
LOG:  Metadata not found for asyncio.events
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/events.pyi (asyncio.events)
LOG:  Metadata not found for asyncio.exceptions
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/exceptions.pyi (asyncio.exceptions)
LOG:  Metadata not found for asyncio.futures
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/futures.pyi (asyncio.futures)
LOG:  Metadata not found for asyncio.locks
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/locks.pyi (asyncio.locks)
LOG:  Metadata not found for asyncio.protocols
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/protocols.pyi (asyncio.protocols)
LOG:  Metadata not found for asyncio.queues
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/queues.pyi (asyncio.queues)
LOG:  Metadata not found for asyncio.runners
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/runners.pyi (asyncio.runners)
LOG:  Metadata not found for asyncio.streams
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/streams.pyi (asyncio.streams)
LOG:  Metadata not found for asyncio.subprocess
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/subprocess.pyi (asyncio.subprocess)
LOG:  Metadata not found for asyncio.tasks
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/tasks.pyi (asyncio.tasks)
LOG:  Metadata not found for asyncio.transports
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/transports.pyi (asyncio.transports)
LOG:  Metadata not found for asyncio.threads
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/threads.pyi (asyncio.threads)
LOG:  Metadata not found for asyncio.taskgroups
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/taskgroups.pyi (asyncio.taskgroups)
LOG:  Metadata not found for asyncio.timeouts
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/timeouts.pyi (asyncio.timeouts)
LOG:  Metadata not found for asyncio.unix_events
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/unix_events.pyi (asyncio.unix_events)
LOG:  Metadata not found for threading
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/threading.pyi (threading)
LOG:  Metadata not found for string
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/string.pyi (string)
LOG:  Metadata not found for coloraide.spaces.srgb.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/srgb/css.py (coloraide.spaces.srgb.css)
LOG:  Metadata not found for coloraide.spaces.hsl.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hsl/css.py (coloraide.spaces.hsl.css)
LOG:  Metadata not found for coloraide.spaces.hwb.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hwb/css.py (coloraide.spaces.hwb.css)
LOG:  Metadata not found for coloraide.spaces.lab.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lab/css.py (coloraide.spaces.lab.css)
LOG:  Metadata not found for coloraide.spaces.lch.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lch/css.py (coloraide.spaces.lch.css)
LOG:  Metadata not found for coloraide.spaces.oklab.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklab/css.py (coloraide.spaces.oklab.css)
LOG:  Metadata not found for coloraide.spaces.oklch.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklch/css.py (coloraide.spaces.oklch.css)
LOG:  Metadata not found for coloraide.spaces.jzazbz.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzazbz/css.py (coloraide.spaces.jzazbz.css)
LOG:  Metadata not found for coloraide.spaces.jzczhz.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzczhz/css.py (coloraide.spaces.jzczhz.css)
LOG:  Metadata not found for coloraide.spaces.ictcp.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/ictcp/css.py (coloraide.spaces.ictcp.css)
LOG:  Metadata not found for coloraide.css.parse
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/parse.py (coloraide.css.parse)
LOG:  Metadata not found for coloraide.spaces.hsv
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hsv.py (coloraide.spaces.hsv)
LOG:  Metadata not found for coloraide.spaces.srgb_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/srgb_linear.py (coloraide.spaces.srgb_linear)
LOG:  Metadata not found for coloraide.spaces.lab_d65
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lab_d65.py (coloraide.spaces.lab_d65)
LOG:  Metadata not found for coloraide.spaces.lch_d65
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lch_d65.py (coloraide.spaces.lch_d65)
LOG:  Metadata not found for coloraide.spaces.display_p3
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/display_p3.py (coloraide.spaces.display_p3)
LOG:  Metadata not found for coloraide.spaces.display_p3_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/display_p3_linear.py (coloraide.spaces.display_p3_linear)
LOG:  Metadata not found for coloraide.spaces.a98_rgb
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/a98_rgb.py (coloraide.spaces.a98_rgb)
LOG:  Metadata not found for coloraide.spaces.a98_rgb_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/a98_rgb_linear.py (coloraide.spaces.a98_rgb_linear)
LOG:  Metadata not found for coloraide.spaces.prophoto_rgb
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/prophoto_rgb.py (coloraide.spaces.prophoto_rgb)
LOG:  Metadata not found for coloraide.spaces.prophoto_rgb_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/prophoto_rgb_linear.py (coloraide.spaces.prophoto_rgb_linear)
LOG:  Metadata not found for coloraide.spaces.rec2020
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2020.py (coloraide.spaces.rec2020)
LOG:  Metadata not found for coloraide.spaces.rec2020_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2020_linear.py (coloraide.spaces.rec2020_linear)
LOG:  Metadata not found for coloraide.spaces.xyz_d65
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/xyz_d65.py (coloraide.spaces.xyz_d65)
LOG:  Metadata not found for coloraide.spaces.xyz_d50
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/xyz_d50.py (coloraide.spaces.xyz_d50)
LOG:  Metadata not found for coloraide.spaces.rec2100_pq
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2100_pq.py (coloraide.spaces.rec2100_pq)
LOG:  Metadata not found for coloraide.spaces.rec2100_hlg
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2100_hlg.py (coloraide.spaces.rec2100_hlg)
LOG:  Metadata not found for coloraide.spaces.rec2100_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2100_linear.py (coloraide.spaces.rec2100_linear)
LOG:  Metadata not found for coloraide.distance.delta_e_76
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_76.py (coloraide.distance.delta_e_76)
LOG:  Metadata not found for coloraide.distance.delta_e_94
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_94.py (coloraide.distance.delta_e_94)
LOG:  Metadata not found for coloraide.distance.delta_e_cmc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_cmc.py (coloraide.distance.delta_e_cmc)
LOG:  Metadata not found for coloraide.distance.delta_e_2000
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_2000.py (coloraide.distance.delta_e_2000)
LOG:  Metadata not found for coloraide.distance.delta_e_hyab
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_hyab.py (coloraide.distance.delta_e_hyab)
LOG:  Metadata not found for coloraide.distance.delta_e_ok
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_ok.py (coloraide.distance.delta_e_ok)
LOG:  Metadata not found for coloraide.distance.delta_e_itp
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_itp.py (coloraide.distance.delta_e_itp)
LOG:  Metadata not found for coloraide.distance.delta_e_z
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_z.py (coloraide.distance.delta_e_z)
LOG:  Metadata not found for coloraide.contrast.wcag21
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/contrast/wcag21.py (coloraide.contrast.wcag21)
LOG:  Metadata not found for coloraide.gamut.fit_minde_chroma
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_minde_chroma.py (coloraide.gamut.fit_minde_chroma)
LOG:  Metadata not found for coloraide.gamut.fit_lch_chroma
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_lch_chroma.py (coloraide.gamut.fit_lch_chroma)
LOG:  Metadata not found for coloraide.gamut.fit_oklch_chroma
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_oklch_chroma.py (coloraide.gamut.fit_oklch_chroma)
LOG:  Metadata not found for coloraide.gamut.fit_raytrace
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_raytrace.py (coloraide.gamut.fit_raytrace)
LOG:  Metadata not found for coloraide.filters.w3c_filter_effects
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/filters/w3c_filter_effects.py (coloraide.filters.w3c_filter_effects)
LOG:  Metadata not found for coloraide.filters.cvd
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/filters/cvd.py (coloraide.filters.cvd)
LOG:  Metadata not found for coloraide.interpolate.linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/linear.py (coloraide.interpolate.linear)
LOG:  Metadata not found for coloraide.interpolate.css_linear
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/css_linear.py (coloraide.interpolate.css_linear)
LOG:  Metadata not found for coloraide.interpolate.continuous
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/continuous.py (coloraide.interpolate.continuous)
LOG:  Metadata not found for coloraide.interpolate.bspline
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/bspline.py (coloraide.interpolate.bspline)
LOG:  Metadata not found for coloraide.interpolate.bspline_natural
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/bspline_natural.py (coloraide.interpolate.bspline_natural)
LOG:  Metadata not found for coloraide.interpolate.monotone
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/monotone.py (coloraide.interpolate.monotone)
LOG:  Metadata not found for coloraide.temperature.ohno_2013
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/ohno_2013.py (coloraide.temperature.ohno_2013)
LOG:  Metadata not found for coloraide.temperature.robertson_1968
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/robertson_1968.py (coloraide.temperature.robertson_1968)
LOG:  Metadata not found for coloraide.cat
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/cat.py (coloraide.cat)
LOG:  Metadata not found for coloraide.distance
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/__init__.py (coloraide.distance)
LOG:  Metadata not found for coloraide.convert
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/convert.py (coloraide.convert)
LOG:  Metadata not found for coloraide.gamut
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/__init__.py (coloraide.gamut)
LOG:  Metadata not found for coloraide.compositing
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/compositing/__init__.py (coloraide.compositing)
LOG:  Metadata not found for coloraide.filters
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/filters/__init__.py (coloraide.filters)
LOG:  Metadata not found for coloraide.contrast
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/contrast/__init__.py (coloraide.contrast)
LOG:  Metadata not found for coloraide.harmonies
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/harmonies.py (coloraide.harmonies)
LOG:  Metadata not found for coloraide.average
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/average.py (coloraide.average)
LOG:  Metadata not found for coloraide.temperature
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/__init__.py (coloraide.temperature)
LOG:  Metadata not found for coloraide.util
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/util.py (coloraide.util)
LOG:  Metadata not found for coloraide.deprecate
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/deprecate.py (coloraide.deprecate)
LOG:  Metadata not found for coloraide.css
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/__init__.py (coloraide.css)
LOG:  Metadata not found for coloraide.types
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/types.py (coloraide.types)
LOG:  Metadata not found for coloraide.spaces
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/__init__.py (coloraide.spaces)
LOG:  Metadata not found for functools
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/functools.pyi (functools)
LOG:  Metadata not found for random
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/random.pyi (random)
LOG:  Metadata not found for math
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/math.pyi (math)
LOG:  Metadata not found for itertools
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/itertools.pyi (itertools)
LOG:  Metadata not found for cmath
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/cmath.pyi (cmath)
LOG:  Metadata not found for operator
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/operator.pyi (operator)
LOG:  Metadata not found for _frozen_importlib
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_frozen_importlib.pyi (_frozen_importlib)
LOG:  Metadata not found for importlib._bootstrap_external
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/_bootstrap_external.pyi (importlib._bootstrap_external)
LOG:  Metadata not found for importlib._abc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/_abc.pyi (importlib._abc)
LOG:  Metadata not found for codecs
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/codecs.pyi (codecs)
LOG:  Metadata not found for posixpath
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/posixpath.pyi (posixpath)
LOG:  Metadata not found for sre_parse
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sre_parse.pyi (sre_parse)
LOG:  Metadata not found for zipfile._path
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/zipfile/_path/__init__.pyi (zipfile._path)
LOG:  Metadata not found for numpy.f2py.__version__
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/__version__.pyi (numpy.f2py.__version__)
LOG:  Metadata not found for numpy.f2py.auxfuncs
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/auxfuncs.pyi (numpy.f2py.auxfuncs)
LOG:  Metadata not found for argparse
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/argparse.pyi (argparse)
LOG:  Metadata not found for pprint
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/pprint.pyi (pprint)
LOG:  Metadata not found for numpy.lib._format_impl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_format_impl.pyi (numpy.lib._format_impl)
LOG:  Metadata not found for numpy.polynomial._polybase
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/_polybase.pyi (numpy.polynomial._polybase)
LOG:  Metadata not found for numpy.polynomial._polytypes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/_polytypes.pyi (numpy.polynomial._polytypes)
LOG:  Metadata not found for numpy.polynomial.polyutils
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/polyutils.pyi (numpy.polynomial.polyutils)
LOG:  Metadata not found for numpy.testing._private
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/_private/__init__.pyi (numpy.testing._private)
LOG:  Metadata not found for unittest.case
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/case.pyi (unittest.case)
LOG:  Metadata not found for unittest.async_case
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/async_case.pyi (unittest.async_case)
LOG:  Metadata not found for unittest.loader
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/loader.pyi (unittest.loader)
LOG:  Metadata not found for unittest.main
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/main.pyi (unittest.main)
LOG:  Metadata not found for unittest.result
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/result.pyi (unittest.result)
LOG:  Metadata not found for unittest.runner
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/runner.pyi (unittest.runner)
LOG:  Metadata not found for unittest.signals
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/signals.pyi (unittest.signals)
LOG:  Metadata not found for unittest.suite
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/suite.pyi (unittest.suite)
LOG:  Metadata not found for textwrap
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/textwrap.pyi (textwrap)
LOG:  Metadata not found for _warnings
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_warnings.pyi (_warnings)
LOG:  Metadata not found for multiprocessing.reduction
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/reduction.pyi (multiprocessing.reduction)
LOG:  Metadata not found for multiprocessing.spawn
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/spawn.pyi (multiprocessing.spawn)
LOG:  Metadata not found for multiprocessing.util
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/util.pyi (multiprocessing.util)
LOG:  Metadata not found for multiprocessing.shared_memory
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/shared_memory.pyi (multiprocessing.shared_memory)
LOG:  Metadata not found for queue
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/queue.pyi (queue)
LOG:  Metadata not found for socket
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/socket.pyi (socket)
LOG:  Metadata not found for concurrent.futures
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/__init__.pyi (concurrent.futures)
LOG:  Metadata not found for ssl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ssl.pyi (ssl)
LOG:  Metadata not found for contextvars
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/contextvars.pyi (contextvars)
LOG:  Metadata not found for _asyncio
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_asyncio.pyi (_asyncio)
LOG:  Metadata not found for concurrent.futures._base
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/_base.pyi (concurrent.futures._base)
LOG:  Metadata not found for asyncio.mixins
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/mixins.pyi (asyncio.mixins)
LOG:  Metadata not found for concurrent
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/__init__.pyi (concurrent)
LOG:  Metadata not found for asyncio.selector_events
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/selector_events.pyi (asyncio.selector_events)
LOG:  Metadata not found for _thread
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_thread.pyi (_thread)
LOG:  Metadata not found for coloraide.spaces.srgb
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/srgb/__init__.py (coloraide.spaces.srgb)
LOG:  Metadata not found for coloraide.css.serialize
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/serialize.py (coloraide.css.serialize)
LOG:  Metadata not found for coloraide.spaces.hsl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hsl/__init__.py (coloraide.spaces.hsl)
LOG:  Metadata not found for coloraide.spaces.hwb
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hwb/__init__.py (coloraide.spaces.hwb)
LOG:  Metadata not found for coloraide.spaces.lab
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lab/__init__.py (coloraide.spaces.lab)
LOG:  Metadata not found for coloraide.spaces.lch
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lch/__init__.py (coloraide.spaces.lch)
LOG:  Metadata not found for coloraide.spaces.oklab
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklab/__init__.py (coloraide.spaces.oklab)
LOG:  Metadata not found for coloraide.spaces.oklch
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklch/__init__.py (coloraide.spaces.oklch)
LOG:  Metadata not found for coloraide.spaces.jzazbz
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzazbz/__init__.py (coloraide.spaces.jzazbz)
LOG:  Metadata not found for coloraide.spaces.jzczhz
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzczhz/__init__.py (coloraide.spaces.jzczhz)
LOG:  Metadata not found for coloraide.spaces.ictcp
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/ictcp/__init__.py (coloraide.spaces.ictcp)
LOG:  Metadata not found for coloraide.css.color_names
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/color_names.py (coloraide.css.color_names)
LOG:  Metadata not found for coloraide.channels
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/channels.py (coloraide.channels)
LOG:  Metadata not found for coloraide.gamut.tools
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/tools.py (coloraide.gamut.tools)
LOG:  Metadata not found for coloraide.temperature.planck
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/planck.py (coloraide.temperature.planck)
LOG:  Metadata not found for coloraide.cmfs
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/cmfs.py (coloraide.cmfs)
LOG:  Metadata not found for coloraide.gamut.pointer
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/pointer.py (coloraide.gamut.pointer)
LOG:  Metadata not found for coloraide.compositing.porter_duff
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/compositing/porter_duff.py (coloraide.compositing.porter_duff)
LOG:  Metadata not found for coloraide.compositing.blend_modes
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/compositing/blend_modes.py (coloraide.compositing.blend_modes)
LOG:  Metadata not found for _random
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_random.pyi (_random)
LOG:  Metadata not found for _operator
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_operator.pyi (_operator)
LOG:  Metadata not found for _frozen_importlib_external
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_frozen_importlib_external.pyi (_frozen_importlib_external)
LOG:  Metadata not found for _codecs
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_codecs.pyi (_codecs)
LOG:  Metadata not found for genericpath
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/genericpath.pyi (genericpath)
LOG:  Metadata not found for numpy.f2py.cfuncs
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/cfuncs.pyi (numpy.f2py.cfuncs)
LOG:  Metadata not found for unittest._log
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/_log.pyi (unittest._log)
LOG:  Metadata not found for copyreg
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/copyreg.pyi (copyreg)
LOG:  Metadata not found for _queue
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_queue.pyi (_queue)
LOG:  Metadata not found for _socket
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_socket.pyi (_socket)
LOG:  Metadata not found for concurrent.futures.process
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/process.pyi (concurrent.futures.process)
LOG:  Metadata not found for concurrent.futures.thread
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/thread.pyi (concurrent.futures.thread)
LOG:  Metadata not found for _ssl
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_ssl.pyi (_ssl)
LOG:  Metadata not found for _contextvars
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_contextvars.pyi (_contextvars)
LOG:  Metadata not found for selectors
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/selectors.pyi (selectors)
LOG:  Metadata not found for signal
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/signal.pyi (signal)
LOG:  Metadata not found for bisect
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/bisect.pyi (bisect)
LOG:  Metadata not found for importlib.metadata
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/metadata/__init__.pyi (importlib.metadata)
LOG:  Metadata not found for importlib.readers
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/readers.pyi (importlib.readers)
LOG:  Metadata not found for weakref
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/weakref.pyi (weakref)
LOG:  Metadata not found for _bisect
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_bisect.pyi (_bisect)
LOG:  Metadata not found for importlib.metadata._meta
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/metadata/_meta.pyi (importlib.metadata._meta)
LOG:  Metadata not found for email.message
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/message.pyi (email.message)
LOG:  Metadata not found for importlib.resources.abc
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/resources/abc.pyi (importlib.resources.abc)
LOG:  Metadata not found for importlib.resources
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/resources/__init__.pyi (importlib.resources)
LOG:  Metadata not found for zipimport
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/zipimport.pyi (zipimport)
LOG:  Metadata not found for _weakref
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_weakref.pyi (_weakref)
LOG:  Metadata not found for _weakrefset
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_weakrefset.pyi (_weakrefset)
LOG:  Metadata not found for email
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/__init__.pyi (email)
LOG:  Metadata not found for email.charset
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/charset.pyi (email.charset)
LOG:  Metadata not found for email.contentmanager
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/contentmanager.pyi (email.contentmanager)
LOG:  Metadata not found for email.errors
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/errors.pyi (email.errors)
LOG:  Metadata not found for email.policy
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/policy.pyi (email.policy)
LOG:  Metadata not found for importlib.resources._common
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/resources/_common.pyi (importlib.resources._common)
LOG:  Metadata not found for email._policybase
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/_policybase.pyi (email._policybase)
LOG:  Metadata not found for email.header
LOG:  Parsing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/header.pyi (email.header)
LOG:  Loaded graph with 394 nodes (1.445 sec)
LOG:  Found 108 SCCs; largest has 92 nodes
LOG:  Processing SCC of size 58 (zipimport zipfile._path zipfile typing_extensions typing types sys._monitoring sys subprocess sre_parse sre_constants sre_compile resource re posixpath pathlib os.path os io importlib.resources.abc importlib.resources._common importlib.resources importlib.readers importlib.metadata._meta importlib.metadata importlib.machinery importlib.abc importlib._bootstrap_external importlib._bootstrap importlib._abc importlib genericpath enum email.policy email.message email.header email.errors email.contentmanager email.charset email._policybase email dataclasses contextlib collections.abc collections codecs ast abc _typeshed.importlib _typeshed _sitebuiltins _io _frozen_importlib_external _frozen_importlib _collections_abc _codecs _ast builtins) as inherently stale
LOG:  Writing zipimport /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/zipimport.pyi zipimport.meta.json zipimport.data.json
LOG:  Cached module zipimport has changed interface
LOG:  Writing zipfile._path /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/zipfile/_path/__init__.pyi zipfile/_path/__init__.meta.json zipfile/_path/__init__.data.json
LOG:  Cached module zipfile._path has changed interface
LOG:  Writing zipfile /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/zipfile/__init__.pyi zipfile/__init__.meta.json zipfile/__init__.data.json
LOG:  Cached module zipfile has changed interface
LOG:  Writing typing_extensions /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/typing_extensions.pyi typing_extensions.meta.json typing_extensions.data.json
LOG:  Cached module typing_extensions has changed interface
LOG:  Writing typing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/typing.pyi typing.meta.json typing.data.json
LOG:  Cached module typing has changed interface
LOG:  Writing types /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/types.pyi types.meta.json types.data.json
LOG:  Cached module types has changed interface
LOG:  Writing sys._monitoring /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sys/_monitoring.pyi sys/_monitoring.meta.json sys/_monitoring.data.json
LOG:  Cached module sys._monitoring has changed interface
LOG:  Writing sys /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sys/__init__.pyi sys/__init__.meta.json sys/__init__.data.json
LOG:  Cached module sys has changed interface
LOG:  Writing subprocess /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/subprocess.pyi subprocess.meta.json subprocess.data.json
LOG:  Cached module subprocess has changed interface
LOG:  Writing sre_parse /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sre_parse.pyi sre_parse.meta.json sre_parse.data.json
LOG:  Cached module sre_parse has changed interface
LOG:  Writing sre_constants /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sre_constants.pyi sre_constants.meta.json sre_constants.data.json
LOG:  Cached module sre_constants has changed interface
LOG:  Writing sre_compile /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/sre_compile.pyi sre_compile.meta.json sre_compile.data.json
LOG:  Cached module sre_compile has changed interface
LOG:  Writing resource /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/resource.pyi resource.meta.json resource.data.json
LOG:  Cached module resource has changed interface
LOG:  Writing re /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/re.pyi re.meta.json re.data.json
LOG:  Cached module re has changed interface
LOG:  Writing posixpath /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/posixpath.pyi posixpath.meta.json posixpath.data.json
LOG:  Cached module posixpath has changed interface
LOG:  Writing pathlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/pathlib.pyi pathlib.meta.json pathlib.data.json
LOG:  Cached module pathlib has changed interface
LOG:  Writing os.path /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/os/path.pyi os/path.meta.json os/path.data.json
LOG:  Cached module os.path has changed interface
LOG:  Writing os /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/os/__init__.pyi os/__init__.meta.json os/__init__.data.json
LOG:  Cached module os has changed interface
LOG:  Writing io /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/io.pyi io.meta.json io.data.json
LOG:  Cached module io has changed interface
LOG:  Writing importlib.resources.abc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/resources/abc.pyi importlib/resources/abc.meta.json importlib/resources/abc.data.json
LOG:  Cached module importlib.resources.abc has changed interface
LOG:  Writing importlib.resources._common /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/resources/_common.pyi importlib/resources/_common.meta.json importlib/resources/_common.data.json
LOG:  Cached module importlib.resources._common has changed interface
LOG:  Writing importlib.resources /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/resources/__init__.pyi importlib/resources/__init__.meta.json importlib/resources/__init__.data.json
LOG:  Cached module importlib.resources has changed interface
LOG:  Writing importlib.readers /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/readers.pyi importlib/readers.meta.json importlib/readers.data.json
LOG:  Cached module importlib.readers has changed interface
LOG:  Writing importlib.metadata._meta /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/metadata/_meta.pyi importlib/metadata/_meta.meta.json importlib/metadata/_meta.data.json
LOG:  Cached module importlib.metadata._meta has changed interface
LOG:  Writing importlib.metadata /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/metadata/__init__.pyi importlib/metadata/__init__.meta.json importlib/metadata/__init__.data.json
LOG:  Cached module importlib.metadata has changed interface
LOG:  Writing importlib.machinery /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/machinery.pyi importlib/machinery.meta.json importlib/machinery.data.json
LOG:  Cached module importlib.machinery has changed interface
LOG:  Writing importlib.abc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/abc.pyi importlib/abc.meta.json importlib/abc.data.json
LOG:  Cached module importlib.abc has changed interface
LOG:  Writing importlib._bootstrap_external /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/_bootstrap_external.pyi importlib/_bootstrap_external.meta.json importlib/_bootstrap_external.data.json
LOG:  Cached module importlib._bootstrap_external has changed interface
LOG:  Writing importlib._bootstrap /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/_bootstrap.pyi importlib/_bootstrap.meta.json importlib/_bootstrap.data.json
LOG:  Cached module importlib._bootstrap has changed interface
LOG:  Writing importlib._abc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/_abc.pyi importlib/_abc.meta.json importlib/_abc.data.json
LOG:  Cached module importlib._abc has changed interface
LOG:  Writing importlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/importlib/__init__.pyi importlib/__init__.meta.json importlib/__init__.data.json
LOG:  Cached module importlib has changed interface
LOG:  Writing genericpath /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/genericpath.pyi genericpath.meta.json genericpath.data.json
LOG:  Cached module genericpath has changed interface
LOG:  Writing enum /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/enum.pyi enum.meta.json enum.data.json
LOG:  Cached module enum has changed interface
LOG:  Writing email.policy /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/policy.pyi email/policy.meta.json email/policy.data.json
LOG:  Cached module email.policy has changed interface
LOG:  Writing email.message /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/message.pyi email/message.meta.json email/message.data.json
LOG:  Cached module email.message has changed interface
LOG:  Writing email.header /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/header.pyi email/header.meta.json email/header.data.json
LOG:  Cached module email.header has changed interface
LOG:  Writing email.errors /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/errors.pyi email/errors.meta.json email/errors.data.json
LOG:  Cached module email.errors has changed interface
LOG:  Writing email.contentmanager /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/contentmanager.pyi email/contentmanager.meta.json email/contentmanager.data.json
LOG:  Cached module email.contentmanager has changed interface
LOG:  Writing email.charset /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/charset.pyi email/charset.meta.json email/charset.data.json
LOG:  Cached module email.charset has changed interface
LOG:  Writing email._policybase /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/_policybase.pyi email/_policybase.meta.json email/_policybase.data.json
LOG:  Cached module email._policybase has changed interface
LOG:  Writing email /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/email/__init__.pyi email/__init__.meta.json email/__init__.data.json
LOG:  Cached module email has changed interface
LOG:  Writing dataclasses /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/dataclasses.pyi dataclasses.meta.json dataclasses.data.json
LOG:  Cached module dataclasses has changed interface
LOG:  Writing contextlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/contextlib.pyi contextlib.meta.json contextlib.data.json
LOG:  Cached module contextlib has changed interface
LOG:  Writing collections.abc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/collections/abc.pyi collections/abc.meta.json collections/abc.data.json
LOG:  Cached module collections.abc has changed interface
LOG:  Writing collections /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/collections/__init__.pyi collections/__init__.meta.json collections/__init__.data.json
LOG:  Cached module collections has changed interface
LOG:  Writing codecs /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/codecs.pyi codecs.meta.json codecs.data.json
LOG:  Cached module codecs has changed interface
LOG:  Writing ast /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ast.pyi ast.meta.json ast.data.json
LOG:  Cached module ast has changed interface
LOG:  Writing abc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/abc.pyi abc.meta.json abc.data.json
LOG:  Cached module abc has changed interface
LOG:  Writing _typeshed.importlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_typeshed/importlib.pyi _typeshed/importlib.meta.json _typeshed/importlib.data.json
LOG:  Cached module _typeshed.importlib has changed interface
LOG:  Writing _typeshed /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_typeshed/__init__.pyi _typeshed/__init__.meta.json _typeshed/__init__.data.json
LOG:  Cached module _typeshed has changed interface
LOG:  Writing _sitebuiltins /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_sitebuiltins.pyi _sitebuiltins.meta.json _sitebuiltins.data.json
LOG:  Cached module _sitebuiltins has changed interface
LOG:  Writing _io /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_io.pyi _io.meta.json _io.data.json
LOG:  Cached module _io has changed interface
LOG:  Writing _frozen_importlib_external /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_frozen_importlib_external.pyi _frozen_importlib_external.meta.json _frozen_importlib_external.data.json
LOG:  Cached module _frozen_importlib_external has changed interface
LOG:  Writing _frozen_importlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_frozen_importlib.pyi _frozen_importlib.meta.json _frozen_importlib.data.json
LOG:  Cached module _frozen_importlib has changed interface
LOG:  Writing _collections_abc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_collections_abc.pyi _collections_abc.meta.json _collections_abc.data.json
LOG:  Cached module _collections_abc has changed interface
LOG:  Writing _codecs /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_codecs.pyi _codecs.meta.json _codecs.data.json
LOG:  Cached module _codecs has changed interface
LOG:  Writing _ast /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_ast.pyi _ast.meta.json _ast.data.json
LOG:  Cached module _ast has changed interface
LOG:  Writing builtins /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/builtins.pyi builtins.meta.json builtins.data.json
LOG:  Cached module builtins has changed interface
LOG:  Processing SCC singleton (_weakrefset) as inherently stale with stale deps (builtins collections.abc sys types typing typing_extensions)
LOG:  Writing _weakrefset /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_weakrefset.pyi _weakrefset.meta.json _weakrefset.data.json
LOG:  Cached module _weakrefset has changed interface
LOG:  Processing SCC singleton (_bisect) as inherently stale with stale deps (_typeshed builtins collections.abc sys typing)
LOG:  Writing _bisect /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_bisect.pyi _bisect.meta.json _bisect.data.json
LOG:  Cached module _bisect has changed interface
LOG:  Processing SCC singleton (signal) as inherently stale with stale deps (_typeshed builtins collections.abc enum sys types typing typing_extensions)
LOG:  Writing signal /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/signal.pyi signal.meta.json signal.data.json
LOG:  Cached module signal has changed interface
LOG:  Processing SCC singleton (selectors) as inherently stale with stale deps (_typeshed abc builtins collections.abc sys typing typing_extensions)
LOG:  Writing selectors /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/selectors.pyi selectors.meta.json selectors.data.json
LOG:  Cached module selectors has changed interface
LOG:  Processing SCC singleton (_contextvars) as inherently stale with stale deps (builtins collections.abc sys types typing typing_extensions)
LOG:  Writing _contextvars /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_contextvars.pyi _contextvars.meta.json _contextvars.data.json
LOG:  Cached module _contextvars has changed interface
LOG:  Processing SCC singleton (_queue) as inherently stale with stale deps (builtins sys types typing)
LOG:  Writing _queue /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_queue.pyi _queue.meta.json _queue.data.json
LOG:  Cached module _queue has changed interface
LOG:  Processing SCC singleton (copyreg) as inherently stale with stale deps (builtins collections.abc typing typing_extensions)
LOG:  Writing copyreg /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/copyreg.pyi copyreg.meta.json copyreg.data.json
LOG:  Cached module copyreg has changed interface
LOG:  Processing SCC singleton (_random) as inherently stale with stale deps (builtins typing_extensions)
LOG:  Writing _random /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_random.pyi _random.meta.json _random.data.json
LOG:  Cached module _random has changed interface
LOG:  Processing SCC singleton (coloraide.cmfs) as inherently stale with stale deps (builtins)
LOG:  Writing coloraide.cmfs /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/cmfs.py coloraide/cmfs.meta.json coloraide/cmfs.data.json
LOG:  Cached module coloraide.cmfs has changed interface
LOG:  Processing SCC singleton (concurrent) as inherently stale with stale deps (builtins)
LOG:  Writing concurrent /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/__init__.pyi concurrent/__init__.meta.json concurrent/__init__.data.json
LOG:  Cached module concurrent has changed interface
LOG:  Processing SCC of size 2 (_socket socket) as inherently stale with stale deps (_typeshed builtins collections.abc enum io sys typing typing_extensions)
LOG:  Writing _socket /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_socket.pyi _socket.meta.json _socket.data.json
LOG:  Cached module _socket has changed interface
LOG:  Writing socket /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/socket.pyi socket.meta.json socket.data.json
LOG:  Cached module socket has changed interface
LOG:  Processing SCC singleton (multiprocessing.shared_memory) as inherently stale with stale deps (builtins collections.abc sys types typing typing_extensions)
LOG:  Writing multiprocessing.shared_memory /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/shared_memory.pyi multiprocessing/shared_memory.meta.json multiprocessing/shared_memory.data.json
LOG:  Cached module multiprocessing.shared_memory has changed interface
LOG:  Processing SCC singleton (multiprocessing.spawn) as inherently stale with stale deps (builtins collections.abc types typing)
LOG:  Writing multiprocessing.spawn /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/spawn.pyi multiprocessing/spawn.meta.json multiprocessing/spawn.data.json
LOG:  Cached module multiprocessing.spawn has changed interface
LOG:  Processing SCC singleton (_warnings) as inherently stale with stale deps (builtins sys typing)
LOG:  Writing _warnings /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_warnings.pyi _warnings.meta.json _warnings.data.json
LOG:  Cached module _warnings has changed interface
LOG:  Processing SCC singleton (textwrap) as inherently stale with stale deps (builtins collections.abc re)
LOG:  Writing textwrap /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/textwrap.pyi textwrap.meta.json textwrap.data.json
LOG:  Cached module textwrap has changed interface
LOG:  Processing SCC singleton (numpy.testing._private) as inherently stale with stale deps (builtins)
LOG:  Writing numpy.testing._private /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/_private/__init__.pyi numpy/testing/_private/__init__.meta.json numpy/testing/_private/__init__.data.json
LOG:  Cached module numpy.testing._private has changed interface
LOG:  Processing SCC singleton (pprint) as inherently stale with stale deps (builtins sys typing)
LOG:  Writing pprint /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/pprint.pyi pprint.meta.json pprint.data.json
LOG:  Cached module pprint has changed interface
LOG:  Processing SCC singleton (argparse) as inherently stale with stale deps (_typeshed builtins collections.abc re sys typing typing_extensions)
LOG:  Writing argparse /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/argparse.pyi argparse.meta.json argparse.data.json
LOG:  Cached module argparse has changed interface
LOG:  Processing SCC of size 2 (_operator operator) as inherently stale with stale deps (_typeshed builtins collections.abc sys typing typing_extensions)
LOG:  Writing _operator /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_operator.pyi _operator.meta.json _operator.data.json
LOG:  Cached module _operator has changed interface
LOG:  Writing operator /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/operator.pyi operator.meta.json operator.data.json
LOG:  Cached module operator has changed interface
LOG:  Processing SCC singleton (cmath) as inherently stale with stale deps (builtins typing typing_extensions)
LOG:  Writing cmath /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/cmath.pyi cmath.meta.json cmath.data.json
LOG:  Cached module cmath has changed interface
LOG:  Processing SCC singleton (itertools) as inherently stale with stale deps (_typeshed builtins collections.abc sys types typing typing_extensions)
LOG:  Writing itertools /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/itertools.pyi itertools.meta.json itertools.data.json
LOG:  Cached module itertools has changed interface
LOG:  Processing SCC singleton (math) as inherently stale with stale deps (_typeshed builtins collections.abc sys typing typing_extensions)
LOG:  Writing math /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/math.pyi math.meta.json math.data.json
LOG:  Cached module math has changed interface
LOG:  Processing SCC singleton (functools) as inherently stale with stale deps (_typeshed builtins collections.abc sys types typing typing_extensions)
LOG:  Writing functools /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/functools.pyi functools.meta.json functools.data.json
LOG:  Cached module functools has changed interface
LOG:  Processing SCC singleton (coloraide.css) as inherently stale with stale deps (builtins)
LOG:  Writing coloraide.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/__init__.py coloraide/css/__init__.meta.json coloraide/css/__init__.data.json
LOG:  Cached module coloraide.css has changed interface
LOG:  Processing SCC singleton (string) as inherently stale with stale deps (_typeshed builtins collections.abc re sys typing typing_extensions)
LOG:  Writing string /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/string.pyi string.meta.json string.data.json
LOG:  Cached module string has changed interface
LOG:  Processing SCC singleton (asyncio.timeouts) as inherently stale with stale deps (builtins types typing typing_extensions)
LOG:  Writing asyncio.timeouts /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/timeouts.pyi asyncio/timeouts.meta.json asyncio/timeouts.data.json
LOG:  Cached module asyncio.timeouts has changed interface
LOG:  Processing SCC singleton (asyncio.threads) as inherently stale with stale deps (builtins collections.abc typing typing_extensions)
LOG:  Writing asyncio.threads /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/threads.pyi asyncio/threads.meta.json asyncio/threads.data.json
LOG:  Cached module asyncio.threads has changed interface
LOG:  Processing SCC singleton (asyncio.exceptions) as inherently stale with stale deps (builtins sys)
LOG:  Writing asyncio.exceptions /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/exceptions.pyi asyncio/exceptions.meta.json asyncio/exceptions.data.json
LOG:  Cached module asyncio.exceptions has changed interface
LOG:  Processing SCC singleton (asyncio.coroutines) as inherently stale with stale deps (builtins collections.abc sys typing typing_extensions)
LOG:  Writing asyncio.coroutines /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/coroutines.pyi asyncio/coroutines.meta.json asyncio/coroutines.data.json
LOG:  Cached module asyncio.coroutines has changed interface
LOG:  Processing SCC singleton (multiprocessing.process) as inherently stale with stale deps (builtins collections.abc typing)
LOG:  Writing multiprocessing.process /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/process.pyi multiprocessing/process.meta.json multiprocessing/process.data.json
LOG:  Cached module multiprocessing.process has changed interface
LOG:  Processing SCC singleton (multiprocessing.queues) as inherently stale with stale deps (builtins sys types typing)
LOG:  Writing multiprocessing.queues /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/queues.pyi multiprocessing/queues.meta.json multiprocessing/queues.data.json
LOG:  Cached module multiprocessing.queues has changed interface
LOG:  Processing SCC singleton (numbers) as inherently stale with stale deps (_typeshed abc builtins typing)
LOG:  Writing numbers /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/numbers.pyi numbers.meta.json numbers.data.json
LOG:  Cached module numbers has changed interface
LOG:  Processing SCC singleton (time) as inherently stale with stale deps (_typeshed builtins sys typing typing_extensions)
LOG:  Writing time /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/time.pyi time.meta.json time.data.json
LOG:  Cached module time has changed interface
LOG:  Processing SCC singleton (numpy.lib._version) as inherently stale with stale deps (builtins)
LOG:  Writing numpy.lib._version /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_version.pyi numpy/lib/_version.meta.json numpy/lib/_version.data.json
LOG:  Cached module numpy.lib._version has changed interface
LOG:  Processing SCC singleton (numpy.lib.introspect) as inherently stale with stale deps (builtins)
LOG:  Writing numpy.lib.introspect /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/introspect.pyi numpy/lib/introspect.meta.json numpy/lib/introspect.data.json
LOG:  Cached module numpy.lib.introspect has changed interface
LOG:  Processing SCC singleton (numpy._typing._shape) as inherently stale with stale deps (builtins collections.abc typing)
LOG:  Writing numpy._typing._shape /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_shape.py numpy/_typing/_shape.meta.json numpy/_typing/_shape.data.json
LOG:  Cached module numpy._typing._shape has changed interface
LOG:  Processing SCC singleton (numpy._typing._nbit_base) as inherently stale with stale deps (builtins typing typing_extensions)
LOG:  Writing numpy._typing._nbit_base /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_nbit_base.pyi numpy/_typing/_nbit_base.meta.json numpy/_typing/_nbit_base.data.json
LOG:  Cached module numpy._typing._nbit_base has changed interface
LOG:  Processing SCC singleton (numpy._typing._char_codes) as inherently stale with stale deps (builtins typing)
LOG:  Writing numpy._typing._char_codes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_char_codes.py numpy/_typing/_char_codes.meta.json numpy/_typing/_char_codes.data.json
LOG:  Cached module numpy._typing._char_codes has changed interface
LOG:  Processing SCC singleton (numpy.lib._datasource) as inherently stale with stale deps (_typeshed builtins pathlib typing)
LOG:  Writing numpy.lib._datasource /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_datasource.pyi numpy/lib/_datasource.meta.json numpy/lib/_datasource.data.json
LOG:  Cached module numpy.lib._datasource has changed interface
LOG:  Processing SCC singleton (numpy._typing._nested_sequence) as inherently stale with stale deps (builtins collections.abc typing)
LOG:  Writing numpy._typing._nested_sequence /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_nested_sequence.py numpy/_typing/_nested_sequence.meta.json numpy/_typing/_nested_sequence.data.json
LOG:  Cached module numpy._typing._nested_sequence has changed interface
LOG:  Processing SCC singleton (numpy._core) as inherently stale with stale deps (builtins)
LOG:  Writing numpy._core /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/__init__.pyi numpy/_core/__init__.meta.json numpy/_core/__init__.data.json
LOG:  Cached module numpy._core has changed interface
LOG:  Processing SCC singleton (_hashlib) as inherently stale with stale deps (_typeshed builtins collections.abc sys types typing typing_extensions)
LOG:  Writing _hashlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_hashlib.pyi _hashlib.meta.json _hashlib.data.json
LOG:  Cached module _hashlib has changed interface
LOG:  Processing SCC singleton (_blake2) as inherently stale with stale deps (_typeshed builtins sys typing typing_extensions)
LOG:  Writing _blake2 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_blake2.pyi _blake2.meta.json _blake2.data.json
LOG:  Cached module _blake2 has changed interface
LOG:  Processing SCC singleton (PIL._imaging) as inherently stale with stale deps (_typeshed builtins collections.abc typing)
LOG:  Writing PIL._imaging /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/_imaging.pyi PIL/_imaging.meta.json PIL/_imaging.data.json
LOG:  Cached module PIL._imaging has changed interface
LOG:  Processing SCC singleton (PIL.PyAccess) as inherently stale with stale deps (_typeshed builtins typing)
LOG:  Writing PIL.PyAccess /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/PyAccess.pyi PIL/PyAccess.meta.json PIL/PyAccess.data.json
LOG:  Cached module PIL.PyAccess has changed interface
LOG:  Processing SCC singleton (uuid) as inherently stale with stale deps (_typeshed builtins enum sys typing_extensions)
LOG:  Writing uuid /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/uuid.pyi uuid.meta.json uuid.data.json
LOG:  Cached module uuid has changed interface
LOG:  Processing SCC singleton (array) as inherently stale with stale deps (_typeshed builtins collections.abc sys types typing typing_extensions)
LOG:  Writing array /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/array.pyi array.meta.json array.data.json
LOG:  Cached module array has changed interface
LOG:  Processing SCC of size 3 (_ctypes ctypes._endian ctypes) as inherently stale with stale deps (_typeshed abc builtins collections.abc sys types typing typing_extensions)
LOG:  Writing _ctypes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_ctypes.pyi _ctypes.meta.json _ctypes.data.json
LOG:  Cached module _ctypes has changed interface
LOG:  Writing ctypes._endian /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ctypes/_endian.pyi ctypes/_endian.meta.json ctypes/_endian.data.json
LOG:  Cached module ctypes._endian has changed interface
LOG:  Writing ctypes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ctypes/__init__.pyi ctypes/__init__.meta.json ctypes/__init__.data.json
LOG:  Cached module ctypes has changed interface
LOG:  Processing SCC singleton (mmap) as inherently stale with stale deps (_typeshed builtins collections.abc sys typing typing_extensions)
LOG:  Writing mmap /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/mmap.pyi mmap.meta.json mmap.data.json
LOG:  Cached module mmap has changed interface
LOG:  Processing SCC singleton (numpy._globals) as inherently stale with stale deps (builtins enum typing)
LOG:  Writing numpy._globals /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_globals.pyi numpy/_globals.meta.json numpy/_globals.data.json
LOG:  Cached module numpy._globals has changed interface
LOG:  Processing SCC singleton (numpy._expired_attrs_2_0) as inherently stale with stale deps (builtins typing)
LOG:  Writing numpy._expired_attrs_2_0 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_expired_attrs_2_0.pyi numpy/_expired_attrs_2_0.meta.json numpy/_expired_attrs_2_0.data.json
LOG:  Cached module numpy._expired_attrs_2_0 has changed interface
LOG:  Processing SCC singleton (numpy.version) as inherently stale with stale deps (builtins typing)
LOG:  Writing numpy.version /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/version.pyi numpy/version.meta.json numpy/version.data.json
LOG:  Cached module numpy.version has changed interface
LOG:  Processing SCC singleton (numpy.exceptions) as inherently stale with stale deps (builtins typing)
LOG:  Writing numpy.exceptions /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/exceptions.pyi numpy/exceptions.meta.json numpy/exceptions.data.json
LOG:  Cached module numpy.exceptions has changed interface
LOG:  Processing SCC singleton (numpy.core) as inherently stale with stale deps (builtins)
LOG:  Writing numpy.core /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/core/__init__.pyi numpy/core/__init__.meta.json numpy/core/__init__.data.json
LOG:  Cached module numpy.core has changed interface
LOG:  Processing SCC singleton (numpy._pytesttester) as inherently stale with stale deps (builtins collections.abc typing)
LOG:  Writing numpy._pytesttester /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_pytesttester.pyi numpy/_pytesttester.meta.json numpy/_pytesttester.data.json
LOG:  Cached module numpy._pytesttester has changed interface
LOG:  Processing SCC singleton (numpy.__config__) as inherently stale with stale deps (builtins enum types typing)
LOG:  Writing numpy.__config__ /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/__config__.pyi numpy/__config__.meta.json numpy/__config__.data.json
LOG:  Cached module numpy.__config__ has changed interface
LOG:  Processing SCC of size 2 (_pickle pickle) as inherently stale with stale deps (_typeshed builtins collections.abc sys typing typing_extensions)
LOG:  Writing _pickle /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_pickle.pyi _pickle.meta.json _pickle.data.json
LOG:  Cached module _pickle has changed interface
LOG:  Writing pickle /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/pickle.pyi pickle.meta.json pickle.data.json
LOG:  Cached module pickle has changed interface
LOG:  Processing SCC singleton (__future__) as inherently stale with stale deps (builtins typing_extensions)
LOG:  Writing __future__ /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/__future__.pyi __future__.meta.json __future__.data.json
LOG:  Cached module __future__ has changed interface
LOG:  Processing SCC singleton (PIL) as inherently stale with stale deps (builtins)
LOG:  Writing PIL /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/__init__.pyi PIL/__init__.meta.json PIL/__init__.data.json
LOG:  Cached module PIL has changed interface
LOG:  Processing SCC singleton (imgcolorshine) as inherently stale with stale deps (builtins)
LOG:  Writing imgcolorshine /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/__init__.py imgcolorshine/__init__.meta.json imgcolorshine/__init__.data.json
LOG:  Cached module imgcolorshine has changed interface
LOG:  Processing SCC of size 2 (_weakref weakref) as inherently stale with stale deps (_typeshed _weakrefset builtins collections.abc sys types typing typing_extensions)
LOG:  Writing _weakref /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_weakref.pyi _weakref.meta.json _weakref.data.json
LOG:  Cached module _weakref has changed interface
LOG:  Writing weakref /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/weakref.pyi weakref.meta.json weakref.data.json
LOG:  Cached module weakref has changed interface
LOG:  Processing SCC singleton (bisect) as inherently stale with stale deps (_bisect builtins)
LOG:  Writing bisect /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/bisect.pyi bisect.meta.json bisect.data.json
LOG:  Cached module bisect has changed interface
LOG:  Processing SCC singleton (coloraide.compositing.porter_duff) as inherently stale with stale deps (__future__ abc builtins)
LOG:  Writing coloraide.compositing.porter_duff /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/compositing/porter_duff.py coloraide/compositing/porter_duff.meta.json coloraide/compositing/porter_duff.data.json
LOG:  Cached module coloraide.compositing.porter_duff has changed interface
LOG:  Processing SCC singleton (coloraide.channels) as inherently stale with stale deps (__future__ builtins)
LOG:  Writing coloraide.channels /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/channels.py coloraide/channels.meta.json coloraide/channels.data.json
LOG:  Cached module coloraide.channels has changed interface
LOG:  Processing SCC singleton (contextvars) as inherently stale with stale deps (_contextvars builtins)
LOG:  Writing contextvars /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/contextvars.pyi contextvars.meta.json contextvars.data.json
LOG:  Cached module contextvars has changed interface
LOG:  Processing SCC of size 2 (_ssl ssl) as inherently stale with stale deps (_typeshed builtins collections.abc enum socket sys typing typing_extensions)
LOG:  Writing _ssl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_ssl.pyi _ssl.meta.json _ssl.data.json
LOG:  Cached module _ssl has changed interface
LOG:  Writing ssl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/ssl.pyi ssl.meta.json ssl.data.json
LOG:  Cached module ssl has changed interface
LOG:  Processing SCC singleton (numpy.f2py.__version__) as inherently stale with stale deps (builtins numpy.version)
LOG:  Writing numpy.f2py.__version__ /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/__version__.pyi numpy/f2py/__version__.meta.json numpy/f2py/__version__.data.json
LOG:  Cached module numpy.f2py.__version__ has changed interface
LOG:  Processing SCC of size 2 (_thread threading) as inherently stale with stale deps (_typeshed builtins collections.abc signal sys types typing typing_extensions)
LOG:  Writing _thread /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_thread.pyi _thread.meta.json _thread.data.json
LOG:  Cached module _thread has changed interface
LOG:  Writing threading /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/threading.pyi threading.meta.json threading.data.json
LOG:  Cached module threading has changed interface
LOG:  Processing SCC singleton (multiprocessing.connection) as inherently stale with stale deps (_typeshed builtins collections.abc socket sys types typing typing_extensions)
LOG:  Writing multiprocessing.connection /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/connection.pyi multiprocessing/connection.meta.json multiprocessing/connection.data.json
LOG:  Cached module multiprocessing.connection has changed interface
LOG:  Processing SCC singleton (warnings) as inherently stale with stale deps (_warnings builtins collections.abc re sys types typing typing_extensions)
LOG:  Writing warnings /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/warnings.pyi warnings.meta.json warnings.data.json
LOG:  Cached module warnings has changed interface
LOG:  Processing SCC singleton (numpy._typing._nbit) as inherently stale with stale deps (builtins numpy._typing._nbit_base typing)
LOG:  Writing numpy._typing._nbit /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_nbit.py numpy/_typing/_nbit.meta.json numpy/_typing/_nbit.data.json
LOG:  Cached module numpy._typing._nbit has changed interface
LOG:  Processing SCC singleton (coloraide.__meta__) as inherently stale with stale deps (__future__ builtins collections re)
LOG:  Writing coloraide.__meta__ /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/__meta__.py coloraide/__meta__.meta.json coloraide/__meta__.data.json
LOG:  Cached module coloraide.__meta__ has changed interface
LOG:  Processing SCC of size 2 (_decimal decimal) as inherently stale with stale deps (builtins collections.abc numbers sys types typing typing_extensions)
LOG:  Writing _decimal /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_decimal.pyi _decimal.meta.json _decimal.data.json
LOG:  Cached module _decimal has changed interface
LOG:  Writing decimal /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/decimal.pyi decimal.meta.json decimal.data.json
LOG:  Cached module decimal has changed interface
LOG:  Processing SCC singleton (datetime) as inherently stale with stale deps (abc builtins sys time typing typing_extensions)
LOG:  Writing datetime /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/datetime.pyi datetime.meta.json datetime.data.json
LOG:  Cached module datetime has changed interface
LOG:  Processing SCC singleton (hashlib) as inherently stale with stale deps (_blake2 _hashlib _typeshed builtins collections.abc sys typing)
LOG:  Writing hashlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/hashlib.pyi hashlib.meta.json hashlib.data.json
LOG:  Cached module hashlib has changed interface
LOG:  Processing SCC of size 3 (PIL.ImagePalette PIL.ImageFilter PIL.Image) as inherently stale with stale deps (PIL.PyAccess PIL._imaging _typeshed builtins collections.abc enum pathlib typing typing_extensions)
LOG:  Writing PIL.ImagePalette /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/ImagePalette.pyi PIL/ImagePalette.meta.json PIL/ImagePalette.data.json
LOG:  Cached module PIL.ImagePalette has changed interface
LOG:  Writing PIL.ImageFilter /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/ImageFilter.pyi PIL/ImageFilter.meta.json PIL/ImageFilter.data.json
LOG:  Cached module PIL.ImageFilter has changed interface
LOG:  Writing PIL.Image /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/PIL-stubs/Image.pyi PIL/Image.meta.json PIL/Image.data.json
LOG:  Cached module PIL.Image has changed interface
LOG:  Processing SCC singleton (numpy.f2py.cfuncs) as inherently stale with stale deps (builtins numpy.f2py.__version__ typing)
LOG:  Writing numpy.f2py.cfuncs /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/cfuncs.pyi numpy/f2py/cfuncs.meta.json numpy/f2py/cfuncs.data.json
LOG:  Cached module numpy.f2py.cfuncs has changed interface
LOG:  Processing SCC singleton (asyncio.mixins) as inherently stale with stale deps (builtins sys threading typing_extensions)
LOG:  Writing asyncio.mixins /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/mixins.pyi asyncio/mixins.meta.json asyncio/mixins.data.json
LOG:  Cached module asyncio.mixins has changed interface
LOG:  Processing SCC singleton (queue) as inherently stale with stale deps (_queue builtins sys threading types typing)
LOG:  Writing queue /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/queue.pyi queue.meta.json queue.data.json
LOG:  Cached module queue has changed interface
LOG:  Processing SCC singleton (coloraide.deprecate) as inherently stale with stale deps (__future__ builtins functools typing warnings)
LOG:  Writing coloraide.deprecate /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/deprecate.py coloraide/deprecate.meta.json coloraide/deprecate.data.json
LOG:  Cached module coloraide.deprecate has changed interface
LOG:  Processing SCC singleton (logging) as inherently stale with stale deps (_typeshed builtins collections.abc io re string sys threading time types typing typing_extensions)
LOG:  Writing logging /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/logging/__init__.pyi logging/__init__.meta.json logging/__init__.data.json
LOG:  Cached module logging has changed interface
LOG:  Processing SCC singleton (fractions) as inherently stale with stale deps (builtins collections.abc decimal numbers sys typing typing_extensions)
LOG:  Writing fractions /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/fractions.pyi fractions.meta.json fractions.data.json
LOG:  Cached module fractions has changed interface
LOG:  Processing SCC singleton (concurrent.futures._base) as inherently stale with stale deps (_typeshed builtins collections.abc logging sys threading types typing typing_extensions)
LOG:  Writing concurrent.futures._base /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/_base.pyi concurrent/futures/_base.meta.json concurrent/futures/_base.data.json
LOG:  Cached module concurrent.futures._base has changed interface
LOG:  Processing SCC singleton (multiprocessing.util) as inherently stale with stale deps (_typeshed builtins collections.abc logging threading typing)
LOG:  Writing multiprocessing.util /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/util.pyi multiprocessing/util.meta.json multiprocessing/util.data.json
LOG:  Cached module multiprocessing.util has changed interface
LOG:  Processing SCC singleton (numpy.f2py.auxfuncs) as inherently stale with stale deps (_typeshed builtins collections.abc numpy.f2py.cfuncs pprint typing)
LOG:  Writing numpy.f2py.auxfuncs /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/auxfuncs.pyi numpy/f2py/auxfuncs.meta.json numpy/f2py/auxfuncs.data.json
LOG:  Cached module numpy.f2py.auxfuncs has changed interface
LOG:  Processing SCC singleton (random) as inherently stale with stale deps (_random _typeshed builtins collections.abc fractions sys typing)
LOG:  Writing random /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/random.pyi random.meta.json random.data.json
LOG:  Cached module random has changed interface
LOG:  Processing SCC singleton (concurrent.futures.thread) as inherently stale with stale deps (builtins collections.abc concurrent.futures._base queue sys threading types typing typing_extensions weakref)
LOG:  Writing concurrent.futures.thread /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/thread.pyi concurrent/futures/thread.meta.json concurrent/futures/thread.data.json
LOG:  Cached module concurrent.futures.thread has changed interface
LOG:  Processing SCC singleton (multiprocessing.popen_spawn_win32) as inherently stale with stale deps (builtins multiprocessing.process multiprocessing.util sys typing)
LOG:  Writing multiprocessing.popen_spawn_win32 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_spawn_win32.pyi multiprocessing/popen_spawn_win32.meta.json multiprocessing/popen_spawn_win32.data.json
LOG:  Cached module multiprocessing.popen_spawn_win32 has changed interface
LOG:  Processing SCC singleton (multiprocessing.popen_fork) as inherently stale with stale deps (builtins multiprocessing.process multiprocessing.util sys typing)
LOG:  Writing multiprocessing.popen_fork /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_fork.pyi multiprocessing/popen_fork.meta.json multiprocessing/popen_fork.data.json
LOG:  Cached module multiprocessing.popen_fork has changed interface
LOG:  Processing SCC singleton (numpy.f2py.f2py2e) as inherently stale with stale deps (argparse builtins collections.abc numpy.f2py.__version__ numpy.f2py.auxfuncs pprint types typing typing_extensions)
LOG:  Writing numpy.f2py.f2py2e /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/f2py2e.pyi numpy/f2py/f2py2e.meta.json numpy/f2py/f2py2e.data.json
LOG:  Cached module numpy.f2py.f2py2e has changed interface
LOG:  Processing SCC of size 85 (coloraide.types coloraide.compositing.blend_modes coloraide.temperature coloraide.contrast coloraide.filters coloraide.convert coloraide.algebra coloraide.contrast.wcag21 coloraide.gamut.tools coloraide.css.color_names coloraide.util coloraide.distance coloraide.filters.cvd coloraide.filters.w3c_filter_effects coloraide.easing coloraide.temperature.planck coloraide.css.serialize coloraide.cat coloraide.distance.delta_e_z coloraide.distance.delta_e_itp coloraide.distance.delta_e_ok coloraide.css.parse coloraide.spaces coloraide.temperature.robertson_1968 coloraide.temperature.ohno_2013 coloraide.spaces.lch coloraide.spaces.lab coloraide.spaces.hwb coloraide.spaces.hsl coloraide.average coloraide.compositing coloraide.distance.delta_e_hyab coloraide.spaces.xyz_d65 coloraide.spaces.srgb_linear coloraide.spaces.hsv coloraide.interpolate coloraide.gamut.pointer coloraide.spaces.ictcp coloraide.spaces.jzczhz coloraide.spaces.jzazbz coloraide.spaces.oklch coloraide.spaces.oklab coloraide.spaces.srgb coloraide.harmonies coloraide.interpolate.continuous coloraide.interpolate.linear coloraide.distance.delta_e_2000 coloraide.distance.delta_e_cmc coloraide.distance.delta_e_94 coloraide.distance.delta_e_76 coloraide.spaces.rec2100_hlg coloraide.spaces.rec2100_pq coloraide.spaces.xyz_d50 coloraide.spaces.rec2020_linear coloraide.spaces.rec2020 coloraide.spaces.prophoto_rgb_linear coloraide.spaces.prophoto_rgb coloraide.spaces.a98_rgb_linear coloraide.spaces.a98_rgb coloraide.spaces.display_p3_linear coloraide.spaces.lch_d65 coloraide.spaces.lab_d65 coloraide.spaces.lch.css coloraide.spaces.lab.css coloraide.spaces.hwb.css coloraide.spaces.hsl.css coloraide.gamut coloraide.interpolate.bspline coloraide.interpolate.css_linear coloraide.spaces.rec2100_linear coloraide.spaces.display_p3 coloraide.spaces.ictcp.css coloraide.spaces.jzczhz.css coloraide.spaces.jzazbz.css coloraide.spaces.oklch.css coloraide.spaces.oklab.css coloraide.spaces.srgb.css coloraide.interpolate.monotone coloraide.interpolate.bspline_natural coloraide.gamut.fit_raytrace coloraide.gamut.fit_minde_chroma coloraide.gamut.fit_oklch_chroma coloraide.gamut.fit_lch_chroma coloraide.color coloraide) as inherently stale with stale deps (__future__ abc bisect builtins cmath coloraide.__meta__ coloraide.channels coloraide.cmfs coloraide.compositing.porter_duff coloraide.css coloraide.deprecate decimal functools itertools math operator random re sys typing)
LOG:  Writing coloraide.types /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/types.py coloraide/types.meta.json coloraide/types.data.json
LOG:  Cached module coloraide.types has changed interface
LOG:  Writing coloraide.compositing.blend_modes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/compositing/blend_modes.py coloraide/compositing/blend_modes.meta.json coloraide/compositing/blend_modes.data.json
LOG:  Cached module coloraide.compositing.blend_modes has changed interface
LOG:  Writing coloraide.temperature /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/__init__.py coloraide/temperature/__init__.meta.json coloraide/temperature/__init__.data.json
LOG:  Cached module coloraide.temperature has changed interface
LOG:  Writing coloraide.contrast /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/contrast/__init__.py coloraide/contrast/__init__.meta.json coloraide/contrast/__init__.data.json
LOG:  Cached module coloraide.contrast has changed interface
LOG:  Writing coloraide.filters /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/filters/__init__.py coloraide/filters/__init__.meta.json coloraide/filters/__init__.data.json
LOG:  Cached module coloraide.filters has changed interface
LOG:  Writing coloraide.convert /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/convert.py coloraide/convert.meta.json coloraide/convert.data.json
LOG:  Cached module coloraide.convert has changed interface
LOG:  Writing coloraide.algebra /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/algebra.py coloraide/algebra.meta.json coloraide/algebra.data.json
LOG:  Cached module coloraide.algebra has changed interface
LOG:  Writing coloraide.contrast.wcag21 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/contrast/wcag21.py coloraide/contrast/wcag21.meta.json coloraide/contrast/wcag21.data.json
LOG:  Cached module coloraide.contrast.wcag21 has changed interface
LOG:  Writing coloraide.gamut.tools /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/tools.py coloraide/gamut/tools.meta.json coloraide/gamut/tools.data.json
LOG:  Cached module coloraide.gamut.tools has changed interface
LOG:  Writing coloraide.css.color_names /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/color_names.py coloraide/css/color_names.meta.json coloraide/css/color_names.data.json
LOG:  Cached module coloraide.css.color_names has changed interface
LOG:  Writing coloraide.util /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/util.py coloraide/util.meta.json coloraide/util.data.json
LOG:  Cached module coloraide.util has changed interface
LOG:  Writing coloraide.distance /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/__init__.py coloraide/distance/__init__.meta.json coloraide/distance/__init__.data.json
LOG:  Cached module coloraide.distance has changed interface
LOG:  Writing coloraide.filters.cvd /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/filters/cvd.py coloraide/filters/cvd.meta.json coloraide/filters/cvd.data.json
LOG:  Cached module coloraide.filters.cvd has changed interface
LOG:  Writing coloraide.filters.w3c_filter_effects /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/filters/w3c_filter_effects.py coloraide/filters/w3c_filter_effects.meta.json coloraide/filters/w3c_filter_effects.data.json
LOG:  Cached module coloraide.filters.w3c_filter_effects has changed interface
LOG:  Writing coloraide.easing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/easing.py coloraide/easing.meta.json coloraide/easing.data.json
LOG:  Cached module coloraide.easing has changed interface
LOG:  Writing coloraide.temperature.planck /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/planck.py coloraide/temperature/planck.meta.json coloraide/temperature/planck.data.json
LOG:  Cached module coloraide.temperature.planck has changed interface
LOG:  Writing coloraide.css.serialize /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/serialize.py coloraide/css/serialize.meta.json coloraide/css/serialize.data.json
LOG:  Cached module coloraide.css.serialize has changed interface
LOG:  Writing coloraide.cat /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/cat.py coloraide/cat.meta.json coloraide/cat.data.json
LOG:  Cached module coloraide.cat has changed interface
LOG:  Writing coloraide.distance.delta_e_z /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_z.py coloraide/distance/delta_e_z.meta.json coloraide/distance/delta_e_z.data.json
LOG:  Cached module coloraide.distance.delta_e_z has changed interface
LOG:  Writing coloraide.distance.delta_e_itp /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_itp.py coloraide/distance/delta_e_itp.meta.json coloraide/distance/delta_e_itp.data.json
LOG:  Cached module coloraide.distance.delta_e_itp has changed interface
LOG:  Writing coloraide.distance.delta_e_ok /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_ok.py coloraide/distance/delta_e_ok.meta.json coloraide/distance/delta_e_ok.data.json
LOG:  Cached module coloraide.distance.delta_e_ok has changed interface
LOG:  Writing coloraide.css.parse /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/css/parse.py coloraide/css/parse.meta.json coloraide/css/parse.data.json
LOG:  Cached module coloraide.css.parse has changed interface
LOG:  Writing coloraide.spaces /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/__init__.py coloraide/spaces/__init__.meta.json coloraide/spaces/__init__.data.json
LOG:  Cached module coloraide.spaces has changed interface
LOG:  Writing coloraide.temperature.robertson_1968 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/robertson_1968.py coloraide/temperature/robertson_1968.meta.json coloraide/temperature/robertson_1968.data.json
LOG:  Cached module coloraide.temperature.robertson_1968 has changed interface
LOG:  Writing coloraide.temperature.ohno_2013 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/temperature/ohno_2013.py coloraide/temperature/ohno_2013.meta.json coloraide/temperature/ohno_2013.data.json
LOG:  Cached module coloraide.temperature.ohno_2013 has changed interface
LOG:  Writing coloraide.spaces.lch /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lch/__init__.py coloraide/spaces/lch/__init__.meta.json coloraide/spaces/lch/__init__.data.json
LOG:  Cached module coloraide.spaces.lch has changed interface
LOG:  Writing coloraide.spaces.lab /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lab/__init__.py coloraide/spaces/lab/__init__.meta.json coloraide/spaces/lab/__init__.data.json
LOG:  Cached module coloraide.spaces.lab has changed interface
LOG:  Writing coloraide.spaces.hwb /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hwb/__init__.py coloraide/spaces/hwb/__init__.meta.json coloraide/spaces/hwb/__init__.data.json
LOG:  Cached module coloraide.spaces.hwb has changed interface
LOG:  Writing coloraide.spaces.hsl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hsl/__init__.py coloraide/spaces/hsl/__init__.meta.json coloraide/spaces/hsl/__init__.data.json
LOG:  Cached module coloraide.spaces.hsl has changed interface
LOG:  Writing coloraide.average /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/average.py coloraide/average.meta.json coloraide/average.data.json
LOG:  Cached module coloraide.average has changed interface
LOG:  Writing coloraide.compositing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/compositing/__init__.py coloraide/compositing/__init__.meta.json coloraide/compositing/__init__.data.json
LOG:  Cached module coloraide.compositing has changed interface
LOG:  Writing coloraide.distance.delta_e_hyab /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_hyab.py coloraide/distance/delta_e_hyab.meta.json coloraide/distance/delta_e_hyab.data.json
LOG:  Cached module coloraide.distance.delta_e_hyab has changed interface
LOG:  Writing coloraide.spaces.xyz_d65 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/xyz_d65.py coloraide/spaces/xyz_d65.meta.json coloraide/spaces/xyz_d65.data.json
LOG:  Cached module coloraide.spaces.xyz_d65 has changed interface
LOG:  Writing coloraide.spaces.srgb_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/srgb_linear.py coloraide/spaces/srgb_linear.meta.json coloraide/spaces/srgb_linear.data.json
LOG:  Cached module coloraide.spaces.srgb_linear has changed interface
LOG:  Writing coloraide.spaces.hsv /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hsv.py coloraide/spaces/hsv.meta.json coloraide/spaces/hsv.data.json
LOG:  Cached module coloraide.spaces.hsv has changed interface
LOG:  Writing coloraide.interpolate /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/__init__.py coloraide/interpolate/__init__.meta.json coloraide/interpolate/__init__.data.json
LOG:  Cached module coloraide.interpolate has changed interface
LOG:  Writing coloraide.gamut.pointer /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/pointer.py coloraide/gamut/pointer.meta.json coloraide/gamut/pointer.data.json
LOG:  Cached module coloraide.gamut.pointer has changed interface
LOG:  Writing coloraide.spaces.ictcp /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/ictcp/__init__.py coloraide/spaces/ictcp/__init__.meta.json coloraide/spaces/ictcp/__init__.data.json
LOG:  Cached module coloraide.spaces.ictcp has changed interface
LOG:  Writing coloraide.spaces.jzczhz /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzczhz/__init__.py coloraide/spaces/jzczhz/__init__.meta.json coloraide/spaces/jzczhz/__init__.data.json
LOG:  Cached module coloraide.spaces.jzczhz has changed interface
LOG:  Writing coloraide.spaces.jzazbz /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzazbz/__init__.py coloraide/spaces/jzazbz/__init__.meta.json coloraide/spaces/jzazbz/__init__.data.json
LOG:  Cached module coloraide.spaces.jzazbz has changed interface
LOG:  Writing coloraide.spaces.oklch /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklch/__init__.py coloraide/spaces/oklch/__init__.meta.json coloraide/spaces/oklch/__init__.data.json
LOG:  Cached module coloraide.spaces.oklch has changed interface
LOG:  Writing coloraide.spaces.oklab /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklab/__init__.py coloraide/spaces/oklab/__init__.meta.json coloraide/spaces/oklab/__init__.data.json
LOG:  Cached module coloraide.spaces.oklab has changed interface
LOG:  Writing coloraide.spaces.srgb /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/srgb/__init__.py coloraide/spaces/srgb/__init__.meta.json coloraide/spaces/srgb/__init__.data.json
LOG:  Cached module coloraide.spaces.srgb has changed interface
LOG:  Writing coloraide.harmonies /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/harmonies.py coloraide/harmonies.meta.json coloraide/harmonies.data.json
LOG:  Cached module coloraide.harmonies has changed interface
LOG:  Writing coloraide.interpolate.continuous /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/continuous.py coloraide/interpolate/continuous.meta.json coloraide/interpolate/continuous.data.json
LOG:  Cached module coloraide.interpolate.continuous has changed interface
LOG:  Writing coloraide.interpolate.linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/linear.py coloraide/interpolate/linear.meta.json coloraide/interpolate/linear.data.json
LOG:  Cached module coloraide.interpolate.linear has changed interface
LOG:  Writing coloraide.distance.delta_e_2000 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_2000.py coloraide/distance/delta_e_2000.meta.json coloraide/distance/delta_e_2000.data.json
LOG:  Cached module coloraide.distance.delta_e_2000 has changed interface
LOG:  Writing coloraide.distance.delta_e_cmc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_cmc.py coloraide/distance/delta_e_cmc.meta.json coloraide/distance/delta_e_cmc.data.json
LOG:  Cached module coloraide.distance.delta_e_cmc has changed interface
LOG:  Writing coloraide.distance.delta_e_94 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_94.py coloraide/distance/delta_e_94.meta.json coloraide/distance/delta_e_94.data.json
LOG:  Cached module coloraide.distance.delta_e_94 has changed interface
LOG:  Writing coloraide.distance.delta_e_76 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/distance/delta_e_76.py coloraide/distance/delta_e_76.meta.json coloraide/distance/delta_e_76.data.json
LOG:  Cached module coloraide.distance.delta_e_76 has changed interface
LOG:  Writing coloraide.spaces.rec2100_hlg /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2100_hlg.py coloraide/spaces/rec2100_hlg.meta.json coloraide/spaces/rec2100_hlg.data.json
LOG:  Cached module coloraide.spaces.rec2100_hlg has changed interface
LOG:  Writing coloraide.spaces.rec2100_pq /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2100_pq.py coloraide/spaces/rec2100_pq.meta.json coloraide/spaces/rec2100_pq.data.json
LOG:  Cached module coloraide.spaces.rec2100_pq has changed interface
LOG:  Writing coloraide.spaces.xyz_d50 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/xyz_d50.py coloraide/spaces/xyz_d50.meta.json coloraide/spaces/xyz_d50.data.json
LOG:  Cached module coloraide.spaces.xyz_d50 has changed interface
LOG:  Writing coloraide.spaces.rec2020_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2020_linear.py coloraide/spaces/rec2020_linear.meta.json coloraide/spaces/rec2020_linear.data.json
LOG:  Cached module coloraide.spaces.rec2020_linear has changed interface
LOG:  Writing coloraide.spaces.rec2020 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2020.py coloraide/spaces/rec2020.meta.json coloraide/spaces/rec2020.data.json
LOG:  Cached module coloraide.spaces.rec2020 has changed interface
LOG:  Writing coloraide.spaces.prophoto_rgb_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/prophoto_rgb_linear.py coloraide/spaces/prophoto_rgb_linear.meta.json coloraide/spaces/prophoto_rgb_linear.data.json
LOG:  Cached module coloraide.spaces.prophoto_rgb_linear has changed interface
LOG:  Writing coloraide.spaces.prophoto_rgb /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/prophoto_rgb.py coloraide/spaces/prophoto_rgb.meta.json coloraide/spaces/prophoto_rgb.data.json
LOG:  Cached module coloraide.spaces.prophoto_rgb has changed interface
LOG:  Writing coloraide.spaces.a98_rgb_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/a98_rgb_linear.py coloraide/spaces/a98_rgb_linear.meta.json coloraide/spaces/a98_rgb_linear.data.json
LOG:  Cached module coloraide.spaces.a98_rgb_linear has changed interface
LOG:  Writing coloraide.spaces.a98_rgb /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/a98_rgb.py coloraide/spaces/a98_rgb.meta.json coloraide/spaces/a98_rgb.data.json
LOG:  Cached module coloraide.spaces.a98_rgb has changed interface
LOG:  Writing coloraide.spaces.display_p3_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/display_p3_linear.py coloraide/spaces/display_p3_linear.meta.json coloraide/spaces/display_p3_linear.data.json
LOG:  Cached module coloraide.spaces.display_p3_linear has changed interface
LOG:  Writing coloraide.spaces.lch_d65 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lch_d65.py coloraide/spaces/lch_d65.meta.json coloraide/spaces/lch_d65.data.json
LOG:  Cached module coloraide.spaces.lch_d65 has changed interface
LOG:  Writing coloraide.spaces.lab_d65 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lab_d65.py coloraide/spaces/lab_d65.meta.json coloraide/spaces/lab_d65.data.json
LOG:  Cached module coloraide.spaces.lab_d65 has changed interface
LOG:  Writing coloraide.spaces.lch.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lch/css.py coloraide/spaces/lch/css.meta.json coloraide/spaces/lch/css.data.json
LOG:  Cached module coloraide.spaces.lch.css has changed interface
LOG:  Writing coloraide.spaces.lab.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/lab/css.py coloraide/spaces/lab/css.meta.json coloraide/spaces/lab/css.data.json
LOG:  Cached module coloraide.spaces.lab.css has changed interface
LOG:  Writing coloraide.spaces.hwb.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hwb/css.py coloraide/spaces/hwb/css.meta.json coloraide/spaces/hwb/css.data.json
LOG:  Cached module coloraide.spaces.hwb.css has changed interface
LOG:  Writing coloraide.spaces.hsl.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/hsl/css.py coloraide/spaces/hsl/css.meta.json coloraide/spaces/hsl/css.data.json
LOG:  Cached module coloraide.spaces.hsl.css has changed interface
LOG:  Writing coloraide.gamut /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/__init__.py coloraide/gamut/__init__.meta.json coloraide/gamut/__init__.data.json
LOG:  Cached module coloraide.gamut has changed interface
LOG:  Writing coloraide.interpolate.bspline /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/bspline.py coloraide/interpolate/bspline.meta.json coloraide/interpolate/bspline.data.json
LOG:  Cached module coloraide.interpolate.bspline has changed interface
LOG:  Writing coloraide.interpolate.css_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/css_linear.py coloraide/interpolate/css_linear.meta.json coloraide/interpolate/css_linear.data.json
LOG:  Cached module coloraide.interpolate.css_linear has changed interface
LOG:  Writing coloraide.spaces.rec2100_linear /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/rec2100_linear.py coloraide/spaces/rec2100_linear.meta.json coloraide/spaces/rec2100_linear.data.json
LOG:  Cached module coloraide.spaces.rec2100_linear has changed interface
LOG:  Writing coloraide.spaces.display_p3 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/display_p3.py coloraide/spaces/display_p3.meta.json coloraide/spaces/display_p3.data.json
LOG:  Cached module coloraide.spaces.display_p3 has changed interface
LOG:  Writing coloraide.spaces.ictcp.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/ictcp/css.py coloraide/spaces/ictcp/css.meta.json coloraide/spaces/ictcp/css.data.json
LOG:  Cached module coloraide.spaces.ictcp.css has changed interface
LOG:  Writing coloraide.spaces.jzczhz.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzczhz/css.py coloraide/spaces/jzczhz/css.meta.json coloraide/spaces/jzczhz/css.data.json
LOG:  Cached module coloraide.spaces.jzczhz.css has changed interface
LOG:  Writing coloraide.spaces.jzazbz.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/jzazbz/css.py coloraide/spaces/jzazbz/css.meta.json coloraide/spaces/jzazbz/css.data.json
LOG:  Cached module coloraide.spaces.jzazbz.css has changed interface
LOG:  Writing coloraide.spaces.oklch.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklch/css.py coloraide/spaces/oklch/css.meta.json coloraide/spaces/oklch/css.data.json
LOG:  Cached module coloraide.spaces.oklch.css has changed interface
LOG:  Writing coloraide.spaces.oklab.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/oklab/css.py coloraide/spaces/oklab/css.meta.json coloraide/spaces/oklab/css.data.json
LOG:  Cached module coloraide.spaces.oklab.css has changed interface
LOG:  Writing coloraide.spaces.srgb.css /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/spaces/srgb/css.py coloraide/spaces/srgb/css.meta.json coloraide/spaces/srgb/css.data.json
LOG:  Cached module coloraide.spaces.srgb.css has changed interface
LOG:  Writing coloraide.interpolate.monotone /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/monotone.py coloraide/interpolate/monotone.meta.json coloraide/interpolate/monotone.data.json
LOG:  Cached module coloraide.interpolate.monotone has changed interface
LOG:  Writing coloraide.interpolate.bspline_natural /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/interpolate/bspline_natural.py coloraide/interpolate/bspline_natural.meta.json coloraide/interpolate/bspline_natural.data.json
LOG:  Cached module coloraide.interpolate.bspline_natural has changed interface
LOG:  Writing coloraide.gamut.fit_raytrace /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_raytrace.py coloraide/gamut/fit_raytrace.meta.json coloraide/gamut/fit_raytrace.data.json
LOG:  Cached module coloraide.gamut.fit_raytrace has changed interface
LOG:  Writing coloraide.gamut.fit_minde_chroma /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_minde_chroma.py coloraide/gamut/fit_minde_chroma.meta.json coloraide/gamut/fit_minde_chroma.data.json
LOG:  Cached module coloraide.gamut.fit_minde_chroma has changed interface
LOG:  Writing coloraide.gamut.fit_oklch_chroma /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_oklch_chroma.py coloraide/gamut/fit_oklch_chroma.meta.json coloraide/gamut/fit_oklch_chroma.data.json
LOG:  Cached module coloraide.gamut.fit_oklch_chroma has changed interface
LOG:  Writing coloraide.gamut.fit_lch_chroma /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/gamut/fit_lch_chroma.py coloraide/gamut/fit_lch_chroma.meta.json coloraide/gamut/fit_lch_chroma.data.json
LOG:  Cached module coloraide.gamut.fit_lch_chroma has changed interface
LOG:  Writing coloraide.color /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/color.py coloraide/color.meta.json coloraide/color.data.json
LOG:  Cached module coloraide.color has changed interface
LOG:  Writing coloraide /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/coloraide/__init__.py coloraide/__init__.meta.json coloraide/__init__.data.json
LOG:  Cached module coloraide has changed interface
LOG:  Processing SCC of size 9 (multiprocessing.reduction multiprocessing.popen_spawn_posix multiprocessing.popen_forkserver multiprocessing.sharedctypes multiprocessing.pool multiprocessing.managers multiprocessing.synchronize multiprocessing.context multiprocessing) as inherently stale with stale deps (_ctypes _pickle _typeshed abc builtins collections.abc copyreg ctypes logging multiprocessing.connection multiprocessing.popen_fork multiprocessing.popen_spawn_win32 multiprocessing.process multiprocessing.queues multiprocessing.shared_memory multiprocessing.spawn multiprocessing.util pickle queue socket sys threading types typing typing_extensions)
LOG:  Writing multiprocessing.reduction /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/reduction.pyi multiprocessing/reduction.meta.json multiprocessing/reduction.data.json
LOG:  Cached module multiprocessing.reduction has changed interface
LOG:  Writing multiprocessing.popen_spawn_posix /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_spawn_posix.pyi multiprocessing/popen_spawn_posix.meta.json multiprocessing/popen_spawn_posix.data.json
LOG:  Cached module multiprocessing.popen_spawn_posix has changed interface
LOG:  Writing multiprocessing.popen_forkserver /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/popen_forkserver.pyi multiprocessing/popen_forkserver.meta.json multiprocessing/popen_forkserver.data.json
LOG:  Cached module multiprocessing.popen_forkserver has changed interface
LOG:  Writing multiprocessing.sharedctypes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/sharedctypes.pyi multiprocessing/sharedctypes.meta.json multiprocessing/sharedctypes.data.json
LOG:  Cached module multiprocessing.sharedctypes has changed interface
LOG:  Writing multiprocessing.pool /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/pool.pyi multiprocessing/pool.meta.json multiprocessing/pool.data.json
LOG:  Cached module multiprocessing.pool has changed interface
LOG:  Writing multiprocessing.managers /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/managers.pyi multiprocessing/managers.meta.json multiprocessing/managers.data.json
LOG:  Cached module multiprocessing.managers has changed interface
LOG:  Writing multiprocessing.synchronize /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/synchronize.pyi multiprocessing/synchronize.meta.json multiprocessing/synchronize.data.json
LOG:  Cached module multiprocessing.synchronize has changed interface
LOG:  Writing multiprocessing.context /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/context.pyi multiprocessing/context.meta.json multiprocessing/context.data.json
LOG:  Cached module multiprocessing.context has changed interface
LOG:  Writing multiprocessing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/multiprocessing/__init__.pyi multiprocessing/__init__.meta.json multiprocessing/__init__.data.json
LOG:  Cached module multiprocessing has changed interface
LOG:  Processing SCC singleton (numpy.f2py) as inherently stale with stale deps (builtins numpy.f2py.f2py2e)
LOG:  Writing numpy.f2py /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/f2py/__init__.pyi numpy/f2py/__init__.meta.json numpy/f2py/__init__.data.json
LOG:  Cached module numpy.f2py has changed interface
LOG:  Processing SCC singleton (concurrent.futures.process) as inherently stale with stale deps (builtins collections.abc concurrent.futures._base multiprocessing.connection multiprocessing.context multiprocessing.queues sys threading types typing typing_extensions weakref)
LOG:  Writing concurrent.futures.process /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/process.pyi concurrent/futures/process.meta.json concurrent/futures/process.data.json
LOG:  Cached module concurrent.futures.process has changed interface
LOG:  Processing SCC singleton (concurrent.futures) as inherently stale with stale deps (builtins concurrent.futures._base concurrent.futures.process concurrent.futures.thread sys)
LOG:  Writing concurrent.futures /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/concurrent/futures/__init__.pyi concurrent/futures/__init__.meta.json concurrent/futures/__init__.data.json
LOG:  Cached module concurrent.futures has changed interface
LOG:  Processing SCC of size 16 (asyncio.selector_events asyncio.subprocess asyncio.protocols _asyncio asyncio.unix_events asyncio.taskgroups asyncio.transports asyncio.tasks asyncio.streams asyncio.runners asyncio.queues asyncio.locks asyncio.futures asyncio.events asyncio.base_events asyncio) as inherently stale with stale deps (_typeshed abc asyncio.coroutines asyncio.exceptions asyncio.mixins asyncio.threads asyncio.timeouts builtins collections collections.abc concurrent concurrent.futures concurrent.futures._base contextvars enum selectors socket ssl subprocess sys types typing typing_extensions)
LOG:  Writing asyncio.selector_events /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/selector_events.pyi asyncio/selector_events.meta.json asyncio/selector_events.data.json
LOG:  Cached module asyncio.selector_events has changed interface
LOG:  Writing asyncio.subprocess /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/subprocess.pyi asyncio/subprocess.meta.json asyncio/subprocess.data.json
LOG:  Cached module asyncio.subprocess has changed interface
LOG:  Writing asyncio.protocols /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/protocols.pyi asyncio/protocols.meta.json asyncio/protocols.data.json
LOG:  Cached module asyncio.protocols has changed interface
LOG:  Writing _asyncio /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/_asyncio.pyi _asyncio.meta.json _asyncio.data.json
LOG:  Cached module _asyncio has changed interface
LOG:  Writing asyncio.unix_events /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/unix_events.pyi asyncio/unix_events.meta.json asyncio/unix_events.data.json
LOG:  Cached module asyncio.unix_events has changed interface
LOG:  Writing asyncio.taskgroups /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/taskgroups.pyi asyncio/taskgroups.meta.json asyncio/taskgroups.data.json
LOG:  Cached module asyncio.taskgroups has changed interface
LOG:  Writing asyncio.transports /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/transports.pyi asyncio/transports.meta.json asyncio/transports.data.json
LOG:  Cached module asyncio.transports has changed interface
LOG:  Writing asyncio.tasks /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/tasks.pyi asyncio/tasks.meta.json asyncio/tasks.data.json
LOG:  Cached module asyncio.tasks has changed interface
LOG:  Writing asyncio.streams /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/streams.pyi asyncio/streams.meta.json asyncio/streams.data.json
LOG:  Cached module asyncio.streams has changed interface
LOG:  Writing asyncio.runners /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/runners.pyi asyncio/runners.meta.json asyncio/runners.data.json
LOG:  Cached module asyncio.runners has changed interface
LOG:  Writing asyncio.queues /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/queues.pyi asyncio/queues.meta.json asyncio/queues.data.json
LOG:  Cached module asyncio.queues has changed interface
LOG:  Writing asyncio.locks /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/locks.pyi asyncio/locks.meta.json asyncio/locks.data.json
LOG:  Cached module asyncio.locks has changed interface
LOG:  Writing asyncio.futures /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/futures.pyi asyncio/futures.meta.json asyncio/futures.data.json
LOG:  Cached module asyncio.futures has changed interface
LOG:  Writing asyncio.events /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/events.pyi asyncio/events.meta.json asyncio/events.data.json
LOG:  Cached module asyncio.events has changed interface
LOG:  Writing asyncio.base_events /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/base_events.pyi asyncio/base_events.meta.json asyncio/base_events.data.json
LOG:  Cached module asyncio.base_events has changed interface
LOG:  Writing asyncio /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/asyncio/__init__.pyi asyncio/__init__.meta.json asyncio/__init__.data.json
LOG:  Cached module asyncio has changed interface
LOG:  Processing SCC of size 10 (unittest.result unittest._log unittest.case unittest.suite unittest.signals unittest.async_case unittest.runner unittest.loader unittest.main unittest) as inherently stale with stale deps (_typeshed asyncio.events builtins collections.abc contextlib logging re sys types typing typing_extensions warnings)
LOG:  Writing unittest.result /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/result.pyi unittest/result.meta.json unittest/result.data.json
LOG:  Cached module unittest.result has changed interface
LOG:  Writing unittest._log /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/_log.pyi unittest/_log.meta.json unittest/_log.data.json
LOG:  Cached module unittest._log has changed interface
LOG:  Writing unittest.case /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/case.pyi unittest/case.meta.json unittest/case.data.json
LOG:  Cached module unittest.case has changed interface
LOG:  Writing unittest.suite /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/suite.pyi unittest/suite.meta.json unittest/suite.data.json
LOG:  Cached module unittest.suite has changed interface
LOG:  Writing unittest.signals /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/signals.pyi unittest/signals.meta.json unittest/signals.data.json
LOG:  Cached module unittest.signals has changed interface
LOG:  Writing unittest.async_case /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/async_case.pyi unittest/async_case.meta.json unittest/async_case.data.json
LOG:  Cached module unittest.async_case has changed interface
LOG:  Writing unittest.runner /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/runner.pyi unittest/runner.meta.json unittest/runner.data.json
LOG:  Cached module unittest.runner has changed interface
LOG:  Writing unittest.loader /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/loader.pyi unittest/loader.meta.json unittest/loader.data.json
LOG:  Cached module unittest.loader has changed interface
LOG:  Writing unittest.main /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/main.pyi unittest/main.meta.json unittest/main.data.json
LOG:  Cached module unittest.main has changed interface
LOG:  Writing unittest /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/mypy/typeshed/stdlib/unittest/__init__.pyi unittest/__init__.meta.json unittest/__init__.data.json
LOG:  Cached module unittest has changed interface
LOG:  Processing SCC singleton (loguru) as inherently stale with stale deps (asyncio builtins datetime logging multiprocessing.context os sys types typing)
LOG:  Writing loguru /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/loguru/__init__.pyi loguru/__init__.meta.json loguru/__init__.data.json
LOG:  Cached module loguru has changed interface
LOG:  Processing SCC of size 92 (numpy.testing.overrides numpy._typing._dtype_like numpy._typing._scalars numpy.dtypes numpy._array_api_info numpy._core._type_aliases numpy.matrixlib.defmatrix numpy._typing._add_docstring numpy.ma.extras numpy.ma.core numpy.ctypeslib._ctypeslib numpy.ma.mrecords numpy.lib._array_utils_impl numpy._typing._ufunc numpy._typing._array_like numpy.matrixlib numpy.typing numpy.ma numpy.ctypeslib numpy._typing numpy.lib._utils_impl numpy.lib._ufunclike_impl numpy.lib._type_check_impl numpy.lib._twodim_base_impl numpy.lib._stride_tricks_impl numpy.lib._shape_base_impl numpy.lib._polynomial_impl numpy.lib._npyio_impl numpy.lib._nanfunctions_impl numpy.lib._index_tricks_impl numpy.lib._histograms_impl numpy.lib._function_base_impl numpy.lib._arraysetops_impl numpy.lib._arraypad_impl numpy._core.shape_base numpy._core.numerictypes numpy._core.numeric numpy._core.multiarray numpy._core.einsumfunc numpy._core.arrayprint numpy._core._ufunc_config numpy._core._asarray numpy._core.fromnumeric numpy._core.function_base numpy._core.records numpy._typing._extended_precision numpy._typing._callable numpy._core._internal numpy numpy.polynomial._polytypes numpy.lib._format_impl numpy.testing._private.utils numpy._core.strings numpy.random.bit_generator numpy.linalg._umath_linalg numpy.lib._arrayterator_impl numpy.lib.stride_tricks numpy.lib.npyio numpy.lib.mixins numpy.lib.array_utils numpy.fft._pocketfft numpy.fft._helper numpy._core.defchararray numpy.lib._scimath_impl numpy.matlib numpy.rec numpy.linalg._linalg numpy.linalg numpy.polynomial.polyutils numpy.polynomial._polybase numpy.random.mtrand numpy.random._sfc64 numpy.random._philox numpy.random._pcg64 numpy.random._mt19937 numpy.linalg.linalg numpy.lib.format numpy.testing numpy.strings numpy.lib numpy.fft numpy.char numpy.lib.scimath numpy.polynomial.polynomial numpy.polynomial.legendre numpy.polynomial.laguerre numpy.polynomial.hermite_e numpy.polynomial.hermite numpy.polynomial.chebyshev numpy.random._generator numpy.random numpy.polynomial) as inherently stale with stale deps (_typeshed abc array ast builtins collections.abc contextlib ctypes datetime decimal fractions mmap numbers numpy.__config__ numpy._expired_attrs_2_0 numpy._globals numpy._pytesttester numpy._typing._char_codes numpy._typing._nbit numpy._typing._nbit_base numpy._typing._nested_sequence numpy._typing._shape numpy.core numpy.exceptions numpy.f2py numpy.lib._datasource numpy.lib._version numpy.lib.introspect numpy.version pathlib re sys textwrap threading types typing typing_extensions unittest unittest.case uuid warnings zipfile)
LOG:  Writing numpy.testing.overrides /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/overrides.pyi numpy/testing/overrides.meta.json numpy/testing/overrides.data.json
LOG:  Cached module numpy.testing.overrides has changed interface
LOG:  Writing numpy._typing._dtype_like /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_dtype_like.py numpy/_typing/_dtype_like.meta.json numpy/_typing/_dtype_like.data.json
LOG:  Cached module numpy._typing._dtype_like has changed interface
LOG:  Writing numpy._typing._scalars /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_scalars.py numpy/_typing/_scalars.meta.json numpy/_typing/_scalars.data.json
LOG:  Cached module numpy._typing._scalars has changed interface
LOG:  Writing numpy.dtypes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/dtypes.pyi numpy/dtypes.meta.json numpy/dtypes.data.json
LOG:  Cached module numpy.dtypes has changed interface
LOG:  Writing numpy._array_api_info /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_array_api_info.pyi numpy/_array_api_info.meta.json numpy/_array_api_info.data.json
LOG:  Cached module numpy._array_api_info has changed interface
LOG:  Writing numpy._core._type_aliases /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_type_aliases.pyi numpy/_core/_type_aliases.meta.json numpy/_core/_type_aliases.data.json
LOG:  Cached module numpy._core._type_aliases has changed interface
LOG:  Writing numpy.matrixlib.defmatrix /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/matrixlib/defmatrix.pyi numpy/matrixlib/defmatrix.meta.json numpy/matrixlib/defmatrix.data.json
LOG:  Cached module numpy.matrixlib.defmatrix has changed interface
LOG:  Writing numpy._typing._add_docstring /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_add_docstring.py numpy/_typing/_add_docstring.meta.json numpy/_typing/_add_docstring.data.json
LOG:  Cached module numpy._typing._add_docstring has changed interface
LOG:  Writing numpy.ma.extras /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/extras.pyi numpy/ma/extras.meta.json numpy/ma/extras.data.json
LOG:  Cached module numpy.ma.extras has changed interface
LOG:  Writing numpy.ma.core /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/core.pyi numpy/ma/core.meta.json numpy/ma/core.data.json
LOG:  Cached module numpy.ma.core has changed interface
LOG:  Writing numpy.ctypeslib._ctypeslib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ctypeslib/_ctypeslib.pyi numpy/ctypeslib/_ctypeslib.meta.json numpy/ctypeslib/_ctypeslib.data.json
LOG:  Cached module numpy.ctypeslib._ctypeslib has changed interface
LOG:  Writing numpy.ma.mrecords /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/mrecords.pyi numpy/ma/mrecords.meta.json numpy/ma/mrecords.data.json
LOG:  Cached module numpy.ma.mrecords has changed interface
LOG:  Writing numpy.lib._array_utils_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_array_utils_impl.pyi numpy/lib/_array_utils_impl.meta.json numpy/lib/_array_utils_impl.data.json
LOG:  Cached module numpy.lib._array_utils_impl has changed interface
LOG:  Writing numpy._typing._ufunc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_ufunc.pyi numpy/_typing/_ufunc.meta.json numpy/_typing/_ufunc.data.json
LOG:  Cached module numpy._typing._ufunc has changed interface
LOG:  Writing numpy._typing._array_like /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_array_like.py numpy/_typing/_array_like.meta.json numpy/_typing/_array_like.data.json
LOG:  Cached module numpy._typing._array_like has changed interface
LOG:  Writing numpy.matrixlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/matrixlib/__init__.pyi numpy/matrixlib/__init__.meta.json numpy/matrixlib/__init__.data.json
LOG:  Cached module numpy.matrixlib has changed interface
LOG:  Writing numpy.typing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/typing/__init__.py numpy/typing/__init__.meta.json numpy/typing/__init__.data.json
LOG:  Cached module numpy.typing has changed interface
LOG:  Writing numpy.ma /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ma/__init__.pyi numpy/ma/__init__.meta.json numpy/ma/__init__.data.json
LOG:  Cached module numpy.ma has changed interface
LOG:  Writing numpy.ctypeslib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/ctypeslib/__init__.pyi numpy/ctypeslib/__init__.meta.json numpy/ctypeslib/__init__.data.json
LOG:  Cached module numpy.ctypeslib has changed interface
LOG:  Writing numpy._typing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/__init__.py numpy/_typing/__init__.meta.json numpy/_typing/__init__.data.json
LOG:  Cached module numpy._typing has changed interface
LOG:  Writing numpy.lib._utils_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_utils_impl.pyi numpy/lib/_utils_impl.meta.json numpy/lib/_utils_impl.data.json
LOG:  Cached module numpy.lib._utils_impl has changed interface
LOG:  Writing numpy.lib._ufunclike_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_ufunclike_impl.pyi numpy/lib/_ufunclike_impl.meta.json numpy/lib/_ufunclike_impl.data.json
LOG:  Cached module numpy.lib._ufunclike_impl has changed interface
LOG:  Writing numpy.lib._type_check_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_type_check_impl.pyi numpy/lib/_type_check_impl.meta.json numpy/lib/_type_check_impl.data.json
LOG:  Cached module numpy.lib._type_check_impl has changed interface
LOG:  Writing numpy.lib._twodim_base_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_twodim_base_impl.pyi numpy/lib/_twodim_base_impl.meta.json numpy/lib/_twodim_base_impl.data.json
LOG:  Cached module numpy.lib._twodim_base_impl has changed interface
LOG:  Writing numpy.lib._stride_tricks_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.pyi numpy/lib/_stride_tricks_impl.meta.json numpy/lib/_stride_tricks_impl.data.json
LOG:  Cached module numpy.lib._stride_tricks_impl has changed interface
LOG:  Writing numpy.lib._shape_base_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_shape_base_impl.pyi numpy/lib/_shape_base_impl.meta.json numpy/lib/_shape_base_impl.data.json
LOG:  Cached module numpy.lib._shape_base_impl has changed interface
LOG:  Writing numpy.lib._polynomial_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.pyi numpy/lib/_polynomial_impl.meta.json numpy/lib/_polynomial_impl.data.json
LOG:  Cached module numpy.lib._polynomial_impl has changed interface
LOG:  Writing numpy.lib._npyio_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_npyio_impl.pyi numpy/lib/_npyio_impl.meta.json numpy/lib/_npyio_impl.data.json
LOG:  Cached module numpy.lib._npyio_impl has changed interface
LOG:  Writing numpy.lib._nanfunctions_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.pyi numpy/lib/_nanfunctions_impl.meta.json numpy/lib/_nanfunctions_impl.data.json
LOG:  Cached module numpy.lib._nanfunctions_impl has changed interface
LOG:  Writing numpy.lib._index_tricks_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_index_tricks_impl.pyi numpy/lib/_index_tricks_impl.meta.json numpy/lib/_index_tricks_impl.data.json
LOG:  Cached module numpy.lib._index_tricks_impl has changed interface
LOG:  Writing numpy.lib._histograms_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_histograms_impl.pyi numpy/lib/_histograms_impl.meta.json numpy/lib/_histograms_impl.data.json
LOG:  Cached module numpy.lib._histograms_impl has changed interface
LOG:  Writing numpy.lib._function_base_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_function_base_impl.pyi numpy/lib/_function_base_impl.meta.json numpy/lib/_function_base_impl.data.json
LOG:  Cached module numpy.lib._function_base_impl has changed interface
LOG:  Writing numpy.lib._arraysetops_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_arraysetops_impl.pyi numpy/lib/_arraysetops_impl.meta.json numpy/lib/_arraysetops_impl.data.json
LOG:  Cached module numpy.lib._arraysetops_impl has changed interface
LOG:  Writing numpy.lib._arraypad_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_arraypad_impl.pyi numpy/lib/_arraypad_impl.meta.json numpy/lib/_arraypad_impl.data.json
LOG:  Cached module numpy.lib._arraypad_impl has changed interface
LOG:  Writing numpy._core.shape_base /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/shape_base.pyi numpy/_core/shape_base.meta.json numpy/_core/shape_base.data.json
LOG:  Cached module numpy._core.shape_base has changed interface
LOG:  Writing numpy._core.numerictypes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/numerictypes.pyi numpy/_core/numerictypes.meta.json numpy/_core/numerictypes.data.json
LOG:  Cached module numpy._core.numerictypes has changed interface
LOG:  Writing numpy._core.numeric /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/numeric.pyi numpy/_core/numeric.meta.json numpy/_core/numeric.data.json
LOG:  Cached module numpy._core.numeric has changed interface
LOG:  Writing numpy._core.multiarray /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/multiarray.pyi numpy/_core/multiarray.meta.json numpy/_core/multiarray.data.json
LOG:  Cached module numpy._core.multiarray has changed interface
LOG:  Writing numpy._core.einsumfunc /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/einsumfunc.pyi numpy/_core/einsumfunc.meta.json numpy/_core/einsumfunc.data.json
LOG:  Cached module numpy._core.einsumfunc has changed interface
LOG:  Writing numpy._core.arrayprint /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/arrayprint.pyi numpy/_core/arrayprint.meta.json numpy/_core/arrayprint.data.json
LOG:  Cached module numpy._core.arrayprint has changed interface
LOG:  Writing numpy._core._ufunc_config /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_ufunc_config.pyi numpy/_core/_ufunc_config.meta.json numpy/_core/_ufunc_config.data.json
LOG:  Cached module numpy._core._ufunc_config has changed interface
LOG:  Writing numpy._core._asarray /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_asarray.pyi numpy/_core/_asarray.meta.json numpy/_core/_asarray.data.json
LOG:  Cached module numpy._core._asarray has changed interface
LOG:  Writing numpy._core.fromnumeric /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/fromnumeric.pyi numpy/_core/fromnumeric.meta.json numpy/_core/fromnumeric.data.json
LOG:  Cached module numpy._core.fromnumeric has changed interface
LOG:  Writing numpy._core.function_base /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/function_base.pyi numpy/_core/function_base.meta.json numpy/_core/function_base.data.json
LOG:  Cached module numpy._core.function_base has changed interface
LOG:  Writing numpy._core.records /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/records.pyi numpy/_core/records.meta.json numpy/_core/records.data.json
LOG:  Cached module numpy._core.records has changed interface
LOG:  Writing numpy._typing._extended_precision /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_extended_precision.py numpy/_typing/_extended_precision.meta.json numpy/_typing/_extended_precision.data.json
LOG:  Cached module numpy._typing._extended_precision has changed interface
LOG:  Writing numpy._typing._callable /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_typing/_callable.pyi numpy/_typing/_callable.meta.json numpy/_typing/_callable.data.json
LOG:  Cached module numpy._typing._callable has changed interface
LOG:  Writing numpy._core._internal /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/_internal.pyi numpy/_core/_internal.meta.json numpy/_core/_internal.data.json
LOG:  Cached module numpy._core._internal has changed interface
LOG:  Writing numpy /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/__init__.pyi numpy/__init__.meta.json numpy/__init__.data.json
LOG:  Cached module numpy has changed interface
LOG:  Writing numpy.polynomial._polytypes /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/_polytypes.pyi numpy/polynomial/_polytypes.meta.json numpy/polynomial/_polytypes.data.json
LOG:  Cached module numpy.polynomial._polytypes has changed interface
LOG:  Writing numpy.lib._format_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_format_impl.pyi numpy/lib/_format_impl.meta.json numpy/lib/_format_impl.data.json
LOG:  Cached module numpy.lib._format_impl has changed interface
LOG:  Writing numpy.testing._private.utils /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/_private/utils.pyi numpy/testing/_private/utils.meta.json numpy/testing/_private/utils.data.json
LOG:  Cached module numpy.testing._private.utils has changed interface
LOG:  Writing numpy._core.strings /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/strings.pyi numpy/_core/strings.meta.json numpy/_core/strings.data.json
LOG:  Cached module numpy._core.strings has changed interface
LOG:  Writing numpy.random.bit_generator /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/bit_generator.pyi numpy/random/bit_generator.meta.json numpy/random/bit_generator.data.json
LOG:  Cached module numpy.random.bit_generator has changed interface
LOG:  Writing numpy.linalg._umath_linalg /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.pyi numpy/linalg/_umath_linalg.meta.json numpy/linalg/_umath_linalg.data.json
LOG:  Cached module numpy.linalg._umath_linalg has changed interface
LOG:  Writing numpy.lib._arrayterator_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_arrayterator_impl.pyi numpy/lib/_arrayterator_impl.meta.json numpy/lib/_arrayterator_impl.data.json
LOG:  Cached module numpy.lib._arrayterator_impl has changed interface
LOG:  Writing numpy.lib.stride_tricks /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/stride_tricks.pyi numpy/lib/stride_tricks.meta.json numpy/lib/stride_tricks.data.json
LOG:  Cached module numpy.lib.stride_tricks has changed interface
LOG:  Writing numpy.lib.npyio /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/npyio.pyi numpy/lib/npyio.meta.json numpy/lib/npyio.data.json
LOG:  Cached module numpy.lib.npyio has changed interface
LOG:  Writing numpy.lib.mixins /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/mixins.pyi numpy/lib/mixins.meta.json numpy/lib/mixins.data.json
LOG:  Cached module numpy.lib.mixins has changed interface
LOG:  Writing numpy.lib.array_utils /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/array_utils.pyi numpy/lib/array_utils.meta.json numpy/lib/array_utils.data.json
LOG:  Cached module numpy.lib.array_utils has changed interface
LOG:  Writing numpy.fft._pocketfft /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/fft/_pocketfft.pyi numpy/fft/_pocketfft.meta.json numpy/fft/_pocketfft.data.json
LOG:  Cached module numpy.fft._pocketfft has changed interface
LOG:  Writing numpy.fft._helper /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/fft/_helper.pyi numpy/fft/_helper.meta.json numpy/fft/_helper.data.json
LOG:  Cached module numpy.fft._helper has changed interface
LOG:  Writing numpy._core.defchararray /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/_core/defchararray.pyi numpy/_core/defchararray.meta.json numpy/_core/defchararray.data.json
LOG:  Cached module numpy._core.defchararray has changed interface
LOG:  Writing numpy.lib._scimath_impl /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/_scimath_impl.pyi numpy/lib/_scimath_impl.meta.json numpy/lib/_scimath_impl.data.json
LOG:  Cached module numpy.lib._scimath_impl has changed interface
LOG:  Writing numpy.matlib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/matlib.pyi numpy/matlib.meta.json numpy/matlib.data.json
LOG:  Cached module numpy.matlib has changed interface
LOG:  Writing numpy.rec /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/rec/__init__.pyi numpy/rec/__init__.meta.json numpy/rec/__init__.data.json
LOG:  Cached module numpy.rec has changed interface
LOG:  Writing numpy.linalg._linalg /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/_linalg.pyi numpy/linalg/_linalg.meta.json numpy/linalg/_linalg.data.json
LOG:  Cached module numpy.linalg._linalg has changed interface
LOG:  Writing numpy.linalg /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/__init__.pyi numpy/linalg/__init__.meta.json numpy/linalg/__init__.data.json
LOG:  Cached module numpy.linalg has changed interface
LOG:  Writing numpy.polynomial.polyutils /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/polyutils.pyi numpy/polynomial/polyutils.meta.json numpy/polynomial/polyutils.data.json
LOG:  Cached module numpy.polynomial.polyutils has changed interface
LOG:  Writing numpy.polynomial._polybase /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/_polybase.pyi numpy/polynomial/_polybase.meta.json numpy/polynomial/_polybase.data.json
LOG:  Cached module numpy.polynomial._polybase has changed interface
LOG:  Writing numpy.random.mtrand /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/mtrand.pyi numpy/random/mtrand.meta.json numpy/random/mtrand.data.json
LOG:  Cached module numpy.random.mtrand has changed interface
LOG:  Writing numpy.random._sfc64 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_sfc64.pyi numpy/random/_sfc64.meta.json numpy/random/_sfc64.data.json
LOG:  Cached module numpy.random._sfc64 has changed interface
LOG:  Writing numpy.random._philox /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_philox.pyi numpy/random/_philox.meta.json numpy/random/_philox.data.json
LOG:  Cached module numpy.random._philox has changed interface
LOG:  Writing numpy.random._pcg64 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_pcg64.pyi numpy/random/_pcg64.meta.json numpy/random/_pcg64.data.json
LOG:  Cached module numpy.random._pcg64 has changed interface
LOG:  Writing numpy.random._mt19937 /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_mt19937.pyi numpy/random/_mt19937.meta.json numpy/random/_mt19937.data.json
LOG:  Cached module numpy.random._mt19937 has changed interface
LOG:  Writing numpy.linalg.linalg /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/linalg/linalg.pyi numpy/linalg/linalg.meta.json numpy/linalg/linalg.data.json
LOG:  Cached module numpy.linalg.linalg has changed interface
LOG:  Writing numpy.lib.format /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/format.pyi numpy/lib/format.meta.json numpy/lib/format.data.json
LOG:  Cached module numpy.lib.format has changed interface
LOG:  Writing numpy.testing /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/testing/__init__.pyi numpy/testing/__init__.meta.json numpy/testing/__init__.data.json
LOG:  Cached module numpy.testing has changed interface
LOG:  Writing numpy.strings /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/strings/__init__.pyi numpy/strings/__init__.meta.json numpy/strings/__init__.data.json
LOG:  Cached module numpy.strings has changed interface
LOG:  Writing numpy.lib /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/__init__.pyi numpy/lib/__init__.meta.json numpy/lib/__init__.data.json
LOG:  Cached module numpy.lib has changed interface
LOG:  Writing numpy.fft /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/fft/__init__.pyi numpy/fft/__init__.meta.json numpy/fft/__init__.data.json
LOG:  Cached module numpy.fft has changed interface
LOG:  Writing numpy.char /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/char/__init__.pyi numpy/char/__init__.meta.json numpy/char/__init__.data.json
LOG:  Cached module numpy.char has changed interface
LOG:  Writing numpy.lib.scimath /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/lib/scimath.pyi numpy/lib/scimath.meta.json numpy/lib/scimath.data.json
LOG:  Cached module numpy.lib.scimath has changed interface
LOG:  Writing numpy.polynomial.polynomial /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/polynomial.pyi numpy/polynomial/polynomial.meta.json numpy/polynomial/polynomial.data.json
LOG:  Cached module numpy.polynomial.polynomial has changed interface
LOG:  Writing numpy.polynomial.legendre /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/legendre.pyi numpy/polynomial/legendre.meta.json numpy/polynomial/legendre.data.json
LOG:  Cached module numpy.polynomial.legendre has changed interface
LOG:  Writing numpy.polynomial.laguerre /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/laguerre.pyi numpy/polynomial/laguerre.meta.json numpy/polynomial/laguerre.data.json
LOG:  Cached module numpy.polynomial.laguerre has changed interface
LOG:  Writing numpy.polynomial.hermite_e /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/hermite_e.pyi numpy/polynomial/hermite_e.meta.json numpy/polynomial/hermite_e.data.json
LOG:  Cached module numpy.polynomial.hermite_e has changed interface
LOG:  Writing numpy.polynomial.hermite /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/hermite.pyi numpy/polynomial/hermite.meta.json numpy/polynomial/hermite.data.json
LOG:  Cached module numpy.polynomial.hermite has changed interface
LOG:  Writing numpy.polynomial.chebyshev /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/chebyshev.pyi numpy/polynomial/chebyshev.meta.json numpy/polynomial/chebyshev.data.json
LOG:  Cached module numpy.polynomial.chebyshev has changed interface
LOG:  Writing numpy.random._generator /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/_generator.pyi numpy/random/_generator.meta.json numpy/random/_generator.data.json
LOG:  Cached module numpy.random._generator has changed interface
LOG:  Writing numpy.random /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/random/__init__.pyi numpy/random/__init__.meta.json numpy/random/__init__.data.json
LOG:  Cached module numpy.random has changed interface
LOG:  Writing numpy.polynomial /Users/adam/Library/Application Support/hatch/env/virtual/imgcolorshine/-wfJnRV4/imgcolorshine-build/lib/python3.12/site-packages/numpy/polynomial/__init__.pyi numpy/polynomial/__init__.meta.json numpy/polynomial/__init__.data.json
LOG:  Cached module numpy.polynomial has changed interface
LOG:  Processing SCC singleton (imgcolorshine.fast_numba.utils) as inherently stale with stale deps (builtins numpy)
LOG:  Writing imgcolorshine.fast_numba.utils /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/utils.py imgcolorshine/fast_numba/utils.meta.json imgcolorshine/fast_numba/utils.data.json
LOG:  Cached module imgcolorshine.fast_numba.utils has changed interface
LOG:  Processing SCC singleton (imgcolorshine.fast_numba.falloff) as inherently stale with stale deps (builtins collections.abc enum numpy)
LOG:  Writing imgcolorshine.fast_numba.falloff /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/falloff.py imgcolorshine/fast_numba/falloff.meta.json imgcolorshine/fast_numba/falloff.data.json
LOG:  Cached module imgcolorshine.fast_numba.falloff has changed interface
LOG:  Processing SCC singleton (imgcolorshine.gpu) as inherently stale with stale deps (builtins loguru numpy)
LOG:  Writing imgcolorshine.gpu /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/gpu.py imgcolorshine/gpu.meta.json imgcolorshine/gpu.data.json
LOG:  Cached module imgcolorshine.gpu has changed interface
LOG:  Processing SCC singleton (imgcolorshine.fast_numba.trans_numba) as inherently stale with stale deps (__future__ builtins numpy typing)
LOG:  Writing imgcolorshine.fast_numba.trans_numba /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/trans_numba.py imgcolorshine/fast_numba/trans_numba.meta.json imgcolorshine/fast_numba/trans_numba.data.json
LOG:  Cached module imgcolorshine.fast_numba.trans_numba has changed interface
LOG:  Processing SCC singleton (imgcolorshine.lut) as inherently stale with stale deps (__future__ builtins hashlib loguru numpy pathlib pickle typing)
LOG:  Writing imgcolorshine.lut /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/lut.py imgcolorshine/lut.meta.json imgcolorshine/lut.data.json
LOG:  Cached module imgcolorshine.lut has changed interface
LOG:  Processing SCC singleton (imgcolorshine.fast_numba.gamut_numba) as inherently stale with stale deps (builtins imgcolorshine.fast_numba.trans_numba numpy)
LOG:  Writing imgcolorshine.fast_numba.gamut_numba /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/gamut_numba.py imgcolorshine/fast_numba/gamut_numba.meta.json imgcolorshine/fast_numba/gamut_numba.data.json
LOG:  Cached module imgcolorshine.fast_numba.gamut_numba has changed interface
LOG:  Processing SCC of size 2 (imgcolorshine.fast_numba.engine_kernels imgcolorshine.fast_numba) as inherently stale with stale deps (__future__ builtins imgcolorshine.fast_numba.falloff imgcolorshine.fast_numba.gamut_numba imgcolorshine.fast_numba.trans_numba imgcolorshine.fast_numba.utils numpy)
LOG:  Writing imgcolorshine.fast_numba.engine_kernels /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/engine_kernels.py imgcolorshine/fast_numba/engine_kernels.meta.json imgcolorshine/fast_numba/engine_kernels.data.json
LOG:  Cached module imgcolorshine.fast_numba.engine_kernels has changed interface
LOG:  Writing imgcolorshine.fast_numba /Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/fast_numba/__init__.py imgcolorshine/fast_numba/__init__.meta.json imgcolorshine/fast_numba/__init__.data.json
LOG:  Cached module imgcolorshine.fast_numba has changed interface
LOG:  Processing SCC of size 11 (imgcolorshine.fast_mypyc.gamut_helpers imgcolorshine.fast_mypyc.engine_helpers imgcolorshine.fast_mypyc.colorshine_helpers imgcolorshine.fast_mypyc imgcolorshine.utils imgcolorshine.io imgcolorshine.gamut imgcolorshine.fast_mypyc.utils imgcolorshine.engine imgcolorshine.colorshine imgcolorshine.cli) as inherently stale with stale deps (PIL PIL.Image __future__ builtins collections.abc coloraide dataclasses imgcolorshine.fast_numba imgcolorshine.fast_numba.engine_kernels imgcolorshine.fast_numba.gamut_numba imgcolorshine.fast_numba.trans_numba imgcolorshine.gpu imgcolorshine.lut importlib loguru numpy pathlib sys typing)
LOG:  Deleting imgcolorshine.fast_mypyc.gamut_helpers src/imgcolorshine/fast_mypyc/gamut_helpers.py imgcolorshine/fast_mypyc/gamut_helpers.meta.json imgcolorshine/fast_mypyc/gamut_helpers.data.json
LOG:  Deleting imgcolorshine.fast_mypyc.engine_helpers src/imgcolorshine/fast_mypyc/engine_helpers.py imgcolorshine/fast_mypyc/engine_helpers.meta.json imgcolorshine/fast_mypyc/engine_helpers.data.json
LOG:  Deleting imgcolorshine.fast_mypyc.colorshine_helpers src/imgcolorshine/fast_mypyc/colorshine_helpers.py imgcolorshine/fast_mypyc/colorshine_helpers.meta.json imgcolorshine/fast_mypyc/colorshine_helpers.data.json
LOG:  Deleting imgcolorshine.fast_mypyc src/imgcolorshine/fast_mypyc/__init__.py imgcolorshine/fast_mypyc/__init__.meta.json imgcolorshine/fast_mypyc/__init__.data.json
LOG:  Deleting imgcolorshine.utils src/imgcolorshine/utils.py imgcolorshine/utils.meta.json imgcolorshine/utils.data.json
LOG:  Deleting imgcolorshine.io src/imgcolorshine/io.py imgcolorshine/io.meta.json imgcolorshine/io.data.json
LOG:  Deleting imgcolorshine.gamut src/imgcolorshine/gamut.py imgcolorshine/gamut.meta.json imgcolorshine/gamut.data.json
LOG:  Deleting imgcolorshine.fast_mypyc.utils src/imgcolorshine/fast_mypyc/utils.py imgcolorshine/fast_mypyc/utils.meta.json imgcolorshine/fast_mypyc/utils.data.json
LOG:  Deleting imgcolorshine.engine src/imgcolorshine/engine.py imgcolorshine/engine.meta.json imgcolorshine/engine.data.json
LOG:  Deleting imgcolorshine.colorshine src/imgcolorshine/colorshine.py imgcolorshine/colorshine.meta.json imgcolorshine/colorshine.data.json
LOG:  Deleting imgcolorshine.cli src/imgcolorshine/cli.py imgcolorshine/cli.meta.json imgcolorshine/cli.data.json
LOG:  No fresh SCCs left in queue
LOG:  Build finished in 10.878 seconds with 394 modules, and 9 errors
src/imgcolorshine/fast_mypyc/gamut_helpers.py:25: error: Name "GamutMapper" is not defined  [name-defined]
src/imgcolorshine/fast_mypyc/gamut_helpers.py:57: error: Name "GamutMapper" is not defined  [name-defined]
src/imgcolorshine/fast_mypyc/utils.py:12: error: Invalid "type: ignore" comment  [syntax]
```
