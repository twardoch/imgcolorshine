# TODO

## Phase 1

```
./testdata/example.sh

Running imgcolorshine examples on louis.jpg...
================================================
1. Basic red attractor (moderate tolerance and strength)
2025-06-15 00:19:01.062 | DEBUG    | imgcolorshine.image_io:<module>:26 - Using OpenCV for image I/O
Loading image: louis.jpg
Loading image: louis.jpg
Transforming colors...
Transforming 946×1280 image with 1 attractors
Traceback (most recent call last):
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/.venv/bin/imgcolorshine", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/cli.py", line 71, in main
    fire.Fire(ImgColorShineCLI)
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/.venv/lib/python3.12/site-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/.venv/lib/python3.12/site-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
                                ^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/.venv/lib/python3.12/site-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/cli.py", line 53, in shine
    process_image(
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/imgcolorshine.py", line 128, in process_image
    transformed = transformer.transform_image(image, attractor_objects, flags)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/transforms.py", line 298, in transform_image
    result = self._transform_tile(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/transforms.py", line 345, in _transform_tile
    result = self.engine.batch_oklab_to_rgb(transformed_lab)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine/src/imgcolorshine/color_engine.py", line 298, in batch_oklab_to_rgb
    return np.array(rgb_list).reshape(h, w, 3)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'float' object cannot be interpreted as an integer
```

- [x] Fix the problem that it shows `Transforming 946×1280 image with 1 attractors` but the image is actually 1280 × 946.
-
- [x] Fix error `TypeError: 'float' object cannot be interpreted as an integer` (cast shape dimensions to `int` before reshape).
-
- [ ] Improve performance. Initial optimizations are in place (Numba kernels, minor clean-ups), but further profiling is still required.

2. Generally it's a bit slow. Do we use optimized code, like numba etc.? 


## Phase 2

Run `./testdata/example.sh` and fix errors until it successfully run, then check the outputs and analyze them, and report the findings into `./testdata/example.md`.


## Phase 3

- [ ] Add comprehensive tests for all modules
- [ ] Plan GPU acceleration support

- Maintain compatibility with the existing pyproject.toml configuration