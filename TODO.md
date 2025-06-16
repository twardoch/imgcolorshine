# TODO

Our working folder is `/Users/adam/Developer/vcs/github.twardoch/pub/imgcolorshine`

TASK: Work through the steps below. Once you've completed a step, mark it with `[x]`. Always use `uv` or `hatch` or `python`, not `python3`.

## [ ] Task 1

1. Read the full codebase (`./llms.txt`) and `./cleanup.log` (the result of `./cleanup.sh`). 

2. In `PLAN.md` write a comprenehsive plan for accomplishing these tasks: 

- Factor out into dedicated modules inside `src/imgcolorshine/fast_mypyc` all code from all modules within `src/imgcolorshine` which are worth compiling with mypyc. 
- Factor out and move into dedicated modules inside `src/imgcolorshine/fast_numba` all code from all modules within `src/imgcolorshine` which use numba. 
- Make sure that the code directly in `src/imgcolorshine` uses pure Python.
- Update `pyproject.toml` so that the code in `src/imgcolorshine/fast_mypyc` is compiled with mypyc 
- Think of other ways to speed up the code. 

You must document your all these things in `PLAN.md`. The `PLAN.md` needs to have itemized lists of the tasks prefixed by `- [ ]`. The plan must be very detailed, and must explain not only what needs to be done, but also specifically how you will do it. 

## [ ] Task 2

Actually read `PLAN.md` and implement the tasks therein. 
