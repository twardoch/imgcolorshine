# TODO

TASK: Work through the steps below. Once you've completed a step, mark it with `[x]`. Always use `uv` or `hatch` or `python`, not `python3`.

## This tool works like so:

Tolerance of 0% influences no pixels, tolerance of 100% influences all pixels, and tolerance of 50% influences that half of the pixels that are more similar to the attractor than the other half. 

The actual influence of the attractor onto a given pixel should always stronger if the pixel is more similar to the attractor, and less strong if it's less similar. 

The strength of 100% means that the influence of the attractor onto the pixels that are most similar to the attractor is full, that is, these pixels take on the hue and/or saturation and/or luminance of the attractor. But for pixels that are less similar, there's a falloff. 

Aa strength of 50% means that the influence is 50% but only on the most similar pixels, that is, the new value of H or S or L becomes 50% of the old one and 50% of the new one. But the strength of the influence always falls off, the less similar the pixel is to the attractor. 

The strength of 200% means there is no falloff: the influence is always full within the tolerance. 

## Task

1. Rewrite `PLAN.md` so that its various subeadings and steps are numbered and include checkable `[ ]` boxes. 

2. Start implementing the tasks described in PLAN.md. As you work, check off the `[x]` boxes in the `PLAN.md` file. 

3. Work tirelessly, without asking me any questions, until the job is complete. 