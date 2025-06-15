
# Task 2

Analyze the proposals by Perplexity, Phind, Grok, Gemini, You.com and ChatGPT. 

Judge each proposal with 1 to 5 emoji stars. Write a brief review for each. 

Then distill the most promising and best approaches into one solid unified detailed specification. The spec needs to be written in a way that is easy to understand and easy to use. A senior developer should be able to read it, understand it, and implement the tool based on the spec.

# Proposal by Gemini

Here is an analysis of the proposals and a unified, detailed specification for the `imgcolorshine` tool.

***

## 1. Analysis of Proposals

### 1.1. Perplexity
**Review:** A decent first attempt that correctly identifies the core concepts like using OKLCH and a physics-inspired model. However, the proposed formula for multi-attractor blending is non-standard and poorly explained, making it difficult to implement correctly. It's a good conceptual start but lacks the detailed rigor needed for implementation.
**Rating:** ⭐⭐⭐☆☆

### 1.2. Phind
**Review:** This proposal is too simplistic. The mathematical model for similarity and transformation is underdeveloped and contains errors, particularly in handling hue distance. It provides a basic structure but misses critical details like gamma correction and offers a less-than-robust algorithm.
**Rating:** ⭐⭐☆☆☆

### 1.3. Grok
**Review:** Grok's proposal correctly identifies the need to compute distance in the Cartesian Oklab space and properly handles hue's angular nature. However, its model for combining attractors is a simple additive approach, which can lead to unpredictable results and color "overshooting," unlike a normalized blending model.
**Rating:** ⭐⭐⭐☆☆

### 1.4. Gemini
**Review:** This is an outstanding proposal. It is exceptionally thorough, scientifically rigorous, and structured like a professional technical paper. It provides a deep justification for using OKLCH, offers a sophisticated model for tolerance and strength with non-linear falloff, and specifies a robust Inverse Distance Weighting method for blending. Most importantly, its implementation pipeline is complete, including critical details like gamma correction and gamut clipping. The pseudocode is clear and directly implementable.
**Rating:** ⭐⭐⭐⭐⭐

### 1.5. You.com
**Review:** A solid, well-researched summary that touches on all the key aspects of the problem, from perceptual color spaces to gamut mapping. It correctly identifies the core components of the solution. However, it functions more as a good literature review than a concrete specification, lacking the specific, unambiguous formulas and algorithmic detail a developer would need.
**Rating:** ⭐⭐⭐☆☆

### 1.6. ChatGPT
**Review:** An excellent, practical, and well-structured proposal. It clearly explains the "why" behind its choices and offers a fantastic discussion of different falloff curves, demonstrating a deep understanding of the implementation nuances. Its description of normalized blending for multiple attractors is clear and correct. Its only minor weakness is the omission of the critical gamma correction step in the final algorithm summary.
**Rating:** ⭐⭐⭐⭐☆

***

## 2. Unified Specification for `imgcolorshine`

This specification synthesizes the best approaches from the provided proposals, with a primary focus on the rigor of the **Gemini** proposal and the practical clarity of the **ChatGPT** proposal.

### 2.1. Overview

`imgcolorshine` is a command-line tool that provides a powerful, physics-inspired method for transforming image colors. It works by "attracting" the colors of a source image toward one or more user-defined "attractor" colors. The tool operates internally in the **OKLCH color space** to ensure all calculations are perceptually uniform, intuitive, and produce high-quality, natural-looking results. Users can precisely control the range and intensity of the effect and specify whether to affect a color's lightness, chroma (saturation), or hue.

### 2.2. Theoretical Foundations

#### 2.2.1. The Choice of OKLCH
To manipulate color in a way that feels intuitive, the underlying color model must align with human perception. Traditional models like RGB (hardware-oriented) and HSL (a flawed early attempt at perceptual modeling) are unsuitable. In HSL, for example, pure yellow (`hsl(60, 100%, 50%)`) and pure blue (`hsl(240, 100%, 50%)`) have the same "Lightness" value but are perceived as having dramatically different brightness.

This specification mandates the use of the **OKLCH** color space for all internal calculations. OKLCH is perceptually uniform, meaning a numerical change of a certain amount in its L (Lightness), C (Chroma), or H (Hue) components results in a consistent perceptual change. This allows us to define "color distance" in a meaningful way.

#### 2.2.2. Perceptual Distance (ΔE) in Oklab
The "tolerance" of an attractor requires a reliable way to measure the perceptual difference between two colors. Because OKLCH is uniform, we can use the **Euclidean distance in its underlying Oklab space** as our perceptual difference metric (ΔEok). The Oklab space represents colors using the same Lightness (L) but replaces the polar coordinates of Chroma and Hue with Cartesian `a` (green-red) and `b` (blue-yellow) axes.

The conversion from OKLCH to Oklab is a standard polar-to-Cartesian conversion:
* `a = C * cos(h)`
* `b = C * sin(h)` *(hue `h` must be in radians)*

The perceptual difference ΔEok between two colors `p1` and `p2` is then:
`ΔEok = sqrt((L₁ - L₂)² + (a₁ - a₂)² + (b₁ - b₂)²)`

This metric is both computationally efficient and perceptually accurate, forming the mathematical basis for the `tolerance` parameter.

### 2.3. The Transformation Model

#### 2.3.1. The Attractor Primitive
Each color attractor is defined by a string: `css_color;tolerance;strength`
* **`css_color`**: A CSS color specifier (e.g., `red`, `#ff8800`, `oklch(70% 0.2 50)`).
* **`tolerance` (0-100)**: Controls the "reach" of the attractor. A higher value affects a wider range of colors.
* **`strength` (0-100)**: Controls the maximum magnitude of the transformation.

#### 2.3.2. Single Attractor Influence
For each pixel, we calculate its interaction with each attractor.

**1. Tolerance Field (Radius of Influence)**
The user's `tolerance` (0-100) is mapped to a maximum perceptual distance, **ΔEmax**. Any pixel with a color difference greater than ΔEmax from the attractor is unaffected. A non-linear mapping is used to give finer control at lower tolerance values.
`ΔEmax = 1.0 * (tolerance / 100)²`
*A scale factor of `1.0` is used, as the distance from black (L=0) to white (L=1) in Oklab is exactly 1.0. This quadratic mapping means a tolerance of 50 corresponds to a ΔEmax of 0.25.*

**2. Attraction Falloff (Influence Weight)**
A pixel's color may be inside the tolerance field but far from the attractor's exact color. Its influence should weaken with distance. We model this with a smooth falloff curve.

First, calculate the **normalized distance** `d_norm` (0 to 1):
`d_norm = ΔEok / ΔEmax`

Next, calculate the **attraction factor** (0 to 1) using a smooth "ease-out" function. A raised cosine curve is recommended for its natural falloff:
`attraction_factor = 0.5 * (cos(d_norm * π) + 1)`

Finally, the **final interpolation weight** `t_interp` is determined by the user's `strength`:
`t_interp = (strength / 100) * attraction_factor`

This `t_interp` value dictates how much the pixel's color will be pulled toward the attractor.

#### 2.3.3. Multi-Attractor Blending
When multiple attractors influence a single pixel, their effects must be blended. We use a **normalized weighted average**, where each attractor's contribution is weighted by its influence.

For a given pixel, we calculate the `t_interp` value from each influential attractor (i.e., each attractor whose tolerance field the pixel falls within). Let's call this value `w_i` for attractor `i`.

The new color `P_final` is a blend of the original pixel color `P_src` and all active attractor colors `C_attri`.

1.  Calculate the total weight: `W_total = Σ w_i`
2.  If `W_total > 1`, normalize all weights: `w_i = w_i / W_total`. The weight of the original color becomes 0.
3.  If `W_total <= 1`, the weight of the original color is `w_src = 1 - W_total`.

The final color is the weighted average:
`P_final = (w_src * P_src) + Σ (w_i * C_attri)`

This calculation must be performed component-wise (for L, C, and H). For **Hue (H)**, a **weighted circular mean** must be used to handle its angular nature correctly.

#### 2.3.4. Selective Channel Application
The `--luminance`, `--saturation` (Chroma), and `--hue` flags restrict the transformation to specific channels. If a flag is disabled, that color component is not changed.

This is implemented by modifying the final blending step. For any disabled channel, the final value is simply the source pixel's original value for that channel, instead of the calculated blended value.

Example: If only `--luminance` and `--hue` are active, the final color will be:
* `L_final` = blended lightness
* `C_final` = `C_src` (original chroma is preserved)
* `H_final` = blended hue

### 2.4. Implementation Specification

#### 2.4.1. CLI Definition
```bash
imgcolorshine --input_image <path> \
              [--output_image <path>] \
              [--luminance] [--saturation] [--chroma] \
              "color1;tol1;str1" ["color2;tol2;str2" ...]
```

#### 2.4.2. End-to-End Processing Pipeline
1.  **Parse Arguments:** Read all CLI arguments. Parse attractor strings and validate them. Convert each attractor's CSS color into OKLCH coordinates.
2.  **Load Image:** Load the input image. Assume it is in the sRGB color space.
3.  **Gamma Decode (Critical Step):** Convert the sRGB image data to **Linear sRGB**. All color math must be done in a linear space.
4.  **Convert to Oklab:** Convert the linear sRGB pixel data to the Oklab color space.
5.  **Allocate Output Buffer:** Create an empty buffer for the transformed Oklab pixel data.
6.  **Per-Pixel Transformation Loop:** Iterate through each pixel of the Oklab image.
    a. Get the source pixel's Oklab color `P_src`.
    b. Apply the **Multi-Attractor Blending** algorithm from section 3.3 to calculate the final Oklab color `P_final`.
    c. Store `P_final` in the output buffer.
7.  **Convert to Linear sRGB:** Convert the Oklab output buffer back to Linear sRGB.
8.  **Gamut Clipping:** The transformation may produce colors outside the sRGB gamut. These colors must be mapped back into gamut. The recommended method is to preserve the color's L and H while progressively reducing its C (Chroma) until it fits.
9.  **Gamma Encode:** Convert the gamut-clipped, linear sRGB data back to standard sRGB by applying the sRGB gamma curve.
10. **Save Image:** Save the final sRGB pixel data to the output file.

#### 2.4.3. Pseudocode for Core Transformation

```python
# Constants
IDW_POWER = 2.0  # For future extension, not used in this simplified blend

function
transform_pixel(p_src_oklab, attractors, flags):
p_src_oklch = convert_oklab_to_oklch(p_src_oklab)

influential_proposals = []
influential_weights = []
total_weight = 0.0

for attractor in attractors:
    # 1. Calculate distance and check if in tolerance
    delta_e = calculate_delta_e_ok(p_src_oklab, attractor.oklab_color)
    delta_e_max = 1.0 * (attractor.tolerance / 100.0) ** 2

    if delta_e <= delta_e_max:
        # 2. Calculate falloff and final weight (t_interp)
        d_norm = delta_e / delta_e_max
        attraction_factor = 0.5 * (cos(d_norm * PI) + 1.0)
        weight = (attractor.strength / 100.0) * attraction_factor

        influential_proposals.append(attractor.oklch_color)
        influential_weights.append(weight)
        total_weight += weight

if not influential_proposals:
    return p_src_oklch  # No change

# 3. Normalize weights and add source color's weight
final_weights = influential_weights
src_weight = 0.0

if total_weight > 1.0:
    # Normalize all proposal weights to sum to 1
    final_weights = [w / total_weight for w in influential_weights]
else:
    # Keep proposal weights and add source color's weight
    src_weight = 1.0 - total_weight

# 4. Calculate weighted average for each enabled channel
final_l, final_c, final_h = p_src_oklch.l, p_src_oklch.c, p_src_oklch.h

# --- Lightness ---
if flags.luminance:
    l_sum = src_weight * p_src_oklch.l
    for i, proposal in enumerate(influential_proposals):
        l_sum += final_weights[i] * proposal.l
    final_l = l_sum

# --- Chroma ---
if flags.saturation:
    c_sum = src_weight * p_src_oklch.c
    for i, proposal in enumerate(influential_proposals):
        c_sum += final_weights[i] * proposal.c
    final_c = c_sum

# --- Hue (using weighted circular mean) ---
if flags.chroma:
    all_colors = [p_src_oklch] + influential_proposals
    all_weights = [src_weight] + final_weights
    final_h = calculate_weighted_circular_mean(all_colors, all_weights)

return OklchColor(final_l, final_c, final_h)

function
calculate_weighted_circular_mean(colors, weights):
sum_sin = 0.0
sum_cos = 0.0
total_w = sum(weights)

for i, color in enumerate(colors):
    hue_rad = to_radians(color.h)
    sum_sin += sin(hue_rad) * weights[i]
    sum_cos += cos(hue_rad) * weights[i]

avg_sin = sum_sin / total_w
avg_cos = sum_cos / total_w

return to_degrees(atan2(avg_sin, avg_cos))
```

# Proposal by Claude

## 3. Proposal Reviews

### 3.1. Perplexity's Proposal ⭐⭐⭐
While reference-rich and covering important concepts, this proposal suffers from being more of a literature review than a specification. The excessive citations ([1][2][3]...) make it difficult to follow, and the mathematical models lack concrete implementation details. The multi-attractor blending section is particularly underdeveloped. It reads like a research paper draft rather than a practical specification.

### 3.2. Phind's Proposal ⭐⭐⭐⭐
A well-structured specification with good balance between theory and practice. The clear sections and algorithm outline are helpful, though there's a mathematical error in the similarity calculation (the `2(C₁·C₂)(1-cos(H₁-H₂))` term is incorrect for hue distance). The practical focus and example usage are strong points, but it could benefit from more detail on falloff functions and edge cases.

### 3.3. Grok's Proposal ⭐⭐⭐⭐
Excellent organization with practical implementation focus. The command-line specification table is particularly useful, and the mathematical detail is appropriate. However, the notation becomes unnecessarily complex in places (especially the modular arithmetic for hue), and the discussion of different falloff functions is missing. The structured approach with clear parameter definitions is exemplary.

### 3.4. Gemini's Proposal ⭐⭐⭐⭐⭐
Outstanding specification that sets the gold standard. The theoretical foundations are impeccable, the mathematical models are rigorous yet accessible, and the pseudocode is comprehensive. The tolerance mapping table and parameter glossary are particularly valuable. The discussion of easing functions for attraction falloff shows deep understanding. Only minor weakness is its length, which might intimidate some readers.

### 3.5. You.com's Proposal ⭐⭐⭐
A decent overview that covers the basics but lacks the depth needed for a robust implementation. The scientific foundation is sound but not thoroughly explored. The mathematical formulas are correct but basic, and the implementation guidance is too high-level. It would serve better as an introduction than a complete specification.

### 3.6. ChatGPT's Proposal ⭐⭐⭐⭐⭐
Excellent balance of depth and clarity. The discussion of different falloff functions (linear, cosine, Gaussian, sigmoid) is particularly strong. The mathematical explanations are clear and correct, and the handling of multiple attractors through normalized blending is well-explained. The algorithm outline is practical and comprehensive. The channel-specific toggle implementation is clearly detailed.

---

# Unified Specification for `imgcolorshine`

## 4. Executive Summary

`imgcolorshine` is a command-line tool that transforms image colors by attracting them toward user-specified target colors in the perceptually uniform OKLCH color space. Each color attractor exerts influence within a tolerance radius, with strength determining the magnitude of transformation. The tool enables precise control over color grading through selective adjustment of lightness, chroma, and hue components.

## 5. Core Concepts

### 5.1. 2.1 Color Space Foundation

The tool operates internally in **OKLCH** (Lightness, Chroma, Hue), the cylindrical representation of Oklab. This space was chosen for:
- **Perceptual uniformity**: Equal numerical changes produce equal perceived changes
- **Hue linearity**: No unexpected color shifts during interpolation  
- **Component independence**: L, C, and H can be adjusted separately without artifacts

### 5.2. 2.2 Color Attractors

Each attractor is defined by three parameters:
- **Color**: Any CSS color specification (e.g., `red`, `#ff0000`, `oklch(0.7 0.2 30)`)
- **Tolerance** (0-100): Radius of influence in perceptual units
- **Strength** (0-100): Maximum transformation intensity

## 6. Mathematical Model

### 6.1. 3.1 Perceptual Distance

Color similarity is measured using Euclidean distance in Oklab space:

```
ΔE_ok = √[(L₁-L₂)² + (a₁-a₂)² + (b₁-b₂)²]
```

Where (a,b) are derived from (C,h) via: `a = C·cos(h)`, `b = C·sin(h)`

### 6.2. 3.2 Tolerance Mapping

User tolerance values (0-100) map to perceptual distances using:

```
ΔE_max = 1.5 × (tolerance/100)²
```

| Tolerance | ΔE_max | Perceptual Meaning |
|-----------|--------|-------------------|
| 0 | 0.0 | Exact matches only |
| 10 | 0.015 | Nearly identical colors |
| 25 | 0.094 | Similar shades |
| 50 | 0.375 | Related colors |
| 75 | 0.844 | Broad color families |
| 100 | 1.500 | Maximum range |

### 6.3. 3.3 Attraction Function

Within the tolerance radius, influence follows a smooth falloff:

```
influence = strength/100 × falloff(d/ΔE_max)
```

Where `falloff(x)` is a raised cosine function:
```
falloff(x) = 0.5 × [1 + cos(π × x)]  for 0 ≤ x ≤ 1
           = 0                        for x > 1
```

### 6.4. 3.4 Multi-Attractor Blending

When multiple attractors influence a pixel, their effects combine via Inverse Distance Weighting:

1. Calculate each attractor's weight: `w_i = influence_i`
2. If Σw_i > 1, normalize: `w_i = w_i / Σw_i`
3. Blend colors:
   ```
   C_final = (1 - Σw_i) × C_original + Σ(w_i × C_attractor_i)
   ```

## 7. Algorithm Implementation

### 7.1. 4.1 Processing Pipeline

```python
def process_image(image, attractors, flags):
    # 1. Convert image to Oklab
    oklab_image = srgb_to_oklab(image)
    
    # 2. Parse attractors
    oklab_attractors = [parse_and_convert(attr) for attr in attractors]
    
    # 3. Transform each pixel
    for pixel in oklab_image:
        # Calculate influences
        weights = []
        for attractor in oklab_attractors:
            d = calculate_delta_e(pixel, attractor)
            if d <= attractor.tolerance:
                w = attractor.strength * falloff(d / attractor.tolerance)
                weights.append((w, attractor))
        
        # Normalize if needed
        total_weight = sum(w for w, _ in weights)
        if total_weight > 1:
            weights = [(w/total_weight, attr) for w, attr in weights]
            total_weight = 1
        
        # Apply transformation
        new_color = pixel * (1 - total_weight)
        for weight, attractor in weights:
            new_color += weight * apply_channel_mask(attractor, pixel, flags)
        
        pixel = new_color
    
    # 4. Convert back to sRGB with gamut mapping
    return oklab_to_srgb(oklab_image, gamut_clip=True)
```

### 7.2. 4.2 Channel-Specific Transformation

When flags restrict transformation to specific channels:

- **`--luminance` only**: Keep C and H from original, blend only L
- **`--saturation` only**: Keep L and H from original, blend only C  
- **`--hue` only**: Keep L and C from original, rotate H along shortest arc

## 8. Command-Line Interface

```bash
imgcolorshine --input_image <path> [--output_image <path>]
              [--luminance] [--saturation] [--chroma]
              <color>;<tolerance>;<strength> [...]
```

### 8.1. 5.1 Arguments

- `--input_image`: Input file path (required)
- `--output_image`: Output file path (auto-generated if omitted)
- `--luminance`: Enable lightness transformation
- `--saturation`: Enable chroma transformation  
- `--hue`: Enable hue transformation
- Color attractors: One or more in format `color;tolerance;strength`

**Note**: If no channel flags are specified, all three are enabled by default.

### 8.2. 5.2 Examples

```bash
# Warm color grade - attract to orange, preserve luminance
imgcolorshine --input_image photo.jpg --saturation --chroma \
              "oklch(0.75 0.15 50);40;60"

# Color harmonization - multiple attractors
imgcolorshine --input_image poster.png \
              "#e74c3c;30;80" "#3498db;30;80" "#f39c12;25;70"

# Subtle enhancement - boost reds only
imgcolorshine --input_image portrait.jpg --saturation \
              "red;15;40"
```

## 9. Implementation Requirements

### 9.1. 6.1 Color Space Conversions

Implement or use libraries for:
- sRGB ↔ Linear RGB (gamma encoding/decoding)
- Linear RGB ↔ XYZ (D65 illuminant)
- XYZ ↔ Oklab ↔ OKLCH

### 9.2. 6.2 CSS Color Parsing

Support standard CSS color formats:
- Named colors: `red`, `blue`, `salmon`
- Hex: `#rgb`, `#rrggbb`, `#rrggbbaa`
- Functions: `rgb()`, `hsl()`, `oklch()`

### 9.3. 6.3 Performance Optimization

- Vectorize operations using NumPy
- Pre-compute attractor Oklab coordinates
- Use lookup tables for gamma conversions
- Consider parallel processing for large images

### 9.4. 6.4 Edge Cases

- **Hue wrapping**: Use circular interpolation (shortest arc)
- **Gamut clipping**: Reduce chroma while preserving L and H
- **Zero weights**: Preserve original pixel color
- **Exact matches**: Handle d=0 gracefully (full attractor weight)

## 10. Quality Assurance

### 10.1. 7.1 Test Cases

1. Single attractor with varying tolerance/strength
2. Multiple overlapping attractors
3. Channel-specific transformations
4. Out-of-gamut color handling
5. Large images (performance testing)

### 10.2. 7.2 Validation

- Compare ΔE calculations against reference implementations
- Verify perceptual uniformity of transformations
- Test CSS color parser against MDN examples
- Ensure round-trip accuracy for color space conversions

## 11. Future Extensions

- **Advanced falloff functions**: User-selectable (Gaussian, sigmoid, linear)
- **Spatial awareness**: Attractors with (x,y) positions using RBF
- **Palette extraction**: Auto-generate attractors from reference image
- **GPU acceleration**: Shader-based implementation
- **Real-time preview**: Interactive parameter adjustment

# Proposal by ChatGPT

### 11.1. 1 Proposal Reviews

| # | Source         | Rating | Short appraisal                                                                                                                                                                                                           |
| - | -------------- | :----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **Perplexity** |  ⭐⭐⭐⭐  | Clear structure, cites sources, uses an exponential fall‑off and weighted blending. However the final “normalisation” formula is mathematically wrong (double counts P) and channel flags are handled only superficially. |
| 2 | **Phind**      |   ⭐⭐⭐  | Succinct and easy to read but over‑simplified: distance metric is half‑baked, multi‑attractor interaction is just “average”, no gamut or edge‑case notes, and no colour‑space conversion details.                         |
| 3 | **Grok**       |  ⭐⭐⭐⭐½ | Very solid: derives distance in Oklab, linear fall‑off, proper IDW weighting, good handling of channel flags and gamut, plus worked example. Minor quibbles: tolerance scaling is hard‑wired and the text is quite dense. |
| 4 | **Gemini**     |  ⭐⭐⭐⭐  | Deep discussion of colour theory and perceptual issues; maps tolerance non‑linearly; introduces easing functions and IDW. Down‑side: extremely long, some digressions, and no concise implementation section.             |
| 5 | **You.com**    |  ⭐⭐⭐⭐  | Balanced: good explanation of ΔE\_ok, fall‑off choices, gamut mapping, and performance hints. Channel flags handled but not fully formalised; no explicit pseudocode.                                                     |
| 6 | **ChatGPT**    |  ⭐⭐⭐⭐½ | Comprehensive and pragmatic. Gives clean IDW normalised blending, clear channel‑flag semantics, raised‑cosine default fall‑off, gamut handling, and vectorised pseudocode. A bit verbose but closest to “ready‑to‑code”.  |

---

## 12. 2 Unified Specification for **`imgcolorshine`**

### 12.1. 2.1 Purpose

Re‑colour an input image by **attracting each pixel’s colour toward one or more user‑defined “colour attractors”** in the perceptually‑uniform **OKLCH** space, with separate switches for lightness (L), chroma (C) and hue (H).

---

### 12.2. 2.2 Command‑line interface

```bash
imgcolorshine \
  --input_image  INPUT_PATH                  # required
  [--output_image OUTPUT_PATH]               # auto‑name if omitted
  [--luminance] [--saturation] [--chroma]       # at least one required
  [--falloff {linear|cosine|gauss}]          # default: cosine
  [--idw-power P]                            # default: 2
  ATTR1 ATTR2 ... ATTRN
```

Each *ATTR* argument is:
`<CSS‑colour>;<tolerance 0‑100>;<strength 0‑100>`

Examples

```bash
imgcolorshine photo.jpg --luminance --chroma \
  "red;40;80"  "oklch(70% 0.20 260);25;60"
```

---

### 12.3. 2.3 Colour maths

| Symbol   | Meaning                                                         |
| -------- | --------------------------------------------------------------- |
| **p**    | pixel colour in cartesian Oklab **v₀ = (L,a,b)**                |
| **aᵢ**   | attractor *i* in Oklab **vᵢ = (Lᵢ,aᵢ,bᵢ)**                      |
| **dᵢ**   | Euclidean distance ‖v₀ − vᵢ‖ (ΔE\_ok)                           |
| **Tᵢ**   | tolerance radius (user % × 1.00)                                |
| **Sᵢ**   | strength factor (user % / 100)                                  |
| **f(x)** | fall‑off curve; default raised‑cosine *0.5(1+cos πx)* for 0≤x≤1 |
| **wᵢ**   | raw weight = Sᵢ · f(dᵢ/Tᵢ) if dᵢ < Tᵢ else 0                    |
| **W**    | Σwᵢ (total raw weight)                                          |

#### 12.3.1. 2.3.1 Channel masking

Before distance and blending, zero‑out components that are **disabled**:

```text
if not --luminance : set ΔL = 0 when computing dᵢ
if not --saturation: ignore chroma difference (i.e. compare only in L,h plane)
if not --hue       : ignore hue angle difference
```

After the blend (below), overwrite the corresponding channel with the original value if it was disabled, ensuring only permitted aspects change.

#### 12.3.2. 2.3.2 Blending algorithm (per pixel)

```text
1. Compute wᵢ for every attractor
2. If W == 0 → leave pixel unchanged
3. If W > 1 → scale all wᵢ ← wᵢ / W   ; set W = 1
4. New colour vₙ = (1 − W)·v₀  +  Σ wᵢ·vᵢ
```

> *Why?*  Steps 3–4 implement **normalised inverse‑distance weighting** (IDW) so contributions never overshoot and the original colour is preserved when total pull is weak.

#### 12.3.3. 2.3.3 Fall‑off options

| Name     | f(x) (0≤x≤1)     | Note           |
| -------- | ---------------- | -------------- |
| linear   | 1 − x            | fastest        |
| cosine\* | 0.5 (1 + cos πx) | smooth default |
| gauss    | exp\[ −4 x² ]    | very soft      |

---

### 12.4. 2.4 Pipeline

1. **Read input** (Pillow/OpenCV). Assume sRGB.
2. **Convert to linear sRGB**, then **to Oklab** (vectorised NumPy).
3. **Parse attractors:**

   * use `colour-science`, `coloraide`, or `culori` to read any CSS colour → OKLCH → Oklab.
   * tolerance % → ΔE units: *T = tol / 100 × 1.0* (full Oklab range).
4. **Process pixels** with algorithm § 2.3 (vectorised where possible).
5. **Optional gamut mapping:** if Oklab→sRGB conversion yields RGB <0 or >1, reduce chroma along constant L,h until in‑gamut.
6. **Convert back to sRGB**, apply gamma (IEC 61966‑2‑1 curve).
7. **Write output**. If `--output_image` absent, append `_colorshine` before extension.

---

### 12.5. 2.5 Complexity & performance

Time O(P·N) where *P* = pixels, *N* = attractors.
Optimise by:

* broadcasting NumPy arrays (whole image vs list of attractors)
* processing in tiles to limit RAM on large images
* optional OpenCL/CUDA kernel for realtime use

---

### 12.6. 2.6 Error handling

| Condition                         | Response                         |
| --------------------------------- | -------------------------------- |
| Bad image path / unreadable       | exit 1 + message                 |
| Malformed attractor string        | exit 2 + show expected syntax    |
| No channel flag given             | error (must choose at least one) |
| All tolerance = 0 or strength = 0 | warn “no effect”                 |

---

### 12.7. 2.7 Worked example

A pixel **p = oklch(0.60 0.12 40°)**, two attractors

| attractor                   | tol | str % | aᵢ (oklab) |
| --------------------------- | --- | ----- | ---------- |
| redish oklch(0.65 0.20 25°) | 30  | 80    | **v₁**     |
| blue #0080ff                | 20  | 60    | **v₂**     |

1. d₁ ≈ 14, T₁ = 0.30 → x₁=0.47 → w₁ = 0.8 · f(0.47)=0.8·0.19≈0.152
2. d₂ ≈ 48, T₂ = 0.20 → outside tolerance → w₂=0
3. W = 0.152 <1: keep 0.848 of original, 0.152 of attractor.
4. Blend → slight warm shift toward red, leaving lightness/chroma almost intact.

---

### 12.8. 2.8 Extensibility (non‑blocking ideas)

* `--preview-scale` down‑samples for a fast preview.
* `--lut 33` dumps a 3‑D LUT for GPU use.
* Spatial attractors (colour + (x,y)) using Radial Basis Functions.
* Alternative distance metric `--metric ciede2000` (slower, high‑accuracy).

---

### 12.9. 2.9 Reference implementation skeleton (Python ≥ 3.12)

```python
#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["pillow", "numpy", "coloraide>=3.0"]
# ///
# this_file: colorshine.py

import fire, numpy as np
from PIL import Image
from coloraide import Color

# ---------- utility ----------------------------------------------------------
def css_to_oklab(css: str) -> tuple[float, float, float]:
    c = Color(css).convert("oklab")  # coloraide uses a, b in range [-1,1]
    return c['l'], c['a'], c['b']

# ---------- main -------------------------------------------------------------
def main(
    input_image: str,
    output_image: str | None = None,
    luminance: bool = False,
    saturation: bool = False,
    hue: bool = False,
    falloff: str = "cosine",
    idw_power: float = 2.0,
    *attractors: str
) -> None:
    if not (luminance or saturation or hue):
        raise SystemExit("choose at least one of --luminance | --saturation | --chroma")

    # load + linearise
    img = Image.open(input_image).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    rgb_lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

    # to oklab
    oklab = rgb_to_oklab(rgb_lin)      # <- implement via matrix formula
    L, a, b = [oklab[..., i] for i in range(3)]

    # parse attractors
    attr = []
    for raw in attractors:
        color, tol, strength = raw.split(';')
        v = np.array(css_to_oklab(color), dtype=np.float32)
        attr.append((v, float(tol) / 100.0, float(strength) / 100.0))
    attr_v = np.stack([v for v, _, _ in attr])            # shape (N,3)
    T      = np.array([t for _, t, _ in attr], dtype=np.float32)
    S      = np.array([s for _, _, s in attr], dtype=np.float32)

    # vectorised per‑pixel processing
    # [... code omitted for brevity ...]

    # back to sRGB and save
    out_rgb = oklab_to_srgb(oklab_new)                    # gamut‑clip inside
    out = Image.fromarray((out_rgb * 255 + 0.5).astype(np.uint8), 'RGB')
    out.save(output_image or derive_name(input_image))

if __name__ == "__main__":
    fire.Fire(main)
```

*(Functions `rgb_to_oklab`, `oklab_to_srgb`, fall‑off kernels, and masking logic are straightforward one‑liners or small helpers; see spec § 2.3.)*

---

### 12.10. 3 Summary

This unified specification keeps the **best ideas** from the six proposals:

* **Oklab Euclidean ΔE** (Grok, ChatGPT, You.com) for speed & perceptual fidelity.
* **Raised‑cosine default fall‑off** (ChatGPT) but pluggable (Gemini).
* **Normalised IDW blending** (ChatGPT & Grok) – smooth, stable.
* **Channel‑flag masking** (all) with explicit pre/post rules (Gemini clarity).
* **Vectorised pipeline, gamut clipping, CLI ergonomics** (Perplexity, You.com).

A senior developer can drop the reference skeleton into a project, plug in the missing one‑liner maths, and have a working, extensible `imgcolorshine` in a few hours.

