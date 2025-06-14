#!/usr/bin/env bash
# this_file: example.sh

# Example script demonstrating various imgcolorshine shine operations on louis.jpg
# Run from the project root directory

# Change working directory to the location of this script
cd "$(dirname "$0")"

set -e # Exit on error

# Create output directory if it doesn't exist
mkdir -p output

echo "Running imgcolorshine examples on louis.jpg..."
echo "================================================"

# Example 1: Basic single attractor - warm red tones
echo "1. Basic red attractor (moderate tolerance and strength)"
imgcolorshine shine louis.jpg "red;50;75" \
    --output_image=output/louis-red-50-75.jpg

# Example 2: Blue shift with high tolerance
echo "2. Blue attractor with high tolerance"
imgcolorshine shine louis.jpg "blue;80;60" \
    --output_image=output/louis-blue-80-60.jpg

# Example 3: Green with only hue transformation
echo "3. Green hue shift only (no luminance/saturation changes)"
imgcolorshine shine louis.jpg "green;60;90" \
    --luminance=False --saturation=False \
    --output_image=output/louis-green-hue-only.jpg

# Example 4: Multiple attractors - warm sunset effect
echo "4. Multiple attractors for sunset effect"
imgcolorshine shine louis.jpg \
    "oklch(80% 0.2 60);40;60" \
    "#ff6b35;30;80" \
    "oklch(70% 0.3 40);25;50" \
    --output_image=output/louis-sunset-multi.jpg

# Example 5: Subtle purple tones
echo "5. Subtle purple transformation"
imgcolorshine shine louis.jpg "#9b59b6;35;40" \
    --output_image=output/louis-purple-subtle.jpg

# Example 6: High strength cyan transformation
echo "6. Strong cyan color replacement"
imgcolorshine shine louis.jpg "cyan;45;95" \
    --output_image=output/louis-cyan-strong.jpg

# Example 7: Multiple complementary colors
echo "7. Complementary color blend (orange and teal)"
imgcolorshine shine louis.jpg \
    "oklch(75% 0.25 50);55;70" \
    "oklch(65% 0.15 200);45;60" \
    --output_image=output/louis-complementary.jpg

# Example 8: Desaturation effect using gray attractor
echo "8. Partial desaturation with gray attractor"
imgcolorshine shine louis.jpg "oklch(60% 0 0);70;80" \
    --output_image=output/louis-desaturated.jpg

# Example 9: Only luminance changes with bright attractor
echo "9. Brightening with luminance-only transformation"
imgcolorshine shine louis.jpg "oklch(90% 0.1 60);60;70" \
    --saturation=False --hue=False \
    --output_image=output/louis-brightened.jpg

# Example 10: Complex multi-attractor artistic effect
echo "10. Artistic multi-color transformation"
imgcolorshine shine louis.jpg \
    "hsl(280 100% 50%);30;60" \
    "#00ff88;25;50" \
    "oklch(55% 0.3 350);40;70" \
    "#ffaa00;35;45" \
    --output_image=output/louis-artistic.jpg

# Example 11: Using verbose mode for debugging
echo "11. Verbose mode example (check console output)"
imgcolorshine shine louis.jpg "magenta;50;50" \
    --verbose=True \
    --output_image=output/louis-magenta-verbose.jpg

# Example 12: Edge case - very low tolerance and strength
echo "12. Minimal effect (low tolerance and strength)"
imgcolorshine shine louis.jpg "yellow;10;20" \
    --output_image=output/louis-yellow-minimal.jpg

echo ""
echo "All examples completed! Check output/ for results."
echo "================================================"

# Optional: Create a comparison montage using ImageMagick if available
if command -v montage &>/dev/null; then
    echo "Creating comparison montage..."
    montage louis.jpg output/louis-*.jpg \
        -tile 4x4 -geometry 200x200+5+5 \
        -label '%f' \
        output/montage-comparison.jpg
    echo "Montage created: output/montage-comparison.jpg"
fi
