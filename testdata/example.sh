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

# Example: Basic single attractor - warm red tones
echo "Basic blue attractors"
for a in 20 40 60 80; do for b in 20 30 60 80; do
    echo "blue;$a;$b"
    imgcolorshine shine louis.jpg "blue;$a;$b" \
        --luminance=False --saturation=False \
        --output_image=output/louis-blue-$a-$b.jpg
done; done

# Optional: Create a comparison montage using ImageMagick if available
if command -v montage &>/dev/null; then
    echo "Creating comparison montage..."
    montage louis.jpg output/louis-*.jpg \
        -tile 4x4 -geometry 200x200+5+5 \
        -label '%f' \
        output/montage-comparison.jpg
    echo "Montage created: output/montage-comparison.jpg"
fi
