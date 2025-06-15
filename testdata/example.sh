#!/usr/bin/env bash
# this_file: example.sh

# Example script demonstrating various imgcolorshine shine operations on louis.jpg
# Run from the project root directory

# Change working directory to the location of this script
cd "$(dirname "$0")"

set -e # Exit on error

# Check if GNU Parallel is available and silence citation notice
if ! command -v parallel &>/dev/null; then
    echo "GNU Parallel is not installed. Running in sequential mode."
    PARALLEL_AVAILABLE=0
else
    # Silence the citation notice
    parallel --citation >/dev/null 2>&1 || true
    PARALLEL_AVAILABLE=1
fi

# Create output directory if it doesn't exist
mkdir -p output

echo "Running imgcolorshine examples on louis.jpg..."
echo "================================================"

# Generate all parameter combinations
generate_params() {
    for a in 50 99; do
        for b in 50 99; do
            for c in blue yellow; do
                echo "$c;$a;$b"
            done
        done
    done
}

# Function to process a single parameter set
process_params() {
    local params=$1
    local luminance=$2
    local saturation=$3
    local chroma=$4
    local suffix=$5

    # Extract color and values from params
    IFS=';' read -r color tolerance strength <<<"$params"

    echo "Processing: $color with tolerance=$tolerance, strength=$strength (l=$luminance,s=$saturation,h=$chroma)"
    imgcolorshine shine louis.jpg "$params" \
        --luminance "$luminance" --saturation "$saturation" --chroma "$chroma" \
        --output_image="output/louis-$suffix-$color-$tolerance-$strength.jpg"
}

export -f process_params

# Generate all parameter combinations
PARAMS=$(generate_params)

# Process with different flag combinations
if [ "$PARALLEL_AVAILABLE" -eq 1 ]; then
    echo "Running in parallel mode..."

    # Luminance only
    echo "$PARAMS" | parallel process_params {} True False False "l"

    # Saturation only
    echo "$PARAMS" | parallel process_params {} False True False "s"

    # chroma only
    echo "$PARAMS" | parallel process_params {} False False True "h"

    # All flags enabled
    echo "$PARAMS" | parallel process_params {} True True True "lsh"

else
    echo "Running in sequential mode..."

    # Luminance only
    echo "$PARAMS" | while read -r params; do
        process_params "$params" True False False "l"
    done

    # Saturation only
    echo "$PARAMS" | while read -r params; do
        process_params "$params" False True False "s"
    done

    # chroma only
    echo "$PARAMS" | while read -r params; do
        process_params "$params" False False True "h"
    done

    # All flags enabled
    echo "$PARAMS" | while read -r params; do
        process_params "$params" True True True "lsh"
    done
fi

# Optional: Create a comparison montage using ImageMagick if available
if command -v montage &>/dev/null; then
    echo "Creating comparison montage..."
    montage louis.jpg output/louis-*.jpg \
        -tile 4x4 -geometry 200x200+5+5 \
        -label '%f' \
        output/montage-comparison.jpg
    echo "Montage created: output/montage-comparison.jpg"
fi
