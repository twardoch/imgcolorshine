#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "=== LUMINANCE-ONLY TRANSFORMATIONS ==="
echo "debug-l50-blue.jpg : Luminance-only (L50): Making blue-tinted areas lighter/darker toward blue's lightness, keeping original colors/hues"
imgcolorshine shine louis.jpg "blue;100;50" --luminance=True --saturation=False --hue=False --output_image=debug-l50-blue.jpg

echo "debug-l99-blue.jpg : Luminance-only (L99): Strongly adjusting brightness of blue-tinted areas toward blue's lightness, dramatic lighting changes"
imgcolorshine shine louis.jpg "blue;100;99" --luminance=True --saturation=False --hue=False --output_image=debug-l99-blue.jpg

echo "=== SATURATION-ONLY TRANSFORMATIONS ==="
echo "debug-s50-blue.jpg : Saturation-only (S50): Making blue-tinted areas more/less vivid toward blue's saturation, keeping original brightness/hue"
imgcolorshine shine louis.jpg "blue;100;50" --luminance=False --saturation=True --hue=False --output_image=debug-s50-blue.jpg

echo "debug-s99-blue.jpg : Saturation-only (S99): Strongly adjusting color vividness of blue-tinted areas toward blue's saturation, dramatic color intensity changes"
imgcolorshine shine louis.jpg "blue;100;99" --luminance=False --saturation=True --hue=False --output_image=debug-s99-blue.jpg

echo "=== HUE-ONLY TRANSFORMATIONS ==="
echo "debug-h50-blue.jpg : Hue-only (H50): Shifting colors toward blue hue while keeping original brightness/saturation - subtle color temperature changes"
imgcolorshine shine louis.jpg "blue;100;50" --luminance=False --saturation=False --hue=True --output_image=debug-h50-blue.jpg

echo "debug-h99-blue.jpg : Hue-only (H99): Strongly shifting colors toward blue hue while keeping original brightness/saturation - dramatic color cast changes"
imgcolorshine shine louis.jpg "blue;100;99" --luminance=False --saturation=False --hue=True --output_image=debug-h99-blue.jpg
