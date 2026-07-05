# this_file: src/imgcolorshine/cli.py
"""Fire-based command-line interface for imgcolorshine.

Usage::

    imgcolorshine shine photo.jpg "orange;50;75"
    imgcolorshine shine landscape.png "oklch(80% 0.2 60);40;60" "#ff6b35;30;80" --output_image=sunset.png
"""

from __future__ import annotations

import fire

from imgcolorshine.colorshine import process_image


class ImgColorShineCLI:
    """Command-line interface for imgcolorshine colour transformations."""

    def shine(
        self,
        input_image: str,
        *attractors: str,
        output_image: str | None = None,
        luminance: bool = True,
        saturation: bool = True,
        hue: bool = True,
        verbose: bool = False,
        tile_size: int = 1024,
        gpu: bool = True,
        lut_size: int = 0,
        fast_hierar: bool = False,
        fast_spatial: bool = True,
        **kwargs: object,
    ) -> None:
        """Transform image colours using colour attractors.

        Parameters
        ----------
        input_image:
            Path to the input image.
        *attractors:
            One or more attractor specifications: ``"color;tolerance;strength"``.
        output_image:
            Destination path (auto-generated if omitted).
        luminance:
            Transform the L (lightness) channel.
        saturation:
            Transform the C (chroma/saturation) channel.
        hue:
            Transform the H (hue) channel.
        verbose:
            Enable detailed logging.
        tile_size:
            Tile size for large-image processing.
        gpu:
            Use GPU acceleration when available.
        lut_size:
            3-D LUT size (0 = disabled, 65 = recommended).
        fast_hierar:
            Use hierarchical multi-resolution processing.
        fast_spatial:
            Use spatial acceleration structures.
        **kwargs:
            Accept case-variant spellings forwarded by Fire
            (e.g. ``LUT_size``, ``Fast_spatial``).
        """
        # Absorb case-variant Fire kwargs (e.g. --LUT_size, --Fast_spatial)
        for key, val in kwargs.items():
            kl = key.lower()
            if kl == "lut_size":
                lut_size = int(str(val))
            elif kl == "fast_hierar":
                fast_hierar = bool(val)
            elif kl == "fast_spatial":
                fast_spatial = bool(val)
            elif kl == "tile_size":
                tile_size = int(str(val))
            elif kl == "gpu":
                gpu = bool(val)

        process_image(
            input_image=input_image,
            attractors=attractors,
            output_image=output_image,
            luminance=luminance,
            saturation=saturation,
            hue=hue,
            verbose=verbose,
            tile_size=tile_size,
            gpu=gpu,
            lut_size=lut_size,
            fast_hierar=fast_hierar,
            fast_spatial=fast_spatial,
        )


def main() -> None:
    """Fire entry point registered as ``imgcolorshine`` console script."""
    fire.Fire(ImgColorShineCLI)


if __name__ == "__main__":
    main()
