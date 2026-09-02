"""Route-aware generation of geometry-preserving initial-SEM patch pools."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image

from .config import CONFIG, ROUTES


def _geometry_preserving_patch(image: Image.Image, rng: random.Random) -> Image.Image:
    """Random square crop, flips, 90-degree rotation, and 224-pixel resize."""
    image = image.convert("RGB")
    width, height = image.size
    shortest = min(width, height)
    crop_size = rng.randint(max(CONFIG.image_size_px, int(0.70 * shortest)), shortest)
    left = rng.randint(0, width - crop_size)
    top = rng.randint(0, height - crop_size)
    patch = image.crop((left, top, left + crop_size, top + crop_size))

    if rng.random() < 0.5:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    patch = patch.rotate(rng.choice((0, 90, 180, 270)), expand=False)
    return patch.resize(
        (CONFIG.image_size_px, CONFIG.image_size_px),
        Image.Resampling.LANCZOS,
    )


def generate_route_patch_pool(
    raw_root: str | Path,
    patch_root: str | Path,
    manifest_path: str | Path,
) -> None:
    """Create exactly 360 augmented patches for each of the four initial routes.

    Expected source layout: ``raw_root/<route>/*.tif`` with exactly 10 TIFF
    fields in each route folder. The original TIFF files are never overwritten.
    """
    raw_root = Path(raw_root)
    patch_root = Path(patch_root)
    manifest_path = Path(manifest_path)
    rng = random.Random(CONFIG.seed)
    records: list[dict[str, object]] = []

    for route in ROUTES:
        source_files = sorted((raw_root / route).glob("*.tif"))
        if len(source_files) != CONFIG.original_fields_per_route:
            raise ValueError(
                f"{route}: expected {CONFIG.original_fields_per_route} original TIFFs, "
                f"found {len(source_files)}"
            )
        route_output = patch_root / route
        route_output.mkdir(parents=True, exist_ok=True)

        for patch_index in range(CONFIG.augmented_patches_per_route):
            source = source_files[patch_index % len(source_files)]
            with Image.open(source) as image:
                patch = _geometry_preserving_patch(image, rng)
            output_name = f"{route.replace('-', '')}_patch_{patch_index:04d}.png"
            output_path = route_output / output_name
            patch.save(output_path, format="PNG")
            records.append(
                {
                    "route": route,
                    "source_file": source.name,
                    "patch_file": output_path.as_posix(),
                    "seed": CONFIG.seed,
                    "width_px": CONFIG.image_size_px,
                    "height_px": CONFIG.image_size_px,
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

