"""
Crop 20 tiles (512×512) from a .tif whole-slide image.

Uses zarr for memory-efficient random access — never loads the full image.
Saves a debug thumbnail + tissue mask so you can verify detection.

Usage:
  python crop_tiles.py image.tif
  python crop_tiles.py image.tif --output ./crops --num-tiles 20 --crop-size 512
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile
import zarr


def pick_thumb_level(series, target_long_edge=2048):
    """Pick a level whose longest edge is closest to target size."""
    best = 0
    best_diff = float("inf")
    for i, lv in enumerate(series.levels):
        long_edge = max(lv.shape[0], lv.shape[1])
        diff = abs(long_edge - target_long_edge)
        if diff < best_diff:
            best_diff = diff
            best = i
    return best


def otsu_threshold(gray: np.ndarray) -> int:
    """Simple Otsu threshold for uint8 grayscale."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0
    max_var, threshold = 0.0, 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold


def main():
    parser = argparse.ArgumentParser(description="Crop tiles from a .tif image")
    parser.add_argument("tif", help="Path to .tif file")
    parser.add_argument("--output", "-o", default="./crops", help="Output directory (default: ./crops)")
    parser.add_argument("--num-tiles", "-n", type=int, default=20, help="Number of tiles (default: 20)")
    parser.add_argument("--crop-size", "-s", type=int, default=512, help="Crop size in pixels (default: 512)")
    args = parser.parse_args()

    tif_path = args.tif
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop = args.crop_size
    num = args.num_tiles

    print(f"Opening: {tif_path}")
    tif = tifffile.TiffFile(tif_path)
    series = tif.series[0]
    num_levels = len(series.levels)
    shape0 = series.levels[0].shape
    H, W = shape0[0], shape0[1]
    print(f"  Size: {W}×{H}, levels: {num_levels}")

    # --- Step 1: read a reasonable thumbnail via zarr ---
    thumb_level = pick_thumb_level(series, target_long_edge=2048)
    thumb_shape = series.levels[thumb_level].shape
    print(f"  Thumbnail from level {thumb_level} ({thumb_shape[1]}×{thumb_shape[0]})")

    store_thumb = tif.aszarr(series=0, level=thumb_level)
    z_thumb = zarr.open(store_thumb, mode="r")
    thumb_data = np.array(z_thumb)

    if thumb_data.ndim == 2:
        thumb_rgb = np.stack([thumb_data] * 3, axis=-1)
    elif thumb_data.shape[2] == 4:
        thumb_rgb = thumb_data[:, :, :3]
    else:
        thumb_rgb = thumb_data[:, :, :3]

    gray = np.mean(thumb_rgb, axis=2).astype(np.uint8)
    th, tw = gray.shape

    # Save debug thumbnail
    Image.fromarray(thumb_rgb).save(out_dir / "_debug_thumbnail.jpg", quality=90)

    # --- Step 2: detect tissue using Otsu ---
    otsu_t = otsu_threshold(gray)
    tissue_mask = gray < otsu_t
    print(f"  Otsu threshold: {otsu_t}, tissue coverage: {tissue_mask.sum() / tissue_mask.size:.1%}")

    Image.fromarray((tissue_mask * 255).astype(np.uint8)).save(out_dir / "_debug_tissue_mask.jpg", quality=90)

    scale_x = W / tw
    scale_y = H / th

    # Grid cell size on thumbnail corresponding to crop_size at level 0
    grid_w = max(4, int(round(crop / scale_x)))
    grid_h = max(4, int(round(crop / scale_y)))
    print(f"  Grid cell on thumbnail: {grid_w}×{grid_h} px")

    candidates = []
    for row in range(0, th - grid_h + 1, grid_h):
        for col in range(0, tw - grid_w + 1, grid_w):
            patch = tissue_mask[row:row + grid_h, col:col + grid_w]
            ratio = patch.sum() / max(patch.size, 1)
            if ratio > 0.7:
                lv0_x = int(col * scale_x)
                lv0_y = int(row * scale_y)
                lv0_x = max(0, min(lv0_x, W - crop))
                lv0_y = max(0, min(lv0_y, H - crop))
                candidates.append((ratio, lv0_x, lv0_y))

    candidates.sort(key=lambda t: t[0], reverse=True)

    selected = []
    for ratio, cx, cy in candidates:
        overlap = any(abs(cx - sx) < crop and abs(cy - sy) < crop for sx, sy in selected)
        if not overlap:
            selected.append((cx, cy))
        if len(selected) >= num:
            break

    print(f"  Candidate regions: {len(candidates)}, selected: {len(selected)}")
    if len(selected) == 0:
        print("  ERROR: No tissue found! Check _debug_thumbnail.jpg and _debug_tissue_mask.jpg")
        return

    # --- Step 3: crop tiles from level 0 via zarr ---
    print(f"  Opening level 0 via zarr...")
    store0 = tif.aszarr(series=0, level=0)
    z0 = zarr.open(store0, mode="r")

    saved = 0
    print(f"Cropping {crop}×{crop} tiles...")
    for i, (x, y) in enumerate(selected):
        x = max(0, min(x, W - crop))
        y = max(0, min(y, H - crop))

        region = np.array(z0[y:y + crop, x:x + crop])
        if region.ndim == 2:
            region = np.stack([region] * 3, axis=-1)
        elif region.shape[2] == 4:
            region = region[:, :, :3]

        # Verify tile is not blank (mean < 240 for tissue)
        tile_mean = np.mean(region)
        if tile_mean > 240:
            print(f"  [skip] @ ({x}, {y}) — blank (mean={tile_mean:.0f})")
            continue

        tile = Image.fromarray(region)
        out_path = out_dir / f"tile_{saved:03d}.jpg"
        tile.save(out_path, "JPEG", quality=95)
        print(f"  [{saved+1:2d}/{num}] tile_{saved:03d}.jpg  @ ({x}, {y})  mean={tile_mean:.0f}")
        saved += 1

        if saved >= num:
            break

    print(f"\nDone! {saved} tiles saved to {out_dir}/")
    if saved < num:
        print(f"  (Only {saved}/{num} tiles had tissue. Try lowering --num-tiles)")


if __name__ == "__main__":
    main()
