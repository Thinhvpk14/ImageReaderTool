"""
Build PathoGen dataset from:
  - crops/          → image.jpg   (tissue tiles cropped from WSI)
  - images/         → condition.jpg (reference tissue patterns)
  - masks/          → mask.jpg     (binary masks)

Each condition PNG is matched with its mask PNG by filename.
A random crop tile is assigned as the image for each sample.

Usage:
  python build_dataset.py
  python build_dataset.py --images ./image_and_mask/images \
                          --masks  ./image_and_mask/masks  \
                          --crops  ./crops \
                          --output ./dataset \
                          --crop-size 512
"""

import argparse
import random
from pathlib import Path
from PIL import Image


def normalize_name(filename: str) -> str:
    """Strip spaces from filename stem for matching."""
    return Path(filename).stem.replace(" ", "")


def main():
    parser = argparse.ArgumentParser(description="Build PathoGen dataset")
    parser.add_argument("--images", default="./image_and_mask/images",
                        help="Condition images directory (PNG)")
    parser.add_argument("--masks", default="./image_and_mask/masks",
                        help="Mask images directory (PNG)")
    parser.add_argument("--crops", default="./crops",
                        help="Cropped tiles directory (JPG) — used as 'image'")
    parser.add_argument("--output", "-o", default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--crop-size", "-s", type=int, default=512,
                        help="Resize all outputs to this size (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    size = args.crop_size
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    crops_dir = Path(args.crops)
    out_dir = Path(args.output)

    # --- Load crop tiles (= image role) ---
    crop_files = sorted([
        f for f in crops_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("_")
    ])
    print(f"Crop tiles (image): {len(crop_files)} files from {crops_dir}")
    if not crop_files:
        print("ERROR: No crop tiles found!")
        return

    # --- Load condition images ---
    cond_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ])
    print(f"Condition images:   {len(cond_files)} files from {images_dir}")

    # --- Load mask images ---
    mask_files = sorted([
        f for f in masks_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ])
    print(f"Mask images:        {len(mask_files)} files from {masks_dir}")

    # --- Match condition ↔ mask by normalized name ---
    cond_map = {normalize_name(f.name): f for f in cond_files}
    mask_map = {normalize_name(f.name): f for f in mask_files}

    matched_keys = sorted(set(cond_map.keys()) & set(mask_map.keys()))
    print(f"\nMatched pairs:      {len(matched_keys)}")

    unmatched_cond = set(cond_map.keys()) - set(mask_map.keys())
    unmatched_mask = set(mask_map.keys()) - set(cond_map.keys())
    if unmatched_cond:
        print(f"  Conditions without mask: {len(unmatched_cond)} — {sorted(unmatched_cond)[:5]}...")
    if unmatched_mask:
        print(f"  Masks without condition: {len(unmatched_mask)} — {sorted(unmatched_mask)[:5]}...")

    if not matched_keys:
        print("ERROR: No matched pairs!")
        return

    # --- Build dataset ---
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = 0

    print(f"\nBuilding dataset ({size}×{size})...")
    for key in matched_keys:
        cond_path = cond_map[key]
        mask_path = mask_map[key]

        cond_img = Image.open(cond_path).convert("RGB").resize((size, size), Image.LANCZOS)
        mask_img = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)

        crop_path = random.choice(crop_files)
        crop_img = Image.open(crop_path).convert("RGB").resize((size, size), Image.LANCZOS)

        sample_dir = out_dir / f"sample_{sample_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        crop_img.save(sample_dir / "image.jpg", "JPEG", quality=95)
        mask_img.save(sample_dir / "mask.jpg", "JPEG", quality=95)
        cond_img.save(sample_dir / "condition.jpg", "JPEG", quality=95)

        print(f"  sample_{sample_idx:04d}  condition={cond_path.name}  mask={mask_path.name}  image={crop_path.name}")
        sample_idx += 1

    print(f"\nDone! {sample_idx} samples saved to {out_dir}/")
    print(f"\nEach sample:")
    print(f"  image.jpg     ← random crop tile ({size}×{size})")
    print(f"  mask.jpg      ← mask PNG ({size}×{size}, grayscale)")
    print(f"  condition.jpg ← condition PNG ({size}×{size}, RGB)")


if __name__ == "__main__":
    main()
