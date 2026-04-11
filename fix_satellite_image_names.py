#!/usr/bin/env python3
"""
Fix satellite image filenames to match their label files.
YOLO matches images to labels by filename stem (without extension).
Current issue: Images are named with _sat suffix but labels are not.

This script renames all _sat.jpg files to remove the suffix.
"""

import os
from pathlib import Path


def fix_satellite_filenames(dataset_root, splits=['train', 'val', 'test']):
    """Rename satellite images to remove _sat suffix."""
    
    dataset_root = Path(dataset_root)
    total_renamed = 0
    
    for split in splits:
        images_dir = dataset_root / 'yolo_dataset' / 'images' / split
        labels_dir = dataset_root / 'yolo_dataset' / 'labels' / split
        
        if not images_dir.exists():
            print(f"⚠️  {split} images directory not found: {images_dir}")
            continue
        
        # Find all satellite images
        sat_images = list(images_dir.glob('*_sat.jpg'))
        
        if not sat_images:
            print(f"✓ {split}: No satellite images found")
            continue
        
        print(f"\n{split.upper()} split:")
        print(f"  Found {len(sat_images)} satellite images")
        
        # Check which ones will be renamed
        to_rename = []
        for sat_img in sat_images:
            # Get the stem without _sat
            new_stem = sat_img.stem.replace('_sat', '')
            new_name = f"{new_stem}.jpg"
            
            # Check if label exists
            label_file = labels_dir / f"{new_stem}.txt"
            if label_file.exists():
                to_rename.append((sat_img, sat_img.parent / new_name, new_stem))
            else:
                print(f"  ⚠️  Missing label for {sat_img.name}: {new_stem}.txt")
        
        # Rename the files
        if to_rename:
            print(f"  Renaming {len(to_rename)} images...")
            for old_path, new_path, stem in to_rename:
                old_path.rename(new_path)
                print(f"    ✓ {old_path.name} → {new_path.name}")
                total_renamed += 1
        
        # Verify all images now have labels
        print(f"  Validating...")
        orphaned = 0
        for img_path in images_dir.glob('*.jpg'):
            stem = img_path.stem
            if not (labels_dir / f"{stem}.txt").exists():
                print(f"    ⚠️  No label for {img_path.name}")
                orphaned += 1
        
        if orphaned == 0:
            print(f"  ✅ All {len(list(images_dir.glob('*.jpg')))} images have labels")
        else:
            print(f"  ⚠️  {orphaned} images missing labels")
    
    print(f"\n{'='*50}")
    print(f"✅ Total images renamed: {total_renamed}")
    print(f"Dataset is now ready for training with proper filename matching!")


if __name__ == "__main__":
    dataset_root = Path(__file__).parent
    print("Fixing satellite image filenames for YOLO matching...")
    print(f"Dataset root: {dataset_root}\n")
    
    fix_satellite_filenames(dataset_root)
