"""
Validate converted unified dataset labels.

Validates:
- Label file format and structure
- Coordinate normalization (0-1 range)
- Class distribution
- Image-label correspondence
- Completeness statistics

Usage:
    python validate_unified_dataset.py --unified-dir unified_dataset
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image


def validate_dataset(unified_dir):
    """Validate the unified dataset"""
    unified_dir = Path(unified_dir)
    
    stats = {
        'total_datasets': 0,
        'total_images': 0,
        'total_labels': 0,
        'images_with_labels': 0,
        'total_objects': 0,
        'class_distribution': defaultdict(int),
        'errors': [],
        'dataset_stats': {}
    }
    
    print(f"\n{'=' * 70}")
    print("Validating Unified Dataset")
    print(f"{'=' * 70}")
    
    # Check each dataset folder
    for dataset_dir in sorted(unified_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        
        if not images_dir.exists():
            print(f"⚠ Skipping {dataset_dir.name}: images dir not found")
            continue
        
        print(f"\nValidating {dataset_dir.name}...")
        print("-" * 70)
        
        dataset_stats = {
            'images': 0,
            'labels': 0,
            'images_with_labels': 0,
            'total_objects': 0,
            'class_distribution': defaultdict(int),
            'errors': 0
        }
        
        # Find all images
        image_files = [f for f in images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]
        dataset_stats['images'] = len(image_files)
        stats['total_images'] += len(image_files)
        stats['total_datasets'] += 1
        
        # Validate each image and its labels
        for idx, image_path in enumerate(image_files):
            base_name = image_path.stem
            # Strip _sat suffix if present (from satellite imagery naming)
            if base_name.endswith('_sat'):
                base_name = base_name[:-4]
            label_path = labels_dir / f"{base_name}.txt"
            
            if not label_path.exists():
                stats['errors'].append(f"{dataset_dir.name}/{base_name}: No label file")
                continue
            
            dataset_stats['labels'] += 1
            stats['total_labels'] += 1
            
            # Read and validate labels
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                dataset_stats['images_with_labels'] += 1
                stats['images_with_labels'] += 1
                
                # Validate each label line
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    
                    if len(parts) < 5:
                        stats['errors'].append(
                            f"{dataset_dir.name}/{base_name}: Line {line_idx} - invalid format"
                        )
                        dataset_stats['errors'] += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= x_center <= 1):
                            stats['errors'].append(
                                f"{dataset_dir.name}/{base_name}: x_center out of range: {x_center}"
                            )
                            dataset_stats['errors'] += 1
                            continue
                        
                        if not (0 <= y_center <= 1):
                            stats['errors'].append(
                                f"{dataset_dir.name}/{base_name}: y_center out of range: {y_center}"
                            )
                            dataset_stats['errors'] += 1
                            continue
                        
                        if not (0 <= width <= 1):
                            stats['errors'].append(
                                f"{dataset_dir.name}/{base_name}: width out of range: {width}"
                            )
                            dataset_stats['errors'] += 1
                            continue
                        
                        if not (0 <= height <= 1):
                            stats['errors'].append(
                                f"{dataset_dir.name}/{base_name}: height out of range: {height}"
                            )
                            dataset_stats['errors'] += 1
                            continue
                        
                        # Validate class ID
                        if not (0 <= class_id <= 4):
                            stats['errors'].append(
                                f"{dataset_dir.name}/{base_name}: Invalid class ID: {class_id}"
                            )
                            dataset_stats['errors'] += 1
                            continue
                        
                        dataset_stats['total_objects'] += 1
                        stats['total_objects'] += 1
                        dataset_stats['class_distribution'][class_id] += 1
                        stats['class_distribution'][class_id] += 1
                    
                    except (ValueError, IndexError) as e:
                        stats['errors'].append(
                            f"{dataset_dir.name}/{base_name}: Line {line_idx} - parse error: {e}"
                        )
                        dataset_stats['errors'] += 1
            
            except Exception as e:
                stats['errors'].append(
                    f"{dataset_dir.name}/{base_name}: {e}"
                )
                dataset_stats['errors'] += 1
            
            if (idx + 1) % 200 == 0:
                print(f"  Validated {idx + 1}/{len(image_files)} images...")
        
        # Print dataset summary
        print(f"\n  Summary for {dataset_dir.name}:")
        print(f"    Images:                {dataset_stats['images']}")
        print(f"    Labels:                {dataset_stats['labels']}")
        print(f"    Images with labels:    {dataset_stats['images_with_labels']}")
        print(f"    Total objects:         {dataset_stats['total_objects']}")
        print(f"    Validation errors:     {dataset_stats['errors']}")
        print(f"    Class distribution:")
        for class_id in sorted(dataset_stats['class_distribution'].keys()):
            count = dataset_stats['class_distribution'][class_id]
            print(f"      Class {class_id}: {count}")
        
        stats['dataset_stats'][dataset_dir.name] = dataset_stats
    
    return stats


def print_summary(stats):
    """Print validation summary"""
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total datasets:           {stats['total_datasets']}")
    print(f"  Total images:             {stats['total_images']}")
    print(f"  Total labels:             {stats['total_labels']}")
    print(f"  Images with labels:       {stats['images_with_labels']}")
    print(f"  Total objects:            {stats['total_objects']}")
    
    print(f"\nClass Distribution:")
    class_names = {
        0: 'building',
        1: 'road',
        2: 'water_body',
        3: 'vegetation',
        4: 'vehicle'
    }
    
    for class_id in sorted(stats['class_distribution'].keys()):
        count = stats['class_distribution'][class_id]
        name = class_names[class_id]
        pct = (count / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
        print(f"  {name:15s} (id={class_id}): {count:6d} ({pct:5.1f}%)")
    
    if stats['errors']:
        print(f"\n⚠ Validation Issues ({len(stats['errors'])} total):")
        for error in stats['errors'][:20]:
            print(f"  - {error}")
        if len(stats['errors']) > 20:
            print(f"  ... and {len(stats['errors']) - 20} more")
    else:
        print(f"\n✓ No validation errors!")
    
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate unified dataset"
    )
    parser.add_argument(
        '--unified-dir',
        type=str,
        default='unified_dataset',
        help='Path to unified dataset directory'
    )
    
    args = parser.parse_args()
    
    unified_dir = Path(args.unified_dir)
    
    if not unified_dir.exists():
        print(f"✗ Error: Unified dataset directory not found: {unified_dir}")
        return 1
    
    # Validate
    stats = validate_dataset(unified_dir)
    
    # Print summary
    print_summary(stats)
    
    return 0 if not stats['errors'] else 1


if __name__ == "__main__":
    exit(main())
