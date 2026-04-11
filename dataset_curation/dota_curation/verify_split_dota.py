"""
Verify and visualize the train/val/test split of DOTA dataset.

This script provides statistics about the dataset split showing:
- Number of images in each split
- Class distribution across splits
- Object counts per image

Note: All labels remain in train/labels. This script reads labels for val/test
images from train/labels using image base names.

Usage:
    python verify_split_dota.py --dataset-dir datasets/dota
"""

import os
import glob
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def load_class_mapping(class_dict_path):
    """Load class index to name mapping."""
    idx_to_class = {}
    if os.path.exists(class_dict_path):
        with open(class_dict_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx_to_class[int(row['idx'])] = row['name']
    return idx_to_class


def analyze_split(dataset_dir, split_name, idx_to_class):
    """
    Analyze a specific split (train, val, or test).
    
    For train split: reads labels from train/labels
    For val/test splits: reads labels from train/labels (same location as train)
    """
    split_dir = Path(dataset_dir) / split_name
    images_dir = split_dir / "images"
    
    # Labels are always in train/labels directory
    labels_dir = Path(dataset_dir) / "train" / "labels"
    
    stats = {
        'split': split_name,
        'total_images': 0,
        'images_with_labels': 0,
        'total_objects': 0,
        'class_distribution': defaultdict(int),
        'avg_objects_per_image': 0
    }
    
    # Count images
    image_files = {}
    if images_dir.exists():
        image_paths = (
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.jpeg"))
        )
        stats['total_images'] = len(image_paths)
        # Store mapping of basename to path
        for img_path in image_paths:
            image_files[img_path.stem] = img_path
    
    # Count and analyze labels (from train/labels for all splits)
    if labels_dir.exists() and image_files:
        for image_basename in image_files.keys():
            label_file = labels_dir / f"{image_basename}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    stats['images_with_labels'] += 1
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(float(parts[0]))
                                if class_id in idx_to_class:
                                    class_name = idx_to_class[class_id]
                                    stats['class_distribution'][class_name] += 1
                                stats['total_objects'] += 1
                            except (ValueError, IndexError):
                                pass
    
    if stats['images_with_labels'] > 0:
        stats['avg_objects_per_image'] = stats['total_objects'] / stats['images_with_labels']
    
    return stats


def print_split_statistics(all_stats):
    """Print formatted split statistics."""
    print("\n" + "=" * 80)
    print("DOTA DATASET SPLIT VERIFICATION")
    print("=" * 80)
    
    # Summary table
    print("\nSplit Summary:")
    print("-" * 80)
    print(f"{'Split':<10} {'Images':>10} {'With Labels':>15} {'Objects':>10}")
    print("-" * 80)
    
    total_images = 0
    total_objects = 0
    
    for stats in all_stats.values():
        split = stats['split']
        images = stats['total_images']
        with_labels = stats['images_with_labels']
        objects = stats['total_objects']
        
        total_images += images
        total_objects += objects
        
        print(f"{split:<10} {images:>10} {with_labels:>15} {objects:>10}")
    
    print("-" * 80)
    print(f"{'TOTAL':<10} {total_images:>10} {sum(s['images_with_labels'] for s in all_stats.values()):>15} {total_objects:>10}")
    
    # Percentages
    print("\nSplit Percentages (by image count):")
    print("-" * 80)
    for stats in all_stats.values():
        split = stats['split']
        images = stats['total_images']
        pct = (images / total_images * 100) if total_images > 0 else 0
        bar_length = int(pct / 2)
        bar = "█" * bar_length
        print(f"{split:<10} {images:>5} ({pct:>5.1f}%) {bar}")
    
    print(f"\n(Note: All labels are stored in train/labels)")
    
    # Class distribution
    print("\nClass Distribution Across Splits:")
    print("-" * 80)
    
    # Collect all classes
    all_classes = set()
    for stats in all_stats.values():
        all_classes.update(stats['class_distribution'].keys())
    
    all_classes = sorted(all_classes)
    
    if all_classes:
        # Header
        print(f"{'Class':<25}", end="")
        for stats in all_stats.values():
            print(f" {stats['split'].upper():>12}", end="")
        print(f" {'TOTAL':>12}")
        print("-" * 80)
        
        # Class counts
        for class_name in all_classes:
            print(f"{class_name:<25}", end="")
            total_for_class = 0
            for stats in all_stats.values():
                count = stats['class_distribution'].get(class_name, 0)
                print(f" {count:>12}", end="")
                total_for_class += count
            print(f" {total_for_class:>12}")
        
        print("-" * 80)
        print(f"{'TOTAL':<25}", end="")
        for stats in all_stats.values():
            print(f" {stats['total_objects']:>12}", end="")
        print(f" {total_objects:>12}")
    
    # Detailed stats
    print("\n" + "=" * 80)
    print("Detailed Split Statistics:")
    print("=" * 80)
    
    for stats in all_stats.values():
        split = stats['split']
        print(f"\n{split.upper()}:")
        print(f"  Images:                    {stats['total_images']}")
        print(f"  Images with labels:        {stats['images_with_labels']}")
        print(f"  Total objects:             {stats['total_objects']}")
        print(f"  Avg objects per image:     {stats['avg_objects_per_image']:.2f}")
        
        if stats['class_distribution']:
            sorted_classes = sorted(
                stats['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            print(f"  Top 5 classes:")
            for class_name, count in sorted_classes[:5]:
                pct = (count / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
                print(f"    - {class_name:<20}: {count:>5} ({pct:>5.1f}%)")
    
    print("=" * 80)
    print(f"(All labels are stored in train/labels)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and analyze DOTA dataset split"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets/dota',
        help='Path to DOTA dataset root directory'
    )
    parser.add_argument(
        '--class-dict',
        type=str,
        default='datasets/dota/class_dict.csv',
        help='Path to class dictionary CSV file'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    # Validate directory
    if not dataset_dir.exists():
        print(f"✗ Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    print("=" * 80)
    print("DOTA Split Verification Tool")
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")
    
    # Load class mapping
    idx_to_class = load_class_mapping(args.class_dict)
    if idx_to_class:
        print(f"Loaded {len(idx_to_class)} classes from class dictionary")
    else:
        print("Warning: No class dictionary found")
    
    # Analyze each split
    print("\nAnalyzing splits...")
    print("-" * 80)
    
    all_stats = {}
    for split_name in ['train', 'val', 'test']:
        split_dir = dataset_dir / split_name
        if split_dir.exists():
            print(f"  Analyzing {split_name.upper()}...", end=" ")
            stats = analyze_split(dataset_dir, split_name, idx_to_class)
            all_stats[split_name] = stats
            print(f"✓ ({stats['total_images']} images)")
        else:
            print(f"  {split_name.upper()} split not found")
    
    # Print statistics
    if all_stats:
        print_split_statistics(all_stats)
    
    print("\n✓ Verification complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
