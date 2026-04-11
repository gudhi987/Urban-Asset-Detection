"""
Verify and visualize the train/val/test split of semantic buildings dataset.

This script provides statistics about the dataset split showing:
- Number of images in each split
- Object counts per split
- Label statistics

Note: All labels remain in train/labels. This script reads labels for val/test
images from train/labels using image base names.

Usage:
    python verify_split_semantic_buildings.py \
        --dataset-dir "../../datasets/semantic_buildings_in_aerial_imagery"
"""

import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_split(dataset_dir, split_name):
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
                    stats['total_objects'] += len(lines)
    
    if stats['images_with_labels'] > 0:
        stats['avg_objects_per_image'] = stats['total_objects'] / stats['images_with_labels']
    
    return stats


def print_split_statistics(all_stats):
    """Print formatted split statistics."""
    print("\n" + "=" * 80)
    print("Semantic Buildings Dataset Split Verification")
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
    
    print("=" * 80)
    print(f"(All labels are stored in train/labels)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and analyze semantic buildings dataset split"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='../../datasets/semantic_buildings_in_aerial_imagery',
        help='Path to dataset root directory'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    # Validate directory
    if not dataset_dir.exists():
        print(f"✗ Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    print("=" * 80)
    print("Semantic Buildings Split Verification Tool")
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")
    
    # Analyze each split
    print("\nAnalyzing splits...")
    print("-" * 80)
    
    all_stats = {}
    for split_name in ['train', 'val', 'test']:
        split_dir = dataset_dir / split_name
        if split_dir.exists():
            print(f"  Analyzing {split_name.upper()}...", end=" ")
            stats = analyze_split(dataset_dir, split_name)
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
