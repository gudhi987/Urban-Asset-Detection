"""
Analyze DOTA dataset structure and statistics.

This script provides detailed information about the DOTA dataset including:
- Image dimensions and statistics
- Label statistics per image
- Class distribution across dataset
- Dataset split information

Usage:
    python analyze_dota_dataset.py --dataset-dir datasets/dota \\
                                   --split train \\
                                   --class-dict datasets/dota/class_dict.csv
"""

import os
import glob
import csv
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter


def load_class_mapping(class_dict_path):
    """Load class index to name mapping from CSV file."""
    idx_to_class = {}
    with open(class_dict_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_class[int(row['idx'])] = row['name']
    return idx_to_class


def analyze_dataset(dataset_dir, split, idx_to_class):
    """
    Analyze DOTA dataset structure and statistics.
    
    Args:
        dataset_dir (str): Path to DOTA dataset root
        split (str): Dataset split ('train', 'valid', 'test')
        idx_to_class (dict): Class index to name mapping
        
    Returns:
        dict: Statistics dictionary
    """
    split_dir = os.path.join(dataset_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    
    stats = {
        'split': split,
        'total_images': 0,
        'total_labels': 0,
        'images_with_labels': 0,
        'images_without_labels': 0,
        'total_objects': 0,
        'image_dims': {'widths': [], 'heights': [], 'areas': []},
        'class_distribution': defaultdict(int),
        'objects_per_image': [],
        'errors': []
    }
    
    # Find all images
    if not os.path.exists(images_dir):
        stats['errors'].append(f"Images directory not found: {images_dir}")
        return None
    
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    stats['total_images'] = len(image_files)
    
    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory not found: {labels_dir}")
        labels_exist = False
    else:
        labels_exist = True
    
    print(f"Analyzing {stats['total_images']} images in {split} split...")
    print("-" * 60)
    
    for idx, image_file in enumerate(image_files):
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        
        # Get image dimensions
        try:
            img = Image.open(image_file)
            width, height = img.size
            stats['image_dims']['widths'].append(width)
            stats['image_dims']['heights'].append(height)
            stats['image_dims']['areas'].append(width * height)
        except Exception as e:
            stats['errors'].append(f"Error reading {os.path.basename(image_file)}: {e}")
            continue
        
        # Check for labels
        label_file = os.path.join(labels_dir, base_name + ".txt")
        
        if labels_exist and os.path.exists(label_file):
            stats['total_labels'] += 1
            
            # Count objects in this image
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                object_count = len([l for l in lines if l.strip()])
                stats['objects_per_image'].append(object_count)
                stats['total_objects'] += object_count
                
                if object_count > 0:
                    stats['images_with_labels'] += 1
                    
                    # Count classes
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(float(parts[0]))
                                if class_id in idx_to_class:
                                    class_name = idx_to_class[class_id]
                                    stats['class_distribution'][class_name] += 1
                            except (ValueError, IndexError):
                                pass
                
                else:
                    stats['images_without_labels'] += 1
            
            except Exception as e:
                stats['errors'].append(f"Error reading {os.path.basename(label_file)}: {e}")
        
        elif labels_exist:
            stats['images_without_labels'] += 1
        
        # Progress update
        if (idx + 1) % 500 == 0:
            print(f"  Analyzed {idx + 1}/{stats['total_images']} images...")
    
    return stats


def print_analysis(stats):
    """Print dataset analysis in a formatted way."""
    if stats is None:
        print("Error: Could not analyze dataset")
        return
    
    print("\n" + "=" * 70)
    print(f"DOTA Dataset Analysis - {stats['split'].upper()} Split")
    print("=" * 70)
    
    print(f"\nImage Statistics:")
    print(f"  Total images:                {stats['total_images']}")
    
    if stats['image_dims']['widths']:
        widths = stats['image_dims']['widths']
        heights = stats['image_dims']['heights']
        areas = stats['image_dims']['areas']
        
        print(f"\n  Image Dimensions:")
        print(f"    Width  - Min: {min(widths):5d}  Max: {max(widths):5d}  " + 
              f"Avg: {sum(widths)//len(widths):5d}")
        print(f"    Height - Min: {min(heights):5d}  Max: {max(heights):5d}  " +
              f"Avg: {sum(heights)//len(heights):5d}")
        print(f"    Area   - Min: {min(areas):10d}  Max: {max(areas):10d}  " +
              f"Avg: {sum(areas)//len(areas):10d}")
    
    print(f"\nLabel Statistics:")
    print(f"  Total label files:           {stats['total_labels']}")
    print(f"  Images with labels:          {stats['images_with_labels']}")
    print(f"  Images without labels:       {stats['images_without_labels']}")
    
    print(f"\nObject Statistics:")
    print(f"  Total objects:               {stats['total_objects']}")
    
    if stats['images_with_labels'] > 0:
        avg_objects = stats['total_objects'] / stats['images_with_labels']
        print(f"  Avg objects per labeled img: {avg_objects:.2f}")
    
    if stats['objects_per_image']:
        print(f"  Min objects per image:       {min(stats['objects_per_image'])}")
        print(f"  Max objects per image:       {max(stats['objects_per_image'])}")
        print(f"  Avg objects per image:       {sum(stats['objects_per_image'])/len(stats['objects_per_image']):.2f}")
    
    print(f"\nClass Distribution:")
    if stats['class_distribution']:
        # Sort by count descending
        sorted_classes = sorted(
            stats['class_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, count in sorted_classes:
            percentage = (count / stats['total_objects']) * 100 if stats['total_objects'] > 0 else 0
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"  {class_name:25s}: {count:6d} {bar:25s} ({percentage:5.2f}%)")
    
    if stats['errors']:
        print(f"\n⚠ Warnings/Errors ({len(stats['errors'])} total):")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    else:
        print(f"\n✓ No errors found!")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DOTA dataset structure and statistics"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets/dota',
        help='Path to DOTA dataset root directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'valid', 'test'],
        help='Dataset split to analyze'
    )
    parser.add_argument(
        '--class-dict',
        type=str,
        default='datasets/dota/class_dict.csv',
        help='Path to class dictionary CSV file'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    if not os.path.exists(args.class_dict):
        print(f"Error: Class dictionary not found: {args.class_dict}")
        return 1
    
    # Load class mapping
    print("=" * 70)
    print("DOTA Dataset Analysis Tool")
    print("=" * 70)
    idx_to_class = load_class_mapping(args.class_dict)
    print(f"Loaded {len(idx_to_class)} classes\n")
    
    # Analyze dataset
    stats = analyze_dataset(args.dataset_dir, args.split, idx_to_class)
    
    # Print results
    print_analysis(stats)
    
    return 0


if __name__ == "__main__":
    exit(main())
