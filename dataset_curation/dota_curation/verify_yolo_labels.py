"""
Verify and analyze YOLO format labels for the DOTA dataset.

This script provides statistics about converted YOLO labels including:
- Total number of objects per class
- Distribution of bounding box sizes
- Files with missing labels
- Normalization validation

Usage:
    python verify_yolo_labels.py --labels-dir datasets/dota/train/labels \\
                                 --images-dir datasets/dota/train/images \\
                                 --class-dict datasets/dota/class_dict.csv
"""

import os
import glob
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image


def load_class_mapping(class_dict_path):
    """
    Load class index to name mapping from CSV file.
    
    Args:
        class_dict_path (str): Path to class dictionary CSV file
        
    Returns:
        dict: Mapping from index to class name
    """
    idx_to_class = {}
    
    with open(class_dict_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_class[int(row['idx'])] = row['name']
    
    return idx_to_class


def verify_labels(labels_dir, images_dir, idx_to_class):
    """
    Verify and analyze YOLO format labels.
    
    Args:
        labels_dir (str): Path to YOLO format labels
        images_dir (str): Path to images
        idx_to_class (dict): Index to class name mapping
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_files': 0,
        'files_with_labels': 0,
        'files_without_labels': 0,
        'total_objects': 0,
        'class_distribution': defaultdict(int),
        'bbox_sizes': {'min': float('inf'), 'max': 0, 'avg': 0},
        'errors': []
    }
    
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    stats['total_files'] = len(label_files)
    
    total_area = 0
    bbox_count = 0
    
    print(f"Verifying {stats['total_files']} label files...")
    print("-" * 60)
    
    for idx, label_file in enumerate(label_files):
        base_name = os.path.basename(label_file)
        
        # Find corresponding image
        image_name = base_name.replace('.txt', '.png')
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            image_name = base_name.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            stats['errors'].append(f"Image not found for {base_name}")
            stats['files_without_labels'] += 1
            continue
        
        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            stats['errors'].append(f"Error reading image {image_name}: {e}")
            continue
        
        # Read and verify labels
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                stats['files_without_labels'] += 1
                continue
            
            stats['files_with_labels'] += 1
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                try:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Verify normalization (should be 0-1)
                    if not (0 <= x_center <= 1) or not (0 <= y_center <= 1):
                        stats['errors'].append(
                            f"{base_name}: Center coordinates out of bounds: "
                            f"({x_center}, {y_center})"
                        )
                    
                    if not (0 <= width <= 1) or not (0 <= height <= 1):
                        stats['errors'].append(
                            f"{base_name}: Dimensions out of bounds: ({width}, {height})"
                        )
                    
                    # Track statistics
                    stats['total_objects'] += 1
                    
                    if class_id in idx_to_class:
                        class_name = idx_to_class[class_id]
                        stats['class_distribution'][class_name] += 1
                    else:
                        stats['errors'].append(f"{base_name}: Unknown class ID {class_id}")
                    
                    bbox_area = width * height
                    stats['bbox_sizes']['min'] = min(stats['bbox_sizes']['min'], bbox_area)
                    stats['bbox_sizes']['max'] = max(stats['bbox_sizes']['max'], bbox_area)
                    total_area += bbox_area
                    bbox_count += 1
                
                except (ValueError, IndexError) as e:
                    stats['errors'].append(f"{base_name}: Could not parse line: {e}")
                    continue
        
        except Exception as e:
            stats['errors'].append(f"Error reading {base_name}: {e}")
            continue
        
        # Progress update
        if (idx + 1) % 500 == 0:
            print(f"  Verified {idx + 1}/{stats['total_files']} files...")
    
    # Calculate average bbox area
    if bbox_count > 0:
        stats['bbox_sizes']['avg'] = total_area / bbox_count
    
    return stats


def print_statistics(stats):
    """Print verification statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("YOLO Label Verification Report")
    print("=" * 60)
    
    print(f"\nFile Statistics:")
    print(f"  Total files:           {stats['total_files']}")
    print(f"  Files with labels:     {stats['files_with_labels']}")
    print(f"  Files without labels:  {stats['files_without_labels']}")
    
    print(f"\nObject Statistics:")
    print(f"  Total objects:         {stats['total_objects']}")
    avg_per_file = stats['total_objects'] / max(stats['files_with_labels'], 1)
    print(f"  Avg objects per file:  {avg_per_file:.2f}")
    
    print(f"\nBounding Box Sizes (normalized, 0-1):")
    print(f"  Minimum area:          {stats['bbox_sizes']['min']:.6f}")
    print(f"  Maximum area:          {stats['bbox_sizes']['max']:.6f}")
    print(f"  Average area:          {stats['bbox_sizes']['avg']:.6f}")
    
    print(f"\nClass Distribution:")
    if stats['class_distribution']:
        for class_name in sorted(stats['class_distribution'].keys()):
            count = stats['class_distribution'][class_name]
            percentage = (count / stats['total_objects']) * 100 if stats['total_objects'] > 0 else 0
            print(f"  {class_name:25s}: {count:6d} ({percentage:5.2f}%)")
    
    if stats['errors']:
        print(f"\n⚠ Warnings/Errors ({len(stats['errors'])} total):")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    else:
        print(f"\n✓ No errors found!")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Verify and analyze YOLO format labels for DOTA dataset"
    )
    parser.add_argument(
        '--labels-dir',
        type=str,
        default='datasets/dota/train/labels',
        help='Path to YOLO format label files'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='datasets/dota/train/images',
        help='Path to images directory'
    )
    parser.add_argument(
        '--class-dict',
        type=str,
        default='datasets/dota/class_dict.csv',
        help='Path to class dictionary CSV file'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.labels_dir):
        print(f"Error: Labels directory not found: {args.labels_dir}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    if not os.path.exists(args.class_dict):
        print(f"Error: Class dictionary not found: {args.class_dict}")
        return 1
    
    # Load class mapping
    print("=" * 60)
    print("YOLO Label Verification Tool")
    print("=" * 60)
    idx_to_class = load_class_mapping(args.class_dict)
    print(f"Loaded {len(idx_to_class)} classes\n")
    
    # Verify labels
    stats = verify_labels(args.labels_dir, args.images_dir, idx_to_class)
    
    # Print results
    print_statistics(stats)
    
    return 0 if not stats['errors'] else 1


if __name__ == "__main__":
    exit(main())
