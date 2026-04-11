"""
Extract unique classes from DOTA dataset labels.

This script scans all label files in the DOTA dataset and extracts unique classes,
then saves them to a CSV file with index-to-class mappings.

Usage:
    python extract_classes_dota.py --input-dir datasets/dota/train/trainset_reclabelTxt --output-file datasets/dota/class_dict.csv
"""

import os
import glob
import csv
import argparse
from pathlib import Path


def extract_unique_classes(label_dir):
    """
    Extract unique classes from all label files in a directory.
    
    Args:
        label_dir (str): Path to directory containing label files
        
    Returns:
        set: Set of unique class names found
    """
    classes = set()
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    print(f"Scanning {len(label_files)} label files for classes...")
    
    for idx, label_file in enumerate(label_files):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(label_files)} files...")
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:  # x1 y1 x2 y2 x3 y3 x4 y4 class difficult
                    class_name = parts[8]
                    classes.add(class_name)
    
    return classes


def save_class_dict(classes, output_file):
    """
    Save class names to CSV file with index mapping.
    
    Args:
        classes (set or list): Collection of unique class names
        output_file (str): Path to output CSV file
    """
    # Sort classes alphabetically for consistent indexing
    sorted_classes = sorted(list(classes))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'name'])
        writer.writeheader()
        for idx, class_name in enumerate(sorted_classes):
            writer.writerow({'idx': idx, 'name': class_name})
    
    print(f"\nClass dictionary saved to: {output_file}")
    print(f"Total classes: {len(sorted_classes)}")
    print("\nClass Mapping:")
    for idx, class_name in enumerate(sorted_classes):
        print(f"  {idx}: {class_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique classes from DOTA dataset and create class dictionary CSV"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='datasets/dota/train/trainset_reclabelTxt',
        help='Path to directory containing label files'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='datasets/dota/class_dict.csv',
        help='Path to output class dictionary CSV file'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Extract classes
    print("=" * 60)
    print("DOTA Class Extraction Tool")
    print("=" * 60)
    classes = extract_unique_classes(args.input_dir)
    
    # Save to file
    save_class_dict(classes, args.output_file)
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
