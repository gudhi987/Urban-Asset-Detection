"""
Convert DOTA dataset from OBB (Oriented Bounding Box) format to YOLO format.

This script converts labeled data from DOTA's oriented bounding box format 
(8 corner coordinates) to YOLO's normalized center-based format.

Input format:  x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
Output format: class_id x_center y_center width height (all normalized 0-1)

Usage:
    python convert_obb_to_yolo.py --input-labels datasets/dota/train/trainset_reclabelTxt \\
                                   --output-labels datasets/dota/train/labels \\
                                   --images-dir datasets/dota/train/images \\
                                   --class-dict datasets/dota/class_dict.csv
"""

import os
import glob
import csv
import argparse
from pathlib import Path
from PIL import Image


def load_class_mapping(class_dict_path):
    """
    Load class name to index mapping from CSV file.
    
    Args:
        class_dict_path (str): Path to class dictionary CSV file
        
    Returns:
        dict: Mapping from class name to index
    """
    class_to_idx = {}
    
    with open(class_dict_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_to_idx[row['name']] = int(row['idx'])
    
    print(f"Loaded {len(class_to_idx)} classes from {class_dict_path}")
    return class_to_idx


def convert_obb_to_yolo(x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
    """
    Convert OBB (Oriented Bounding Box) coordinates to YOLO format.
    
    OBB format: 4 corner points in clockwise order
    YOLO format: normalized center coordinates and dimensions (0-1)
    
    Args:
        x1-y4: OBB corner coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        tuple: (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    # Get bounding box from OBB corners
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to 0-1 range
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clamp values to 0-1 range for safety
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_dataset(input_labels_dir, output_labels_dir, images_dir, class_to_idx):
    """
    Convert all label files from OBB to YOLO format.
    
    Args:
        input_labels_dir (str): Path to input OBB format labels
        output_labels_dir (str): Path to output YOLO format labels
        images_dir (str): Path to images directory
        class_to_idx (dict): Class name to index mapping
        
    Returns:
        tuple: (total_files, successfully_converted, errors)
    """
    # Create output directory
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Get all input label files
    label_files = sorted(glob.glob(os.path.join(input_labels_dir, "*.txt")))
    total_files = len(label_files)
    converted_count = 0
    error_count = 0
    
    print(f"\nProcessing {total_files} label files...")
    print("-" * 60)
    
    for idx, label_file in enumerate(label_files):
        base_name = os.path.basename(label_file)
        
        # Find corresponding image
        image_name = base_name.replace('.txt', '.png')
        image_path = os.path.join(images_dir, image_name)
        
        # Try jpg if png not found
        if not os.path.exists(image_path):
            image_name = base_name.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"[{idx+1}/{total_files}] ⚠ Image not found for {base_name}")
            error_count += 1
            continue
        
        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"[{idx+1}/{total_files}] ✗ Error reading image {image_name}: {e}")
            error_count += 1
            continue
        
        # Read and convert labels
        yolo_boxes = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    
                    # Parse OBB coordinates
                    try:
                        x1, y1 = float(parts[0]), float(parts[1])
                        x2, y2 = float(parts[2]), float(parts[3])
                        x3, y3 = float(parts[4]), float(parts[5])
                        x4, y4 = float(parts[6]), float(parts[7])
                        class_name = parts[8]
                        difficult = int(parts[9]) if len(parts) > 9 else 0
                    except (ValueError, IndexError) as e:
                        print(f"  Warning: Could not parse line in {base_name}: {e}")
                        continue
                    
                    # Check if class exists
                    if class_name not in class_to_idx:
                        print(f"  Warning: Unknown class '{class_name}' in {base_name}")
                        continue
                    
                    class_idx = class_to_idx[class_name]
                    
                    # Convert to YOLO format
                    x_center, y_center, width, height = convert_obb_to_yolo(
                        x1, y1, x2, y2, x3, y3, x4, y4,
                        img_width, img_height
                    )
                    
                    yolo_boxes.append(
                        f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )
        
        except Exception as e:
            print(f"[{idx+1}/{total_files}] ✗ Error processing {base_name}: {e}")
            error_count += 1
            continue
        
        # Write YOLO format labels
        output_file = os.path.join(output_labels_dir, base_name)
        try:
            with open(output_file, 'w') as f:
                for box in yolo_boxes:
                    f.write(box + '\n')
            converted_count += 1
            
            # Progress update
            if (idx + 1) % 200 == 0:
                print(f"[{idx+1}/{total_files}] ✓ Converted {converted_count} files...")
        
        except Exception as e:
            print(f"[{idx+1}/{total_files}] ✗ Error writing {output_file}: {e}")
            error_count += 1
    
    return total_files, converted_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert DOTA dataset from OBB to YOLO format"
    )
    parser.add_argument(
        '--input-labels',
        type=str,
        default='datasets/dota/train/trainset_reclabelTxt',
        help='Path to input OBB format label files'
    )
    parser.add_argument(
        '--output-labels',
        type=str,
        default='datasets/dota/train/labels',
        help='Path to output YOLO format label files'
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
    if not os.path.exists(args.input_labels):
        print(f"Error: Input labels directory not found: {args.input_labels}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    if not os.path.exists(args.class_dict):
        print(f"Error: Class dictionary not found: {args.class_dict}")
        return 1
    
    # Load class mapping
    print("=" * 60)
    print("DOTA OBB to YOLO Conversion Tool")
    print("=" * 60)
    class_to_idx = load_class_mapping(args.class_dict)
    
    # Convert dataset
    total, converted, errors = convert_dataset(
        args.input_labels,
        args.output_labels,
        args.images_dir,
        class_to_idx
    )
    
    # Print summary
    print("-" * 60)
    print(f"\nConversion Summary:")
    print(f"  Total files:       {total}")
    print(f"  Successfully converted: {converted}")
    print(f"  Errors:            {errors}")
    print(f"  Output directory:  {args.output_labels}")
    print("=" * 60)
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
