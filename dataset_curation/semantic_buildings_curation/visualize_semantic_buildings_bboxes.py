"""
Visualization script for semantic buildings dataset - displays sample images with bounding boxes
"""
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import csv
import argparse


def load_class_names(class_dict_path):
    """Load class names from CSV file"""
    class_names = {}
    with open(class_dict_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names[int(row['idx'])] = row['name']
    return class_names


def visualize_semantic_buildings_samples_with_bboxes(dataset_dir, num_samples=4, split='train'):
    """
    Visualize semantic buildings dataset samples with their bounding boxes
    
    Args:
        dataset_dir: Path to the semantic buildings dataset directory
        num_samples: Number of samples to visualize
        split: 'train', 'val', or 'test'
    """
    split_dir = os.path.join(dataset_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(dataset_dir, "train", "labels")  # All labels in train/labels
    class_dict_path = os.path.join(dataset_dir, "class_dict.csv")
    
    # Load class names
    if os.path.exists(class_dict_path):
        class_names = load_class_names(class_dict_path)
    else:
        print(f"Warning: class_dict.csv not found at {class_dict_path}")
        class_names = {}
    
    # Get list of image files
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return
    
    # Randomly sample images
    image_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    rows = len(image_files)
    cols = 2  # image, bbox image
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    
    # Handle case of single sample
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for row_idx, image_file in enumerate(image_files):
        base_name = os.path.splitext(image_file)[0]
        
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        # Load image
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            axes[row_idx, 0].text(10, 10, f"Error loading image", color="red")
            axes[row_idx, 1].text(10, 10, f"Error loading image", color="red")
            continue
        
        h, w = image.shape[:2]
        
        # --- Column 1: Original Image ---
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"Original Image\n{image_file}\nSize: {w}x{h}")
        axes[row_idx, 0].axis("off")
        
        # --- Column 2: Image + BBoxes ---
        axes[row_idx, 1].imshow(image)
        axes[row_idx, 1].set_title(f"Bounding Boxes\n{image_file}")
        axes[row_idx, 1].axis("off")
        
        # Load and draw bounding boxes
        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                bbox_count = 0
                colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'orange']
                
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                        
                        # Convert YOLO (normalized) → pixel coords
                        x_center_px = x_center * w
                        y_center_px = y_center * h
                        bw_px = bw * w
                        bh_px = bh * h
                        
                        x_min = x_center_px - bw_px / 2
                        y_min = y_center_px - bh_px / 2
                        
                        # Draw rectangle
                        color = colors[bbox_count % len(colors)]
                        rect = patches.Rectangle(
                            (x_min, y_min),
                            bw_px,
                            bh_px,
                            linewidth=2,
                            edgecolor=color,
                            facecolor='none',
                            alpha=0.7
                        )
                        axes[row_idx, 1].add_patch(rect)
                        
                        # Add class label
                        class_name = class_names.get(class_id, f"Class {class_id}")
                        axes[row_idx, 1].text(
                            x_min,
                            y_min - 8,
                            class_name,
                            color=color,
                            fontsize=9,
                            fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.6, edgecolor=color, pad=2)
                        )
                        
                        bbox_count += 1
                    
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line {line_idx} in {base_name}.txt: {e}")
                        continue
                
                if bbox_count > 0:
                    axes[row_idx, 1].text(
                        5, 20,
                        f"Total boxes: {bbox_count}",
                        color='white',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.7)
                    )
            
            except Exception as e:
                print(f"Error reading labels for {base_name}: {e}")
                axes[row_idx, 1].text(10, 10, f"Error reading labels: {e}", color="red", fontsize=8)
        else:
            axes[row_idx, 1].text(10, 10, "No labels found", color="red", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize semantic buildings dataset with bounding boxes"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='../../datasets/semantic_buildings_in_aerial_imagery',
        help='Path to semantic buildings dataset directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=4,
        help='Number of samples to visualize (default: 4)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to visualize (default: train)'
    )
    
    args = parser.parse_args()
    
    print(f"Visualizing semantic buildings dataset from {args.dataset_dir}")
    print("=" * 60)
    
    visualize_semantic_buildings_samples_with_bboxes(
        dataset_dir=args.dataset_dir,
        num_samples=args.num_samples,
        split=args.split
    )
