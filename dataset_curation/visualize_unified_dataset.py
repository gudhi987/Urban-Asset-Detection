"""
Visualize converted unified dataset with bounding boxes.

Shows sample images from each dataset with overlaid bounding boxes
in the unified format.

Usage:
    python visualize_unified_dataset.py --unified-dir unified_dataset --num-samples 2
"""

import os
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from PIL import Image


def visualize_unified_dataset(unified_dir, num_samples=2):
    """Visualize converted unified dataset"""
    unified_dir = Path(unified_dir)
    
    class_names = {
        0: 'building',
        1: 'road',
        2: 'water_body',
        3: 'vegetation',
        4: 'vehicle'
    }
    
    colors = ['red', 'green', 'blue', 'yellow', 'cyan']
    
    print(f"\n{'=' * 70}")
    print("Visualizing Unified Dataset")
    print(f"{'=' * 70}")
    
    # Get all datasets
    dataset_dirs = sorted([d for d in unified_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')])
    
    total_rows = len(dataset_dirs)
    total_cols = 2  # Original and with bboxes
    
    fig, axes = plt.subplots(total_rows, total_cols, figsize=(14, 6 * total_rows))
    
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for row_idx, dataset_dir in enumerate(dataset_dirs):
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        
        if not images_dir.exists():
            continue
        
        print(f"\nVisualizing {dataset_dir.name}...")
        
        # Get random image from this dataset
        image_files = [f for f in images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]
        
        if not image_files:
            axes[row_idx, 0].text(0.5, 0.5, f"No images in {dataset_dir.name}", 
                                 ha='center', va='center')
            axes[row_idx, 1].text(0.5, 0.5, f"No images in {dataset_dir.name}", 
                                 ha='center', va='center')
            continue
        
        image_path = random.choice(image_files)
        base_name = image_path.stem
        # Strip _sat suffix if present (from satellite imagery naming)
        if base_name.endswith('_sat'):
            base_name = base_name[:-4]
        label_path = labels_dir / f"{base_name}.txt"
        
        # Load image
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
        
        h, w = image.shape[:2]
        
        # Column 1: Original image
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(f"{dataset_dir.name}\nOriginal\n{image_path.name}")
        axes[row_idx, 0].axis("off")
        
        # Column 2: Image with bboxes
        axes[row_idx, 1].imshow(image)
        axes[row_idx, 1].set_title(f"{dataset_dir.name}\nWith Unified Labels\n{image_path.name}")
        axes[row_idx, 1].axis("off")
        
        # Draw bboxes
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                bbox_count = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        b_width = float(parts[3])
                        b_height = float(parts[4])
                        
                        # Convert to pixel coordinates
                        x_min = (x_center - b_width / 2) * w
                        y_min = (y_center - b_height / 2) * h
                        b_w_px = b_width * w
                        b_h_px = b_height * h
                        
                        # Draw rectangle
                        color = colors[class_id % len(colors)]
                        rect = patches.Rectangle(
                            (x_min, y_min),
                            b_w_px,
                            b_h_px,
                            linewidth=2,
                            edgecolor=color,
                            facecolor='none',
                            alpha=0.8
                        )
                        axes[row_idx, 1].add_patch(rect)
                        
                        # Add class label
                        class_name = class_names[class_id]
                        axes[row_idx, 1].text(
                            x_min,
                            y_min - 8,
                            f"{class_name}",
                            color=color,
                            fontsize=8,
                            fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7, pad=1)
                        )
                        
                        bbox_count += 1
                    
                    except (ValueError, IndexError):
                        continue
                
                if bbox_count > 0:
                    axes[row_idx, 1].text(
                        5, 15,
                        f"{bbox_count} objects",
                        color='white',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.7)
                    )
            
            except Exception as e:
                print(f"  Error reading labels: {e}")
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize unified dataset conversions"
    )
    parser.add_argument(
        '--unified-dir',
        type=str,
        default='unified_dataset',
        help='Path to unified dataset directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of samples per dataset to show'
    )
    
    args = parser.parse_args()
    
    unified_dir = Path(args.unified_dir)
    
    if not unified_dir.exists():
        print(f"✗ Error: Unified dataset directory not found: {unified_dir}")
        return 1
    
    visualize_unified_dataset(unified_dir, args.num_samples)
    
    return 0


if __name__ == "__main__":
    exit(main())
