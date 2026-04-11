"""
Convert unified dataset to YOLO format with train/val/test splits.

Structure:
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

Splits: 80% train, 10% val, 10% test
"""

import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict


def create_yolo_dataset(unified_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Convert unified dataset to YOLO format with splits"""
    
    unified_dir = Path(unified_dir)
    output_dir = Path(output_dir)
    
    # Create output structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("Converting Unified Dataset to YOLO Format")
    print(f"{'=' * 70}")
    print(f"Input: {unified_dir}")
    print(f"Output: {output_dir}")
    print(f"Train/Val/Test split: {train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f}")
    
    # Collect all image-label pairs from each dataset
    all_pairs = []
    dataset_stats = defaultdict(int)
    
    # Get all datasets
    dataset_dirs = sorted([d for d in unified_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')])
    
    for dataset_dir in dataset_dirs:
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        
        if not images_dir.exists():
            print(f"\n⚠ Skipping {dataset_dir.name}: images dir not found")
            continue
        
        # Get all image files
        image_files = sorted([f for f in images_dir.iterdir() 
                             if f.suffix in ['.png', '.jpg', '.jpeg']])
        
        print(f"\n{dataset_dir.name}:")
        print(f"  Found {len(image_files)} images")
        
        # Collect pairs
        for image_path in image_files:
            base_name = image_path.stem
            # Strip _sat suffix if present
            if base_name.endswith('_sat'):
                base_name = base_name[:-4]
            label_path = labels_dir / f"{base_name}.txt"
            
            if label_path.exists():
                all_pairs.append((image_path, label_path, dataset_dir.name))
                dataset_stats[dataset_dir.name] += 1
        
        print(f"  Total valid pairs: {dataset_stats[dataset_dir.name]}")
    
    # Shuffle and split
    random.shuffle(all_pairs)
    total_images = len(all_pairs)
    
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    
    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:train_count + val_count]
    test_pairs = all_pairs[train_count + val_count:]
    
    split_data = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    print(f"\n{'=' * 70}")
    print("Dataset Split:")
    print(f"  Train: {len(train_pairs)} images ({len(train_pairs)/total_images*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} images ({len(val_pairs)/total_images*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/total_images*100:.1f}%)")
    print(f"  Total: {total_images} images")
    print(f"{'=' * 70}")
    
    # Copy files to respective splits
    for split, pairs in split_data.items():
        print(f"\nCopying {split} data...")
        
        for idx, (image_path, label_path, dataset_name) in enumerate(pairs):
            # Copy image
            output_image_path = output_dir / 'images' / split / image_path.name
            shutil.copy2(image_path, output_image_path)
            
            # Copy label (with original base name without _sat)
            base_name = image_path.stem
            if base_name.endswith('_sat'):
                base_name = base_name[:-4]
            output_label_path = output_dir / 'labels' / split / f"{base_name}.txt"
            shutil.copy2(label_path, output_label_path)
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(pairs)} images...")
        
        print(f"  [OK] {split} complete: {len(pairs)} images")
    
    # Create data.yaml for training
    data_yaml = output_dir / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"\n")
        f.write(f"nc: 5\n")
        f.write(f"names: ['building', 'road', 'water_body', 'vegetation', 'vehicle']\n")
    
    print(f"\n[OK] Created data.yaml at {data_yaml}")
    
    # Print class distribution
    print(f"\n{'=' * 70}")
    print("Class Distribution by Split:")
    print(f"{'=' * 70}")
    
    for split, pairs in split_data.items():
        class_dist = defaultdict(int)
        for image_path, label_path, _ in pairs:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        class_dist[class_id] += 1
        
        print(f"\n{split}:")
        class_names = ['building', 'road', 'water_body', 'vegetation', 'vehicle']
        total_objects = sum(class_dist.values())
        for class_id in sorted(class_dist.keys()):
            count = class_dist[class_id]
            print(f"  {class_names[class_id]:15s}: {count:6d} ({count/total_objects*100:5.1f}%)")
    
    print(f"\n{'=' * 70}")
    print("[OK] YOLO dataset creation complete!")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert unified dataset to YOLO format"
    )
    parser.add_argument(
        '--unified-dir',
        type=str,
        default='unified_dataset',
        help='Path to unified dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='yolo_dataset',
        help='Output directory for YOLO format dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio'
    )
    
    args = parser.parse_args()
    
    unified_dir = Path(args.unified_dir)
    if not unified_dir.exists():
        print(f"✗ Error: Unified dataset directory not found: {unified_dir}")
        return 1
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"✗ Error: Ratios must sum to 1.0, got {total_ratio}")
        return 1
    
    create_yolo_dataset(
        unified_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
