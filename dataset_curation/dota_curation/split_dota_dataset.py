"""
Split DOTA dataset into train/val/test sets.

This script splits the DOTA training dataset into:
- 80% training set (stays in train/images)
- 10% validation set (images moved to val/images)
- 10% test set (images moved to test/images)

All labels remain in train/labels.

Usage:
    python split_dota_dataset.py \
        --dataset-dir datasets/dota \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1 \
        --seed 42
"""

import os
import glob
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict


def create_split_directories(dataset_dir):
    """Create necessary directories for train/val/test splits."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = Path(dataset_dir) / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {images_dir}")
    
    # Ensure train/labels directory exists
    labels_dir = Path(dataset_dir) / "train" / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")


def validate_ratios(train_ratio, val_ratio, test_ratio):
    """Validate that ratios sum to 1.0"""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:  # Allow small floating point errors
        raise ValueError(
            f"Ratios must sum to 1.0, got {total}. "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )


def get_image_label_pairs(images_dir, labels_dir):
    """
    Get all image-label pairs from the images and labels directories.
    
    Returns:
        dict: Mapping of base_name -> {'image': image_path, 'label': label_path}
    """
    pairs = {}
    
    # Find all images
    image_files = (
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    
    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(labels_dir, base_name + ".txt")
        
        pairs[base_name] = {
            'image': image_file,
            'label': label_file if os.path.exists(label_file) else None
        }
    
    return pairs


def split_dataset(pairs, train_ratio, val_ratio, test_ratio, random_seed):
    """
    Split image-label pairs into train/val/test sets.
    
    Args:
        pairs (dict): Image-label pairs
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Split data with keys 'train', 'val', 'test'
    """
    random.seed(random_seed)
    
    # Get list of base names and shuffle
    base_names = list(pairs.keys())
    random.shuffle(base_names)
    
    total = len(base_names)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    # test_count = int(total * test_ratio)  # Remaining go to test
    
    splits = {
        'train': base_names[:train_count],
        'val': base_names[train_count:train_count + val_count],
        'test': base_names[train_count + val_count:]
    }
    
    return splits


def move_files(source_dir, dest_dir, file_path, description):
    """
    Move a file from source to destination.
    
    Args:
        source_dir (str): Source directory
        dest_dir (str): Destination directory
        file_path (str): Path to file to move
        description (str): Description for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"  ⚠ Warning: {description} not found: {file_path}")
        return False
    
    try:
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        shutil.move(file_path, dest_path)
        return True
    except Exception as e:
        print(f"  ✗ Error moving {description}: {e}")
        return False


def perform_split(dataset_dir, pairs, splits, source_images_dir, source_labels_dir):
    """
    Move image files according to the split configuration.
    Labels remain in train/labels for all splits.
    
    Args:
        dataset_dir (str): Root dataset directory
        pairs (dict): Image-label pairs
        splits (dict): Split assignments
        source_images_dir (str): Source images directory
        source_labels_dir (str): Source labels directory (unused but kept for interface compatibility)
        
    Returns:
        dict: Statistics about the split
    """
    stats = {
        'train': {'images': 0},
        'val': {'images': 0},
        'test': {'images': 0},
        'errors': []
    }
    
    print("\n" + "=" * 70)
    print("Moving Image Files")
    print("=" * 70)
    print("Note: All labels remain in train/labels")
    
    # Train images stay in place, just count them
    print(f"\nTRAIN ({len(splits['train'])} files):")
    print("-" * 70)
    stats['train']['images'] = len(splits['train'])
    print(f"✓ TRAIN: {stats['train']['images']} images (staying in train/images)")
    
    # Move val and test images only
    for split_name in ['val', 'test']:
        base_names = splits[split_name]
        dest_images_dir = os.path.join(dataset_dir, split_name, "images")
        
        print(f"\n{split_name.upper()} ({len(base_names)} files):")
        print("-" * 70)
        
        for idx, base_name in enumerate(base_names):
            pair = pairs[base_name]
            
            # Move only image (not label)
            if move_files(source_images_dir, dest_images_dir, pair['image'], f"Image {base_name}"):
                stats[split_name]['images'] += 1
            else:
                stats[split_name]['errors'].append(f"Failed to move image: {base_name}")
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"  Moved {idx + 1}/{len(base_names)} files...")
        
        print(f"✓ {split_name.upper()}: {stats[split_name]['images']} images moved to {split_name}/images")
    
    return stats


def print_summary(splits, stats):
    """Print summary statistics of the split."""
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    
    total_files = sum(len(files) for files in splits.values())
    
    print(f"\nDataset Split Summary:")
    for split_name in ['train', 'val', 'test']:
        count = len(splits[split_name])
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"  {split_name.upper():5s}: {count:4d} files ({percentage:5.1f}%)")
    
    print(f"\nImage Movement Summary:")
    for split_name in ['train', 'val', 'test']:
        images = stats[split_name]['images']
        location = "train/images" if split_name == 'train' else f"{split_name}/images"
        print(f"  {split_name.upper():5s}: {images:4d} images → {location}")
    
    print(f"\nLabels Location: train/labels (all {stats['train']['images'] + stats['val']['images'] + stats['test']['images']} labels)")
    
    total_errors = sum(len(errors) for errors in [stats[s].get('errors', []) for s in ['train', 'val', 'test']])
    if total_errors > 0:
        print(f"\n⚠ Errors encountered: {total_errors}")
    else:
        print(f"\n✓ No errors encountered!")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Split DOTA dataset into train/val/test sets"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='datasets/dota',
        help='Path to DOTA dataset root directory'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of files for training set (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of files for validation set (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Ratio of files for test set (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"✗ Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    
    if not train_images_dir.exists():
        print(f"✗ Error: Training images directory not found: {train_images_dir}")
        return 1
    
    # Validate ratios
    try:
        validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    except ValueError as e:
        print(f"✗ Error: {e}")
        return 1
    
    print("=" * 70)
    print("DOTA Dataset Split Tool")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio:   {args.val_ratio}")
    print(f"Test ratio:  {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Dry run:     {args.dry_run}")
    
    # Create directories
    print("\n" + "=" * 70)
    print("Setting Up Directories")
    print("=" * 70)
    
    if not args.dry_run:
        create_split_directories(dataset_dir)
    else:
        print("(Dry run: directories would be created)")
    
    # Get image-label pairs
    print("\n" + "=" * 70)
    print("Scanning Images and Labels")
    print("=" * 70)
    
    pairs = get_image_label_pairs(str(train_images_dir), str(train_labels_dir))
    print(f"Found {len(pairs)} image-label pairs in training set")
    
    if len(pairs) == 0:
        print("✗ Error: No images found in training set")
        return 1
    
    # Generate splits
    print("\n" + "=" * 70)
    print("Generating Splits")
    print("=" * 70)
    
    splits = split_dataset(
        pairs,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"Train set: {len(splits['train'])} files ({len(splits['train'])/len(pairs)*100:.1f}%)")
    print(f"Val set:   {len(splits['val'])} files ({len(splits['val'])/len(pairs)*100:.1f}%)")
    print(f"Test set:  {len(splits['test'])} files ({len(splits['test'])/len(pairs)*100:.1f}%)")
    
    # Show preview
    print(f"\nPreview of split assignments:")
    print(f"  Train: {', '.join(sorted(splits['train'])[:3])} ...")
    print(f"  Val:   {', '.join(sorted(splits['val'])[:3])} ...")
    print(f"  Test:  {', '.join(sorted(splits['test'])[:3])} ...")
    
    # Perform split
    if args.dry_run:
        print("\n(Dry run: files would be moved here)")
        stats = {
            'train': {'images': len(splits['train'])},
            'val': {'images': len(splits['val'])},
            'test': {'images': len(splits['test'])},
            'errors': []
        }
    else:
        # Only move images for val and test (train stays in place, labels stay in train)
        print("\n" + "=" * 70)
        print("Splitting Dataset")
        print("=" * 70)
        
        stats = {
            'train': {'images': len(splits['train'])},
            'val': {'images': 0},
            'test': {'images': 0}
        }
        
        # Move val and test images only
        for split_name in ['val', 'test']:
            dest_images_dir = Path(dataset_dir) / split_name / "images"
            
            print(f"\n{split_name.upper()} ({len(splits[split_name])} files):")
            print("-" * 70)
            
            for idx, base_name in enumerate(splits[split_name]):
                pair = pairs[base_name]
                
                # Move only image (not label)
                if move_files(str(train_images_dir), str(dest_images_dir), 
                             pair['image'], f"Image {base_name}"):
                    stats[split_name]['images'] += 1
                
                # Progress update
                if (idx + 1) % 100 == 0:
                    print(f"  Moved {idx + 1}/{len(splits[split_name])} files...")
            
            print(f"✓ {split_name.upper()}: {stats[split_name]['images']} images moved")
    
    # Print summary
    print_summary(splits, stats)
    
    return 0


if __name__ == "__main__":
    exit(main())
