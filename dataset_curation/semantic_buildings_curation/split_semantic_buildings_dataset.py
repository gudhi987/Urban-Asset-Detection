import os
import glob
import shutil
import argparse
import random
from pathlib import Path


def create_split_directories(dataset_dir):
    """Create necessary directories for train/val/test splits."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = Path(dataset_dir) / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {images_dir}")
    
    # Ensure train/labels directory exists
    labels_dir = Path(dataset_dir) / "train" / "labels"
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}. Please ensure labels are in train/labels before splitting.")


def validate_ratios(train_ratio, val_ratio, test_ratio):
    """Validate that ratios sum to 1.0"""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:  # Allow small floating point errors
        raise ValueError(
            f"Ratios must sum to 1.0, got {total}. "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )


def get_image_files(images_dir):
    """Get all image files from the images directory."""
    image_files = (
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    return image_files


def split_images(image_files, train_ratio, val_ratio, test_ratio, random_seed):
    """
    Split image files into train/val/test sets.
    
    Returns:
        dict: Split data with keys 'train', 'val', 'test' containing file paths
    """
    random.seed(random_seed)
    
    # Shuffle and split
    shuffled = image_files.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    splits = {
        'train': shuffled[:train_count],
        'val': shuffled[train_count:train_count + val_count],
        'test': shuffled[train_count + val_count:]
    }
    
    return splits


def move_file(source_path, dest_dir):
    """Move a file from source to destination."""
    if not os.path.exists(source_path):
        print(f"  ⚠ Warning: File not found: {source_path}")
        return False
    
    try:
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
        shutil.move(source_path, dest_path)
        return True
    except Exception as e:
        print(f"  ✗ Error moving file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Split semantic buildings dataset into train/val/test sets"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='../../datasets/semantic_buildings_in_aerial_imagery',
        help='Path to dataset root directory'
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
    print("Semantic Buildings Dataset Split Tool")
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
    
    # Get image files
    print("\n" + "=" * 70)
    print("Scanning Images")
    print("=" * 70)
    
    image_files = get_image_files(str(train_images_dir))
    print(f"Found {len(image_files)} image files in training set")
    
    if len(image_files) == 0:
        print("✗ Error: No images found in training set")
        return 1
    
    # Generate splits
    print("\n" + "=" * 70)
    print("Generating Splits")
    print("=" * 70)
    
    splits = split_images(
        image_files,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"Train set: {len(splits['train'])} files ({len(splits['train'])/len(image_files)*100:.1f}%)")
    print(f"Val set:   {len(splits['val'])} files ({len(splits['val'])/len(image_files)*100:.1f}%)")
    print(f"Test set:  {len(splits['test'])} files ({len(splits['test'])/len(image_files)*100:.1f}%)")
    
    # Perform split
    if args.dry_run:
        print("\n(Dry run: files would be moved here)")
        stats = {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        }
    else:
        print("\n" + "=" * 70)
        print("Splitting Dataset")
        print("=" * 70)
        print("Note: Train images remain in train/images, only val and test are moved")
        
        stats = {
            'train': len(splits['train']),
            'val': 0,
            'test': 0
        }
        
        # Move val and test files only
        for split_name in ['val', 'test']:
            dest_images_dir = Path(dataset_dir) / split_name / "images"
            
            print(f"\n{split_name.upper()} ({len(splits[split_name])} files):")
            print("-" * 70)
            
            for idx, image_path in enumerate(splits[split_name]):
                if move_file(image_path, str(dest_images_dir)):
                    stats[split_name] += 1
                
                # Progress update
                if (idx + 1) % 50 == 0:
                    print(f"  Moved {idx + 1}/{len(splits[split_name])} files...")
            
            print(f"✓ {split_name.upper()}: {stats[split_name]} images moved")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    
    total_files = sum(len(files) for files in splits.values())
    
    print(f"\nDataset Split Summary:")
    for split_name in ['train', 'val', 'test']:
        count = len(splits[split_name])
        percentage = (count / total_files * 100) if total_files > 0 else 0
        location = "train/images" if split_name == 'train' else f"{split_name}/images"
        print(f"  {split_name.upper():5s}: {count:4d} files ({percentage:5.1f}%) → {location}")
    
    print(f"\nLabels Location: train/labels (all {total_files} labels)")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
