#!/usr/bin/env python
"""
DOTA Dataset Processing Setup Script

This is a unified entry point for processing DOTA dataset labels.
It runs the complete pipeline: extract classes, convert OBB to YOLO, verify labels, and analyze.

Usage:
    python setup_dota_pipeline.py
    
Or with custom paths:
    python setup_dota_pipeline.py --dataset-dir /path/to/dota --split train
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print(f"⚠ Warning: Command completed with return code {result.returncode}")
            return False
        print(f"✓ {description} completed successfully!\n")
        return True
    except FileNotFoundError as e:
        print(f"✗ Error: Python script not found. {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="DOTA Dataset Processing Pipeline Setup"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='../../datasets/dota',
        help='Path to DOTA dataset root directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'valid', 'test', 'all'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--class-dict',
        type=str,
        default=None,
        help='Path to class dictionary CSV (auto-generated if not provided)'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip dataset analysis step'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization step'
    )
    
    args = parser.parse_args()
    
    # Resolve relative paths
    script_dir = Path(__file__).parent
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.is_absolute():
        dataset_dir = script_dir / dataset_dir
    
    dataset_dir = dataset_dir.resolve()
    
    print("\n" + "=" * 70)
    print("DOTA DATASET PROCESSING PIPELINE")
    print("=" * 70)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Split(s) to process: {args.split}")
    
    # Validate dataset directory
    if not dataset_dir.exists():
        print(f"\n✗ Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    # Auto-determine class dictionary path
    if args.class_dict is None:
        class_dict = dataset_dir / "class_dict.csv"
    else:
        class_dict = Path(args.class_dict).resolve()
    
    # Determine splits to process
    splits = [args.split] if args.split != 'all' else ['train', 'valid', 'test']
    
    # Step 1: Extract classes (only if class_dict doesn't exist)
    if not class_dict.exists():
        print(f"\nClass dictionary not found at {class_dict}")
        print("Extracting classes from training labels...")
        
        train_labels_dir = dataset_dir / "train" / "trainset_reclabelTxt"
        if not train_labels_dir.exists():
            print(f"✗ Error: Training labels directory not found: {train_labels_dir}")
            return 1
        
        cmd = [
            sys.executable,
            str(script_dir / "extract_classes_dota.py"),
            "--input-dir", str(train_labels_dir),
            "--output-file", str(class_dict)
        ]
        
        if not run_command(cmd, "Extract Classes"):
            return 1
    else:
        print(f"\n✓ Class dictionary already exists: {class_dict}")
    
    # Step 2: Convert OBB to YOLO for each split
    for split in splits:
        split_dir = dataset_dir / split
        
        if not split_dir.exists():
            print(f"\n⚠ Warning: {split.upper()} split directory not found: {split_dir}")
            continue
        
        input_labels = split_dir / "trainset_reclabelTxt"
        output_labels = split_dir / "labels"
        images_dir = split_dir / "images"
        
        # Check if already converted
        if output_labels.exists() and len(list(output_labels.glob("*.txt"))) > 0:
            print(f"\n✓ {split.upper()} labels already converted")
            continue
        
        if not input_labels.exists():
            print(f"\n⚠ Warning: Input labels not found for {split}: {input_labels}")
            continue
        
        cmd = [
            sys.executable,
            str(script_dir / "convert_obb_to_yolo.py"),
            "--input-labels", str(input_labels),
            "--output-labels", str(output_labels),
            "--images-dir", str(images_dir),
            "--class-dict", str(class_dict)
        ]
        
        if not run_command(cmd, f"Convert {split.upper()} Set (OBB → YOLO)"):
            print(f"⚠ Warning: Conversion failed for {split} split")
    
    # Step 3: Verify labels
    for split in splits:
        split_dir = dataset_dir / split
        labels_dir = split_dir / "labels"
        images_dir = split_dir / "images"
        
        if not labels_dir.exists():
            continue
        
        cmd = [
            sys.executable,
            str(script_dir / "verify_yolo_labels.py"),
            "--labels-dir", str(labels_dir),
            "--images-dir", str(images_dir),
            "--class-dict", str(class_dict)
        ]
        
        if not run_command(cmd, f"Verify {split.upper()} Labels"):
            print(f"⚠ Warning: Verification issues found for {split} split")
    
    # Step 4: Analyze dataset (optional)
    if not args.skip_analysis:
        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            cmd = [
                sys.executable,
                str(script_dir / "analyze_dota_dataset.py"),
                "--dataset-dir", str(dataset_dir),
                "--split", split,
                "--class-dict", str(class_dict)
            ]
            
            if not run_command(cmd, f"Analyze {split.upper()} Dataset"):
                print(f"⚠ Warning: Analysis failed for {split} split")
    
    # Step 5: Visualization (optional)
    if not args.skip_visualization and 'train' in splits:
        print("\n" + "=" * 70)
        print("OPTIONAL: Visualize Dataset")
        print("=" * 70)
        print("\nTo visualize sample images with bounding boxes, run:")
        print(f"  python {script_dir}/visualize_dota_bboxes.py")
        print("\nOr programmatically:")
        print(f"  from visualize_dota_bboxes import visualize_dota_samples_with_bboxes")
        print(f"  visualize_dota_samples_with_bboxes('{dataset_dir}', num_samples=4, split='train')")
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nDataset Location: {dataset_dir}")
    print(f"Class Dictionary: {class_dict}")
    
    for split in splits:
        labels_dir = dataset_dir / split / "labels"
        if labels_dir.exists():
            num_files = len(list(labels_dir.glob("*.txt")))
            print(f"  {split.upper()} - {num_files} label files converted")
    
    print("\n✓ All steps completed successfully!")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
