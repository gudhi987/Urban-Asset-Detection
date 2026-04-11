#!/usr/bin/env python
"""
Quick script to split DOTA dataset and verify the split.

This is a convenience script that:
1. Runs the split (80% train, 10% val, 10% test)
2. Verifies the split
3. Shows statistics

Usage:
    python split_and_verify_dota.py
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(script_name, args=None, description=None):
    """Run a Python script."""
    script_dir = Path(__file__).parent
    script_path = script_dir / script_name
    
    if not script_path.exists():
        print(f"✗ Error: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    if description:
        print("\n" + "=" * 80)
        print(f"STEP: {description}")
        print("=" * 80)
        print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("DOTA DATASET SPLIT AND VERIFICATION")
    print("=" * 80)
    
    # Step 1: Split dataset
    print("\nStep 1: Split Dataset")
    if not run_command(
        "split_dota_dataset.py",
        args=["--train-ratio", "0.8", "--val-ratio", "0.1", "--test-ratio", "0.1"],
        description="Split DOTA dataset (80% train, 10% val, 10% test)"
    ):
        print("\n✗ Failed to split dataset")
        return 1
    
    # Step 2: Verify split
    print("\n\nStep 2: Verify Split")
    if not run_command(
        "verify_split_dota.py",
        description="Verify dataset split"
    ):
        print("\n⚠ Verification had issues")
        # Don't return error, as this is just informational
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print("=" * 80)
    print("\nDataset is now split into:")
    print("  - train/ (80% of original training data)")
    print("  - val/   (10% of original training data)")
    print("  - test/  (10% of original training data)")
    print("\nEach split contains images/ and labels/ subdirectories.")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
