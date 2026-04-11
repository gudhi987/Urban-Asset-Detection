"""
Complete pipeline for sampling, converting, validating and visualizing unified dataset.

This script runs the full workflow:
1. Sample and convert all datasets to unified format
2. Validate the conversion
3. Visualize results

Usage:
    python unified_dataset_pipeline.py --unified-dir unified_dataset
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(script_name, args=None, description=None):
    """Run a Python script with arguments"""
    if description:
        print(f"\n{'=' * 70}")
        print(description)
        print(f"{'=' * 70}")
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete unified dataset pipeline"
    )
    parser.add_argument(
        '--unified-dir',
        type=str,
        default='unified_dataset',
        help='Output directory for unified dataset'
    )
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip conversion step (use existing unified dataset)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization step'
    )
    parser.add_argument(
        '--landcover-dir',
        type=str,
        default='../datasets/deep_globe_land_cover_dataset',
        help='Path to LandCover dataset'
    )
    parser.add_argument(
        '--road-dir',
        type=str,
        default='../datasets/deep_globe_road_extraction',
        help='Path to Road extraction dataset'
    )
    parser.add_argument(
        '--dota-dir',
        type=str,
        default='../datasets/dota',
        help='Path to DOTA dataset'
    )
    parser.add_argument(
        '--semantic-dir',
        type=str,
        default='../datasets/semantic_buildings_in_aerial_imagery',
        help='Path to Semantic buildings dataset'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("UNIFIED DATASET PIPELINE")
    print("=" * 70)
    
    # Step 1: Conversion
    if not args.skip_conversion:
        conversion_args = [
            '--output-dir', args.unified_dir,
            '--landcover-dir', args.landcover_dir,
            '--road-dir', args.road_dir,
            '--dota-dir', args.dota_dir,
            '--semantic-dir', args.semantic_dir
        ]
        if not run_command(
            'sample_and_convert_datasets.py',
            conversion_args,
            'Step 1: Sample and Convert Datasets'
        ):
            print("✗ Conversion failed!")
            return 1
    else:
        print("\nSkipping conversion (--skip-conversion flag set)")
    
    # Step 2: Validation
    if not args.skip_validation:
        validation_args = ['--unified-dir', args.unified_dir]
        if not run_command(
            'validate_unified_dataset.py',
            validation_args,
            'Step 2: Validate Unified Dataset'
        ):
            print("✗ Validation failed!")
            return 1
    else:
        print("\nSkipping validation (--skip-validation flag set)")
    
    # Step 3: Visualization
    if not args.skip_visualization:
        viz_args = ['--unified-dir', args.unified_dir, '--num-samples', '1']
        if not run_command(
            'visualize_unified_dataset.py',
            viz_args,
            'Step 3: Visualize Unified Dataset'
        ):
            print("⚠ Visualization encountered an issue (may be normal)")
    else:
        print("\nSkipping visualization (--skip-visualization flag set)")
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nUnified dataset location: {args.unified_dir}")
    print("Ready for training!")
    
    return 0


if __name__ == "__main__":
    exit(main())
