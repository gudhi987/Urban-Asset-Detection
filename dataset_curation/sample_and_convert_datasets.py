"""
Sample and filter labels from multiple datasets to unified format.

This script:
1. Samples images from each dataset (respecting folder limits)
2. Filters existing YOLO labels to keep only required classes
3. Copies filtered labels to output location
4. Validates filtering results

All datasets already have YOLO-format labels, so we just:
- Copy images and labels
- Filter by class mapping
- No conversion needed

Sampling limits:
- LandCover: all images
- Road extraction: 1000 images
- DOTA: all images
- Semantic buildings: 2000 images

Usage:
    python sample_and_convert_datasets.py \
        --output-dir unified_dataset \
        --landcover-dir ../datasets/deep_globe_land_cover_dataset \
        --road-dir ../datasets/deep_globe_road_extraction \
        --dota-dir ../datasets/dota \
        --semantic-dir "../datasets/semantic_buildings_in_aerial_imagery"
"""

import os
import csv
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
from class_mapping_config import LANDCOVER_MAPPING, ROAD_MAPPING, DOTA_MAPPING, SEMANTIC_BUILDINGS_MAPPING


class DatasetConverter:
    """Base class for dataset conversion"""
    
    def __init__(self, output_dir, dataset_name):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.dataset_output_dir = self.output_dir / dataset_name
        self.images_output_dir = self.dataset_output_dir / "images"
        self.labels_output_dir = self.dataset_output_dir / "labels"
        self.temp_mappings_file = self.dataset_output_dir / "label_mappings.csv"
        
        self.images_output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_images_sampled': 0,
            'images_with_valid_labels': 0,
            'total_objects': 0,
            'filtered_objects': 0,
            'class_distribution': {}
        }
    
    def log_mapping(self, original_id, mapped_id, reason):
        """Log label mapping to CSV for validation"""
        pass
    
    def save_stats(self):
        """Save statistics to file"""
        stats_file = self.dataset_output_dir / "conversion_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total images sampled: {self.stats['total_images_sampled']}\n")
            f.write(f"Images with valid labels: {self.stats['images_with_valid_labels']}\n")
            f.write(f"Total objects detected: {self.stats['total_objects']}\n")
            f.write(f"Filtered objects: {self.stats['filtered_objects']}\n")
            f.write(f"\nClass Distribution:\n")
            for class_id, count in sorted(self.stats['class_distribution'].items()):
                f.write(f"  Class {class_id}: {count}\n")


class LandcoverConverter(DatasetConverter):
    """Filter and copy LandCover dataset labels (already in YOLO format)"""
    
    def __init__(self, output_dir):
        super().__init__(output_dir, "landcover")
    
    def convert(self, dataset_dir):
        """Filter landcover labels to required classes and copy"""
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "train" / "images"
        labels_dir = dataset_dir / "train" / "labels"
        
        if not images_dir.exists():
            print(f"[ERROR] LandCover images dir not found: {images_dir}")
            return
        
        if not labels_dir.exists():
            print(f"[ERROR] LandCover labels dir not found: {labels_dir}")
            return
        
        print(f"\n{'=' * 70}")
        print(f"Filtering LandCover Dataset")
        print(f"{'=' * 70}")
        
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
        print(f"Found {len(image_files)} images")
        
        for idx, image_path in enumerate(image_files):
            base_name = image_path.stem
            # Strip _sat suffix if present (from satellite imagery naming)
            if base_name.endswith('_sat'):
                base_name = base_name[:-4]
            label_path = labels_dir / f"{base_name}.txt"
            
            # Filter labels first - only copy image if it has valid objects
            has_valid_labels = False
            yolo_lines = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        try:
                            class_id = int(float(parts[0]))
                            
                            # Map class (filter unmapped classes)
                            required_class_id = LANDCOVER_MAPPING.get(class_id, None)
                            
                            if required_class_id is None:
                                self.stats['filtered_objects'] += 1
                                continue
                            
                            # Keep line with new class ID
                            x_center, y_center, width, height = parts[1:5]
                            yolo_lines.append(f"{required_class_id} {x_center} {y_center} {width} {height}")
                            self.stats['total_objects'] += 1
                            self.stats['class_distribution'][required_class_id] = \
                                self.stats['class_distribution'].get(required_class_id, 0) + 1
                            has_valid_labels = True
                        
                        except (ValueError, IndexError):
                            continue
                
                except Exception as e:
                    print(f"⚠ Error processing {base_name}: {e}")
            
            # Only copy image if it has valid labels after filtering
            if has_valid_labels:
                output_image_path = self.images_output_dir / image_path.name
                shutil.copy2(image_path, output_image_path)
                self.stats['total_images_sampled'] += 1
                
                output_label_path = self.labels_output_dir / f"{base_name}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                self.stats['images_with_valid_labels'] += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"[OK] LandCover filtering complete: {self.stats['total_images_sampled']} images, " +
              f"{self.stats['total_objects']} objects retained")
        self.save_stats()


class RoadConverter(DatasetConverter):
    """Filter and copy Road dataset labels (already in YOLO format)"""
    
    def __init__(self, output_dir):
        super().__init__(output_dir, "road_extraction")
    
    def convert(self, dataset_dir, limit=1000):
        """Sample images and filter labels to required classes"""
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "train" / "images"
        labels_dir = dataset_dir / "train" / "labels"
        
        if not images_dir.exists():
            print(f"[ERROR] Road images dir not found: {images_dir}")
            return
        
        print(f"\n{'=' * 70}")
        print(f"Filtering Road Extraction Dataset (sampling {limit} images)")
        print(f"{'=' * 70}")
        
        image_files = [f for f in images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]
        image_files = random.sample(image_files, min(limit, len(image_files)))
        image_files = sorted(image_files)
        print(f"Sampled {len(image_files)} images")
        
        for idx, image_path in enumerate(image_files):
            base_name = image_path.stem
            # Strip _sat suffix if present (from satellite imagery naming)
            if base_name.endswith('_sat'):
                base_name = base_name[:-4]
            label_path = labels_dir / f"{base_name}.txt"
            
            # Filter labels first - only copy image if it has valid objects
            has_valid_labels = False
            yolo_lines = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        try:
                            class_id = int(float(parts[0]))
                            
                            # Map class (filter unmapped classes)
                            required_class_id = ROAD_MAPPING.get(class_id, None)
                            
                            if required_class_id is None:
                                self.stats['filtered_objects'] += 1
                                continue
                            
                            # Keep line with new class ID
                            x_center, y_center, width, height = parts[1:5]
                            yolo_lines.append(f"{required_class_id} {x_center} {y_center} {width} {height}")
                            self.stats['total_objects'] += 1
                            self.stats['class_distribution'][required_class_id] = \
                                self.stats['class_distribution'].get(required_class_id, 0) + 1
                            has_valid_labels = True
                        
                        except (ValueError, IndexError):
                            continue
                
                except Exception as e:
                    print(f"[ERROR] processing {base_name}: {e}")
            
            # Only copy image if it has valid labels after filtering
            if has_valid_labels:
                output_image_path = self.images_output_dir / image_path.name
                shutil.copy2(image_path, output_image_path)
                self.stats['total_images_sampled'] += 1
                
                output_label_path = self.labels_output_dir / f"{base_name}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                self.stats['images_with_valid_labels'] += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"[OK] Road filtering complete: {self.stats['total_images_sampled']} images, " +
              f"{self.stats['total_objects']} objects retained")
        self.save_stats()


class DOTAConverter(DatasetConverter):
    """Filter and copy DOTA dataset labels (already in YOLO format)"""
    
    def __init__(self, output_dir):
        super().__init__(output_dir, "dota")
    
    def convert(self, dataset_dir):
        """Filter DOTA labels to required classes and copy"""
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "train" / "images"
        labels_dir = dataset_dir / "train" / "labels"
        
        if not images_dir.exists():
            print(f"[ERROR] DOTA images dir not found: {images_dir}")
            return
        
        print(f"\n{'=' * 70}")
        print(f"Filtering DOTA Dataset")
        print(f"{'=' * 70}")
        
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])
        print(f"Found {len(image_files)} images")
        
        for idx, image_path in enumerate(image_files):
            base_name = image_path.stem
            label_path = labels_dir / f"{base_name}.txt"
            
            # Filter labels first - only copy image if it has valid objects
            has_valid_labels = False
            yolo_lines = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        try:
                            class_id = int(float(parts[0]))
                            
                            # Map class (filter unmapped classes)
                            required_class_id = DOTA_MAPPING.get(class_id, None)
                            
                            if required_class_id is None:
                                self.stats['filtered_objects'] += 1
                                continue
                            
                            # Keep line with new class ID
                            x_center, y_center, width, height = parts[1:5]
                            yolo_lines.append(f"{required_class_id} {x_center} {y_center} {width} {height}")
                            self.stats['total_objects'] += 1
                            self.stats['class_distribution'][required_class_id] = \
                                self.stats['class_distribution'].get(required_class_id, 0) + 1
                            has_valid_labels = True
                        
                        except (ValueError, IndexError):
                            continue
                
                except Exception as e:
                    print(f"⚠ Error processing {base_name}: {e}")
            
            # Only copy image if it has valid labels after filtering
            if has_valid_labels:
                output_image_path = self.images_output_dir / image_path.name
                shutil.copy2(image_path, output_image_path)
                self.stats['total_images_sampled'] += 1
                
                output_label_path = self.labels_output_dir / f"{base_name}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                self.stats['images_with_valid_labels'] += 1
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"[OK] DOTA filtering complete: {self.stats['total_images_sampled']} images, " +
              f"{self.stats['total_objects']} objects retained")
        self.save_stats()


class SemanticBuildingsConverter(DatasetConverter):
    """Filter and copy Semantic Buildings labels (already in YOLO format)"""
    
    def __init__(self, output_dir):
        super().__init__(output_dir, "semantic_buildings")
    
    def convert(self, dataset_dir, limit=2000):
        """Sample images and filter labels to required classes"""
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "train" / "images"
        labels_dir = dataset_dir / "train" / "labels"
        
        if not images_dir.exists():
            print(f"[ERROR] Semantic buildings images dir not found: {images_dir}")
            return
        
        print(f"\n{'=' * 70}")
        print(f"Filtering Semantic Buildings Dataset (sampling {limit} images)")
        print(f"{'=' * 70}")
        
        image_files = [f for f in images_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]
        image_files = random.sample(image_files, min(limit, len(image_files)))
        image_files = sorted(image_files)
        print(f"Sampled {len(image_files)} images")
        
        for idx, image_path in enumerate(image_files):
            base_name = image_path.stem
            label_path = labels_dir / f"{base_name}.txt"
            
            # Filter labels first - only copy image if it has valid objects
            has_valid_labels = False
            yolo_lines = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        try:
                            class_id = int(float(parts[0]))
                            
                            # Map class (filter unmapped classes)
                            required_class_id = SEMANTIC_BUILDINGS_MAPPING.get(class_id, None)
                            
                            if required_class_id is None:
                                self.stats['filtered_objects'] += 1
                                continue
                            
                            # Keep line with new class ID
                            x_center, y_center, width, height = parts[1:5]
                            yolo_lines.append(f"{required_class_id} {x_center} {y_center} {width} {height}")
                            self.stats['total_objects'] += 1
                            self.stats['class_distribution'][required_class_id] = \
                                self.stats['class_distribution'].get(required_class_id, 0) + 1
                            has_valid_labels = True
                        
                        except (ValueError, IndexError):
                            continue
                
                except Exception as e:
                    print(f"⚠ Error processing {base_name}: {e}")
            
            # Only copy image if it has valid labels after filtering
            if has_valid_labels:
                output_image_path = self.images_output_dir / image_path.name
                shutil.copy2(image_path, output_image_path)
                self.stats['total_images_sampled'] += 1
                
                output_label_path = self.labels_output_dir / f"{base_name}.txt"
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                self.stats['images_with_valid_labels'] += 1
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"[OK] Semantic Buildings filtering complete: {self.stats['total_images_sampled']} images, " +
              f"{self.stats['total_objects']} objects retained")
        self.save_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Sample and convert multiple datasets to unified format"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='unified_dataset',
        help='Output directory for unified dataset'
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
    print("Multi-Dataset Sampling and Conversion Tool")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load required class dict
    required_class_dict = {
        0: 'building',
        1: 'road',
        2: 'water_body',
        3: 'vegetation',
        4: 'vehicle'
    }
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Required classes: {required_class_dict}")
    
    # Convert each dataset
    converters_stats = {}
    
    # LandCover
    lc = LandcoverConverter(output_dir)
    lc.convert(args.landcover_dir)
    converters_stats['landcover'] = lc.stats
    
    # Road extraction
    rd = RoadConverter(output_dir)
    rd.convert(args.road_dir, limit=1000)
    converters_stats['road_extraction'] = rd.stats
    
    # DOTA
    dota = DOTAConverter(output_dir)
    dota.convert(args.dota_dir)
    converters_stats['dota'] = dota.stats
    
    # Semantic buildings
    sb = SemanticBuildingsConverter(output_dir)
    sb.convert(args.semantic_dir, limit=2000)
    converters_stats['semantic_buildings'] = sb.stats
    
    # Print overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL CONVERSION SUMMARY")
    print("=" * 70)
    
    total_images = 0
    total_objects = 0
    total_filtered = 0
    all_class_dist = {}
    
    for dataset_name, stats in converters_stats.items():
        total_images += stats['total_images_sampled']
        total_objects += stats['total_objects']
        total_filtered += stats['filtered_objects']
        for class_id, count in stats['class_distribution'].items():
            all_class_dist[class_id] = all_class_dist.get(class_id, 0) + count
    
    print(f"\nTotal images: {total_images}")
    print(f"Total objects retained: {total_objects}")
    print(f"Total objects filtered: {total_filtered}")
    print(f"\nFinal class distribution:")
    class_names = {
        0: 'building',
        1: 'road',
        2: 'water_body',
        3: 'vegetation',
        4: 'vehicle'
    }
    for class_id in sorted(all_class_dist.keys()):
        count = all_class_dist[class_id]
        print(f"  {class_names[class_id]:15s} (id={class_id}): {count:6d}")
    
    print("\n[OK] Conversion complete!")


if __name__ == "__main__":
    exit(main())
