"""
Annotation Utilities for YOLO Format Bounding Box Generation
Converts semantic segmentation masks (RGB) to instance-level YOLO annotations
Works with any dataset by accepting class metadata as input
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from typing import Dict, List, Tuple, Optional


class MaskAnnotationPipeline:
    """
    Pipeline to convert RGB semantic masks to YOLO format bounding box annotations
    """
    
    def __init__(self, class_metadata: pd.DataFrame):
        """
        Initialize with class metadata
        
        Args:
            class_metadata (pd.DataFrame): DataFrame with columns ['name', 'r', 'g', 'b']
                Example:
                    name              r    g    b
                0   urban_land        0   255  255
                1   agriculture_land 255  255    0
        """
        self.class_metadata = class_metadata
        self.class_colors = self._build_color_mapping()
        self.class_names = self._build_name_mapping()
        self.num_classes = len(class_metadata)
        
        print(f"✓ Initialized pipeline for {self.num_classes} classes")
        print(f"  Classes: {', '.join(self.class_names.values())}")
    
    def _build_color_mapping(self) -> Dict[Tuple[int, int, int], int]:
        """Build RGB color → class ID mapping"""
        color_map = {}
        for idx, row in self.class_metadata.iterrows():
            rgb = (int(row['r']), int(row['g']), int(row['b']))
            color_map[rgb] = idx
        return color_map
    
    def _build_name_mapping(self) -> Dict[int, str]:
        """Build class ID → class name mapping"""
        name_map = {}
        for idx, row in self.class_metadata.iterrows():
            name_map[idx] = row['name']
        return name_map
    
    def rgb_to_class_index(self, mask_rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to class index mask
        
        Args:
            mask_rgb (np.ndarray): RGB mask image (H, W, 3)
            
        Returns:
            np.ndarray: Class index mask (H, W) with values 0 to num_classes-1
        """
        h, w = mask_rgb.shape[:2]
        # class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # for rgb_color, class_id in self.class_colors.items():
        #     # Find pixels matching this RGB color
        #     match = np.all(mask_rgb[:, :, :3] == np.array(rgb_color), axis=2)
        #     class_mask[match] = class_id

        mask_int = (
            mask_rgb[:, :, 0].astype(np.int32) * 256 * 256 +
            mask_rgb[:, :, 1].astype(np.int32) * 256 +
            mask_rgb[:, :, 2].astype(np.int32)
        )

        # Build color → class_id mapping in integer space
        color_to_class = {}
        for rgb, class_id in self.class_colors.items():
            r, g, b = rgb
            key = r * 256 * 256 + g * 256 + b
            color_to_class[key] = class_id

        # Initialize output
        class_mask = np.zeros((h, w), dtype=np.uint8)

        # Map values
        for color_key, class_id in color_to_class.items():
            class_mask[mask_int == color_key] = class_id
            
        return class_mask
    
    def extract_bboxes_yolo(self, mask_path: str, image_width: int, image_height: int) -> List[List[float]]:
        """
        Extract bounding boxes from mask in YOLO format
        
        Args:
            mask_path (str): Path to mask image
            image_width (int): Width of original image
            image_height (int): Height of original image
            
        Returns:
            List[List[float]]: List of [class_id, x_center, y_center, width, height]
                              where all coordinates are normalized to 0-1
        """
        # Load and convert mask to class indices
        mask_rgb = np.array(Image.open(mask_path))
        class_mask = self.rgb_to_class_index(mask_rgb)
        
        bboxes = []
        
        # Process each class
        for class_id in range(self.num_classes):
            # Binary mask for this class
            binary_mask = (class_mask == class_id)
            
            if not binary_mask.any():
                continue
            
            # Find connected components (separate objects/instances)
            labeled_mask, num_objects = ndimage.label(binary_mask)
            
            # Extract bbox for each object
            for obj_id in range(1, num_objects + 1):
                obj_mask = labeled_mask == obj_id
                
                # Get pixel coordinates
                y_indices, x_indices = np.where(obj_mask)
                
                if len(y_indices) == 0:
                    continue
                
                # Pixel-level bounding box
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                
                # Convert to YOLO format (normalized to 0-1)
                x_center = (x_min + x_max + 1) / (2 * image_width)
                y_center = (y_min + y_max + 1) / (2 * image_height)
                width = (x_max - x_min + 1) / image_width
                height = (y_max - y_min + 1) / image_height
                
                # Clamp to valid range [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                bboxes.append([class_id, x_center, y_center, width, height])
        
        return bboxes
    
    def process_image_pair(self, image_path: str, mask_path: str, output_label_path: str) -> Tuple[bool, int, str]:
        """
        Process a single image-mask pair and save YOLO annotations
        
        Args:
            image_path (str): Path to image file
            mask_path (str): Path to corresponding mask file
            output_label_path (str): Path to save annotation txt file
            
        Returns:
            Tuple[bool, int, str]: (success, num_objects, message)
        """
        try:
            if not os.path.exists(mask_path):
                return False, 0, f"Mask not found: {mask_path}"
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Extract bounding boxes
            bboxes = self.extract_bboxes_yolo(mask_path, img_width, img_height)
            
            # Save annotations
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            with open(output_label_path, 'w') as f:
                for bbox in bboxes:
                    class_id, x_center, y_center, width, height = bbox
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            return True, len(bboxes), f"✓ Extracted {len(bboxes)} objects"
            
        except Exception as e:
            return False, 0, f"Error: {str(e)}"
    
    def process_split(self, split_dir: str, mask_suffix: str = "_mask.png", verbose: bool = True) -> Dict:
        """
        Process a dataset split (train/test/val)
        
        Expected directory structure:
            split_dir/
            ├── images/
            │   ├── image_001.jpg
            │   ├── image_002.jpg
            │   └── ...
            └── masks/
                ├── image_001_mask.png
                ├── image_002_mask.png
                └── ...
        
        Args:
            split_dir (str): Path to dataset split directory
            mask_suffix (str): Suffix for mask files (default: "_mask.png")
            verbose (bool): Print progress
            
        Returns:
            Dict: Processing statistics
        """
        images_dir = os.path.join(split_dir, "images")
        masks_dir = os.path.join(split_dir, "masks")
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(images_dir):
            return {'split': os.path.basename(split_dir), 'successful': 0, 'failed': 0, 'error': 'images/ not found'}
        
        if not os.path.exists(masks_dir):
            return {'split': os.path.basename(split_dir), 'successful': 0, 'failed': 0, 'error': 'masks/ not found'}
        
        os.makedirs(labels_dir, exist_ok=True)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {os.path.basename(split_dir).upper()}")
            print(f"{'='*70}")
        
        stats = {
            'split': os.path.basename(split_dir),
            'successful': 0,
            'failed': 0,
            'total_objects': 0,
            'total_images': len(image_files),
            'failed_files': []
        }
        
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            
            # Determine mask file
            base_name = os.path.splitext(image_file)[0].split("_")[0]  # Remove suffix if present
            mask_file = base_name + mask_suffix
            mask_path = os.path.join(masks_dir, mask_file)
            
            # Output label file
            label_file = base_name + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            
            # Process image-mask pair
            success, num_objects, message = self.process_image_pair(image_path, mask_path, label_path)
            
            if success:
                stats['successful'] += 1
                stats['total_objects'] += num_objects
                if verbose and (idx % 10 == 0 or idx == len(image_files) - 1):
                    print(f"  [{idx+1:4d}/{len(image_files)}] {image_file:40s} → {num_objects:3d} objects")
            else:
                stats['failed'] += 1
                stats['failed_files'].append({'file': image_file, 'error': message})
                if verbose:
                    print(f"  ✗ [{idx+1:4d}/{len(image_files)}] {image_file}: {message}")
        
        if verbose:
            print(f"\n📊 Summary:")
            print(f"  ✓ Successful: {stats['successful']}/{len(image_files)}")
            print(f"  ✗ Failed: {stats['failed']}/{len(image_files)}")
            print(f"  📦 Total objects extracted: {stats['total_objects']}")
            print(f"  📁 Annotations: {labels_dir}")
        
        return stats
    
    def process_dataset(self, base_dir: str, splits: List[str] = ['train', 'test', 'val'], 
                       mask_suffix: str = "_mask.png", verbose: bool = True) -> Dict:
        """
        Process entire dataset (all splits)
        
        Args:
            base_dir (str): Base dataset directory
            splits (List[str]): List of split names to process
            mask_suffix (str): Suffix for mask files
            verbose (bool): Print progress
            
        Returns:
            Dict: Combined statistics for all splits
        """
        all_results = {}
        
        for split in splits:
            split_path = os.path.join(base_dir, split)
            if not os.path.exists(split_path):
                if verbose:
                    print(f"⚠ {split}/ not found, skipping...")
                continue
            
            result = self.process_split(split_path, mask_suffix, verbose)
            all_results[split] = result
        
        # Summary
        if verbose:
            print(f"\n{'='*70}")
            print("OVERALL SUMMARY")
            print(f"{'='*70}")
            total_successful = sum(r.get('successful', 0) for r in all_results.values())
            total_failed = sum(r.get('failed', 0) for r in all_results.values())
            total_objects = sum(r.get('total_objects', 0) for r in all_results.values())
            
            for split, result in all_results.items():
                print(f"{split.upper():6s}: {result['successful']:4d} ✓ | {result['failed']:4d} ✗ | {result.get('total_objects', 0):6d} objects")
            
            print(f"\n📊 Grand Total:")
            print(f"   Images: {total_successful + total_failed}")
            print(f"   Successful: {total_successful}")
            print(f"   Failed: {total_failed}")
            print(f"   Objects: {total_objects}")
        
        return all_results
    
    def visualize_sample(self, split_dir: str, num_samples: int = 2):
        """
        Display sample annotations from a split
        
        Args:
            split_dir (str): Path to dataset split
            num_samples (int): Number of samples to display
        """
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(labels_dir):
            print(f"No labels directory found at {labels_dir}")
            return
        
        label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        
        print(f"\n📋 Sample Annotations ({num_samples} files):")
        print(f"   {os.path.basename(split_dir).upper()}")
        
        for label_file in label_files[:num_samples]:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            print(f"\n   📄 {label_file}")
            if not lines:
                print(f"      (empty - no objects)")
            else:
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    class_name = self.class_names.get(class_id, '?')
                    print(f"      [{line_idx+1}] {class_name:20s} → center=({x_center:.3f},{y_center:.3f}) size=({width:.3f}×{height:.3f})")


# Example usage (for reference)
if __name__ == "__main__":
    # Load class metadata
    class_dict_path = r"C:\Users\JAGADEESH\Downloads\deep_globe_land_cover_dataset\class_dict.csv"
    class_metadata = pd.read_csv(class_dict_path)
    
    # Initialize pipeline
    pipeline = MaskAnnotationPipeline(class_metadata)
    
    # Process entire dataset
    base_dir = r"C:\Users\JAGADEESH\Downloads\deep_globe_land_cover_dataset"
    results = pipeline.process_dataset(base_dir)
    
    # View samples
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(base_dir, split)
        if os.path.exists(split_path):
            pipeline.visualize_sample(split_path, num_samples=2)
