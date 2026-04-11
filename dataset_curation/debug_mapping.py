"""Debug script to check label mapping"""
from pathlib import Path
from collections import Counter
from class_mapping_config import LANDCOVER_MAPPING, ROAD_MAPPING, DOTA_MAPPING

# Check LandCover labels
print("=" * 70)
print("LANDCOVER DATASET DEBUG")
print("=" * 70)

lc_labels_dir = Path("c:/Users/JAGADEESH/Documents/Urban-Asset-Detection/datasets/deep_globe_land_cover_dataset/train/labels")
if not lc_labels_dir.exists():
    print(f"✗ Path not found: {lc_labels_dir}")
else:
    class_ids_found = Counter()
    total_objects = 0
    sample_file = None
    
    for label_file in list(lc_labels_dir.iterdir())[:5]:  # Check first 5 files
        print(f"\n{label_file.name}:")
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not sample_file:
            sample_file = label_file
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                class_ids_found[class_id] += 1
                total_objects += 1
    
    print(f"\n\nUnique class IDs found: {dict(class_ids_found)}")
    print(f"Total objects in sample: {total_objects}")
    print(f"\nMapping for found classes:")
    for class_id, count in sorted(class_ids_found.items()):
        mapped = LANDCOVER_MAPPING.get(class_id, None)
        status = "✓ KEEP" if mapped is not None else "✗ FILTER"
        print(f"  Class {class_id}: {count} objects → maps to {mapped} {status}")

# Check DOTA
print(f"\n\n{'=' * 70}")
print("DOTA DATASET DEBUG")
print("=" * 70)

dota_labels_dir = Path("c:/Users/JAGADEESH/Documents/Urban-Asset-Detection/datasets/dota/train/labels")
if not dota_labels_dir.exists():
    print(f"✗ Path not found: {dota_labels_dir}")
else:
    class_ids_found = Counter()
    total_objects = 0
    
    for label_file in list(dota_labels_dir.iterdir())[:5]:  # Check first 5 files
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                class_ids_found[class_id] += 1
                total_objects += 1
    
    print(f"Unique class IDs found: {dict(class_ids_found)}")
    print(f"Total objects in sample: {total_objects}")
    print(f"\nMapping for found classes:")
    for class_id, count in sorted(class_ids_found.items()):
        mapped = DOTA_MAPPING.get(class_id, None)
        status = "✓ KEEP" if mapped is not None else "✗ FILTER"
        print(f"  Class {class_id}: {count} objects → maps to {mapped} {status}")
