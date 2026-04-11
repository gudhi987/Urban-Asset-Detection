"""Debug road converter path and filtering"""
from pathlib import Path

# Simulate the paths as the converter would see them
dataset_dir = Path("../datasets/deep_globe_road_extraction")
print(f"Dataset dir: {dataset_dir}")
print(f"Dataset dir exists: {dataset_dir.exists()}")
print(f"Dataset dir absolute: {dataset_dir.resolve()}")

images_dir = dataset_dir / "train" / "images"
labels_dir = dataset_dir / "train" / "labels"

print(f"\nImages dir: {images_dir}")
print(f"Images dir exists: {images_dir.exists()}")
print(f"Images dir absolute: {images_dir.resolve()}")

print(f"\nLabels dir: {labels_dir}")
print(f"Labels dir exists: {labels_dir.exists()}")
print(f"Labels dir absolute: {labels_dir.resolve()}")

# Check a sample file
if labels_dir.exists():
    label_files = list(labels_dir.glob("*.txt"))
    print(f"\nTotal label files: {len(label_files)}")
    if label_files:
        sample = label_files[0]
        print(f"\nSample file: {sample.name}")
        print(f"Sample exists: {sample.exists()}")
        with open(sample) as f:
            lines = f.readlines()
        print(f"Lines in sample: {len(lines)}")
        print(f"First line: {lines[0].strip()}")
        
        # Test mapping
        from class_mapping_config import ROAD_MAPPING
        print(f"\nROAD_MAPPING: {ROAD_MAPPING}")
        
        kept = 0
        filtered = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            mapped = ROAD_MAPPING.get(class_id, None)
            if mapped is not None:
                kept += 1
            else:
                filtered += 1
        
        print(f"In sample file {sample.name}:")
        print(f"  Kept (class 0 -> 1): {kept}")
        print(f"  Filtered (class 1 -> None): {filtered}")
