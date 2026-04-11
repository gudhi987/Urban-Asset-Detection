"""Quick test to read a single label file"""
from pathlib import Path

# Test paths
label_path = Path("../datasets/deep_globe_land_cover_dataset/train/labels/100694.txt")
print(f"Testing path: {label_path}")
print(f"Absolute path: {label_path.resolve()}")
print(f"Exists: {label_path.exists()}")

if label_path.exists():
    with open(label_path, 'r') as f:
        lines = f.readlines()
    print(f"Lines in file: {len(lines)}")
    if lines:
        print(f"First line: {lines[0]}")
        parts = lines[0].strip().split()
        print(f"Parts: {parts}")
        if len(parts) >= 5:
            class_id = int(float(parts[0]))
            print(f"Class ID: {class_id}")
