# DOTA Dataset Processing Scripts

This collection contains Python scripts for processing and analyzing the DOTA (A Large-scale Object Detection in Aerial Images) dataset, specifically for converting from OBB (Oriented Bounding Box) format to YOLO format.

## Scripts Overview

### 1. `extract_classes_dota.py`
Extracts unique classes from DOTA labels and creates a class dictionary CSV file.

**Purpose**: Scan all label files and identify all unique object classes in the dataset.

**Output**: `class_dict.csv` with format:
```
idx,name
0,baseball-diamond
1,basketball-court
...
14,tennis-court
```

**Usage**:
```bash
python extract_classes_dota.py \
    --input-dir datasets/dota/train/trainset_reclabelTxt \
    --output-file datasets/dota/class_dict.csv
```

**Arguments**:
- `--input-dir`: Path to OBB format label files (default: `datasets/dota/train/trainset_reclabelTxt`)
- `--output-file`: Path to output class dictionary CSV (default: `datasets/dota/class_dict.csv`)

---

### 2. `convert_obb_to_yolo.py`
Converts DOTA labels from OBB format to YOLO format.

**Purpose**: Convert oriented bounding box coordinates (8 corner points) to YOLO normalized center-based format.

**Input Format** (DOTA OBB):
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```
- 4 corner points in clockwise order (pixel coordinates)
- class_name: string identifier
- difficult: 0 or 1 flag

**Output Format** (YOLO):
```
class_id x_center y_center width height
```
- class_id: index from class dictionary
- All coordinates normalized to 0-1 range
- class_id, x_center, y_center, width, height

**Usage**:
```bash
python convert_obb_to_yolo.py \
    --input-labels datasets/dota/train/trainset_reclabelTxt \
    --output-labels datasets/dota/train/labels \
    --images-dir datasets/dota/train/images \
    --class-dict datasets/dota/class_dict.csv
```

**Arguments**:
- `--input-labels`: Input OBB format labels directory (default: `datasets/dota/train/trainset_reclabelTxt`)
- `--output-labels`: Output YOLO format labels directory (default: `datasets/dota/train/labels`)
- `--images-dir`: Images directory for dimension lookup (default: `datasets/dota/train/images`)
- `--class-dict`: Class dictionary CSV file (default: `datasets/dota/class_dict.csv`)

**Output**: One `.txt` file per image in YOLO format, placed in `--output-labels` directory

---

### 3. `verify_yolo_labels.py`
Validates and analyzes YOLO format labels.

**Purpose**: Quality check on converted labels, verify normalization, and get statistics.

**Checks Performed**:
- Verify coordinate normalization (0-1 range)
- Check for missing images or labels
- Validate class IDs
- Calculate bounding box statistics

**Usage**:
```bash
python verify_yolo_labels.py \
    --labels-dir datasets/dota/train/labels \
    --images-dir datasets/dota/train/images \
    --class-dict datasets/dota/class_dict.csv
```

**Arguments**:
- `--labels-dir`: YOLO format labels directory (default: `datasets/dota/train/labels`)
- `--images-dir`: Images directory (default: `datasets/dota/train/images`)
- `--class-dict`: Class dictionary CSV file (default: `datasets/dota/class_dict.csv`)

**Output**: 
- Detailed verification report with statistics
- Warnings for any issues found
- Class distribution summary

---

### 4. `analyze_dota_dataset.py`
Provides comprehensive statistics about the DOTA dataset.

**Purpose**: Understand dataset composition including image statistics and class distribution.

**Statistics Provided**:
- Image dimensions (width, height, area) - min/max/avg
- Total images and labeled images
- Objects per image statistics
- Class distribution across dataset
- Potential data issues

**Usage**:
```bash
python analyze_dota_dataset.py \
    --dataset-dir datasets/dota \
    --split train \
    --class-dict datasets/dota/class_dict.csv
```

**Arguments**:
- `--dataset-dir`: DOTA dataset root directory (default: `datasets/dota`)
- `--split`: Dataset split - 'train', 'valid', or 'test' (default: `train`)
- `--class-dict`: Class dictionary CSV file (default: `datasets/dota/class_dict.csv`)

**Output**: Formatted analysis report with visualizations

---

## Typical Workflow

### Step 1: Extract Classes (Optional, if class_dict.csv doesn't exist)
```bash
python extract_classes_dota.py \
    --input-dir datasets/dota/train/trainset_reclabelTxt \
    --output-file datasets/dota/class_dict.csv
```

### Step 2: Convert OBB to YOLO
```bash
python convert_obb_to_yolo.py \
    --input-labels datasets/dota/train/trainset_reclabelTxt \
    --output-labels datasets/dota/train/labels \
    --images-dir datasets/dota/train/images \
    --class-dict datasets/dota/class_dict.csv
```

### Step 3: Verify Conversion
```bash
python verify_yolo_labels.py \
    --labels-dir datasets/dota/train/labels \
    --images-dir datasets/dota/train/images \
    --class-dict datasets/dota/class_dict.csv
```

### Step 4: Analyze Dataset (Optional)
```bash
python analyze_dota_dataset.py \
    --dataset-dir datasets/dota \
    --split train \
    --class-dict datasets/dota/class_dict.csv
```

---

## DOTA Dataset Structure

Expected directory structure:
```
datasets/dota/
├── class_dict.csv                    # Class index to name mapping
├── train/
│   ├── images/                       # Training images (PNG/JPG)
│   │   ├── P0000.png
│   │   ├── P0001.png
│   │   └── ...
│   ├── labels/                       # Output YOLO format labels
│   │   ├── P0000.txt
│   │   ├── P0001.txt
│   │   └── ...
│   └── trainset_reclabelTxt/         # Input OBB format labels
│       ├── P0000.txt
│       ├── P0001.txt
│       └── ...
├── valid/
│   ├── images/
│   ├── labels/
│   └── ...
└── test/
    ├── images/
    └── ...
```

---

## DOTA Classes (15 Total)

| ID  | Class Name          |
|-----|-------------------|
| 0   | baseball-diamond  |
| 1   | basketball-court  |
| 2   | bridge            |
| 3   | ground-track-field|
| 4   | harbor            |
| 5   | helicopter        |
| 6   | large-vehicle     |
| 7   | plane             |
| 8   | roundabout        |
| 9   | ship              |
| 10  | small-vehicle     |
| 11  | soccer-ball-field |
| 12  | storage-tank      |
| 13  | swimming-pool     |
| 14  | tennis-court      |

---

## Error Handling

All scripts include comprehensive error handling:

- **Missing files**: Scripts will warn and skip files
- **Invalid coordinates**: Out-of-bounds values are clamped to 0-1
- **Corrupt images**: Files with reading errors are skipped with warning
- **Invalid labels**: Lines that can't be parsed are reported

Check script output for warnings and errors.

---

## Requirements

- Python 3.7+
- Pillow (for image processing)

Install dependencies:
```bash
pip install Pillow
```

---

## Notes

1. **Coordinate Conversion**: OBB oriented bounding boxes are converted to axis-aligned bounding boxes by finding the min/max coordinates across the 4 corners.

2. **Normalization**: All YOLO coordinates are normalized to 0-1 range based on image dimensions.

3. **Class Mapping**: Automatically handles class name to index mapping using the class dictionary CSV.

4. **Image Formats**: Scripts support PNG and JPG image formats.

5. **Parallelization**: For very large datasets, consider implementing parallelization in the conversion script.

---

## Future Enhancements

Possible improvements:
- Multi-processing support for faster conversion
- Support for rotated bounding boxes in YOLO format
- Data validation and cleaning tools
- Visualization tools for inspection
- Split dataset generation (train/val/test)

