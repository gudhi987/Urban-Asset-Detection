# DOTA Dataset Processing - Quick Start Guide

A complete collection of Python scripts for processing DOTA (A Large-scale Object Detection in Aerial Images) dataset from OBB (Oriented Bounding Box) format to YOLO format.

## 📦 What's Included

```
dataset_curation/dota_curation/
├── setup_dota_pipeline.py          # ⭐ Main entry point (run this!)
├── extract_classes_dota.py         # Extract unique classes
├── convert_obb_to_yolo.py          # Convert OBB to YOLO format
├── verify_yolo_labels.py           # Verify conversion quality
├── analyze_dota_dataset.py         # Dataset statistics
├── visualize_dota_bboxes.py        # Visualize images with bboxes
├── README.md                       # Detailed documentation
└── QUICKSTART.md                   # This file
```

## 🚀 Quick Start (Easiest Way)

### Option 1: Automatic Setup (Recommended)

Run the complete pipeline with one command:

```bash
cd dataset_curation/dota_curation
python setup_dota_pipeline.py
```

This automatically:
1. Extracts class dictionary (if needed)
2. Converts all label files from OBB to YOLO format
3. Verifies conversions
4. Analyzes dataset statistics

### Option 2: Manual Steps

If you prefer to run steps individually:

#### Step 1: Extract Classes
```bash
python extract_classes_dota.py
```

#### Step 2: Convert OBB to YOLO
```bash
python convert_obb_to_yolo.py
```

#### Step 3: Verify Conversion
```bash
python verify_yolo_labels.py
```

#### Step 4: Analyze Dataset
```bash
python analyze_dota_dataset.py
```

#### Step 5: Visualize Results
```bash
python visualize_dota_bboxes.py
```

## ⚙️ Custom Configuration

### Using Different Paths

```bash
python setup_dota_pipeline.py \
    --dataset-dir /path/to/dota \
    --split train
```

### Process Multiple Splits

```bash
python setup_dota_pipeline.py --split all
```

This processes: train, valid, and test splits

### Skip Optional Steps

```bash
python setup_dota_pipeline.py --skip-analysis --skip-visualization
```

## 🔧 Advanced Usage

### Extract Classes Only

```bash
python extract_classes_dota.py \
    --input-dir datasets/dota/train/trainset_reclabelTxt \
    --output-file datasets/dota/class_dict.csv
```

### Convert with Custom Class Dictionary

```bash
python convert_obb_to_yolo.py \
    --input-labels datasets/dota/train/trainset_reclabelTxt \
    --output-labels datasets/dota/train/labels \
    --images-dir datasets/dota/train/images \
    --class-dict datasets/dota/class_dict.csv
```

### Verify Labels with Details

```bash
python verify_yolo_labels.py \
    --labels-dir datasets/dota/train/labels \
    --images-dir datasets/dota/train/images \
    --class-dict datasets/dota/class_dict.csv
```

### Analyze Valid Split

```bash
python analyze_dota_dataset.py \
    --dataset-dir datasets/dota \
    --split valid \
    --class-dict datasets/dota/class_dict.csv
```

## 📊 What Gets Created

After running the pipeline, you'll have:

```
datasets/dota/
├── class_dict.csv                          # ✨ NEW
├── train/
│   ├── labels/                             # ✨ NEW (YOLO format)
│   │   ├── P0000.txt
│   │   ├── P0001.txt
│   │   └── ...
│   ├── trainset_reclabelTxt/               # Original OBB format
│   │   ├── P0000.txt
│   │   └── ...
│   └── images/
├── valid/
│   ├── labels/                             # ✨ NEW (if processed)
│   └── ...
└── test/
    └── images/
```

### Sample YOLO Format Output

Original OBB format:
```
2753.0 2385.0 2888.0 2385.0 2888.0 2502.0 2753.0 2502.0 plane 0
```

Converted YOLO format:
```
7 0.727871 0.444111 0.034839 0.021265
```

Breakdown:
- `7` = class_id (plane, from class_dict.csv)
- `0.727871` = normalized x_center (0-1)
- `0.444111` = normalized y_center (0-1)
- `0.034839` = normalized width (0-1)
- `0.021265` = normalized height (0-1)

## 📈 Example Output

Running the pipeline produces reports like:

```
==================================================
Conversion completed!
Total files processed: 1411
Successfully converted: 1411
Errors: 0
Output directory: datasets/dota/train/labels
==================================================

==================================================
YOLO Label Verification Report
==================================================

File Statistics:
  Total files:           1411
  Files with labels:     1411
  Files without labels:  0

Object Statistics:
  Total objects:         98990
  Avg objects per file:  70.13

Class Distribution:
  plane               : 16221 ███████████ (16.39%)
  large-vehicle       : 13462 ██████████  (13.60%)
  small-vehicle       :  9803 ███████     (9.90%)
  ...
```

## 🐍 Using Scripts in Python Code

Import and use functions directly in your code:

```python
from convert_obb_to_yolo import load_class_mapping, convert_obb_to_yolo
from analyze_dota_dataset import analyze_dataset
from visualize_dota_bboxes import visualize_dota_samples_with_bboxes

# Load classes
class_to_idx = load_class_mapping('datasets/dota/class_dict.csv')

# Convert coordinates
x_c, y_c, w, h = convert_obb_to_yolo(
    2753.0, 2385.0, 2888.0, 2385.0, 
    2888.0, 2502.0, 2753.0, 2502.0,
    img_width=3875, img_height=5502
)

# Visualize
visualize_dota_samples_with_bboxes(
    dataset_dir='datasets/dota',
    num_samples=4,
    split='train'
)
```

## 🐛 Troubleshooting

### Error: "Class dictionary not found"
Run Step 1 first:
```bash
python extract_classes_dota.py
```

### Error: "Image not found"
Verify your images directory path matches the label files naming (e.g., P0000.txt ↔ P0000.png)

### Error: "Out of bounds coordinates"
The scripts automatically clamp values. This is usually safe. Check the verification report for details.

### Memory Issues with Large Datasets
The scripts process files sequentially, so memory usage is minimal. If issues persist, process splits separately.

### Permissions Error
Ensure the output directory is writable:
```bash
chmod -R u+w datasets/dota/
```

## 📝 Requirements

- Python 3.7 or higher
- Pillow (for image processing)
- matplotlib (for visualization)

Install requirements:
```bash
pip install Pillow matplotlib
```

## 🎯 Typical Workflow

1. **First Time Setup**
   ```bash
   python setup_dota_pipeline.py
   ```

2. **Verify Everything Works**
   - Check output in `datasets/dota/train/labels/`
   - Review verification report

3. **Visualize Sample Data**
   ```bash
   python visualize_dota_bboxes.py
   ```

4. **Use in Your Project**
   - Load YOLO format labels with your training framework
   - Use `class_dict.csv` for class name mapping

## 📚 DOTA 15 Classes

```
0:  baseball-diamond
1:  basketball-court
2:  bridge
3:  ground-track-field
4:  harbor
5:  helicopter
6:  large-vehicle
7:  plane
8:  roundabout
9:  ship
10: small-vehicle
11: soccer-ball-field
12: storage-tank
13: swimming-pool
14: tennis-court
```

## 🔗 Related Resources

- [DOTA Dataset Paper](https://arxiv.org/abs/1711.10398)
- [YOLO Format Documentation](https://docs.ultralytics.com/datasets/detect/)
- [Oriented Bounding Box Concepts](https://en.wikipedia.org/wiki/Minimum_bounding_rectangle)

## 💡 Tips & Tricks

- **Batch Processing**: Use `--split all` to process all splits at once
- **Progress Tracking**: Watch terminal output for file processing progress
- **Dry Run**: Run verification script to check quality without modifying data
- **Custom Classes**: Manually edit `class_dict.csv` if needed
- **Integration**: Use converted labels with YOLO v5/v8 directly

## ❓ Common Questions

**Q: Can I use the original OBB labels with YOLO models?**
A: No, YOLO models expect normalized center-based coordinates. Use these scripts to convert.

**Q: How are oriented boxes converted to axis-aligned boxes?**
A: By finding the min/max coordinates across all 4 corners of the OBB.

**Q: Are "difficult" annotations preserved?**
A: Currently no - all objects are included. You can modify the conversion script if needed.

**Q: Can I process the dataset incrementally?**
A: Yes, run scripts for specific splits or re-run conversion for updated labels.

**Q: What if image dimensions are very different?**
A: Normalization handles any image size correctly (0-1 range).

## 📞 Support

For issues or questions:
1. Check the detailed [README.md](README.md)
2. Review script docstrings
3. Check console output for specific error messages
4. Verify file paths and directory structure

---

**Last Updated**: April 2026
**Version**: 1.0
**Status**: Production Ready ✓

