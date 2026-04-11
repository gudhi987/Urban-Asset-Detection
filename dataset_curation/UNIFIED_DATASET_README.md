# Unified Dataset Pipeline

Complete system for sampling, converting, validating, and visualizing multiple geospatial datasets into a unified format.

## Overview

This pipeline converts 4 different datasets with different label formats and class hierarchies into a single unified dataset with 5 main classes:

- **Class 0**: building
- **Class 1**: road
- **Class 2**: water_body
- **Class 3**: vegetation
- **Class 4**: vehicle

## Dataset Sampling

Each dataset is sampled with specific limits to create a balanced unified dataset:

| Dataset | Source | Samples | Format | Classes → Required |
|---------|--------|---------|--------|-------------------|
| LandCover | `deep_globe_land_cover_dataset/train` | All | Masks (RGB) | 7 classes → 3 kept |
| Road | `deep_globe_road_extraction/train` | 1,000 | Masks (RGB) | 2 classes → 1 kept |
| DOTA | `dota/train` | All (~1,400) | YOLO Bboxes | 15 classes → 1 kept |
| Semantic Buildings | `semantic_buildings_in_aerial_imagery/train` | 2,000 | YOLO Bboxes | 1 class → 1 kept |

## Class Mappings

### LandCover Dataset
```
urban_land (0, 255, 255)       → FILTERED
agriculture_land (255, 255, 0) → Class 3 (vegetation)
rangeland (255, 0, 255)        → Class 3 (vegetation)
forest_land (0, 255, 0)        → Class 3 (vegetation)
water (0, 0, 255)              → Class 2 (water_body)
barren_land (255, 255, 255)    → FILTERED
unknown (0, 0, 0)              → FILTERED
```

### Road Extraction Dataset
```
road (255, 255, 255)    → Class 1 (road)
background (0, 0, 0)    → FILTERED
```

### DOTA Dataset
```
0: baseball-diamond      → FILTERED
1: basketball-court      → FILTERED
2: bridge                → FILTERED
3: ground-track-field    → FILTERED
4: harbor                → FILTERED
5: helicopter            → Class 4 (vehicle)
6: large-vehicle         → Class 4 (vehicle)
7: plane                 → Class 4 (vehicle)
8: roundabout            → FILTERED
9: ship                  → Class 4 (vehicle)
10: small-vehicle        → Class 4 (vehicle)
11: soccer-ball-field    → FILTERED
12: storage-tank         → FILTERED
13: swimming-pool        → FILTERED
14: tennis-court         → FILTERED
```

### Semantic Buildings Dataset
```
0: building → Class 0 (building)
```

## Pipeline Scripts

### 1. `class_mapping_config.py`
Configuration file defining all class mappings. Edit this to change mappings.

```python
from class_mapping_config import LANDCOVER_MAPPING, ROAD_MAPPING, DOTA_MAPPING, SEMANTIC_BUILDINGS_MAPPING
```

### 2. `sample_and_convert_datasets.py`
**Purpose**: Sample images and convert labels to unified format

**Features**:
- Samples images from each dataset with configurable limits
- Converts masks → YOLO format for LandCover and Road datasets
- Maps class IDs according to configuration
- Filters out unmapped classes
- Generates temporary statistics per dataset
- Creates folder structure: `<dataset>/images/`, `<dataset>/labels/`

**Usage**:
```bash
python sample_and_convert_datasets.py \
  --output-dir unified_dataset \
  --landcover-dir ../../datasets/deep_globe_land_cover_dataset \
  --road-dir ../../datasets/deep_globe_road_extraction \
  --dota-dir ../../datasets/dota \
  --semantic-dir "../../datasets/semantic_buildings_in_aerial_imagery"
```

**Output**:
```
unified_dataset/
├── landcover/
│   ├── images/           (all LandCover images)
│   ├── labels/           (converted YOLO labels)
│   └── conversion_stats.txt
├── road_extraction/
│   ├── images/           (1000 sampled road images)
│   ├── labels/           (converted YOLO labels)
│   └── conversion_stats.txt
├── dota/
│   ├── images/           (all DOTA images)
│   ├── labels/           (remapped YOLO labels)
│   └── conversion_stats.txt
└── semantic_buildings/
    ├── images/           (2000 sampled images)
    ├── labels/           (class remapped)
    └── conversion_stats.txt
```

### 3. `validate_unified_dataset.py`
**Purpose**: Validate converted dataset quality

**Validates**:
- Label file format and structure
- Coordinate normalization (must be 0-1 range)
- Class ID validity (must be 0-4)
- Image-label correspondence
- Parsing errors and anomalies

**Usage**:
```bash
python validate_unified_dataset.py --unified-dir unified_dataset
```

**Output**:
- Per-dataset statistics (images, labels, objects)
- Per-class distribution analysis
- Error report with line counts
- Overall dataset health check

### 4. `visualize_unified_dataset.py`
**Purpose**: Visualize sample images with unified labels

**Shows**:
- Original image + image with bounding boxes
- One sample per dataset
- Colored class labels on bboxes
- Object counts per image

**Usage**:
```bash
python visualize_unified_dataset.py \
  --unified-dir unified_dataset \
  --num-samples 1
```

### 5. `unified_dataset_pipeline.py`
**Purpose**: Run complete pipeline automatically

**Steps**:
1. Sample and convert all datasets
2. Validate results
3. Visualize outputs

**Usage**:
```bash
# Full pipeline
python unified_dataset_pipeline.py --unified-dir unified_dataset

# Skip specific steps
python unified_dataset_pipeline.py \
  --unified-dir unified_dataset \
  --skip-conversion        # Use existing conversions
  
# Custom data paths
python unified_dataset_pipeline.py \
  --unified-dir my_dataset \
  --landcover-dir /custom/path/landcover \
  --road-dir /custom/path/road
```

## YOLO Format

All outputs are in standard YOLO v5 format:

```
class_id center_x center_y width height
```

Where:
- `class_id`: 0-4 (unified classes)
- `center_x`, `center_y`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized box dimensions (0-1)

One line per object. Multiple objects per image supported.

## Workflow

### Quick Start
```bash
# Full pipeline with defaults
python unified_dataset_pipeline.py

# Output: unified_dataset/ directory with all data
```

### Step by Step

```bash
# 1. Sample and convert
python sample_and_convert_datasets.py --output-dir my_unified

# 2. Validate quality
python validate_unified_dataset.py --unified-dir my_unified

# 3. Visualize results
python visualize_unified_dataset.py --unified-dir my_unified --num-samples 1
```

### Troubleshooting

**"LandCover images dir not found"**
- Check path: `datasets/deep_globe_land_cover_dataset/train/images/`

**"Validation errors in coordinate ranges"**
- Some original labels may be malformed
- Check conversion stats for filtered object counts

**"No images found in dataset"**
- Dataset path incorrect or folder structure different
- Use `--landcover-dir`, `--road-dir`, etc. to specify custom paths

## Output Statistics

After conversion, check conversion statistics:

```bash
cat unified_dataset/landcover/conversion_stats.txt
cat unified_dataset/road_extraction/conversion_stats.txt
# ... etc for each dataset
```

Example output:
```
Dataset: landcover
==================================================
Total images sampled: 674
Images with valid labels: 674
Total objects detected: 2847
Filtered objects: 412

Class Distribution:
  Class 0: 0
  Class 1: 0
  Class 2: 456
  Class 3: 2391
  Class 4: 0
```

## Customization

### Changing Class Mappings

Edit `class_mapping_config.py`:

```python
# Example: Map building to vehicle
SEMANTIC_BUILDINGS_MAPPING = {
    0: 4  # building → vehicle (was: 0)
}
```

Then re-run conversion:
```bash
python sample_and_convert_datasets.py --output-dir unified_dataset
```

### Changing Sample Limits

In `sample_and_convert_datasets.py`, modify converter calls:

```python
# Change road limit from 1000 to 500
rd.convert(args.road_dir, required_class_dict, limit=500)

# Change semantic buildings limit from 2000 to 3000
sb.convert(args.semantic_dir, limit=3000)
```

## Next Steps

After creating the unified dataset:

1. **Split dataset**: Use existing split scripts
   ```bash
   cd ../dota_curation
   python split_dota_dataset.py --dataset-dir ../unified_dataset
   ```

2. **Train model**: Use with YOLOv5/v8
   ```bash
   yolo detect train data=data.yaml model=yolov8m.pt epochs=100
   ```

3. **Evaluate**: Benchmark on converted datasets

## Quality Checklist

- [ ] All conversions complete without errors
- [ ] Validation shows 0 errors
- [ ] Class distribution is balanced
- [ ] Visualization shows correct labels
- [ ] No images missing labels
- [ ] Coordinates all in 0-1 range

## Performance Notes

- **LandCover**: Mask → YOLO conversion takes ~2-5 minutes (color detection)
- **Road extraction**: 1000 image sampling takes ~1 minute
- **DOTA**: 1400 images process in ~30 seconds
- **Semantic Buildings**: 2000 image sampling takes ~1 minute

Total runtime: ~5-10 minutes for full pipeline

## See Also

- `class_mapping_config.py` - Mapping configuration
- `required_class_dict.csv` - Required unified classes
- Dataset-specific folders in `dataset_curation/`
