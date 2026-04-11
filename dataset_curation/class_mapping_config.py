"""
Mapping configuration for converting dataset-specific labels to required unified labels.

This file defines how to map from each dataset's original class IDs to the 
required unified class dictionary.

Required classes:
- 0: building
- 1: road
- 2: water_body
- 3: vegetation
- 4: vehicle

All datasets use class ID indices (0-based) for consistent mapping.
"""

# LandCover Dataset (mask-based)
# Class index → required class id (based on class_dict.csv order)
# 0: urban_land, 1: agriculture_land, 2: rangeland, 3: forest_land, 4: water, 5: barren_land, 6: unknown
LANDCOVER_MAPPING = {
    0: None,                        # urban_land → filter
    1: 3,                          # agriculture_land → vegetation
    2: 3,                          # rangeland → vegetation
    3: 3,                          # forest_land → vegetation
    4: 2,                          # water → water_body
    5: None,                       # barren_land → filter
    6: None                        # unknown → filter
}

# Road Extraction Dataset (mask-based)
# Class index → required class id (based on class_dict.csv order)
# 0: road, 1: background
ROAD_MAPPING = {
    0: 1,                          # road → road
    1: None                        # background → filter
}

# DOTA Dataset (bounding boxes, class index mapping)
# Map from original class_id to required class id, or None to filter
DOTA_MAPPING = {
    0: None,                        # baseball-diamond → filter
    1: None,                        # basketball-court → filter
    2: None,                        # bridge → filter
    3: None,                        # ground-track-field → filter
    4: None,                        # harbor → filter
    5: 4,                          # helicopter → vehicle
    6: 4,                          # large-vehicle → vehicle
    7: 4,                          # plane → vehicle
    8: None,                        # roundabout → filter
    9: 4,                          # ship → vehicle
    10: 4,                         # small-vehicle → vehicle
    11: None,                      # soccer-ball-field → filter
    12: None,                      # storage-tank → filter
    13: None,                      # swimming-pool → filter
    14: None                       # tennis-court → filter
}

# Semantic Buildings Dataset (bounding boxes, class index mapping)
# Map from original class_id to required class id, or None to filter
SEMANTIC_BUILDINGS_MAPPING = {
    0: 0                           # building → building
}

# Summary of mappings
DATASET_MAPPINGS = {
    'landcover': LANDCOVER_MAPPING,
    'road': ROAD_MAPPING,
    'dota': DOTA_MAPPING,
    'semantic_buildings': SEMANTIC_BUILDINGS_MAPPING
}
