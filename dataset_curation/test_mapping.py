"""Test the mapping"""
from class_mapping_config import LANDCOVER_MAPPING

print("LANDCOVER_MAPPING:", LANDCOVER_MAPPING)
print()

# Test specific mappings
test_classes = [0, 1, 2, 3, 4, 5, 6]
for class_id in test_classes:
    mapped = LANDCOVER_MAPPING.get(class_id, None)
    print(f"Class {class_id} → {mapped}")
