"""
@file target_class.py
@brief 検出するクラスを設定
@reference Library/YOLOX/yolox/data/datasets/coco_classes.py
"""
import numpy as np

TARGET_CLASSES = (
    "person"
)

COLORS = np.array(
    [
        0.000, 0.447, 0.741,
    ]
).astype(np.float32).reshape(-1, 3)
