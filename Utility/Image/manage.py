"""
@file manage.py
@brief 画像データの処理関係
"""
import cv2
import numpy as np
from PIL import Image

def cv2_2_pillow(cv2_image):
    rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return pil_img

def pillow_2_cv2(pil_image):
    rgb_image = np.asarray(pil_image)
    cv2_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return cv2_image