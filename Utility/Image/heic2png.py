"""
@file heic2png.py
@brief heic拡張子の画像データをpngファイルに変換する
"""
import argparse
import glob
import os
import pyheif
from PIL import Image

def convert(path: str) -> Image:
    """拡張子を変更する関数

    Args:
        path (str): 拡張子変更前の画像データのパス

    Returns:
        Image: 変換後の画像データ
    """
    heifImg = pyheif.read(path)
    img = Image.frombytes(
        heifImg.mode,
        heifImg.size,
        heifImg.data,
        "raw",
        heifImg.mode,
        heifImg.stride
    )
    return img

if __name__ == "__main__":
    targetExt = "png"
    resolution = (1512, 1134)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputDir", type=str, help="Dir in HEIC images")
    parser.add_argument("-o", "--outputDir", type=str, help="Output directory")
    args = parser.parse_args()

    heicPathList = glob.glob(os.path.join(args.inputDir, "**/*.HEIC"), recursive=True)
    
    for path in heicPathList:
        originalImg = convert(path)
        resizedImg = originalImg.resize(resolution)
        os.makedirs(args.outputDir, exist_ok=True)
        savePath = os.path.join(args.outputDir, os.path.splitext(os.path.basename(path))[0]) + f".{targetExt}"
        resizedImg.save(savePath, targetExt.upper())