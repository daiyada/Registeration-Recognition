"""
@file change_resolution.py
@brief 動画の解像度を変更する
"""
import argparse
import glob
import os

from tqdm import tqdm

import cv2

def change_resolution(video_path: str, resolution: tuple, save_path: str) -> None:
    """解像度を変更する

    Args:
        video_path (str): 解像度を変更する動画のパス
        resolution (tuple): 解像度 ( width, height )
        save_path (str): 保存先のパス
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, resolution)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resolution)
        writer.write(frame)

    writer.release()
    cap.release()

if __name__ == "__main__":
    target_ext = "MOV"
    resolution = (1920, 1080)
    ext_after_change = "mp4"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",
                        type=str, default="../../TestData/Movie_",
                        help="Dir in HEIC images")
    parser.add_argument("-o", "--output_dir",
                        type=str, default="../../TestData/Movie",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_path_list = glob.glob(os.path.join(args.input_dir, "**",
                                f"*.{target_ext}"),
                                recursive=True)
    for path in tqdm(video_path_list):
        save_path = os.path.join(args.output_dir,
                f"{os.path.splitext(os.path.basename(path))[0]}.{ext_after_change}")
        change_resolution(path, resolution, save_path)