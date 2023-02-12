"""
@file movie2gif.py
@brief 動画ファイルからgifデータに変更
"""
import argparse
import os

from moviepy.editor import VideoFileClip



def main(input_path: str, save_path: str) -> None:
    clip = VideoFileClip(input_path)
    # 動画をGIFアニメに変換
    clip.write_gif(save_path)
    clip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        type=str, default="RegisteredTracker/IMG_2468.mp4",
                        help="Dir in HEIC images")                    
    parser.add_argument("-o", "--output_dir",
                        type=str, default="Sample/",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir,
            f"{os.path.splitext(os.path.basename(args.input))[0]}.gif")
    main(args.input, save_path)