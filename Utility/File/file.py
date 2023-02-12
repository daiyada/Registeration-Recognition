"""
@file file.py
@brief ファイル操作を扱う関数・クラスを扱う
"""
import glob
import os
import sys

def get_file_path_list(dir_path: str, ext_list: list = ["png"], recursive : bool = True) -> list:
    """指定のディレクトリ下の指定の拡張子のファイルパスを取得する

    Args:
        die_path (str): ディレクトリのパス
        ext (str):  取得するファイルの拡張子 
                    初期値は "png"
        recursive  (bool):  ファイルを再帰的に取得するか
                            初期値は True

    Returns:
        ret_list: ファイルパスを格納したリスト
    """
    ret_list = []
    for ext in ext_list:
        if recursive:
            img_path = os.path.join(dir_path, "**", f"*.{ext}")
        else:
            img_path = os.path.join(dir_path, f"*.{ext}")
        img_list = glob.glob(img_path, recursive=recursive)
        ret_list.extend(img_list)
    ret_list = sorted(ret_list)
    if not len(ret_list):
        raise FileNotFoundError(
            f"[{__file__}][{sys._getframe().f_code.co_name}]\nファイルが見つかりません [指定のディレクトリ:{dir_path}] [指定の拡張子:{ext_list}]"
            )
    return ret_list