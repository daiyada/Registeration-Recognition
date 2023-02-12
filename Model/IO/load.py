"""
@file load.py
@brief ファイル読み込み用のスクリプト
"""
from abc import ABCMeta, abstractmethod
import json
import os
import re
from typing import Optional, Union

import torch
import yaml

class SettingsLoader(metaclass=ABCMeta):

    def _read_extension(self, cls_name: str) -> str:
        """クラス名から拡張子の読み取り

        Args:
            str: cls_name クラス名
        Returns:
            str: ext 拡張子 ( 文字列は全て小文字 ) 
        """
        ext = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', cls_name)).split()[0]
        ext = ext.lower()
        return ext

    @abstractmethod
    def load(self) -> None:
        pass


class YamlSettingsLoader(SettingsLoader):

    def __init__(self, dir: str, file_name: str) -> None:
        """Constructor

        Args:
            dir (str): 設定ファイルが格納してあるディレクトリ  
            file_name (str): 設定ファイル名 (拡張子なし)
        """
        super().__init__()
        self.__path = os.path.join(dir, file_name) + "." \
                    + self._read_extension(str(__class__.__name__))
        self.load()

    def load(self) -> None:
        """Yamlファイルをロードする
        """
        with open(self.__path, mode="r", encoding="utf-8") as f:
            self._settings = yaml.load(f, Loader=yaml.SafeLoader)

class DetectionYamlSettingsLoader(YamlSettingsLoader):

    @property
    def demo(self) -> str:
        return self._settings["demo"]

    @property
    def experiment_name(self) -> Optional[str]:
        return self._settings["experiment_name"]

    @experiment_name.setter
    def experiment_name(self, experiment_name) -> Optional[str]:
        self._settings["experiment_name"] = experiment_name

    @property
    def data_path(self) -> str:
        return self._settings["data_path"]

    @property
    def name(self) -> Optional[str]:
        return self._settings["name"]

    @property
    def camid(self) -> Optional[int]:
        return self._settings["camid"]

    @property
    def save_result(self) -> bool:
        return self._settings["save_result"]

    @property
    def output_dir(self) -> str:
        return self._settings["output_dir"]

    @property
    def exp_file(self) -> str:
        return self._settings["exp_file"]

    @property
    def weights_path(self) -> str:
        return self._settings["weights_path"]

    @property
    def device(self) -> Union[str, torch.device]:
        return self._settings["device"]

    @device.setter
    def device(self, device) -> None:
        self._settings["device"] = device

    @property
    def confidence(self) -> float:
        return self._settings["confidence"]

    @property
    def nms(self) -> float:
        return self._settings["nms"]

    @property
    def img_size(self) -> Optional[int]:
        return self._settings["img_size"]

    @property
    def fp16(self) -> bool:
        return self._settings["fp16"]

    @property
    def trt(self) -> bool:
        return self._settings["trt"]

    def __init__(self, dir: str, file_name: str) -> None:
        super().__init__(dir, file_name)


class YoloxYamlSettingsLoader(DetectionYamlSettingsLoader):
    DIR = "Config/Registration"
    FILE_NAME = "yolox"

    def __init__(self) -> None:
        super().__init__(self.DIR, self.FILE_NAME)


class ByteTrackYamlSettingsLoader(DetectionYamlSettingsLoader):
    DIR = "Config/Identification"
    FILE_NAME = "bytetrack"

    @property
    def fps(self) -> int:
        return self._settings["fps"]

    @property
    def track_thresh(self) -> float:
        return self._settings["track_thresh"]

    @property
    def track_buffer(self) -> int:
        return self._settings["track_buffer"]

    @property
    def match_thresh(self) -> float:
        return self._settings["match_thresh"]

    @property
    def aspect_ratio_thresh(self) -> float:
        return self._settings["aspect_ratio_thresh"]

    @property
    def min_box_area(self) -> int:
        return self._settings["min_box_area"]

    @property
    def mot20(self) -> bool:
        return self._settings["mot20"]

    def __init__(self) -> None:
        super().__init__(self.DIR, self.FILE_NAME)


class TransReIdYamlSettingsLoader(YamlSettingsLoader):
    DIR = "Config/Identification"
    FILE_NAME = "trans_reid"

    @property
    def tracker_dir(self) -> str:
        return self._settings["tracker_dir"]

    @property
    def model_settings_path(self) -> str:
        return self._settings["model_settings_path"]

    @property
    def weights_path(self) -> str:
        return self._settings["weights_path"]

    @property
    def device(self) -> str:
        return self._settings["device"]

    @property
    def reid_threshold(self) -> int:
        return self._settings["reid_threshold"]

    @property
    def input_size(self) -> tuple:
        return tuple(self._settings["input_size"])

    def __init__(self) -> None:
        super().__init__(self.DIR, self.FILE_NAME)


class JsonSettingsLoader(SettingsLoader):

    @property
    def settings(self) -> dict:
        return self._settings

    def __init__(self, dir: str, file_name: str) -> None:
        """Constructor

        Args:
            dir (str): 設定ファイルが格納してあるディレクトリ  
            file_name (str): 設定ファイル名 (拡張子なし)
        """
        super().__init__()
        self.__path = os.path.join(dir, file_name) + "." \
                    + self._read_extension(str(__class__.__name__))
        self.load()

    def load(self) -> None:
        """Constructor
        """
        with open(self.__path, mode="r", encoding="utf-8") as f:
            self._settings = json.load(f)


class RegisteredTrackerJsonLoader(JsonSettingsLoader):
    CHILD_DIR = "Information"
    FILE_NAME = "registration"

    def __init__(self, parent_dir: str) -> None:
        super().__init__(os.path.join(parent_dir, self.CHILD_DIR), self.FILE_NAME)
