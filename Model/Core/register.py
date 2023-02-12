"""
@file register.py
@brief 登録関係の処理を行う
"""
from abc import ABCMeta, abstractmethod
import json
import os

from Model.Core.tracker import Tracker

class Registration(metaclass=ABCMeta):

    @abstractmethod
    def register(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass
    

class JsonRegistration(Registration):
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.__tracker_info = {}
        self.__json_dir = os.path.join(output_dir, "Information")
        os.makedirs(self.__json_dir, exist_ok=True)

    def register(self, tracker: Tracker) -> None:
        """trackerの情報をjsonに登録する

        Args:
            tracker (Tracker): Trackerクラスのインスタンス
        """
        detailed_info = {}
        detailed_info["img_name"] = tracker.img_name
        detailed_info["group"] = tracker.group
        detailed_info["bbox"] = tracker.bbox
        if tracker.nickname not in self.__tracker_info:
            self.__tracker_info[f"{tracker.nickname}"] = []
        self.__tracker_info[f"{tracker.nickname}"].append(detailed_info)

    def save(self) -> None:
        """登録したトラッカー情報をjsonファイルとして保存
        """
        save_path = os.path.join(
                    self.__json_dir,
                    "registration.json"
                    )
        with open(save_path, "w") as f:
            json.dump(self.__tracker_info, f, indent=4)
            

class DBRegistration(Registration):
    def __init__(self) -> None:
        super().__init__()

    def register(self) -> None:
        pass

    def save(self) -> None:
        pass
