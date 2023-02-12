"""
@file controller.py
@brief 登録や検出を制御するコントローラ
"""
from abc import ABCMeta, abstractmethod
from typing import Type
from loguru import logger
import os
import shutil
import sys

import torch

sys.path.append("../")
from Library.ByteTrack.yolox.exp import get_exp
from Library.ByteTrack.yolox.exp.yolox_base import Exp
from Library.ByteTrack.yolox.utils import get_model_info
from Model.Core.register import JsonRegistration
from Model.Core.tracker import LocalTracker, CloudTracker
from Model.IO.load import ByteTrackYamlSettingsLoader, YoloxYamlSettingsLoader
from Model.Prediction.registration import TrackerPredictor
from Model.Prediction.identification import RegisteredTrackerPredictor
from Utility.File.file import get_file_path_list

class Controller(metaclass=ABCMeta):

    def _set_variable(self, settings: dict, exp: Exp) -> None:
        """パラメータのセット
        """
        if not settings.experiment_name:
            settings.experiment_name = exp.exp_name

        if settings.trt:
            settings.device = "gpu"
        settings.device = torch.device("cuda" if settings.device == "gpu" else "cpu")

        if settings.confidence is not None:
            exp.test_conf = settings.confidence
        if settings.nms is not None:
            exp.nmsthre = settings.nms
        if settings.img_size is not None:
            exp.test_size = (settings.img_size, settings.img_size)

        self._model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self._model, exp.test_size)))

        self._model = exp.get_model().to(settings.device)
        if settings.fp16:
            self._model.half()  # to FP16
        self._model.eval()

        if not settings.trt:
            if settings.weights_path is None:
                file_name = os.path.join(exp.output_dir, settings.experiment_name)
                weights_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                weights_file = settings.weights_path
            logger.info("loading checkpoint")
            weights = torch.load(weights_file, map_location="cpu")
            # load the model state dict
            self._model.load_state_dict(weights["model"])
            logger.info("loaded checkpoint done.")

    @abstractmethod
    def predict_on_local(self) -> None:
        """ローカルで推論
        """
        pass

    @abstractmethod
    def predict_on_cloud(self) -> None:
        """クラウド上で推論
        """

class RegistrationController(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.__ysl = YoloxYamlSettingsLoader()
        self.__exp = get_exp(self.__ysl.exp_file, self.__ysl.name)
        self._set_variable(self.__ysl, self.__exp)
        self.__predictor = TrackerPredictor(
            model=self._model, exp=self.__exp,
            device=self.__ysl.device, fp16=self.__ysl.fp16
        )

    def predict_on_local(self, copy: bool = True) -> None:
        """ローカル環境に保存してある画像でprediction

        Args:
            copy (bool) :   元画像群をtracker情報などを出力するフォルダに
                            コピーするか否か
        """
        img_list = get_file_path_list(self.__ysl.data_path, recursive=True)
        jr = JsonRegistration(self.__ysl.output_dir)
        for img_path in img_list:
            try:
                tracker = LocalTracker(os.path.basename(img_path))
                self.__predictor.predict(img_path, tracker)
                if self.__ysl.save_result:
                    self.__predictor.save(self.__ysl.output_dir, img_path)
                jr.register(tracker)
            except TypeError:
                logger.info(f"[TypeError] / {os.path.basename(img_path)}: 未検出画像")
        jr.save()
        if copy:
            copy_dir = os.path.join(self.__ysl.output_dir, "Image")
            if os.path.isdir(copy_dir):
                shutil.rmtree(copy_dir)
            shutil.copytree(
                self.__ysl.data_path, os.path.join(self.__ysl.output_dir, "Image")
            )

    def predict_on_cloud(self, img_path, nickname: str, group: str):
        """クラウド上の画像でprediction

        Args:
            img_path (str): 画像データ
            nickname (str): ニックネーム
            group (str): グループ名
        """
        tracker = CloudTracker(os.path.basename(img_path))
        self.__predictor.predict(img_path, tracker)


class IdentificationController(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.__isl = ByteTrackYamlSettingsLoader()
        self.__exp = get_exp(self.__isl.exp_file, self.__isl.name)
        self._set_variable(self.__isl, self.__exp)
        self.__predictor = RegisteredTrackerPredictor(
            model=self._model, exp=self.__exp,
            device=self.__isl.device, fp16=self.__isl.fp16            
        )

    def predict_on_local(self) -> None:
        """ローカルで登録人物の検出
        """
        func = getattr(self.__predictor, sys._getframe().f_code.co_name)
        # RegisteredTrackerPredictor の predict_on_local 関数の実行
        func(self.__isl, self.__exp)

    def predict_on_cloud(self) -> None:
        """クラウド上で登録人物の検出
        """
        pass