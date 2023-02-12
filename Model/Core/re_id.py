"""
@file re_id.py
@brief Trans-ReIDモデルを扱うファイル
"""
import os

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from Utility.Image.manage import cv2_2_pillow
from Library.TransReID import model
from Library.TransReID.config import cfg
from Library.ByteTrack.yolox.tracker.basetrack import TrackState
from Model.IO.load import TransReIdYamlSettingsLoader, RegisteredTrackerJsonLoader
from Model.Core.tracker import RegisteredLocalTracker


class TransReIDModel(object):

    @property
    def tracker_list(self) -> list:
        return self.__tracker_list

    @property
    def reid_threshold(self) -> int:
        return self.__tidysl.reid_threshold

    def __init__(self, place: str) -> None:
        """Constructor
        """
        self.__tidysl = TransReIdYamlSettingsLoader()

        if place == "local":
            rtj = RegisteredTrackerJsonLoader(parent_dir=self.__tidysl.tracker_dir)
            self.set_tracker_on_local(rtj.settings)
        # place == "cloud"の場合
        else:
            self.set_tracker_on_cloud()

        self.__setup_model()

    def init_tracker_state(self) -> None:
        """tracker_list内のtracker (Union[RegisteredLocalTracker, RegisteredCloudTracker])
        """
        for registered_tracker in self.__tracker_list:
            registered_tracker.state = TrackState.Lost

    def set_tracker_on_local(self, tracker_info: dict) -> None:
        """ローカル上に登録したトラッカーを読み取る

        Args:
            tracker_info (dict): execute_registration_on_local.pyで登録したtracker
                                 情報を含んだdictionary
        """
        self.__tracker_list: list(RegisteredLocalTracker) = []
        for nickname in tracker_info:
            for info in tracker_info[nickname]:
                rltracker = RegisteredLocalTracker(
                    img_name=info["img_name"],
                    nickname=nickname,
                    group= info["group"],
                    bbox=info["bbox"]
                    )
                self.__tracker_list.append(rltracker)

    def set_tracker_on_cloud(self) -> None:
        """クラウド上に登録したトラッカーを読み取る
        
        Note:
            - 追加したところまでの番号を記憶しておいて、それ以降をself.__tracker_listにappend
            させるイメージ
            - かつそのtrackerのdb上のstatusがactiveであることも条件
        """
        pass


    def __setup_model(self) -> None:
        """モデルのセットアップ
        """
        cfg.merge_from_file(self.__tidysl.model_settings_path)
        cfg.freeze()

        if self.__tidysl.model_settings_path != "":
            with open(self.__tidysl.model_settings_path, 'r') as cf:
                config_str = "\n" + cf.read()

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

        self.__model = model.make_model(cfg, num_class=751, camera_num=6, view_num = 1)
        self.__model.load_param(self.__tidysl.weights_path)
        self.__model.to("cuda" if self.__tidysl.device == "gpu" else "cpu").eval()

    def inference(self, img: np.ndarray) -> torch.Tensor:
        """推論

        Args:
            img (np.ndarray): 特徴量の計算をするPILの画像データ

        Returns:
            feature (torch.Tensor): 特徴量 
        """
        ifcf = ImgForCalculatingFeature(img, self.__tidysl.input_size)
        with torch.no_grad():
            img = ifcf.img.to("cuda" if self.__tidysl.device == "gpu" else "cpu")
            feature = self.__model(img, 1, 1)
        return feature

    def __euclidean_distance(self, qf: Image, gf: Image):
        """ユークリッド距離を算出

        Args:
            qf (Image): 比較対象1の画像データ
            gf (Image): 比較対象2の画像データ

        Returns:
            dist_mat.cpu().numpy() (np.ndarray): ユークリッド距離
        """
        m = qf.shape[0]
        n = gf.shape[0]
        dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_mat.addmm_(1, -2, qf, gf.t())
        return dist_mat.cpu().numpy()

    def get_similarity_score(self, feature_1, feature_2) -> np.float32:
        """_summary_

        Args:
            feature_1 (torch.Tensor): 特徴量1
            feature_2 (torch.Tensor): 特徴量2

        Returns:
            np.float32: ユークリッド距離
        """
        return self.__euclidean_distance(feature_1, feature_2)[0][0]
        

class ImgForCalculatingFeature:
    @property
    def img(self) -> Image:
        return self.__img

    def __init__(self , img: np.ndarray, input_size: tuple):
        """Constructor

        Args:
            img (np.ndarray): 画像データ
            input_size(tuple): モデルへのinput_size
        """
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean= [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5] )
        ])
        resized_img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
        img_pil = cv2_2_pillow(resized_img)
        processed_image = transforms(img_pil)
        self.__img = processed_image.unsqueeze(0)