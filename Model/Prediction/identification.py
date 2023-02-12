"""
@file identification.py
@brief identificationの推論関係の処理を扱う
"""
import os
import sys

import cv2
import numpy as np
import torch

from Config.target_class import TARGET_CLASSES
from Library.ByteTrack.tools.demo_track import Predictor
from Library.ByteTrack.yolox.data.data_augment import preproc
from Library.ByteTrack.yolox.exp.yolox_base import Exp
from Library.ByteTrack.yolox.tracking_utils.timer import Timer
from Library.ByteTrack.yolox.utils import postprocess
from Model.Core.tracker import ReIDByteTracker
from Model.Core.re_id import TransReIDModel
from Model.IO.load import ByteTrackYamlSettingsLoader
from Utility.Coordinate.bbox import BoundingBox


class RegisteredTrackerPredictor(Predictor):

    def __init__(self, model, exp,
                cls_names=TARGET_CLASSES,
                trt_file=None,decoder=None,
                device="cpu", fp16=False
                ):
        super().__init__(model, exp, trt_file, decoder, device, fp16)
        self.num_classes = len([cls_names])

    def predict_on_local(self, settings: ByteTrackYamlSettingsLoader, exp: Exp) -> None:
        """ローカル上で推論

        Args:
            settings (ByteTrackYamlSettingsLoader): ByteTrack関係の設定値
            exp (Exp): yoloxの基本の設定値
        """
        _, _, place = (sys._getframe().f_code.co_name).split("_")
        cap = cv2.VideoCapture(
            settings.data_path if settings.demo == "video" else settings.camid
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if settings.save_result:
            os.makedirs(settings.output_dir, exist_ok=True)
            save_path = os.path.join(settings.output_dir, os.path.basename(settings.data_path))
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, (width, height)
            )
        reid_model = TransReIDModel(place)
        tracker = ReIDByteTracker(settings, reid_model, frame_rate=30)
        timer = Timer()

        frame_id = 0
        while True:
            ret, frame = cap.read()
            timer.tic()
            if ret:
                outputs, img_info = self.inference(frame)
                if outputs[0] is not None:
                    online_targets = tracker.update(
                        outputs[0],
                        [img_info['height'], img_info['width']],
                        exp.test_size,
                        frame
                        )
                    group_nicknames, bboxes = self.__extract_info(online_targets)
                    timer.toc()
                    online_img = self.__draw_frame(
                        img_info['raw_img'], bboxes, group_nicknames, frame_id=frame_id + 1, fps=1. / timer.diff
                    )
                else:
                    timer.toc()
                    online_img = self.__draw_frame(
                        img_info['raw_img'], bboxes=[], group_nicknames=[], frame_id=frame_id + 1, fps=1. / timer.diff
                    )
                if settings.save_result:
                    writer.write(online_img)
                resized_img = cv2.resize(online_img, dsize=None, fx=1/2, fy=1/2)
                cv2.imshow("output", resized_img)
                if cv2.waitKey(10) == 5:
                    break
            else:
                break
            frame_id += 1
        cap.release()
        cv2.destroyAllWindows()

    def predict_on_cloud(self, settings: ByteTrackYamlSettingsLoader, exp: Exp) -> None:
        """クラウド上で推論

        Args:
            settings (ByteTrackYamlSettingsLoader): ByteTrack関係の設定値
            exp (Exp): yoloxの基本の設定値
        """
        pass


    def inference(self, img: np.ndarray):
        """画像内の人を推論する (override)

        Args:
            img (np.ndarray): 画像データ

        Returns:
            outputs (np.ndarray): 推論結果を格納したnp配列
            img_info (dict): 推論した画像情報を格納したdict
        """
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

    def __extract_info(self, targets: list) -> tuple:
        """target (type: ReIDStrack) の情報を抽出する

        Args:
            targets (list[ReIDStrack]): 1画像内の検出対象情報を格納したReIDStrack
                                        のインスタンスを格納したリスト
        
        Returns:
            groups_nicknames (list): 検出対象各々の"group名_nickname"を格納したリスト
            bboxes (list[BoundingBox]): 検出対象各々のbbox (type: BoundingBox) を格納したリスト
        """
        groups_nicknames = []
        bboxes = []
        # target ( type: ReIDSTrack )
        for target in targets:
            groups_nicknames.append(f"{target.group}_{target.nickname}")
            bboxes.append(BoundingBox(target.tlwh))
        return groups_nicknames, bboxes

    def __draw_frame(
        self,
        image: np.ndarray,
        bboxes: list,
        group_nicknames: list,
        frame_id=0,
        fps=0.,
        color=(0, 255, 0),
        offset: int = 10
        ):
        """bytetrack用に検出結果を描画する関数

        Args:
            image (np.ndarray): 画像データ
            bboxes (list): bbox情報
            group_nicknames (list): "group_nickname"の文字列
            frame_id (int, optional): 動画のフレーム番号 Defaults to 0.
            fps (float, optional): 動画のfps Defaults to 0..
            color (tuple, optional): 書き出す色 Defaults to (0, 255, 0).
            offset (int, optional): オフセット Defaults to 10.

        Returns:
            描画した画像データ (np.ndarray)
        """
        img = np.ascontiguousarray(np.copy(image))

        text_scale = 2
        text_thickness = 2
        line_thickness = 3

        cv2.putText(img, 'frame: %d num: %d fps: %.2f' % (frame_id, len(bboxes), fps),
                    (0, img.shape[0] - int(10 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)

        # bbox (type: BoundingBox)
        # group_nickname (type: str)
        for bbox, group_nickname in zip(bboxes, group_nicknames):
            # 登録されていないトラッカーが検出された場合は赤色(B, G, R) = (0, 0, 255)で表示
            if group_nickname == "Not_Registered":
                color = (0, 0, 255)
            cv2.rectangle(img, bbox.bbox_xymax[0:2], bbox.bbox_xymax[2:4], color=color, thickness=line_thickness)
            cv2.putText(img, group_nickname, (bbox.xmin, bbox.ymin - offset), cv2.FONT_HERSHEY_PLAIN, text_scale, color,
                        thickness=text_thickness)
            color = (0, 255, 0)
        return img