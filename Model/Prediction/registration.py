"""
@file registration.py
@brief rgistrationの推論関係の処理を扱う
"""
import os

import cv2
import numpy as np
import torch, torchvision

from Config.target_class import TARGET_CLASSES, COLORS
from Library.ByteTrack.tools.demo_track import Predictor
from Model.Core.tracker import Tracker
from Utility.Coordinate.bbox import BoundingBox


class TrackerPredictor(Predictor):

    @property
    def tracker(self) -> Tracker:
        return self.__tracker

    def __init__(self, model, exp, cls_names=TARGET_CLASSES, 
                trt_file=None, decoder=None, device="cpu", 
                fp16=False, legacy=False
                ):
        super().__init__(model, exp, trt_file, 
                        decoder, device, fp16)
        # 検出はPersonクラスのみとする
        self.num_classes = len([cls_names])
        self.__tracker = None

    def predict(self, img_path: str , tracker: Tracker) -> None:
        outputs, img_info = self.inference(img_path)
        self.__rt = RepresentativeTarget(outputs, img_info)
        self.__tracker = tracker
        self.__tracker.bbox = self.__rt.bbox.bbox_wh

    def inference(self, img) -> tuple:
        """推論 (override)

        Args:
            img (np.ndarray): 推論対象の画像データ

        Returns:
            outputs (list[tensor.Tensor]) : 推論結果を格納したリスト
            img_info (dict) : 画像情報を格納したdict
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

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = self.postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        """画像の前処理 (override)
           YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)の
           YOLOX/yolox/data/data_augment.py 142行目を参考

        Args:
            img (np.ndarray): 前処理をかける画像データ
            input_size (list): モデルへの入力サイズ
            swap (tuple, optional): スワップ  初期値: (2, 0, 1)

        Returns:
            padded_img (np.ndarray): 前処理後の画像データ
            r (float): 縮尺割合 
        """
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False) -> torch:
        """推論結果の後処理 (override)
            YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)の
            YOLOX/yolox/utils/boxes.py 32行目を参考

        Args:
            prediction (list[tensor.Tensor]) : 推論結果を格納したリスト
            num_classes (int): 検出するクラス数
            conf_thre (float, optional): 確信度の閾値 初期値: 0.7.
            nms_thre (float, optional): non maximum supressionの閾値 初期値: 0.45.
            class_agnostic (bool, optional): _description_. Defaults to False.

        Returns:
            output (torch): 後処理後の推論結果
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output

    def save(self, output_dir: str, img_path: str, ext: str = "png") -> None:
        """重畳画像の保存

        Args:
            output_dir (str): 保存先のディレクトリパス
            img_path (str): 重畳する元の画像のパス
            ext (str): 重畳画像の拡張子 初期値は"png"
        """
        save_dir = os.path.join(output_dir, "ResultImg")
        os.makedirs(save_dir, exist_ok=True)
        img = cv2.imread(img_path)
        result_img = self.__visualize(
            img, self.__rt.bbox, score=self.__rt.score, cls_id=self.__rt.cls_id,
            conf_thresh=self.confthre, cls_name=TARGET_CLASSES
            )
        save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.{ext}")
        cv2.imwrite(save_path, result_img)

    def __visualize(self, img: np.ndarray, bbox: BoundingBox, score: float, cls_id: int,
                    conf_thresh: float, cls_name: tuple
                    ) -> None:
        if score < conf_thresh:
            return img

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(cls_name[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.width, bbox.height), color, 2)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (bbox.xmin, bbox.ymin + 1),
            (bbox.xmin + txt_size[0] + 1, bbox.ymin + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (bbox.xmin, bbox.ymin + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img


class RepresentativeTarget(object):
    @property
    def score(self) -> float:
        return self.__score

    @property
    def bbox(self) -> BoundingBox:
        return self.__bbox

    @property
    def cls_id(self) -> int:
        return self.__cls_id


    def __init__(self, result: torch.Tensor, img_info: dict) -> None:
        result = result[0]
        scores = result[:, 4] * result[:, 5]
        repr_id = self.__decide_representative_id(scores)

        self.__score = float(scores[repr_id])
        self.__bbox = self.__read_bbox(result[repr_id], img_info["ratio"])
        self.__cls_id = int(result[repr_id, 6])

    def __decide_representative_id(self, scores: torch.Tensor) -> int:
        """scoreが一番高いidを返す

        Args:
            scores (torch.Tensor): 複数のオブジェクトの確信度を収めたTensor

        Returns:
            int: repr_id (id)
        """
        scores = scores.tolist()
        repr_id = scores.index(max(scores))
        return repr_id

    def __read_bbox(self, output: torch.Tensor, ratio: float) -> BoundingBox:
        """推論結果からbbox ( = [x=min, y_min, w, h] )を読み取る

        Args:
            output (torch.Tensor): 1つのオブジェクトの推論結果を収めたTensor
            ratio (float): resize率
        Returns:
            BoundinbBox: 推論結果を格納した、Boundingboxクラスのインスタンス
        """
        bbox_list = output[0:4] / ratio
        bbox = BoundingBox(bbox_list)
        return bbox

