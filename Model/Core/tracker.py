"""
@file tracker.py
@brief トラッキング対象のデータを扱う
"""
from loguru import logger
import os
import sys
from typing import Optional, Union, List

import cv2
import numpy as np

from Library.ByteTrack.yolox.tracker import matching
from Library.ByteTrack.yolox.tracker.basetrack import TrackState
from Library.ByteTrack.yolox.tracker.byte_tracker import (
    BYTETracker, STrack, joint_stracks, sub_stracks, remove_duplicate_stracks
    )
from Utility.Coordinate.bbox import BoundingBox, check_bbox


class Tracker(object):

    @property
    def img_name(self) -> str:
        return self._img_name

    @img_name.setter
    def img_name(self, img_name) -> None:
        self._img_name = img_name

    @property
    def nickname(self) -> str:
        return self._nickname

    @property
    def group(self) -> str:
        return self._group

    @property
    def bbox(self) -> list:
        return self._bbox

    @bbox.setter
    def bbox(self, bbox) -> None:
        self._bbox = bbox

    def __init__(
        self, 
        img_name: Optional[str] = None,
        nickname: Optional[str] = None,
        group: Optional[str] = None,
        bbox: Optional[list] = None
        ) -> None:

        self._nickname = nickname
        self._group = group
        self._img_name = img_name
        self._bbox = bbox


class LocalTracker(Tracker):
    """ローカル環境でテストする際に画像名から
       グループ名、ニックネームを登録する
    """

    def __init__(
        self,
        img_name: Optional[str],
        nickname: Optional[str] = None,
        group: Optional[str] = None, 
        bbox: Optional[list] = None,
        ) -> None:
        super().__init__(img_name, nickname, group, bbox)
        self.__read_tracker_info()


    def __read_tracker_info(self) -> None:
        """画像名からgroup名とnicknameを読み取る
        """
        self._group, self._nickname, _ = self.img_name.split("_")


class RegisteredLocalTracker(LocalTracker):
    """ローカル上に登録済みのTrackerを扱う
    """
    IMG_DIR = "RegisteredTracker/Image"

    @property
    def bbox(self) -> BoundingBox:
        return self.__bbox

    @property
    def tracker_img(self) -> np.ndarray:
        return self.__tracker_img

    @property
    def state(self) -> TrackState:
        return self.__state
    
    @state.setter
    def state(self, state) -> None:
        self.__state = state

    def __init__(
        self,
        img_name: str,
        nickname: str,
        group: str,
        bbox: list,
        state: TrackState = TrackState.Lost
        ) -> None:
        super().__init__(img_name, nickname, group, bbox)
        self.__state = state
        self.__bbox = BoundingBox(self._bbox)
        self.__cut_tracker_img()

    def __repr__(self) -> object:
        return f"{self._group}_{self._nickname}"

    def __cut_tracker_img(self) -> None:
        """bboxを元にTrackerを切り出す
        """
        img_path = os.path.join(self.IMG_DIR, self._img_name)
        img = cv2.imread(img_path)
        self.__tracker_img = img[
            self.__bbox.ymin: self.__bbox.ymax,
            self.__bbox.xmin: self.__bbox.xmax
            ]


class CloudTracker(Tracker):
    """クラウド環境でトラッカーを登録する際、
       UIからグループ名、ニックネームを登録する
    """

    def __init__(
        self,
        img_name: str,
        nickname: Optional[str] = None,
        group: Optional[str] = None, 
        bbox: Optional[list] = None,
        ) -> None:
        super().__init__(img_name, nickname, group, bbox)
        self.__checker(nickname, "nickname")
        self.__checker(group, "group")


    def checker(self, target: str, var_name: str) -> None:
        """対象の変数に値が格納されているか確認する

        Args:
            target (str): 確認対象の変数
            var_name (str): 変数名
        """
        if target is None:
            raise ValueError(f"[{__file__}][{sys._getframe().f_code.co_name}] \
                [{var_name}] \n 変数を入力してください")


class RegisteredCloudTracker(CloudTracker):
    """Cloud上に登録済みのTrackerを扱う
    """
    @property
    def state(self) -> TrackState:
        return self.__state

    @state.setter
    def state(self, state) -> None:
        self.__state = state

    def __init__(
        self,
        img_name: str,
        nickname: str,
        group: str,
        bbox: list,
        state: TrackState = TrackState.Lost
        ) -> None:
        super().__init__(img_name, nickname, group, bbox)
        self.__state = state


class ReIDByteTracker(BYTETracker):

    def __init__(
                self,
                args,
                model,
                frame_rate=30
                ) -> None:
        super().__init__(args, frame_rate)
        self.__model = model

    def update(
                self, output_results: np.ndarray,
                img_info: list, img_size: list,
                frame: int
                ):
        """Tracking情報を更新 (override)
           ( return前のfor loopのみを更新し、BYTETrackerのupdate関数を更新 )

        Args:
            outputs (np.ndarray): 推論結果を格納したnp配列
            img_info (list): 推論した画像サイズを格納したlist [height, width]
            img_size (list): テスト用の画像サイズ [height, width]
            frame (np.ndarray): キャプションした画像
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [ReIDSTrack(ReIDSTrack.tlbr_to_tlwh(tlbr), score, img_info) for
                        (tlbr, score) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        ReIDSTrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [ReIDSTrack(ReIDSTrack.tlbr_to_tlwh(tlbr), s, img_info) for
                        (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # 最終のReIDSTrackのインスタンスにnicknameとgoupのメンバー変数が
        # 登録されていない場合に登録
        # 切り出し範囲のwidth, heightのどちらも0でないときに限る
        for reid_strack in output_stracks:
            if not (reid_strack.nickname or reid_strack.group):
                reid_strack.cut_tracker_img(frame)
                if reid_strack.tracker_img is not None:
                    self.__update_tracker_data(reid_strack, self.__model)
            self.__update_tracker_state(self.__model.tracker_list, reid_strack)
        self.__model.init_tracker_state()

        return output_stracks

    def __update_tracker_data(self, reid_strack, model) -> None:
        """trackerとre_id ( nickname, group ) 対象の特徴量を比較し、一番スコアが高い
        ものにnicknameとgroup名を入れる

        Args:
            reid_strack (ReIDSTrack): キャプション画像から検出した人を切り出した画像
            model (TransReIDModel): TransReIDModelクラスのインスタンス

        Raise:
            ValueError: 登録されたトラッカー情報がない場合 (i.e. scoresが空) に発出
        """
        scores: Union[float, np.nan] = []
        t_feature = model.inference(reid_strack.tracker_img)
        for registered_tracker in model.tracker_list:
            if registered_tracker.state == TrackState.Tracked:
                scores.append(np.nan)
                continue
            rt_feature = model.inference(registered_tracker.tracker_img)
            score = model.get_similarity_score(t_feature, rt_feature)
            scores.append(score)
        # 特徴量 ( ユークリッド距離 ) が最小のものが一番類似度が高い
        try:
            min_score = np.nanmin(np.array(scores))
        except ValueError:
            logger.info(f"[ValueError] 登録されたトラッカー情報がない")
            min_score = None
        if (min_score is None) or (min_score > self.__model.reid_threshold):
            # logger.info(f"[min_score]: {min_score} / 特徴量の最小値が閾値より大きい")
            reid_strack.nickname = "Registered"
            reid_strack.group = "Not"
        else:
            index = scores.index(min_score)
            # logger.info(f"[min_score]: {min_score} / 特徴量の最小値が閾値未満")
            rep_tracker = model.tracker_list[index]
            reid_strack.nickname = rep_tracker.nickname
            reid_strack.group = rep_tracker.group

    def __update_tracker_state(self, 
                tracker_list: List[Union[RegisteredLocalTracker, RegisteredCloudTracker]], reid_strack
                                ) -> None:
        """登録されたtrackerのメンバー変数のstate (RegisteredLocalTracker or
        RegisteredVloudTrackerのメンバー変数) を更新する

        Args:
            tracker_list (list[Union[RegisteredLocalTracker, RegisteredVloudTracker]]): 
                                                登録したトラッカーの情報を格納したインスタンス
            reid_strack ()
        Note:
            登録する画像枚数が1人あたり複数枚になるためこちらの関数を設けた
            1枚ならば __update_tracker_data関数の最後の行に rep_tracker = TrackState.Tracked 
            とするだけでよい
        """
        for registered_tracker in tracker_list:
            if str(reid_strack) == str(registered_tracker):
                registered_tracker.state = TrackState.Tracked


class ReIDSTrack(STrack):

    @property
    def tracker_img(self) -> Optional[np.ndarray]:
        return self.__tracker_img

    @property
    def bbox(self) -> BoundingBox:
        return self.__bbox

    @property
    def nickname(self) -> Optional[str]:
        return self.__nickname

    @nickname.setter
    def nickname(self, nickname: str) -> None:
        self.__nickname = nickname

    @property
    def group(self) -> Optional[str]:
        return self.__group

    @group.setter
    def group(self, group: str) -> None:
        self.__group = group

    def __init__(
                self, 
                bbox: np.ndarray,
                score: np.float32,
                img_size: list,
                nickname: Optional[str] = None,
                group: Optional[str] = None
                ):
        """Constructor

        Args:
            bbox (np.ndarray): numpyに格納されたbbox [xmin, ymin, width, height]
            score (np.float32): 確信度
        """
        super().__init__(bbox, score)
        bbox = bbox.tolist()
        check_bbox(bbox, img_size)
        self.__bbox = BoundingBox(bbox)
        self.__nickname = nickname
        self.__group = group

    def __repr__(self):
        return f"{self.__group}_{self.__nickname}"

    def cut_tracker_img(self, img: np.ndarray) -> None:
        """bboxを元にTrackerを切り出す

        Args:
            img (np.ndarray): キャプチャーした画像データ
        """
        self.__tracker_img = None
        if (self.__bbox.height != 0 and self.__bbox.width != 0): 
            self.__tracker_img = img[
                self.__bbox.ymin: self.__bbox.ymax,
                self.__bbox.xmin: self.__bbox.xmax
                ]