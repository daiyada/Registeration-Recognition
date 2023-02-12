"""
@file bbox.py
@brief Bounding Boxを扱う
"""
class BoundingBox(object):
    @property
    def xmin(self) -> int:
        return self.__xmin

    @property
    def ymin(self) -> int:
        return self.__ymin

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def xmax(self) -> int:
        return (self.__xmin + self.__width)

    @property
    def ymax(self) -> int:
        return (self.__ymin + self.__height)

    @property
    def bbox_wh(self) -> list:
        return [self.__xmin, self.__ymin, self.__width, self.__height]

    @property
    def bbox_xymax(self) -> list:
        return [
            self.__xmin, self.__ymin, self.__xmin + self.__width,
            self.__ymin + self.__height
            ]

    def __init__(self, bbox: list) -> None:
        """Constructor

        Args:
            bbox (list): [xmin, ymin, width, height]
        """
        self.__xmin = int(bbox[0])
        self.__ymin = int(bbox[1])
        self.__width = int(bbox[2])
        self.__height = int(bbox[3])


def check_bbox(bbox: list, img_size: list) -> None:
    """bboxの構成が画像範囲に収まるっているか確認し、
       収まっていない場合は更新する

    Args:
        bbox (list): bounding box [xmin, ymin, xmax, ymax]
        img_size (list): 画像サイズ [w, h]
    """
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] > img_size[0]:
        bbox[2] = img_size[0]
    if bbox[3] < img_size[1]:
        bbox[3] = img_size[1]
