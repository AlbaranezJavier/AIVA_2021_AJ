import cv2
from abc import ABC, abstractmethod
import numpy as np

class AbstractSegmentator(ABC):

    @abstractmethod
    def do_segmentation(self, image, background):
        pass


class BackgroundSegmentator(AbstractSegmentator):

    def __init__(self, kernel):
        self.__kernel = kernel

    def do_segmentation(self, hsv, background=None):
        """
        Extracts the background of the image.
        :param hsv: image of interest
        :param background: None
        :return: background mask
        """
        background = np.ones((hsv.shape[0], hsv.shape[1]))
        _mask = cv2.threshold(hsv[..., 2], np.mean(hsv[..., 2]) - np.std(hsv[..., 2]), 255, cv2.THRESH_BINARY)[
            1]
        _contours, _ = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _max_contour = [0, 0]
        for c in range(len(_contours)):
            _temp = cv2.contourArea(_contours[c])
            if _temp > _max_contour[1]:
                _max_contour = [_contours[c], _temp]

        cv2.fillPoly(background, pts=[_max_contour[0]], color=0)
        background = cv2.erode(background, self.__kernel, iterations=2)
        background = cv2.dilate(background, self.__kernel, iterations=10)
        return background


class ImperfectionSegmentator(AbstractSegmentator):

    def __init__(self, threshold_factor, kernel):
        self.__threshold_factor = threshold_factor
        self.__kernel = kernel


    def do_segmentation(self, hsv, background):
        """
        Generates the mask of imperfections taking into account the background
        :param hsv: image of interest
        :param background: background mask
        :return: imperfections mask
        """
        # Background is not an imperfection
        _imperfections_candidates = \
        cv2.threshold(hsv[..., 2], np.mean(hsv[..., 2]) - np.std(hsv[..., 2]) / self.__threshold_factor, 255, cv2.THRESH_BINARY_INV)[1]
        _imperfections = np.logical_and(_imperfections_candidates, np.logical_not(background)).astype(np.uint8)

        # The area between contiguous imperfections is an imperfection
        _imperfections = cv2.dilate(_imperfections, self.__kernel, iterations=1)
        _imperfections = cv2.morphologyEx(_imperfections, cv2.MORPH_CLOSE, self.__kernel, iterations=3)
        _imperfections = cv2.morphologyEx(_imperfections, cv2.MORPH_OPEN, self.__kernel, iterations=3)

        return _imperfections
