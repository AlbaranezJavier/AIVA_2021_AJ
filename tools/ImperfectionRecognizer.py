import cv2
import numpy as np
from tools.Analyzer import MaskQuantificator
from tools.Segmentator import BackgroundSegmentator, ImperfectionSegmentator


class ImperfectionRecognizer:
    """
    Controls the whole process of segmentation and quantification of the detected imperfections.
    """

    def __init__(self, threshold_factor=20,
                 kernel_background=np.array([[0, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 1, 0, 0],
                                             [0, 0, 1, 1, 0, 0],
                                             [0, 0, 1, 1, 0, 0],
                                             [0, 0, 1, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0]], dtype='uint8'),
                 kernel_imperfections=np.array([[0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0],
                                                [0, 1, 1, 1, 1, 0],
                                                [0, 1, 1, 1, 1, 0],
                                                [0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0]], dtype='uint8')
                 ):
        self.__bs = BackgroundSegmentator(kernel_background)
        self.__is = ImperfectionSegmentator(threshold_factor, kernel_imperfections)
        self.__mc = MaskQuantificator()
        self.imperfections_mask = None
        self.background_mask = None

    def imperfections(self, image):
        """
        Generates background masks and imperfections.
        :param image: bgr image
        :return: None
        """
        # From BGR to HSV
        _hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Background
        self.background_mask = self.__bs.do_segmentation(_hsv)

        # Imperfections
        self.imperfections_mask = self.__is.do_segmentation(_hsv, self.background_mask)

    def quantify_imperfections(self):
        """
        Quantifies mask imperfections.
        :return: percentage: float
        """
        if self.imperfections_mask is None or self.background_mask is None:
            raise ValueError('Detect imperfections first with "imperfections()"')
        return self.__mc.do_quantification(self.imperfections_mask, self.background_mask)

    def imperfections_and_quantification(self, image):
        """
        Executes segmentation and quantification and returns values.
        :param image: bgr image
        :return: imperfections_mask: image, imperfections_quantification: float
        """
        self.imperfections(image)
        return self.imperfections_mask, self.quantify_imperfections()
