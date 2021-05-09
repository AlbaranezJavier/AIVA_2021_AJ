import cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from tools.ImperfectionRecognizer import ImperfectionRecognizer
from tools.Metrics import Metrics
from tools.LabelsManager import label2mask

def calculate_statistics():
    # Variables
    paths = glob.glob(f'./Samples/Test/*.png')
    ir = ImperfectionRecognizer(threshold_factor=2)
    s = Metrics(p=0.01)
    type = "cumulative"

    # Load image
    for path in paths:
        img = cv2.imread(path)
        path_gt = path.split('.')
        path_gt = '.' + path_gt[1] + '.reg'

        # Get imperfections mask and quantification
        mask, percent = ir.imperfections_and_quantification(img)
        mask_gt = label2mask(path_gt, img.shape)

        # s.cal_basic_stats(mask, mask_gt)
        s.cal_basic_stats_modified(mask, mask_gt, path_gt)
        s.update_cumulative_stats()

    s.cal_complex_stats(type)
    s.print_table(type)

calculate_statistics()