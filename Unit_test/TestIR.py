import unittest
from tools.ImperfectionRecognizer import *
from tools.LabelsManager import label2mask
import cv2


class TestImperfectionRecognizer(unittest.TestCase):

    def test_comparation(self):
        path_image = '../Samples/Test/1.png'
        path_label = '../Samples/Test/1.reg'
        threshold_factor = 10
        threshold_test = 10

        img = cv2.imread(path_image)
        mask_label = label2mask(path_label, img.shape)
        ir = ImperfectionRecognizer(threshold_factor = threshold_factor)
        ir.imperfections(img)
        ir.imperfections_mask = mask_label
        percent_label = ir.quantify_imperfections()

        _, percent_ours = ir.imperfections_and_quantification(img)

        self.assertLess(abs(percent_label - percent_ours), threshold_test)

    def test_binary_mask(self):
        path_image = '../Samples/Train/25.png'
        path_label = '../Samples/Train/25.reg'
        img = cv2.imread(path_image)

        mask_gt = np.zeros((img.shape[0], img.shape[1]))
        mask_gt[120:178, 263:299] = 1
        mask_gt[296:332, 1:481] = 1
        mask_gt[71:146, 126:164] = 1
        mask_label = label2mask(path_label, img.shape)
        result = (mask_gt == mask_label)
        self.assertEqual(result.all(), True)


if __name__ == '__main__':
    unittest.main()