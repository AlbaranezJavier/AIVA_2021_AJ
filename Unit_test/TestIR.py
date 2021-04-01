import unittest, cv2
from tools.ImperfectionRecognizer import *
from tools.LabelsManager import label2mask
from tools.StatisticalData import Statistics
import matplotlib.pyplot as plt


class TestImperfectionRecognizer(unittest.TestCase):

    def test_prediction_vs_gt(self):
        """
        Load an example and its groundthruth and display the statistical data.
        If verbose = 0, no print nothing
        If verbose = 1, print statistical table
        If verbose = 2, print predicted mask and gt mask adn statistical table
        :return:check if the difference is not greater than a threshold.
        """
        # Variables
        _verbose = 2
        _example = 1
        _sd = Statistics(p=0.01)
        _threshold_factor = 0.99
        _threshold_test = 23
        _ir = ImperfectionRecognizer(threshold_factor=_threshold_factor)
        _path_example = f'../Samples/Test/{_example}.png'
        _path_gt = f'../Samples/Test/{_example}.reg'

        # Load example
        _example = cv2.imread(_path_example)

        # Get groundthruth
        _mask_gt = label2mask(_path_gt, _example.shape)
        _ir.imperfections(_example)
        _ir.imperfections_mask = _mask_gt
        _percent_gt = _ir.quantify_imperfections()

        # Get prediction
        _mask_predicted, _percent_predicted = _ir.imperfections_and_quantification(_example)

        if _verbose > 0:
            # Show statistics
            _sd.tp_fn_fp_tn(_mask_predicted, _mask_gt)
            _sd.others()
            _sd.print_table()
            if _verbose > 1:
                # Show masks
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(_mask_predicted)
                axs[0].set_title("Predicted")
                axs[1].imshow(_mask_gt)
                axs[1].set_title("Gt")
                fig.suptitle(f'{_path_example}')
                plt.show()


        # Make test
        self.assertLess(abs(_percent_gt - _percent_predicted), _threshold_test)

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