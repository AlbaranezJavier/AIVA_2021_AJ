import unittest
from tools.StatisticalData import Statistics

"""
Checks that the results obtained in statistical data are correct.
"""

class TestStatistics(unittest.TestCase):
    def test_statistics(self):
        """
        This test checks that the numbers calculated for each parameter are correct.
        :return:
        """
        tocheck = {
            "population": 280, "pp": 60, "bias": 21.43, "pn": 220, "ib": 78.57, "rp": 70, "tp": 50, "recall": 71.43, "fn": 20, "fnr": 28.57,
            "lr+": 15.01, "prevalence": 25.0, "precision": 83.33, "performance": 17.86, "fna": 9.09, "irr": 7.14, "lr-": 0.3, "rn": 210, "fp": 10,
            "fpr": 4.76, "tn": 200, "specifity": 95.24, "dor": 50.03, "ner": 75.0, "fdr": 16.67, "der": 3.57, "ip": 90.91, "crr": 71.43,
            "informedness": 66.67, "chi": 139.8, "correlation": 0.73, "pra": 64.0, "markedness": 74.24, "accuracy": 89.29, "mcc": 0.71, "iou": 62.0,
            "ck": 0.7, "mr": 11.0, "f1": 0.77, "p": 0.0
        }
        s = Statistics(p=0.01)
        s.test(50, 20, 10, 200)
        s.others()

        for check in list(tocheck.keys()):
            msg = f'The test failed in {check}'
            with self.subTest():
                self.assertEqual(s.mydict[check], tocheck[check], msg)


if __name__ == '__main__':
    unittest.main()
