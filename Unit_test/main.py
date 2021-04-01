import unittest
from Unit_test.TestSD import TestStatistics
from Unit_test.TestIR import TestImperfectionRecognizer

"""
This script executes a set of tests
"""

class SuiteTest(unittest.TestSuite):

    def run(self):
        """
        Run the TestStatistics and TestImperfectionRecognizer tests.
        :return: results
        """
        self.addTest(TestStatistics("test_statistics"))
        self.addTest(TestImperfectionRecognizer("test_prediction_vs_gt"))
        self.addTest(TestImperfectionRecognizer("test_binary_mask"))

if __name__ == '__main__':
    unittest.main()
