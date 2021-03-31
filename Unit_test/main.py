import unittest
from TestSD import TestStatistics
from TestIR import TestImperfectionRecognizer

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
        self.addTest(TestImperfectionRecognizer("test_comparation"))
        self.addTest(TestImperfectionRecognizer("test_binary_mask"))

if __name__ == '__main__':
    unittest.main()
